/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "core/layers/npu/rotary_embedding.h"
#include "framework/model_loader.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/qwen3.h"
#include "models/model_registry.h"
#include "util/tensor_helper.h"

namespace xllm::npu::model {

class DFlashQwen3ModelImpl : public QWen3ModelImpl {
 public:
  explicit DFlashQwen3ModelImpl(const ModelContext& context)
      : QWen3ModelImpl(context) {
    const ModelArgs& model_args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    const int32_t dp_size = parallel_args.dp_size();
    const int32_t cp_size = parallel_args.cp_size();
    CHECK_GT(dp_size, 0) << "DFlash dp_size must be positive.";
    CHECK_GT(cp_size, 0) << "DFlash cp_size must be positive.";
    CHECK_EQ(parallel_args.world_size() % (dp_size * cp_size), 0)
        << "DFlash world_size must be divisible by dp_size * cp_size.";
    tp_size_ = parallel_args.world_size() / (dp_size * cp_size);
    CHECK_GT(tp_size_, 0) << "DFlash tp_size must be positive.";
    tp_rank_ = parallel_args.rank() % tp_size_;
    head_dim_ = model_args.head_dim();
    rms_norm_eps_ = model_args.rms_norm_eps();
    tensor_options_ = context.get_tensor_options();
    rotary_embedding_ = std::make_shared<RotaryEmbeddingGeneric>(
        head_dim_,
        model_args.max_position_embeddings(),
        layer::rotary::compute_inv_freq(
            head_dim_, model_args.rope_theta(), tensor_options_),
        /*interleaved=*/false,
        tensor_options_);
    const int64_t fc_input_size =
        static_cast<int64_t>(model_args.layers_to_capture().size()) *
        model_args.hidden_size();
    CHECK_GT(fc_input_size, 0)
        << "DFlash requires dflash_config.target_layer_ids.";
    fc_ =
        register_module("fc",
                        layer::ReplicatedLinear(fc_input_size,
                                                model_args.hidden_size(),
                                                /*bias=*/false,
                                                QuantArgs(),
                                                context.get_tensor_options()));
    hidden_norm_ =
        register_module("hidden_norm",
                        layer::RMSNorm(model_args.hidden_size(),
                                       model_args.rms_norm_eps(),
                                       context.get_tensor_options()));
  }

  ModelOutput forward(torch::Tensor tokens,
                      torch::Tensor positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) override {
    torch::Tensor input_embedding = input_params.embedding.input_embedding;
    if (input_embedding.defined()) {
      CHECK(input_params.embedding.dflash_context_hidden)
          << "DFlashDraftModel only accepts target hidden as context K/V "
             "input.";
      return materialize_context_kv(
          input_embedding, positions, kv_caches, input_params);
    }

    return QWen3ModelImpl::forward(tokens, positions, kv_caches, input_params);
  }

  void load_state_dict(const StateDict& state_dict) override {
    if (state_dict.has("fc.weight")) {
      fc_loaded_ = true;
    }
    if (state_dict.has("hidden_norm.weight")) {
      hidden_norm_loaded_ = true;
    }

    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    load_context_kv_weights(state_dict);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    CHECK(fc_loaded_) << "Failed to find DFlash draft fc weight.";
    CHECK(hidden_norm_loaded_)
        << "Failed to find DFlash draft hidden_norm weight.";
    verify_context_kv_weights();
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() override {
    for (QWen3DecoderLayer& layer : layers_) {
      layer->merge_loaded_weights();
    }
    build_fused_context_kv_weight();
    norm_->merge_loaded_weights();
  }

 protected:
  torch::Tensor gen_append_attn_mask(int32_t q_len,
                                     int32_t kv_len,
                                     int32_t max_kv_len,
                                     torch::Dtype dtype,
                                     torch::Device device) override {
    torch::Tensor non_causal_mask = attn_mask_.gen_append_mask(
        /*q_len=*/1, kv_len, max_kv_len, dtype, device);
    return non_causal_mask.repeat({q_len, 1});
  }

 private:
  void load_context_kv_weights(const StateDict& state_dict) {
    if (k_proj_weights_.empty()) {
      const int32_t num_layers = static_cast<int32_t>(layers_.size());
      k_proj_weights_.resize(num_layers);
      v_proj_weights_.resize(num_layers);
      k_norm_weights_.resize(num_layers);
    }

    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      StateDict layer_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      torch::Tensor k_proj_weight = layer_dict.get_sharded_tensor(
          "self_attn.k_proj.weight", /*dim=*/0, tp_rank_, tp_size_);
      torch::Tensor v_proj_weight = layer_dict.get_sharded_tensor(
          "self_attn.v_proj.weight", /*dim=*/0, tp_rank_, tp_size_);
      torch::Tensor k_norm_weight =
          layer_dict.get_tensor("self_attn.k_norm.weight");

      if (k_proj_weight.defined()) {
        k_proj_weights_[i] = k_proj_weight.to(tensor_options_);
      }
      if (v_proj_weight.defined()) {
        v_proj_weights_[i] = v_proj_weight.to(tensor_options_);
      }
      if (k_norm_weight.defined()) {
        k_norm_weights_[i] =
            k_norm_weight.to(tensor_options_).to(torch::kFloat32);
      }
    }
  }

  void verify_context_kv_weights() const {
    CHECK_EQ(k_proj_weights_.size(), layers_.size());
    CHECK_EQ(v_proj_weights_.size(), layers_.size());
    CHECK_EQ(k_norm_weights_.size(), layers_.size());
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      CHECK(k_proj_weights_[i].defined())
          << "Failed to find DFlash draft layers." << i
          << ".self_attn.k_proj.weight.";
      CHECK(v_proj_weights_[i].defined())
          << "Failed to find DFlash draft layers." << i
          << ".self_attn.v_proj.weight.";
      CHECK(k_norm_weights_[i].defined())
          << "Failed to find DFlash draft layers." << i
          << ".self_attn.k_norm.weight.";
    }
  }

  void build_fused_context_kv_weight() {
    std::vector<torch::Tensor> kv_weights;
    kv_weights.reserve(layers_.size() * 2);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      CHECK_EQ(k_proj_weights_[i].dim(), 2)
          << "DFlash k_proj weight must be 2D.";
      CHECK_EQ(v_proj_weights_[i].dim(), 2)
          << "DFlash v_proj weight must be 2D.";
      CHECK_EQ(k_proj_weights_[i].size(0), v_proj_weights_[i].size(0))
          << "DFlash k/v projection output size mismatch.";
      CHECK_EQ(k_proj_weights_[i].size(1), v_proj_weights_[i].size(1))
          << "DFlash k/v projection weight shape mismatch.";
      CHECK_EQ(k_proj_weights_[i].size(0) % head_dim_, 0)
          << "DFlash k_proj output size must align to head_dim.";
      const int64_t layer_local_kv_heads =
          k_proj_weights_[i].size(0) / head_dim_;
      if (i == 0) {
        local_kv_heads_ = layer_local_kv_heads;
      } else {
        CHECK_EQ(local_kv_heads_, layer_local_kv_heads)
            << "DFlash local KV heads mismatch.";
      }
      kv_weights.emplace_back(k_proj_weights_[i]);
      kv_weights.emplace_back(v_proj_weights_[i]);
    }
    fused_kv_weight_ = torch::cat(kv_weights, /*dim=*/0).contiguous();
    k_proj_weights_.clear();
    v_proj_weights_.clear();
  }

  torch::Tensor apply_k_norm(const torch::Tensor& key,
                             const torch::Tensor& weight) const {
    torch::Tensor key_fp32 = key.to(torch::kFloat32);
    torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
    torch::Tensor normed_key =
        key_fp32 * torch::rsqrt(variance + rms_norm_eps_);
    return (normed_key * weight).to(key.scalar_type());
  }

  torch::Tensor apply_rope(const torch::Tensor& key,
                           const torch::Tensor& positions) const {
    CHECK(rotary_embedding_ != nullptr)
        << "DFlash rotary embedding is not initialized.";
    return std::get<1>(rotary_embedding_->forward(key, key, positions));
  }

  void write_cache_slots(KVCache& kv_cache,
                         const torch::Tensor& key,
                         const torch::Tensor& value,
                         const torch::Tensor& block_indices,
                         const torch::Tensor& block_offsets) const {
    torch::Tensor k_cache = kv_cache.get_k_cache();
    torch::Tensor v_cache = kv_cache.get_v_cache();
    CHECK_EQ(k_cache.dim(), 4) << "DFlash requires dense Qwen3 KV cache.";
    CHECK_EQ(v_cache.dim(), 4) << "DFlash requires dense Qwen3 KV cache.";
    CHECK_EQ(block_indices.numel(), key.size(0));
    CHECK_EQ(block_offsets.numel(), key.size(0));
    CHECK_EQ(value.size(0), key.size(0));

    k_cache.index_put_({block_indices,
                        block_offsets,
                        torch::indexing::Slice(),
                        torch::indexing::Slice()},
                       key);
    v_cache.index_put_({block_indices,
                        block_offsets,
                        torch::indexing::Slice(),
                        torch::indexing::Slice()},
                       value);
  }

  void build_cache_slot_tensors(KVCache& kv_cache,
                                const std::vector<int32_t>& new_cache_slots,
                                torch::Tensor& block_indices,
                                torch::Tensor& block_offsets) const {
    torch::Tensor k_cache = kv_cache.get_k_cache();
    CHECK_EQ(k_cache.dim(), 4) << "DFlash requires dense Qwen3 KV cache.";
    const int64_t block_size = k_cache.size(1);
    std::vector<int64_t> block_indices_vec;
    std::vector<int64_t> block_offsets_vec;
    block_indices_vec.reserve(new_cache_slots.size());
    block_offsets_vec.reserve(new_cache_slots.size());
    for (int32_t slot : new_cache_slots) {
      CHECK_GE(slot, 0);
      block_indices_vec.emplace_back(slot / block_size);
      block_offsets_vec.emplace_back(slot % block_size);
    }

    torch::TensorOptions index_options =
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
    torch::TensorOptions device_index_options =
        torch::TensorOptions().dtype(torch::kLong).device(k_cache.device());
    block_indices = safe_to(torch::tensor(block_indices_vec, index_options),
                            device_index_options,
                            /*non_blocking=*/true);
    block_offsets = safe_to(torch::tensor(block_offsets_vec, index_options),
                            device_index_options,
                            /*non_blocking=*/true);
  }

  ModelOutput materialize_context_kv(const torch::Tensor& target_hidden,
                                     const torch::Tensor& positions,
                                     std::vector<KVCache>& kv_caches,
                                     const ModelInputParams& input_params) {
    CHECK_EQ(kv_caches.size(), layers_.size());
    CHECK_EQ(input_params.attention.host.new_cache_slots.size(),
             static_cast<size_t>(target_hidden.size(0)));
    CHECK(!kv_caches.empty()) << "DFlash requires non-empty KV cache.";

    torch::Tensor block_indices;
    torch::Tensor block_offsets;
    build_cache_slot_tensors(kv_caches.front(),
                             input_params.attention.host.new_cache_slots,
                             block_indices,
                             block_offsets);

    torch::Tensor projected_hidden = fc_(target_hidden);
    projected_hidden = std::get<0>(hidden_norm_->forward(projected_hidden));
    CHECK(fused_kv_weight_.defined())
        << "DFlash fused K/V weight is not initialized.";
    CHECK_GT(local_kv_heads_, 0) << "DFlash local KV heads is invalid.";

    const int64_t num_layers = static_cast<int64_t>(layers_.size());
    const int64_t num_context = projected_hidden.size(0);
    namespace F = torch::nn::functional;
    torch::Tensor all_kv = F::linear(projected_hidden, fused_kv_weight_);
    all_kv =
        all_kv.view({num_context, num_layers, 2, local_kv_heads_, head_dim_})
            .permute({2, 1, 0, 3, 4})
            .contiguous();

    torch::Tensor all_key = all_kv.select(/*dim=*/0, /*index=*/0);
    torch::Tensor all_value = all_kv.select(/*dim=*/0, /*index=*/1);
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      torch::Tensor key = all_key.select(/*dim=*/0, /*index=*/i);
      torch::Tensor value = all_value.select(/*dim=*/0, /*index=*/i);
      key = apply_rope(apply_k_norm(key, k_norm_weights_[i]), positions);
      write_cache_slots(kv_caches[i], key, value, block_indices, block_offsets);
    }

    return ModelOutput(projected_hidden);
  }

  layer::ReplicatedLinear fc_{nullptr};
  layer::RMSNorm hidden_norm_{nullptr};
  std::vector<torch::Tensor> k_proj_weights_;
  std::vector<torch::Tensor> v_proj_weights_;
  std::vector<torch::Tensor> k_norm_weights_;
  torch::Tensor fused_kv_weight_;
  torch::TensorOptions tensor_options_;
  std::shared_ptr<NpuRotaryEmbedding> rotary_embedding_;
  int32_t tp_rank_ = 0;
  int32_t tp_size_ = 1;
  int64_t head_dim_ = 0;
  int64_t local_kv_heads_ = 0;
  double rms_norm_eps_ = 0.0;
  bool fc_loaded_ = false;
  bool hidden_norm_loaded_ = false;
};
TORCH_MODULE(DFlashQwen3Model);

class DFlashQwen3ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DFlashQwen3Model> {
 public:
  explicit DFlashQwen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DFlashQwen3Model>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      CHECK(!state_dict->has("d2t") &&
            !state_dict->has("draft_id_to_target_id"))
          << "xLLM DFlash currently shares the target lm_head and does not "
             "support d2t vocabulary remapping.";
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      CHECK(!sub_dict.has("d2t") && !sub_dict.has("draft_id_to_target_id"))
          << "xLLM DFlash currently shares the target lm_head and does not "
             "support d2t vocabulary remapping.";
      model_->load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
  }
};
TORCH_MODULE(DFlashQwen3ForCausalLM);

REGISTER_CAUSAL_MODEL_WITH_VARNAME(dflash_draft_model,
                                   DFlashDraftModel,
                                   DFlashQwen3ForCausalLM);

REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(
    dflash_draft_model,
    DFlashDraftModel,
    [](const JsonReader& json, ModelArgs* args) {
      ModelArgsLoader loader =
          ModelRegistry::get_model_args_loader("qwen3_atb");
      CHECK(loader != nullptr)
          << "qwen3_atb args loader must be registered before "
             "DFlashDraftModel.";
      CHECK(loader(json, args)) << "Failed to load DFlashDraftModel args.";
      const std::optional<int64_t> draft_vocab_size =
          json.value<int64_t>("draft_vocab_size");
      if (draft_vocab_size.has_value() && draft_vocab_size.value() > 0) {
        CHECK_EQ(draft_vocab_size.value(), args->vocab_size())
            << "xLLM DFlash currently shares the target lm_head and does not "
               "support reduced draft vocabulary.";
        args->draft_vocab_size(draft_vocab_size.value());
      }
      std::optional<std::vector<int32_t>> target_layer_ids =
          json.value<std::vector<int32_t>>("dflash_config.target_layer_ids");
      CHECK(target_layer_ids.has_value())
          << "DFlashDraftModel requires dflash_config.target_layer_ids.";
      args->model_type("DFlashDraftModel");
      args->layers_to_capture(target_layer_ids.value());
      return true;
    });

}  // namespace xllm::npu::model
