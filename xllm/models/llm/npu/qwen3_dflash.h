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
#include <string>
#include <vector>

#include "core/layers/npu/npu_column_parallel_linear_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "framework/model_loader.h"
#include "models/llm/draft_context_kv.h"
#include "models/llm/npu/qwen3.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

class DFlashQwen3ModelImpl : public QWen3ModelImpl {
 public:
  explicit DFlashQwen3ModelImpl(const ModelContext& context)
      : QWen3ModelImpl(context) {
    ::xllm::init_context_kv_state(kv_state_, context);
    fc_ = register_module("fc", layer::NpuColumnParallelLinear(context));
    hidden_norm_ = register_module("hidden_norm", layer::NpuRMSNorm(context));
  }

  void load_state_dict(const StateDict& state_dict) override {
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    ::xllm::load_context_kv_shard(
        kv_state_, state_dict, static_cast<int32_t>(layers_.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  void verify_loaded_weights(const std::string& prefix) const override {
    fc_->verify_loaded_weights(prefix + "fc.");
    hidden_norm_->verify_loaded_weights(prefix + "hidden_norm.");
    ::xllm::verify_context_kv_shards(kv_state_,
                                     static_cast<int32_t>(layers_.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    norm_->verify_loaded_weights(prefix + "norm.");
  }

  void merge_loaded_weights() override {
    fc_->merge_loaded_weights();
    hidden_norm_->merge_loaded_weights();
    ::xllm::build_fused_context_kv(kv_state_,
                                   static_cast<int32_t>(layers_.size()));
    for (QWen3DecoderLayer& layer : layers_) {
      layer->merge_loaded_weights();
    }
    norm_->merge_loaded_weights();
  }

 protected:
  torch::Tensor gen_append_attn_mask(int32_t q_len,
                                     int32_t kv_len,
                                     int32_t max_kv_len,
                                     torch::Dtype dtype,
                                     torch::Device device) override {
    // Block-diffusion draft attends the full context non-causally: every draft
    // position sees the whole block, so all q_len rows share one kv-only mask.
    // Do not restore a causal (per-row diagonal) mask here.
    torch::Tensor non_causal_mask = attn_mask_.gen_append_mask(
        /*q_len=*/1, kv_len, max_kv_len, dtype, device);
    return non_causal_mask.repeat({q_len, 1});
  }

 public:
  // fc/hidden_norm are ATB-backend modules; the shared context-KV write
  // (fused K/V linear -> k_norm -> RoPE -> scatter) lives in
  // draft_context_kv.h.
  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    torch::Tensor projected_hidden = fc_(target_hidden, 0);
    projected_hidden = hidden_norm_(projected_hidden, 0);
    return ::xllm::write_context_kv(kv_state_,
                                    projected_hidden,
                                    positions,
                                    device_cache_slots,
                                    kv_caches,
                                    input_params);
  }

 private:
  layer::NpuColumnParallelLinear fc_{nullptr};
  layer::NpuRMSNorm hidden_norm_{nullptr};
  DraftContextKvState kv_state_;
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
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }
};
TORCH_MODULE(DFlashQwen3ForCausalLM);

// Draft config carries model_type="qwen3" (loaded via the qwen3_atb args
// loader); worker_impl overwrites it to "qwen3_dflash", which resolves to
// "qwen3_dflash_atb" under ATB and selects this factory.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_dflash_atb,
                                   qwen3_dflash_atb,
                                   DFlashQwen3ForCausalLM);

}  // namespace xllm::npu::model
