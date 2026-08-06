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

#include "core/framework/model_loader.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/common/linear.h"
#include "core/layers/common/rms_norm.h"
#include "models/llm/draft_context_kv.h"
#include "models/llm/qwen3.h"
#include "models/model_registry.h"

namespace xllm {

// Torch-backend counterpart to the ATB DFlash draft
// (models/llm/npu/qwen3_dflash.h). Derives from QWen3ModelImpl so the draft can
// reuse a torch-backend target's lm_head/word_embedding by instance sharing.
// Context-K/V accumulation/fusion/scatter lives in DraftContextKvState
// (draft_context_kv.h); this class wires only fc/hidden_norm.
class DFlashQwen3TorchModelImpl final : public QWen3ModelImpl {
 public:
  explicit DFlashQwen3TorchModelImpl(const ModelContext& context)
      : QWen3ModelImpl(context) {
    init_context_kv_state(kv_state_, context);
    const ModelArgs& model_args = context.get_model_args();
    const int64_t hidden_size = model_args.hidden_size();
    const int64_t num_captured =
        static_cast<int64_t>(model_args.layers_to_capture().size());

    fc_ = register_module("fc",
                          layer::ReplicatedLinear(hidden_size * num_captured,
                                                  hidden_size,
                                                  /*bias=*/false,
                                                  QuantArgs(),
                                                  kv_state_.tensor_options));
    hidden_norm_ = register_module(
        "hidden_norm",
        layer::RMSNorm(
            hidden_size, kv_state_.rms_norm_eps, kv_state_.tensor_options));
  }

  void load_state_dict(const StateDict& state_dict) override {
    fc_->load_state_dict(state_dict.get_dict_with_prefix("fc."));
    hidden_norm_->load_state_dict(
        state_dict.get_dict_with_prefix("hidden_norm."));
    load_context_kv_shard(
        kv_state_, state_dict, static_cast<int32_t>(layers_.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(layers_.size()); ++i) {
      layers_[i]->load_state_dict(
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + "."));
    }
    norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  }

  // Torch backend has no ATB-style verify/merge phase, so ForCausalLM calls
  // this once after every (possibly sharded) state dict is loaded.
  void build_fused_context_kv() {
    ::xllm::build_fused_context_kv(kv_state_,
                                   static_cast<int32_t>(layers_.size()));
  }

 protected:
  // Non-causal attention: every draft position sees the whole KV range. Flag
  // it so the attention kernel picks sparse_mode=0 (no mask); block boundary
  // is enforced by the KV range, not a mask tensor.
  layer::AttentionMetadata get_attention_metadata(
      const ModelInputParams& params,
      const torch::Tensor& h) override {
    layer::AttentionMetadata attn_metadata =
        layer::AttentionMetadataBuilder::build(params,
                                               model_args_.enable_mla());
    attn_metadata.is_causal = false;
    return attn_metadata;
  }

 public:
  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    torch::Tensor projected_hidden = fc_(target_hidden);
    projected_hidden = std::get<0>(hidden_norm_->forward(projected_hidden));
    return ::xllm::write_context_kv(kv_state_,
                                    projected_hidden,
                                    positions,
                                    device_cache_slots,
                                    kv_caches,
                                    input_params);
  }

 private:
  layer::ReplicatedLinear fc_{nullptr};
  layer::RMSNorm hidden_norm_{nullptr};
  DraftContextKvState kv_state_;
};
TORCH_MODULE(DFlashQwen3TorchModel);

class DFlashQwen3TorchForCausalLMImpl final
    : public LlmForCausalLMImplBase<DFlashQwen3TorchModel> {
 public:
  explicit DFlashQwen3TorchForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DFlashQwen3TorchModel>(context) {}

  // Draft checkpoint has no lm_head/embed_tokens — those come from the target
  // at runtime. Load body across all (possibly sharded) dicts, then fuse once.
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
    model_->build_fused_context_kv();
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
TORCH_MODULE(DFlashQwen3TorchForCausalLM);

// Torch-backend factory (key "qwen3_dflash"); the ATB body registers
// "qwen3_dflash_atb" separately and resolution picks by npu_kernel_backend.
// NPU-only because the DFlash worker's scatter/PD-PUSH/expanded-decode graph
// inputs are all NPU-gated.
#if defined(USE_NPU)
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_dflash,
                                   qwen3_dflash,
                                   DFlashQwen3TorchForCausalLM);
#endif

}  // namespace xllm
