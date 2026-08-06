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
#include <string>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/rotary_embedding_util.h"
#include "core/layers/npu/rotary_embedding.h"

namespace xllm {

// Draft-model context-KV state: projects a target's captured hidden states
// into the draft's KV cache. Compose one, call load_context_kv_shard() per
// shard, build_fused_context_kv() once, then write_context_kv() per validate
// step. Shared by the ATB and torch draft bodies (each wires only
// fc/hidden_norm).
struct DraftContextKvState {
  // Accumulated across (possibly sharded) state dicts; cleared after fusion.
  std::vector<torch::Tensor> per_layer_k_proj;
  std::vector<torch::Tensor> per_layer_v_proj;
  std::vector<torch::Tensor> per_layer_k_norm;

  // Layer-major layout: [l0_k, l0_v, l1_k, l1_v, ...]; k_norm stacked as
  // [num_layers, 1, 1, head_dim].
  torch::Tensor fused_kv_weight;
  torch::Tensor k_norm_weight;

  torch::TensorOptions tensor_options;
  std::shared_ptr<NpuRotaryEmbedding> rotary_embedding;

  int64_t head_dim = 0;
  double rms_norm_eps = 1e-6;
  int32_t tp_rank = 0;
  int32_t tp_size = 1;
  int64_t local_kv_heads = 0;
};

// Fill the fields that don't depend on backend wrapper types. Callers still
// register their own fc / hidden_norm.
inline void init_context_kv_state(DraftContextKvState& state,
                                  const ModelContext& context) {
  const ModelArgs& model_args = context.get_model_args();
  const ParallelArgs& parallel_args = context.get_parallel_args();
  const int32_t dp_size = parallel_args.dp_size();
  const int32_t cp_size = parallel_args.cp_size();
  CHECK_GT(dp_size, 0) << "DFlash dp_size must be positive.";
  CHECK_GT(cp_size, 0) << "DFlash cp_size must be positive.";
  CHECK_EQ(parallel_args.world_size() % (dp_size * cp_size), 0)
      << "DFlash world_size must be divisible by dp_size * cp_size.";
  state.tp_size = parallel_args.world_size() / (dp_size * cp_size);
  CHECK_GT(state.tp_size, 0) << "DFlash tp_size must be positive.";
  state.tp_rank = parallel_args.rank() % state.tp_size;

  state.tensor_options = context.get_tensor_options();
  state.head_dim = model_args.head_dim();
  state.rms_norm_eps = model_args.rms_norm_eps();
  CHECK_GT(state.head_dim, 0) << "DFlash head_dim must be positive.";
  CHECK_GT(model_args.layers_to_capture().size(), 0u)
      << "DFlash requires dflash_config.target_layer_ids.";

  state.rotary_embedding = std::make_shared<RotaryEmbeddingGeneric>(
      state.head_dim,
      model_args.max_position_embeddings(),
      layer::rotary::compute_inv_freq(
          state.head_dim, model_args.rope_theta(), state.tensor_options),
      // DFlash draft is fixed Qwen3-dense: NeoX-style, non-interleaved rope.
      /*interleaved=*/false,
      state.tensor_options);
}

// A sharded checkpoint delivers draft layers across multiple state dicts;
// each call only fills layers present in this shard.
inline void load_context_kv_shard(DraftContextKvState& state,
                                  const StateDict& state_dict,
                                  int32_t num_layers) {
  if (state.per_layer_k_proj.empty()) {
    state.per_layer_k_proj.resize(num_layers);
    state.per_layer_v_proj.resize(num_layers);
    state.per_layer_k_norm.resize(num_layers);
  }
  for (int32_t i = 0; i < num_layers; ++i) {
    StateDict layer_dict =
        state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
    torch::Tensor k_proj = layer_dict.get_sharded_tensor(
        "self_attn.k_proj.weight", /*dim=*/0, state.tp_rank, state.tp_size);
    torch::Tensor v_proj = layer_dict.get_sharded_tensor(
        "self_attn.v_proj.weight", /*dim=*/0, state.tp_rank, state.tp_size);
    torch::Tensor k_norm = layer_dict.get_tensor("self_attn.k_norm.weight");
    if (!k_proj.defined() && !v_proj.defined() && !k_norm.defined()) {
      continue;  // this draft layer lives in another shard
    }
    CHECK(k_proj.defined()) << "Failed to find DFlash draft layers." << i
                            << ".self_attn.k_proj.weight.";
    CHECK(v_proj.defined()) << "Failed to find DFlash draft layers." << i
                            << ".self_attn.v_proj.weight.";
    CHECK(k_norm.defined()) << "Failed to find DFlash draft layers." << i
                            << ".self_attn.k_norm.weight.";
    CHECK_EQ(k_proj.dim(), 2) << "DFlash k_proj weight must be 2D.";
    CHECK_EQ(v_proj.dim(), 2) << "DFlash v_proj weight must be 2D.";
    CHECK_EQ(k_proj.size(0), v_proj.size(0))
        << "DFlash k/v projection output size mismatch.";
    CHECK_EQ(k_proj.size(1), v_proj.size(1))
        << "DFlash k/v projection weight shape mismatch.";
    CHECK_EQ(k_proj.size(0) % state.head_dim, 0)
        << "DFlash k_proj output size must align to head_dim.";
    const int64_t layer_local_kv_heads = k_proj.size(0) / state.head_dim;
    if (state.local_kv_heads == 0) {
      state.local_kv_heads = layer_local_kv_heads;
    } else {
      CHECK_EQ(state.local_kv_heads, layer_local_kv_heads)
          << "DFlash local KV heads mismatch.";
    }
    state.per_layer_k_proj[i] = k_proj.to(state.tensor_options);
    state.per_layer_v_proj[i] = v_proj.to(state.tensor_options);
    state.per_layer_k_norm[i] =
        k_norm.to(state.tensor_options).to(torch::kFloat32);
  }
}

inline void verify_context_kv_shards(const DraftContextKvState& state,
                                     int32_t num_layers) {
  CHECK_EQ(static_cast<int32_t>(state.per_layer_k_proj.size()), num_layers)
      << "DFlash context K/V weights were not accumulated.";
  CHECK_GT(state.local_kv_heads, 0) << "DFlash local KV heads is invalid.";
  for (int32_t i = 0; i < num_layers; ++i) {
    CHECK(state.per_layer_k_proj[i].defined())
        << "Missing DFlash draft layers." << i << ".self_attn.k_proj.weight.";
    CHECK(state.per_layer_v_proj[i].defined())
        << "Missing DFlash draft layers." << i << ".self_attn.v_proj.weight.";
    CHECK(state.per_layer_k_norm[i].defined())
        << "Missing DFlash draft layers." << i << ".self_attn.k_norm.weight.";
  }
}

// Runs verify_context_kv_shards internally, so callers do not need to.
inline void build_fused_context_kv(DraftContextKvState& state,
                                   int32_t num_layers) {
  verify_context_kv_shards(state, num_layers);
  std::vector<torch::Tensor> kv_weights;
  kv_weights.reserve(static_cast<size_t>(num_layers) * 2);
  std::vector<torch::Tensor> k_norm_weights;
  k_norm_weights.reserve(num_layers);
  for (int32_t i = 0; i < num_layers; ++i) {
    kv_weights.emplace_back(state.per_layer_k_proj[i]);
    kv_weights.emplace_back(state.per_layer_v_proj[i]);
    k_norm_weights.emplace_back(state.per_layer_k_norm[i]);
  }
  state.fused_kv_weight = torch::cat(kv_weights, /*dim=*/0).contiguous();
  state.k_norm_weight =
      torch::stack(k_norm_weights, /*dim=*/0).view({num_layers, 1, 1, -1});
  state.per_layer_k_proj.clear();
  state.per_layer_v_proj.clear();
  state.per_layer_k_norm.clear();
  // rotary_embedding comes from init_context_kv_state, not the loaded shards,
  // so verify_context_kv_shards can't cover it; assert once here to keep it out
  // of the hot write_context_kv path.
  CHECK(state.rotary_embedding != nullptr)
      << "DFlash rotary embedding is not initialized.";
}

// Runs the fused K/V linear + k_norm + RoPE + per-layer scatter on the target
// hidden already projected through the caller's fc/hidden_norm. Returns an
// empty ModelOutput if the per-layer synchronizer's record_event fails, which
// would otherwise deadlock a PD-PUSH transfer.
inline ModelOutput write_context_kv(const DraftContextKvState& state,
                                    const torch::Tensor& projected_hidden,
                                    const torch::Tensor& positions,
                                    const torch::Tensor& device_cache_slots,
                                    std::vector<KVCache>& kv_caches,
                                    const ModelInputParams& input_params) {
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  CHECK(device_cache_slots.defined())
      << "DFlash context K/V requires device new_cache_slots.";
  CHECK_EQ(device_cache_slots.numel(), projected_hidden.size(0))
      << "DFlash device cache slot count mismatch.";

  const int64_t num_context = projected_hidden.size(0);
  torch::Tensor all_kv =
      torch::nn::functional::linear(projected_hidden, state.fused_kv_weight);
  // View + permute so k/v peels off first, then layer. Single contiguous()
  // is intentional: the per-layer scatter needs a contiguous view; doing it
  // once beats re-materializing per layer.
  all_kv = all_kv
               .view({num_context,
                      num_layers,
                      2,
                      state.local_kv_heads,
                      state.head_dim})
               .permute({2, 1, 0, 3, 4})
               .contiguous();

  torch::Tensor all_key = all_kv.select(/*dim=*/0, /*index=*/0);
  torch::Tensor all_value = all_kv.select(/*dim=*/0, /*index=*/1);

  // Batched fp32 RMSNorm: k_norm_weight ([L,1,1,head_dim]) broadcasts over
  // [num_layers, num_context, local_kv_heads, head_dim] so all layers
  // normalize in one shot. Qwen3 scales by raw weight (not gemma 1+weight).
  torch::Tensor key_fp32 = all_key.to(torch::kFloat32);
  torch::Tensor variance = key_fp32.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  torch::Tensor all_key_normed =
      (key_fp32 * torch::rsqrt(variance + state.rms_norm_eps) *
       state.k_norm_weight)
          .to(all_key.scalar_type());

  // RoPE is layer-invariant: flatten [num_layers, num_context, ...] and
  // repeat positions per layer to do all layers in one call.
  torch::Tensor flat_key = all_key_normed.reshape(
      {num_layers * num_context, state.local_kv_heads, state.head_dim});
  torch::Tensor repeated_positions = positions.repeat({num_layers});
  flat_key = std::get<1>(
      state.rotary_embedding->forward(flat_key, flat_key, repeated_positions));
  all_key_normed = flat_key.view(
      {num_layers, num_context, state.local_kv_heads, state.head_dim});

#if defined(USE_NPU)
  const int32_t device_index = all_key_normed.device().index();
#endif
  for (int64_t i = 0; i < num_layers; ++i) {
    kernel::ReshapePagedCacheParams scatter_params;
    scatter_params.key = all_key_normed[i];
    scatter_params.value = all_value[i];
    scatter_params.k_cache = kv_caches[i].get_k_cache();
    scatter_params.v_cache = kv_caches[i].get_v_cache();
    scatter_params.slot_mapping = device_cache_slots;
    kernel::reshape_paged_cache(scatter_params);
#if defined(USE_NPU)
    // Standard attention op records this event inside ATB; this custom
    // scatter path must record it explicitly so PD-PUSH's per-layer
    // synchronizer does not stall.
    if (input_params.parallel.layer_synchronizer != nullptr &&
        !input_params.parallel.layer_synchronizer->record_event(i,
                                                                device_index)) {
      return ModelOutput();
    }
#endif
  }
  return ModelOutput(projected_hidden);
}

}  // namespace xllm
