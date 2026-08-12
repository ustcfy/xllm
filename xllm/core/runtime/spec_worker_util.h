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

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/kv_cache/kv_cache_capacity.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/platform/stream.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/llm_worker_impl.h"

namespace xllm {
namespace spec {

// Free helpers shared by every SpeculativeWorkerImpl subclass (MTP, DFlash,
// and any future spec algorithm). Deliberately header-only + free functions
// rather than methods on SpeculativeWorkerImpl — hoisting to the base would
// widen its public surface and add link deps; a header of inline helpers
// keeps the sharing at the smallest invariant.

// Picks the process group across which sampled draft/accepted tokens must be
// unified. TP group takes priority; falls back to the DP-local process group.
// Per-rank sampling RNG diverges across the TP dimension, so broadcasting to
// rank 0 keeps every rank's cached draft probs and accepted prefixes
// identical. Returns nullptr for a single-process run.
inline ProcessGroup* broadcast_group(const ParallelArgs& parallel_args) {
  return parallel_args.tp_group_ != nullptr ? parallel_args.tp_group_
                                            : parallel_args.process_group_;
}

// Broadcasts `tokens` from `root_rank` across `pg`, in place. No-op when
// `pg` is null / single-rank / `tokens` undefined.
inline void broadcast_tokens(torch::Tensor& tokens,
                             ProcessGroup* pg,
                             int32_t root_rank = 0) {
  if (pg == nullptr || pg->world_size() <= 1 || !tokens.defined()) {
    return;
  }
  tokens = tokens.contiguous();
  pg->broadcast(tokens, root_rank);
}

inline bool use_orthogonal_cp_consensus(const ParallelArgs& parallel_args) {
  return parallel_args.cp_size() > 1 && parallel_args.tp_group_ != nullptr &&
         parallel_args.cp_group_ != nullptr &&
         parallel_args.cp_group_ != parallel_args.tp_group_;
}

inline bool should_broadcast_tokens(const ParallelArgs& parallel_args,
                                    bool enable_spec_token_broadcast,
                                    bool all_greedy_sample) {
  return use_orthogonal_cp_consensus(parallel_args) ||
         (enable_spec_token_broadcast && !all_greedy_sample);
}

// Speculative state is replicated across every model rank within one DP
// replica. With orthogonal CP x TP, tp_group_ covers only one CP shard, while
// cp_group_ connects the same TP rank across CP shards. Broadcast along both
// axes so every replica caches and consumes the same sampled token, without
// crossing into another DP replica through the world group.
inline void broadcast_spec_tokens(torch::Tensor& tokens,
                                  const ParallelArgs& parallel_args,
                                  int32_t root_rank = 0) {
  if (!use_orthogonal_cp_consensus(parallel_args)) {
    broadcast_tokens(tokens, broadcast_group(parallel_args), root_rank);
    return;
  }

  broadcast_tokens(tokens, parallel_args.tp_group_, root_rank);
  broadcast_tokens(tokens, parallel_args.cp_group_, root_rank);
}

// Records a stream event on `stream` and stashes it on
// `input.metadata_ready_event`. If the stream's record_event fails (returns
// nullptr), synchronously flushes the stream instead so the metadata is
// observable by the next consumer.
inline void record_metadata_ready(Stream& stream, ForwardInput& input) {
  StreamEventPtr event = stream.record_event();
  if (event == nullptr) {
    stream.synchronize();
  }
  input.metadata_ready_event = event;
}

// Blocks `stream` until the metadata_ready_event previously recorded via
// `record_metadata_ready` fires. CHECK-fails if the wait cannot be issued.
inline void wait_metadata_ready(const ForwardInput& input, Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait speculative metadata ready event";
}

// Clears just the selected-token embeddings field. Used when a caller keeps
// the full embedding block but no longer needs the per-selected-token slice
// (e.g. after packing a bootstrap subset).
inline void clear_selected_embeddings(ForwardOutput& output) {
  output.sample_output.selected_embeddings = torch::Tensor();
}

// Clears both the full sample embedding and the selected-token embeddings.
// Used when the caller has consumed everything they need from the sample
// output and wants to release the memory before returning.
inline void clear_all_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  clear_selected_embeddings(output);
}

// Effective TP degree derived from `ParallelArgs`: world_size / (dp * cp),
// with dp/cp clamped to at least 1. This is the same "how many ranks share a
// single-copy tensor along the TP dimension" quantity every spec worker needs
// (KV cache shape estimation, draft-body KV shape, per-head slicing). Returns
// at least 1.
inline int64_t dp_local_tp_size(const ParallelArgs& parallel_args) {
  const int64_t dp_size = std::max<int64_t>(parallel_args.dp_size(), 1);
  const int64_t cp_size = std::max<int64_t>(parallel_args.cp_size(), 1);
  return std::max<int64_t>(parallel_args.world_size() / dp_size / cp_size, 1);
}

inline KVCacheShape draft_kv_cache_shape(const KVCacheShape& target_shape,
                                         const ModelArgs& draft_model_args,
                                         int64_t block_size,
                                         int64_t world_size) {
  CHECK_GT(block_size, 0)
      << "draft_kv_cache_shape block_size must be positive.";
  CHECK_GT(world_size, 0)
      << "draft_kv_cache_shape world_size must be positive.";
  KVCacheCapacity draft_capacity;
  draft_capacity.n_blocks(target_shape.key_cache_shape()[0])
      .block_size(block_size);
  return KVCacheShape(draft_capacity, draft_model_args, world_size);
}

// Launch a no-sync target/draft forward on the two-stream layout: prepare on
// `prepare_stream`, compute on `compute_stream`. The prepare stage records
// its metadata-ready event only when the two streams are distinct (a
// same-stream FIFO already orders them). `execute_no_sync_on_stream` never
// records its output ready event here — spec workers manage their own output
// handoff and never consume `ForwardOutput::ready_event` on this path. When
// `processed_output` is non-null, the prepared ForwardInput is moved into it
// so the caller can inspect processed metadata (e.g. `positions_host`) after
// launch.
inline std::optional<ForwardOutput> run_llm_no_sync(
    LLMWorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput* processed_output = nullptr) {
  ForwardInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input,
      processed_input,
      prepare_stream,
      /*record_ready_event=*/&prepare_stream != &compute_stream);
  std::optional<ForwardOutput> output = worker.execute_no_sync_on_stream(
      processed_input, compute_stream, /*record_ready_event=*/false);
  if (processed_output != nullptr) {
    *processed_output = std::move(processed_input);
  }
  return output;
}

// Overload for callers that always want the processed input (MTP path).
inline std::optional<ForwardOutput> run_llm_no_sync(
    LLMWorkerImpl& worker,
    const ForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    ForwardInput& processed_input) {
  return run_llm_no_sync(
      worker, input, prepare_stream, compute_stream, &processed_input);
}

#if defined(USE_NPU)
// Reset the expanded-decode graph inputs on `input_params`. Both build_*
// callers invoke this so they can early-return without leaking stale device
// tensors from a prior step.
inline void clear_expanded_paged_attention_input(
    ModelInputParams& input_params) {
  input_params.graph.use_expanded_decode_for_spec_verify_attention = false;
  input_params.graph.expanded_kv_seq_lens = torch::Tensor();
  input_params.graph.expanded_block_tables = torch::Tensor();
  input_params.graph.expanded_tiling_data = torch::Tensor();
  input_params.graph.expanded_kv_seq_lens_vec.clear();
}

// Expand a per-sequence spec input (one row, q_len query tokens) into one
// PagedAttention row per query token, routing attention through the
// expanded-decode path instead of the chunked-prefill FIA. The query is
// decode-shaped (a few tokens attending to a much longer context+block KV),
// which the CANN split-fuse (chunked-prefill) FIA tiling rejects. The
// PagedAttention decode kernel has no split-fuse tiling and no attention
// mask: each expanded row reads a contiguous KV prefix bounded only by its
// kv_seq_len.
//   - causal=false (DFlash draft block diffusion): every query token sees the
//     WHOLE block, so each expanded row gets the FLAT full kv_len.
//   - causal=true (MTP / DFlash spec verify): validate token i sees only
//     itself and the earlier tokens, so its kv_len follows a staircase
//     (kv_len - q_len + i + 1).
inline void build_expanded_paged_attention_input(ModelInputParams& input_params,
                                                 const torch::Device& device,
                                                 bool causal) {
  clear_expanded_paged_attention_input(input_params);
  const std::vector<int32_t>& q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const std::vector<int32_t>& kv_seq_lens =
      input_params.attention.host.kv_seq_lens;
  if (q_seq_lens.empty() || kv_seq_lens.empty()) {
    return;
  }
  CHECK_EQ(q_seq_lens.size(), kv_seq_lens.size())
      << "spec verify q/kv seq lens must both be sequence-scoped";
  CHECK(input_params.attention.device.block_tables.defined())
      << "spec verify block tables must be rebuilt before expanded input";

  const int64_t batch_size = static_cast<int64_t>(q_seq_lens.size());
  CHECK_GE(input_params.attention.device.block_tables.size(0), batch_size)
      << "spec verify block table rows are fewer than sequences";

  int64_t total_expanded_rows = 0;
  for (int32_t q_len : q_seq_lens) {
    total_expanded_rows += q_len;
  }
  CHECK_GE(total_expanded_rows, 0) << "expanded row count is negative";
  std::vector<int32_t> expanded_kv_seq_lens;
  std::vector<int32_t> expanded_row_indices;
  expanded_kv_seq_lens.reserve(static_cast<size_t>(total_expanded_rows));
  expanded_row_indices.reserve(static_cast<size_t>(total_expanded_rows));
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int32_t q_len = q_seq_lens[static_cast<size_t>(seq_idx)];
    const int32_t kv_len = kv_seq_lens[static_cast<size_t>(seq_idx)];
    CHECK_GE(q_len, 1) << "expanded q_len must be positive";
    CHECK_GE(kv_len, q_len) << "kv_len must include the block's query tokens";
    for (int32_t token_idx = 0; token_idx < q_len; ++token_idx) {
      expanded_kv_seq_lens.emplace_back(causal ? kv_len - q_len + token_idx + 1
                                               : kv_len);
      expanded_row_indices.emplace_back(static_cast<int32_t>(seq_idx));
    }
  }
  if (expanded_kv_seq_lens.empty()) {
    return;
  }

  // Pack a host int vector into a pinned CPU tensor and stage an async H2D
  // copy onto the caller's active stream.
  auto to_device_pinned = [&device](const auto& values,
                                    torch::ScalarType dtype) {
    return torch::tensor(values,
                         torch::TensorOptions()
                             .dtype(dtype)
                             .device(torch::kCPU)
                             .pinned_memory(true))
        .to(device, /*non_blocking=*/true);
  };

  input_params.graph.use_expanded_decode_for_spec_verify_attention = true;
  input_params.graph.expanded_kv_seq_lens =
      to_device_pinned(expanded_kv_seq_lens, torch::kInt);
  // One gather over the (already-on-device) block table replaces the per-row
  // select + host-side torch::stack: expanded_row_indices maps each expanded
  // query row back to its sequence, so index_select emits the same
  // [total_rows, stride] tensor the stacked per-row selects produced.
  torch::Tensor row_index_device =
      to_device_pinned(expanded_row_indices, torch::kLong);
  input_params.graph.expanded_block_tables =
      input_params.attention.device.block_tables.index_select(/*dim=*/0,
                                                              row_index_device);
  input_params.graph.expanded_kv_seq_lens_vec = std::move(expanded_kv_seq_lens);
}
#endif

}  // namespace spec
}  // namespace xllm
