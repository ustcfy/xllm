/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>
#include <torch/types.h>

#include "sampling_params.h"

namespace xllm {

class RejectionSampler final {
 public:
  RejectionSampler(const torch::Tensor& do_sample,
                   bool all_random_sample,
                   bool all_greedy_sample,
                   bool logprobs,
                   int64_t max_top_logprobs,
                   bool enable_fused_kernel = false,
                   const torch::Tensor& temperatures = {},
                   const torch::Tensor& top_k = {},
                   const torch::Tensor& top_p = {},
                   const torch::Tensor& frequency_penalties = {},
                   const torch::Tensor& presence_penalties = {},
                   const torch::Tensor& repetition_penalties = {},
                   const torch::Tensor& unique_token_ids = {},
                   const torch::Tensor& unique_token_counts = {});

  // operator() allows us to use the module as a function.
  template <typename... Args>
  auto operator()(Args&&... args) const {
    return this->forward(::std::forward<Args>(args)...);
  }

  // Sample tokens ids using rejection sampling.
  // draft_token_ids: [batch_size, n_speculative_tokens]
  // draft_probs:
  //   1) undefined: greedy draft (NO_DRAFT_PROBS), q treated as one-hot
  //   2) dense format: [batch_size, n_speculative_tokens, vocab_size]
  //      (probabilistic draft, exact (p-q)+ rejection)
  // target_logits: [batch_size, n_speculative_tokens + 1, vocab_size]
  //   RAW model logits; the RejectionSampler applies penalties then
  //   temperature/top-k/top-p internally. Do NOT pass logits already shaped by
  //   the sampler.
  // bonus_token_ids: [batch_size, 1]
  SampleOutput forward(const torch::Tensor& draft_token_ids,
                       const torch::Tensor& draft_probs,
                       const torch::Tensor& target_logits,
                       const torch::Tensor& bonus_token_ids,
                       bool mask_out_rejected_tokens = false) const;

  // build mask from accepted matrix
  // for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
  //               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
  static torch::Tensor build_accepted_mask(const torch::Tensor& accepted);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

  static std::tuple<torch::Tensor, torch::Tensor> random_sample_fused(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& draft_probs,
      const torch::Tensor& target_probs,
      const torch::Tensor& uniform_rand,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

  static std::tuple<torch::Tensor, torch::Tensor> greedy_sample(
      const torch::Tensor& draft_token_ids,
      const torch::Tensor& target_probs,
      const torch::Tensor& bonus_token_ids,
      bool mask_out_rejected_tokens);

 private:
  // whether to return logprobs
  bool logprobs_ = false;

  // max number of top logprobs in the batch
  int64_t max_top_logprobs_ = 0;

  // [batch_size]
  torch::Tensor do_sample_;
  bool all_random_sample_ = true;
  bool all_greedy_sample_ = true;

  // Per-request sampling params applied to the RAW target logits inside
  // forward() to shape p. [batch_size] each; may be undefined.
  torch::Tensor temperatures_;
  torch::Tensor top_k_;
  torch::Tensor top_p_;

  // Per-request penalties applied to the RAW target logits inside forward(),
  // BEFORE temperature/top-k/top-p, mirroring the target Sampler order so the
  // verification distribution p matches what the target model actually samples
  // from (Leviathan 2023 requires p to be the true target sampling
  // distribution). [batch_size] each; unique_token_ids/counts are
  // [batch_size, max_unique] token-history tables. All may be undefined (no
  // penalties). NOTE: a single per-request history is broadcast to all
  // n_speculative + 1 positions; for n_speculative > 1 the later positions do
  // NOT fold in the earlier draft tokens (t1..t_{i-1}), so their penalties are
  // approximate. This matches the pre-existing xLLM behavior (the old in-place
  // path had the same limitation). Exact per-position penalty history is left
  // as a follow-up (see PR description).
  torch::Tensor frequency_penalties_;
  torch::Tensor presence_penalties_;
  torch::Tensor repetition_penalties_;
  torch::Tensor unique_token_ids_;
  torch::Tensor unique_token_counts_;

  // whether to use fused kernel
  bool enable_fused_kernel_ = false;
};

}  // namespace xllm
