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

#include "rejection_sampler.h"

#include <ATen/core/TensorBody.h>
#include <ATen/ops/stack.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include "kernels/ops_api.h"
#include "logits_utils.h"
#include "sampler.h"

namespace xllm {

namespace {
// index_select that supports multiple dimensions index
torch::Tensor index_select_2d(const torch::Tensor& input,
                              int64_t dim,
                              const torch::Tensor& index) {
  return input.gather(dim, index.unsqueeze(dim)).squeeze(dim);
}

}  // namespace

RejectionSampler::RejectionSampler(const torch::Tensor& do_sample,
                                   bool all_random_sample,
                                   bool all_greedy_sample,
                                   bool logprobs,
                                   int64_t max_top_logprobs,
                                   bool enable_fused_kernel,
                                   const torch::Tensor& temperatures,
                                   const torch::Tensor& top_k,
                                   const torch::Tensor& top_p,
                                   const torch::Tensor& frequency_penalties,
                                   const torch::Tensor& presence_penalties,
                                   const torch::Tensor& repetition_penalties,
                                   const torch::Tensor& unique_token_ids,
                                   const torch::Tensor& unique_token_counts)
    : logprobs_(logprobs),
      max_top_logprobs_(max_top_logprobs),
      all_random_sample_(all_random_sample),
      all_greedy_sample_(all_greedy_sample),
      temperatures_(temperatures),
      top_k_(top_k),
      top_p_(top_p),
      frequency_penalties_(frequency_penalties),
      presence_penalties_(presence_penalties),
      repetition_penalties_(repetition_penalties),
      unique_token_ids_(unique_token_ids),
      unique_token_counts_(unique_token_counts),
      enable_fused_kernel_(enable_fused_kernel) {
  CHECK(do_sample.defined());
  // Keep a private expanded view and do not mutate the caller-owned tensor.
  // The same SamplingParameters object is reused later by MTP draft extend.
  // An in-place unsqueeze here corrupts Sampler::forward() mixed-mode shape
  // assumptions and can broadcast sampled token ids into 2D.
  do_sample_ = do_sample.unsqueeze(/*dim=*/-1);
}

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
// returns accepted tokens. [batch_size, n_speculative_tokens + 1]
SampleOutput RejectionSampler::forward(const torch::Tensor& draft_token_ids,
                                       const torch::Tensor& draft_probs,
                                       const torch::Tensor& target_logits,
                                       const torch::Tensor& bonus_token_ids,
                                       bool mask_out_rejected_tokens) const {
  CHECK_EQ(draft_token_ids.size(0), do_sample_.size(0))
      << "batch size mismatch";
  DCHECK(!draft_probs.defined() ||
         draft_token_ids.size(1) == draft_probs.size(1));

  // Shape the target distribution p from raw logits here: target_logits is the
  // RAW model output (the target worker stores output.logits = logits.clone()
  // before its sampler mutates logits in place), so the RejectionSampler owns
  // the penalty + T/top-k/top-p shaping instead of relying on the sampler's
  // in-place side effect. The order mirrors Sampler::forward: penalties first,
  // then temperature/top-k/top-p, yielding the correct sampled target
  // distribution p = softmax(shaped).
  //
  // Penalty caveat (n_speculative > 1): a single per-request token history
  // (unique_token_ids_/counts_) is broadcast to all n_speculative + 1
  // positions, so positions after the first do NOT fold in the earlier draft
  // tokens (t1..t_{i-1}). Their penalties are therefore approximate. This
  // matches the pre-existing xLLM behavior; exact per-position penalty history
  // is a follow-up (see PR description). For n_speculative == 1 the single
  // position uses the exact prefix history and is unbiased.
  torch::Tensor shaped_logits = target_logits;
  const bool has_penalties =
      frequency_penalties_.defined() || repetition_penalties_.defined();
  if (has_penalties || temperatures_.defined() || top_k_.defined() ||
      top_p_.defined()) {
    const int64_t batch = shaped_logits.size(0);
    const int64_t n_pos = shaped_logits.size(1);  // n_speculative + 1
    const int64_t vocab = shaped_logits.size(2);
    // apply_* are in-place and target_logits is a const ref, so clone.
    auto logits_2d = shaped_logits.reshape({batch * n_pos, vocab}).clone();
    // Broadcast per-request [batch] params to per-position [batch * n_pos].
    auto expand_param = [n_pos](const torch::Tensor& t) {
      return t.defined() ? t.repeat_interleave(n_pos, /*dim=*/0) : t;
    };
    // Penalties before top-k/top-p, mirroring Sampler::forward. Both penalty
    // kernels index the same token history, so expand it once.
    if (has_penalties) {
      auto token_ids = expand_param(unique_token_ids_);
      if (frequency_penalties_.defined()) {
        apply_frequency_presence_penalties(logits_2d,
                                           token_ids,
                                           expand_param(unique_token_counts_),
                                           expand_param(frequency_penalties_),
                                           expand_param(presence_penalties_));
      }
      if (repetition_penalties_.defined()) {
        apply_repetition_penalties(
            logits_2d, token_ids, expand_param(repetition_penalties_));
      }
    }
    apply_top_k_top_p(logits_2d,
                      expand_param(temperatures_),
                      expand_param(top_k_),
                      expand_param(top_p_));
    shaped_logits = logits_2d.reshape({batch, n_pos, vocab});
  }
  // [batch_size, n_speculative_tokens + 1, vocab_size] FloatTensor
  auto target_probs =
      torch::softmax(shaped_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);
  // filter out probs for bonus tokens
  target_probs = target_probs.slice(
      /*dim=*/1, /*start=*/0, /*end=*/target_probs.size(1) - 1);

  // Determine whether we need to restore rejected tokens.
  // IMPORTANT: The fused kernel implementation only supports masking out
  // rejected tokens,
  //            and does not support restoring their original values. Only use
  //            fused path if logprobs are NOT needed and
  //            mask_out_rejected_tokens is true.
  bool use_fused_kernel =
      enable_fused_kernel_ && (!logprobs_ && mask_out_rejected_tokens);

  // select the random sampler function based on the use_fused_kernel flag
  auto random_sampler_func = use_fused_kernel
                                 ? &RejectionSampler::random_sample_fused
                                 : &RejectionSampler::random_sample;

  // [batch_size, n_speculative_tokens + 1]
  torch::Tensor accepted_token_ids;
  torch::Tensor masked_accepted_token_ids;
  if (all_greedy_sample_) {
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        greedy_sample(draft_token_ids,
                      target_probs,
                      bonus_token_ids,
                      mask_out_rejected_tokens);
  } else if (all_random_sample_) {
    auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), target_probs.options());
    std::tie(accepted_token_ids, masked_accepted_token_ids) =
        random_sampler_func(draft_token_ids,
                            draft_probs,
                            target_probs,
                            uniform_rand,
                            bonus_token_ids,
                            mask_out_rejected_tokens);
  } else {
    auto uniform_rand =
        torch::rand(draft_token_ids.sizes(), target_probs.options());
    // mixed sample, sample both then choose based on do_sample_
    auto [random, masked_random] =
        random_sampler_func(draft_token_ids,
                            draft_probs,
                            target_probs,
                            uniform_rand,
                            bonus_token_ids,
                            mask_out_rejected_tokens);
    auto [greedy, masked_greedy] = greedy_sample(draft_token_ids,
                                                 target_probs,
                                                 bonus_token_ids,
                                                 mask_out_rejected_tokens);
    accepted_token_ids = torch::where(do_sample_, random, greedy);
    if (mask_out_rejected_tokens) {
      masked_accepted_token_ids =
          torch::where(do_sample_, masked_random, masked_greedy);
    }
  }

  SampleOutput output;
  output.next_tokens =
      mask_out_rejected_tokens ? masked_accepted_token_ids : accepted_token_ids;

  if (logprobs_) {
    // log_softmax is equivalent to log(softmax) but more numerically stable
    // [batch_size, n_speculative_tokens + 1, vocab_size]
    // Use the shaped logits so returned logprobs reflect the actual sampled
    // distribution (tokens truncated by top-k/top-p get -inf).
    auto target_logprobs = torch::log_softmax(
        shaped_logits, /*dim=*/-1, /*dtype=*/torch::kFloat32);

    // select the logprobs for each sequence
    const auto selected_logprobs =
        index_select_2d(target_logprobs, /*dim=*/-1, accepted_token_ids);
    // output.probs = selected_probs;
    output.logprobs = selected_logprobs;

    if (max_top_logprobs_ > 0) {
      auto [values, indices] =
          target_logprobs.topk(max_top_logprobs_, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }
  return output;
}

// build mask from accepted matrix
// for example: [[1, 1, 0, 1],   ->   [[1, 1, 1, 0, 0],
//               [1, 0, 0, 0]]         [1, 1, 0, 0, 0]]
torch::Tensor RejectionSampler::build_accepted_mask(
    const torch::Tensor& accepted) {
  // build the mask for the first rejected token
  const auto batch_size = accepted.size(0);
  const auto n_tokens = accepted.size(1);

  // use LongTensor since argmax does not support bool
  auto accepted_int64 = accepted.to(torch::kInt64);
  auto bonus_mask = torch::zeros({batch_size, 1}, accepted_int64.options());
  auto combined_mask = torch::cat({accepted_int64, bonus_mask}, /*dim=*/-1);
  // [batch_size, 1]
  auto first_rejected_mask =
      (1 - combined_mask).argmax(/*dim=*/1, /*keepdim=*/true);

  // [1, n_speculative_tokens + 1]
  auto indices =
      torch::arange(n_tokens + 1, accepted.device()).unsqueeze(/*dim=*/0);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_mask = indices <= first_rejected_mask;
  return accepted_mask;
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::random_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  // Two lossless modes, distinguished by whether the draft carried a full
  // dense distribution q:
  //   * NO_DRAFT_PROBS (draft_probs undefined): greedy draft, q is a point
  //     mass at the draft token. Acceptance = min(1, p(t)/1) = p(t); the
  //     residual (p - q)+ reduces exactly to "p with the draft token zeroed".
  //   * dense (draft_probs [batch, n_spec, vocab]): probabilistic draft.
  //     Acceptance = min(1, p(t)/q(t)); residual = (p - q)+ over the full
  //     vocab.
  const bool no_draft_probs = !draft_probs.defined();

  auto selected_target_probs =
      index_select_2d(target_probs, /*dim=*/-1, draft_token_ids);

  torch::Tensor acceptance_probs;
  if (no_draft_probs) {
    // q(t) = 1 -> acceptance prob = min(1, p(t)) = p(t) (p is a valid prob).
    acceptance_probs = selected_target_probs;
  } else {
    CHECK_EQ(draft_probs.dim(), 3)
        << "probabilistic draft_probs must be [batch, n_spec, vocab], got "
        << draft_probs.sizes();
    auto selected_draft_probs =
        index_select_2d(draft_probs, /*dim=*/-1, draft_token_ids);
    acceptance_probs = selected_target_probs / selected_draft_probs;
  }
  auto accepted = (uniform_rand < acceptance_probs);

  // construct recovered probs = normalize((p - q)+)
  auto recovered_probs = target_probs.clone();
  if (no_draft_probs) {
    // (p - onehot_t)+ : zero out the draft token, keep p elsewhere.
    recovered_probs.scatter_(
        /*dim=*/-1,
        draft_token_ids.unsqueeze(-1),
        torch::zeros_like(draft_token_ids)
            .unsqueeze(-1)
            .to(recovered_probs.dtype()));
  } else {
    recovered_probs.sub_(draft_probs).clamp_min_(0);
  }
  // a small value to avoid division by zero
  const auto epsilon = 1e-6f;
  auto sum = recovered_probs.sum(-1, /*keepdim=*/true).clamp_min_(epsilon);
  recovered_probs.div_(sum);

  // resample on the recovered probs
  torch::Tensor recovered_token_ids = Sampler::random_sample(recovered_probs);

  auto combined = torch::where(accepted, draft_token_ids, recovered_token_ids);
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids = torch::cat({combined, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    // build the mask for the first rejected token
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    masked_accepted_token_ids =
        torch::where(accepted_mask,
                     accepted_token_ids,
                     -torch::ones_like(accepted_token_ids));
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::random_sample_fused(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& draft_probs,
    const torch::Tensor& target_probs,
    const torch::Tensor& uniform_rand,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  CHECK_EQ(draft_probs.dim(), 3)
      << "Fused rejection sampler requires dense draft_probs [batch, n_spec, "
         "vocab]. Ensure validate passes dense draft_probs, i.e. run with "
         "--enable_probabilistic_draft=true.";

  const auto device = draft_token_ids.device();
  const int64_t batch_size = draft_token_ids.size(0);
  const int64_t n_spec = draft_token_ids.size(1);
  const int64_t vocab_size = target_probs.size(2);

  // Strictly check device consistency for bonus_token_ids and draft_token_ids
  CHECK_EQ(bonus_token_ids.device().type(), device.type())
      << "bonus_token_ids must be on the same device as draft_token_ids";

  // Check that bonus_token_ids has at least batch_size elements
  CHECK_GE(bonus_token_ids.numel(), batch_size)
      << "bonus_token_ids numel (" << bonus_token_ids.numel()
      << ") is smaller than batch_size (" << batch_size << ")";

  // Prepare input Tensors and ensure they are contiguous where needed
  // If draft_token_ids is already int32 and contiguous, no copy occurs
  torch::Tensor draft_token_ids_int32 =
      draft_token_ids.reshape({-1}).to(torch::kInt32).contiguous();
  torch::Tensor bonus_token_ids_int32 =
      bonus_token_ids.reshape({-1}).to(torch::kInt32).contiguous();

  // Ensure large probability matrices are in the correct shape and contiguous
  torch::Tensor draft_probs_flat =
      draft_probs.reshape({-1, vocab_size}).contiguous();
  torch::Tensor target_probs_flat =
      target_probs.reshape({-1, vocab_size}).contiguous();
  torch::Tensor uniform_rand_flat =
      uniform_rand.to(torch::kFloat32).flatten().contiguous();

  // Create auxiliary tensors directly on the target device to avoid unnecessary
  // copies
  torch::TensorOptions options_int32 =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor num_draft_tokens =
      torch::full({batch_size}, n_spec, options_int32);
  torch::Tensor cu_num_draft_tokens =
      torch::arange(n_spec, (batch_size + 1) * n_spec, n_spec, options_int32);

  // Always create recovery probability matrix here, as kernel requires it
  torch::Tensor uniform_probs =
      torch::empty({batch_size * n_spec, vocab_size},
                   target_probs.options().dtype(torch::kFloat32))
          .exponential_();

  // Call the fused kernel
  kernel::RejectionSampleParams params;
  params.draft_token_ids = draft_token_ids_int32;
  params.num_draft_tokens = num_draft_tokens;
  params.cu_num_draft_tokens = cu_num_draft_tokens;
  params.draft_probs = draft_probs_flat;
  params.target_probs = target_probs_flat;
  params.bonus_token_ids = bonus_token_ids_int32;
  params.uniform_rand = uniform_rand_flat;
  params.uniform_probs = uniform_probs;
  params.max_spec_len = n_spec;

  // The result is flattened, and positions of rejected tokens are set to -1
  torch::Tensor output_token_ids = kernel::rejection_sample(params);

  // Reshape result to [batch, n_spec + 1]
  torch::Tensor masked_result =
      output_token_ids.reshape({batch_size, n_spec + 1}).to(torch::kInt64);

  // When mask_out_rejected_tokens=true and logprobs_=false,
  // we can safely return masked_result for both outputs.
  return {masked_result, masked_result};
}

std::tuple<torch::Tensor, torch::Tensor> RejectionSampler::greedy_sample(
    const torch::Tensor& draft_token_ids,
    const torch::Tensor& target_probs,
    const torch::Tensor& bonus_token_ids,
    bool mask_out_rejected_tokens) {
  auto target_token_ids = Sampler::greedy_sample(target_probs);

  // mask out the rejected tokens with -1
  // [batch_size, n_speculative_tokens + 1]
  auto accepted_token_ids =
      torch::cat({target_token_ids, bonus_token_ids}, /*dim=*/-1);
  torch::Tensor masked_accepted_token_ids;
  if (mask_out_rejected_tokens) {
    // [batch_size, n_speculative_tokens + 1]
    auto accepted = (target_token_ids == draft_token_ids);
    auto accepted_mask = build_accepted_mask(accepted);
    // mask out the rejected tokens with -1
    masked_accepted_token_ids =
        torch::where(accepted_mask,
                     accepted_token_ids,
                     -torch::ones_like(accepted_token_ids));
  }
  return {accepted_token_ids, masked_accepted_token_ids};
}

}  // namespace xllm
