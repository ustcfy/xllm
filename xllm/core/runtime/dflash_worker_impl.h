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

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "runtime/mtp_worker_impl.h"

namespace xllm {

class DFlashWorkerImpl : public MTPWorkerImpl {
 public:
  DFlashWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DFlashWorkerImpl() override = default;

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;

  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;

  std::vector<ForwardOutput> run_decode_draft(
      const ForwardInput& input,
      const std::vector<EmbeddingCache::DecodeState>& last_states,
      ForwardInput& validate_input) override;

 private:
  void prepare_dflash_query_inputs(const ForwardInput& input,
                                   ForwardInput& query_input);

  void materialize_dflash_context_kv(const ForwardInput& input,
                                     const torch::Tensor& context_hidden,
                                     const std::vector<int32_t>& positions,
                                     std::vector<int32_t> new_cache_slots);

  void write_target_context_to_cache(
      const ForwardInput& input,
      const SampleOutput& validate_output) override;

  std::vector<ForwardOutput> split_dflash_draft_output(
      const ForwardOutput& draft_output);

  int32_t mask_token_id_ = -1;
  int64_t expected_context_hidden_size_ = 0;
};

}  // namespace xllm
