/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/config/speculative_config.h"

#include <glog/logging.h>

#include "core/common/global_flags.h"
#include "core/framework/config/config_utils.h"

DEFINE_string(draft_model, "", "draft hf model path to the model file.");

DEFINE_string(draft_devices,
              "",
              "Devices to run the draft model on, e.g. npu:0, npu:0,npu:1. "
              "If omitted, uses the target model devices when speculative "
              "decoding is enabled.");

DEFINE_int32(num_speculative_tokens, 0, "Number of speculative tokens.");

DEFINE_string(speculative_algorithm,
              "MTP",
              "Speculative decoding algorithm. Supported options: MTP, Eagle3, "
              "Suffix. Default is MTP.");

DEFINE_int32(speculative_suffix_cache_max_depth,
             64,
             "Maximum suffix tree depth for suffix speculative decoding.");

DEFINE_double(speculative_suffix_max_spec_factor,
              1.0,
              "Suffix speculation max tokens factor relative to match length.");

DEFINE_double(speculative_suffix_max_spec_offset,
              0.0,
              "Suffix speculation max tokens additive offset.");

DEFINE_double(speculative_suffix_min_token_prob,
              0.1,
              "Minimum token probability used in suffix speculation.");

DEFINE_int32(speculative_suffix_max_cached_requests,
             -1,
             "Maximum globally cached requests for suffix speculation (-1 "
             "unlimited, 0 disabled).");

DEFINE_bool(speculative_suffix_use_tree_spec,
            false,
            "Whether to use tree-based suffix speculation instead of path "
            "speculation.");

DEFINE_bool(
    enable_probabilistic_draft,
    false,
    "Draft sampling method for speculative decoding. false (default) = "
    "greedy: draft always picks argmax and draft probabilities are "
    "treated as one-hot during rejection sampling (NO_DRAFT_PROBS) -- "
    "acceptance uses q=1 and the residual zeroes the draft token "
    "(exact, zero extra HBM). true = probabilistic: draft samples from "
    "its own distribution and carries full dense draft probabilities to "
    "validate for exact (p-q)+ rejection (higher acceptance, costs "
    "full-vocab HBM). Both are lossless.");

DEFINE_bool(enable_atb_spec_kernel,
            false,
            "Whether to use ATB speculative kernel.");

namespace xllm {

void SpeculativeConfig::from_flags() {
  XLLM_CONFIG_ASSIGN_FROM_FLAG(draft_model);
  if (config::is_flag_specified("draft_devices")) {
    LOG(WARNING) << "--draft_devices is deprecated and will be removed in a "
                    "future release. Because it's same as --devices.";
  }
  XLLM_CONFIG_ASSIGN_FROM_FLAG(draft_devices);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_probabilistic_draft);
  XLLM_CONFIG_ASSIGN_FROM_FLAG(enable_atb_spec_kernel);
}

void SpeculativeConfig::from_json(const JsonReader& json) {
  XLLM_CONFIG_ASSIGN_FROM_JSON(draft_model);
  // don't read rank-related config
  // XLLM_CONFIG_ASSIGN_FROM_JSON(draft_devices);
  XLLM_CONFIG_ASSIGN_FROM_JSON(num_speculative_tokens);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_algorithm);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_cache_max_depth);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_factor);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_spec_offset);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_min_token_prob);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_max_cached_requests);
  XLLM_CONFIG_ASSIGN_FROM_JSON(speculative_suffix_use_tree_spec);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_probabilistic_draft);
  XLLM_CONFIG_ASSIGN_FROM_JSON(enable_atb_spec_kernel);
}

void SpeculativeConfig::append_config_json(
    nlohmann::ordered_json& config_json) const {
  const SpeculativeConfig default_config;
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, draft_model);
  // don't dump rank-related config
  //  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
  //      config_json, default_config, draft_devices);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, num_speculative_tokens);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_algorithm);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_cache_max_depth);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_factor);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_spec_offset);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_min_token_prob);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_max_cached_requests);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, speculative_suffix_use_tree_spec);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_probabilistic_draft);
  APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT(
      config_json, default_config, enable_atb_spec_kernel);
}

SpeculativeConfig& SpeculativeConfig::get_instance() {
  static SpeculativeConfig config;
  return config;
}

void SpeculativeConfig::initialize() {
  from_flags();
  if (const auto& json_config = config::get_parsed_json_config()) {
    from_json(*json_config);
  }
}

}  // namespace xllm
