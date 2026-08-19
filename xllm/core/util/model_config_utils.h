/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xllm {

class JsonReader;

namespace util {

// RedHat/vLLM "speculators"-format draft config vocabulary. Single source of
// truth so the config loader, model-type resolver, and worker draft-config
// readers never diverge on key spellings.
inline constexpr const char* kSpeculatorsModelTypeKey =
    "speculators_model_type";
inline constexpr const char* kSpeculatorsTransformerConfigKey =
    "transformer_layer_config";
inline constexpr const char* kSpeculatorsBackboneModelTypeKey =
    "transformer_layer_config.model_type";
inline constexpr const char* kDsparkDraftArchitecture = "DSparkDraftModel";
inline constexpr const char* kDflashDraftArchitecture = "DFlashDraftModel";
inline constexpr const char* kQwen3DsparkModelType = "qwen3_dspark";

// Draft config keys carrying the target layer ids to capture, by precedence.
// (Hooks run before a layer, so the consumer offsets the id by +1.)
inline const std::vector<std::string> kSpeculatorsCaptureLayerIdKeys = {
    "dspark_target_layer_ids",
    "target_layer_ids",
    "dflash_config.target_layer_ids",
    "aux_hidden_state_layer_ids",
};

std::string get_model_type(const JsonReader& reader,
                           const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

std::string get_model_type(const std::filesystem::path& model_path,
                           std::optional<std::string> backend = std::nullopt);

}  // namespace util
}  // namespace xllm
