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

#include "runtime/dflash_worker_impl.h"

#include <glog/logging.h>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "framework/model/model_args.h"
#include "runtime/spec_input_builder.h"
#include "util/json_reader.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {
namespace {

runtime::Options dflash_main_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options dflash_draft_options(const runtime::Options& options) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .enable_graph(false)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options) {
  input.device_tensors_ready = false;
  input.token_ids_host = make_cpu_int_tensor(token_ids);
  input.positions_host = make_cpu_int_tensor(positions);
  input.token_ids =
      safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
  input.positions =
      safe_to(input.positions_host, position_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

void set_positions_tensor(ForwardInput& input,
                          const std::vector<int32_t>& positions,
                          const torch::TensorOptions& position_options) {
  input.device_tensors_ready = false;
  input.positions_host = make_cpu_int_tensor(positions);
  input.positions =
      safe_to(input.positions_host, position_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

void repeat_sampling_tensor(torch::Tensor& tensor, int32_t repeats) {
  if (tensor.defined()) {
    tensor = tensor.repeat_interleave(/*repeats=*/repeats, /*dim=*/0);
  }
}

void repeat_sampling_params(SamplingParameters& sampling_params,
                            int32_t repeats) {
  repeat_sampling_tensor(sampling_params.frequency_penalties, repeats);
  repeat_sampling_tensor(sampling_params.presence_penalties, repeats);
  repeat_sampling_tensor(sampling_params.repetition_penalties, repeats);
  repeat_sampling_tensor(sampling_params.temperatures, repeats);
  repeat_sampling_tensor(sampling_params.top_p, repeats);
  repeat_sampling_tensor(sampling_params.top_k, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_counts, repeats);
  repeat_sampling_tensor(sampling_params.unique_token_ids_lens, repeats);
  repeat_sampling_tensor(sampling_params.do_sample, repeats);
}

void clear_output_embeddings(ForwardOutput& output) {
  output.sample_output.embeddings = torch::Tensor();
  output.sample_output.selected_embeddings = torch::Tensor();
}

void scale_dp_global_token_nums(ModelInputParams& input_params,
                                int32_t multiplier) {
  for (int32_t& token_num : input_params.parallel.dp_global_token_nums) {
    token_num *= multiplier;
  }
}

std::optional<ForwardOutput> run_worker_no_sync(LLMWorkerImpl& worker,
                                                const ForwardInput& input,
                                                Stream& prepare_stream,
                                                Stream& compute_stream) {
  ForwardInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream);
  return worker.execute_no_sync_on_stream(processed_input, compute_stream);
}

int32_t build_dflash_query_rows(const ForwardInput& input,
                                int32_t mask_token_id,
                                int32_t num_speculative_tokens,
                                int32_t block_size,
                                specBuilder::DecodeBuildBuffers& buf,
                                std::vector<int32_t>& selected_idxes,
                                std::vector<int32_t>& q_cu_seq_lens) {
  CHECK_GE(mask_token_id, 0) << "DFlash mask_token_id must be initialized.";
  CHECK_GT(num_speculative_tokens, 0)
      << "DFlash num_speculative_tokens must be positive.";

  const int32_t num_sequences = input.input_params.meta.num_sequences;
  const int32_t query_width = num_speculative_tokens + 1;
  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  Slice<int32_t> token_ids = {
      input.token_ids_host.data_ptr<int32_t>(),
      static_cast<size_t>(input.token_ids_host.numel())};
  CHECK_GE(static_cast<int32_t>(token_ids.size()), num_sequences)
      << "DFlash input token_ids size is smaller than num_sequences.";

  buf.out_token_ids.reserve(num_sequences * query_width);
  buf.out_positions.reserve(num_sequences * query_width);
  buf.out_new_cache_slots.reserve(num_sequences * query_width);
  buf.out_kv_seq_lens.reserve(num_sequences);
  buf.out_q_seq_lens.reserve(num_sequences);

  selected_idxes.reserve(num_sequences * num_speculative_tokens);
  q_cu_seq_lens.reserve(num_sequences + 1);
  q_cu_seq_lens.emplace_back(0);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    for (int32_t query_idx = 0; query_idx < query_width; ++query_idx) {
      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.token_id = query_idx == 0 ? token_ids[seq_id] : mask_token_id;
      row.position_offset = query_idx;
      row.append_kv_len = false;
      row.append_q_len_one = false;
      row.append_block_table = false;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      if (query_idx > 0) {
        selected_idxes.emplace_back(seq_id * query_width + query_idx);
      }
    }

    specBuilder::append_seq_len_by_layout(buf.out_q_seq_lens, query_width);
    q_cu_seq_lens.emplace_back(q_cu_seq_lens.back() + query_width);
    const int32_t kv_len =
        specBuilder::calc_kv_len(input.input_params.attention.host.kv_seq_lens,
                                 seq_id,
                                 /*offset=*/0) +
        num_speculative_tokens;
    specBuilder::update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
  }

  return query_width;
}

std::vector<int64_t> build_dflash_accepted_context_rows(
    const ForwardInput& input,
    const torch::Tensor& accepted_tokens_cpu,
    int32_t block_size,
    specBuilder::DecodeBuildBuffers& buf) {
  CHECK(accepted_tokens_cpu.defined())
      << "DFlash accepted tokens are undefined.";
  CHECK(accepted_tokens_cpu.device().is_cpu())
      << "DFlash accepted tokens must be on CPU for host row building.";
  CHECK_EQ(accepted_tokens_cpu.scalar_type(), torch::kInt64)
      << "DFlash accepted tokens must be int64.";
  CHECK(accepted_tokens_cpu.is_contiguous())
      << "DFlash accepted tokens must be contiguous.";
  CHECK_EQ(accepted_tokens_cpu.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";

  CHECK_LE(accepted_tokens_cpu.size(0),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "DFlash accepted token batch is too large.";
  CHECK_LE(accepted_tokens_cpu.size(1),
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "DFlash accepted token width is too large.";
  const int32_t batch_size = static_cast<int32_t>(accepted_tokens_cpu.size(0));
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  CHECK_EQ(input.input_params.meta.num_sequences, batch_size)
      << "DFlash accepted token batch mismatch.";

  specBuilder::DecodeRowContext row_ctx =
      specBuilder::make_decode_row_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_positions.reserve(buf.out_positions.size() +
                            static_cast<size_t>(accepted_tokens_cpu.numel()));
  buf.out_new_cache_slots.reserve(
      buf.out_new_cache_slots.size() +
      static_cast<size_t>(accepted_tokens_cpu.numel()));

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      specBuilder::RowSpec row;
      row.seq_id = seq_id;
      row.position_offset = token_idx;
      row.append_token = false;
      row.append_kv_len = false;
      specBuilder::append_decode_row(row_ctx, row, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "DFlash accepted context slots/positions mismatch.";
  return accepted_idxes;
}

}  // namespace

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    dflash_main_options(options),
                    dflash_draft_options(options),
                    /*enable_opt_validate_probs=*/true) {}

bool DFlashWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  bool result =
      MTPWorkerImpl::init_model(model_weights_path, random_seed, master_status);
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    JsonReader reader;
    const std::string config_path = model_weights_path + "/config.json";
    CHECK(reader.parse(config_path))
        << "Failed to parse DFlash draft config: " << config_path;
    std::optional<int32_t> mask_token_id =
        reader.value<int32_t>("dflash_config.mask_token_id");
    CHECK(mask_token_id.has_value())
        << "DFlash draft config requires dflash_config.mask_token_id.";
    mask_token_id_ = mask_token_id.value();

    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    const int64_t num_target_layers =
        static_cast<int64_t>(draft_args.layers_to_capture().size());
    CHECK_GT(num_target_layers, 0)
        << "DFlash requires dflash_config.target_layer_ids.";
    expected_context_hidden_size_ =
        static_cast<int64_t>(draft_args.hidden_size()) * num_target_layers;
  }
  return result;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const ForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    std::optional<ForwardOutput> output =
        run_worker_no_sync(*impl_, input, *prepare_stream_, *compute_stream_);
    std::optional<ForwardOutput> draft_output = run_worker_no_sync(
        *draft_impl_, input, *prepare_stream_, *compute_stream_);
    (void)draft_output;
    if (output.has_value()) {
      clear_output_embeddings(output.value());
    }
    return output;
  }

  const int32_t query_width = options_.num_speculative_tokens() + 1;
  ForwardInput query_input = input;
  query_input.input_params.meta.batch_forward_type =
      BatchForwardType::CHUNKED_PREFILL;
  query_input.input_params.meta.q_max_seq_len = query_width;
  scale_dp_global_token_nums(query_input.input_params, query_width);
  ForwardOutput draft_output =
      run_worker_no_sync(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_)
          .value();
  (void)draft_output;

  ForwardInput validate_input = input;
  scale_dp_global_token_nums(validate_input.input_params, query_width);
  ForwardOutput output =
      run_worker_no_sync(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  clear_output_embeddings(output);
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const ForwardInput& input) {
  Timer timer;
  ForwardOutput output =
      run_worker_no_sync(*impl_, input, *prepare_stream_, *compute_stream_)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              timer.elapsed_seconds());

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    CHECK(input.positions_host.defined())
        << "DFlash prefill requires positions_host.";
    Slice<int32_t> positions = {
        input.positions_host.data_ptr<int32_t>(),
        static_cast<size_t>(input.positions_host.numel())};
    CHECK_EQ(positions.size(), static_cast<size_t>(embeddings.size(0)))
        << "DFlash prefill hidden/position count mismatch.";
    CHECK_EQ(input.input_params.attention.host.new_cache_slots.size(),
             positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    std::vector<int32_t> positions_vec(positions.begin(), positions.end());
    std::vector<int32_t> new_cache_slots =
        input.input_params.attention.host.new_cache_slots;
    timer.reset();
    materialize_dflash_context_kv(
        input, embeddings, positions_vec, std::move(new_cache_slots));
    COUNTER_ADD(speculative_execution_latency_seconds_draft,
                timer.elapsed_seconds());
  }

  if (input.sampling_params.selected_token_idxes.defined()) {
    const torch::Tensor& target_hidden =
        output.sample_output.selected_embeddings.defined()
            ? output.sample_output.selected_embeddings
            : embeddings;
    embedding_cache_->write_prefill_target_context(
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        output.sample_output.next_tokens,
        target_hidden,
        input.sampling_params.selected_token_idxes);
  }
  clear_output_embeddings(output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::vector<ForwardOutput> DFlashWorkerImpl::run_decode_draft(
    const ForwardInput& input,
    const std::vector<EmbeddingCache::DecodeState>& last_states,
    ForwardInput& validate_input) {
  Timer timer;
  CHECK_GE(mask_token_id_, 0) << "DFlash mask_token_id is not initialized.";
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  CHECK_EQ(last_states.size(), static_cast<size_t>(num_sequences))
      << "DFlash decode target state count mismatch.";

  ForwardInput query_input;
  prepare_dflash_query_inputs(input, query_input);
  prepare_validate_inputs(input, validate_input);

  ForwardOutput draft_output =
      run_worker_no_sync(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_)
          .value();
  process_draft_sample_output(draft_output.sample_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return split_dflash_draft_output(draft_output);
}

void DFlashWorkerImpl::prepare_dflash_query_inputs(const ForwardInput& input,
                                                   ForwardInput& query_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input;
  query_input.device_tensors_ready = false;
  ModelInputParams& input_params = query_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  input_params.embedding.dflash_context_hidden = false;

  specBuilder::DecodeBuildBuffers buf;
  std::vector<int32_t> selected_idxes;
  std::vector<int32_t> q_cu_seq_lens;
  const int32_t query_width =
      build_dflash_query_rows(input,
                              mask_token_id_,
                              options_.num_speculative_tokens(),
                              options_.block_size(),
                              buf,
                              selected_idxes,
                              q_cu_seq_lens);

  set_token_position_tensors(query_input,
                             buf.out_token_ids,
                             buf.out_positions,
                             input.token_ids.options(),
                             input.positions.options());
  input_params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
  specBuilder::update_input_params(input_params,
                                   buf,
                                   query_width,
                                   std::move(buf.out_q_seq_lens),
                                   std::move(q_cu_seq_lens),
                                   buf.meta.kv_max_seq_len,
                                   std::move(buf.out_kv_seq_lens),
                                   /*update_block_tables=*/false);
  scale_dp_global_token_nums(input_params, query_width);
  input_params.attention.rebuild_device_buffer(device_);

  torch::TensorOptions idx_options =
      torch::TensorOptions().dtype(torch::kInt).device(device_);
  query_input.sampling_params.selected_token_idxes =
      torch::tensor(selected_idxes, idx_options);
  query_input.sampling_params.sample_idxes =
      torch::arange(static_cast<int64_t>(selected_idxes.size()), idx_options);
  repeat_sampling_params(query_input.sampling_params,
                         options_.num_speculative_tokens());
  query_input.device_tensors_ready = true;
}

void DFlashWorkerImpl::materialize_dflash_context_kv(
    const ForwardInput& input,
    const torch::Tensor& context_hidden,
    const std::vector<int32_t>& positions,
    std::vector<int32_t> new_cache_slots) {
  CHECK(context_hidden.defined()) << "DFlash context hidden is undefined.";
  CHECK_EQ(context_hidden.dim(), 2) << "DFlash context hidden must be 2D.";
  CHECK_EQ(context_hidden.size(1), expected_context_hidden_size_)
      << "DFlash context hidden size must be hidden_size * "
      << "target_layer_ids.size().";
  CHECK_EQ(positions.size(), static_cast<size_t>(context_hidden.size(0)))
      << "DFlash context hidden/position count mismatch.";
  CHECK_EQ(new_cache_slots.size(), positions.size())
      << "DFlash context hidden/cache slot count mismatch.";

  StreamEventPtr context_hidden_ready_event = compute_stream_->record_event();
  if (context_hidden_ready_event == nullptr) {
    compute_stream_->synchronize();
  }
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  CHECK(prepare_stream_->wait_event(context_hidden_ready_event))
      << "failed to wait DFlash context hidden ready event";
  ForwardInput context_input = input;
  context_input.device_tensors_ready = false;
  ModelInputParams& input_params = context_input.input_params;
  input_params.embedding.input_embedding = context_hidden.to(device_);
  input_params.embedding.dflash_context_hidden = true;
  set_positions_tensor(context_input, positions, input.positions.options());
  input_params.attention.host.new_cache_slots = std::move(new_cache_slots);
  input_params.attention.rebuild_device_buffer(device_);
  context_input.sampling_params.selected_token_idxes = torch::Tensor();
  context_input.sampling_params.sample_idxes = torch::Tensor();
  context_input.device_tensors_ready = true;
  run_worker_no_sync(
      *draft_impl_, context_input, *prepare_stream_, *compute_stream_)
      .value();
}

void DFlashWorkerImpl::write_target_context_to_cache(
    const ForwardInput& input,
    const SampleOutput& validate_output) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  torch::Tensor accepted_tokens = validate_output.next_tokens;
  CHECK(accepted_tokens.defined()) << "DFlash accepted tokens are undefined.";
  if (!accepted_tokens.device().is_cpu()) {
    accepted_tokens = accepted_tokens.to(torch::kCPU);
  }
  if (accepted_tokens.scalar_type() != torch::kInt64) {
    accepted_tokens = accepted_tokens.to(torch::kInt64);
  }
  accepted_tokens = accepted_tokens.contiguous();

  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  specBuilder::DecodeBuildBuffers buf;
  std::vector<int64_t> accepted_idxes = build_dflash_accepted_context_rows(
      input, accepted_tokens, options_.block_size(), buf);
  torch::TensorOptions host_index_options =
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  torch::TensorOptions device_index_options =
      torch::TensorOptions()
          .dtype(torch::kLong)
          .device(accepted_embeddings.device());
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  torch::Tensor accepted_index =
      safe_to(torch::tensor(accepted_idxes, host_index_options),
              device_index_options,
              /*non_blocking=*/true);
  torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, accepted_index);
  materialize_dflash_context_kv(input,
                                context_hidden,
                                buf.out_positions,
                                std::move(buf.out_new_cache_slots));
  MTPWorkerImpl::write_target_context_to_cache(input, validate_output);
}

std::vector<ForwardOutput> DFlashWorkerImpl::split_dflash_draft_output(
    const ForwardOutput& draft_output) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_draft_tokens =
      draft_output.sample_output.next_tokens.numel();
  CHECK_EQ(num_draft_tokens % num_speculative_tokens, 0)
      << "DFlash draft token count mismatch.";
  const int32_t batch_size = num_draft_tokens / num_speculative_tokens;

  torch::Tensor draft_token_ids = draft_output.sample_output.next_tokens.view(
      {batch_size, num_speculative_tokens});
  torch::Tensor draft_probs = draft_output.sample_output.probs;
  CHECK_EQ(draft_probs.numel(), num_draft_tokens)
      << "DFlash draft output requires selected draft probs.";
  draft_probs = draft_probs.view({batch_size, num_speculative_tokens});
  std::vector<ForwardOutput> draft_outputs;
  draft_outputs.reserve(num_speculative_tokens);
  for (int32_t step = 0; step < num_speculative_tokens; ++step) {
    ForwardOutput step_output;
    step_output.sample_output.next_tokens =
        draft_token_ids.select(/*dim=*/1, /*index=*/step);
    step_output.sample_output.probs =
        draft_probs.select(/*dim=*/1, /*index=*/step);
    draft_outputs.emplace_back(std::move(step_output));
  }
  return draft_outputs;
}

}  // namespace xllm
