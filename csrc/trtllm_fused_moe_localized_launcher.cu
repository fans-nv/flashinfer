/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Localized (split-N) NVFP4 MoE launcher. It includes the stock launcher so the module also
// carries a same-ABI stock path, which is the bit-exact reference (mode 1). The partition
// streams are the caller's (green-context streams created by the framework).
#include <mutex>

#include "trtllm_fused_moe_kernel_launcher.cu"

namespace flashinfer {

namespace {

constexpr int kLocalizedPartitions = 2;

// Fork/join events, created once per device on the first (eager, pre-capture) call and kept for
// the process. Reuse is safe: every record/wait pair is stream ordered.
struct LocalizedEvents {
  cudaEvent_t fork{};
  cudaEvent_t fc1[kLocalizedPartitions]{};
  cudaEvent_t fc2[kLocalizedPartitions]{};
};

LocalizedEvents& localizedEvents(int device) {
  static std::mutex mutex;
  static std::map<int, LocalizedEvents> events;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = events.find(device);
  if (it == events.end()) {
    LocalizedEvents e;
    auto mk = [](cudaEvent_t* ev) {
      FLASHINFER_CHECK(cudaEventCreateWithFlags(ev, cudaEventDisableTiming) == cudaSuccess,
                       "cudaEventCreate failed");
    };
    mk(&e.fork);
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      mk(&e.fc1[p]);
      mk(&e.fc2[p]);
    }
    it = events.emplace(device, e).first;
  }
  return it->second;
}

// Test-only skew injection for the join/PDL test: spins on one partition's stream right before
// its FC1 or FC2. Production passes cycles == 0, so nothing is launched.
__global__ void localizedDebugSpin(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

struct DebugDelay {
  int partition{-1};
  int phase{0};  // 1 = before FC1, 2 = before FC2
  long long cycles{0};
};

void checkCuda(cudaError_t err, char const* what) {
  FLASHINFER_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

}  // namespace

class FP4BlockScaleLocalizedLauncher : public FP4BlockScaleLauncher {
 public:
  struct Partition {
    TensorView gemm1_weights;
    TensorView gemm1_weights_scale;
    TensorView gemm2_weights;
    TensorView gemm2_weights_scale;
    cudaStream_t stream{};
    int32_t sm_count{};
  };

  // mode 0: localized fork/join over the two partitions.
  // mode 1: stock single-stream run inside this module (full weights passed as partition 0);
  //         the same-ABI bit-exact reference.
  template <typename... Ts>
  FP4BlockScaleLocalizedLauncher(int64_t mode, std::array<Partition, kLocalizedPartitions> parts,
                                 bool poison_intermediates, DebugDelay delay, Ts&&... base)
      : FP4BlockScaleLauncher(std::forward<Ts>(base)...),
        mode_(mode),
        parts_(parts),
        poison_(poison_intermediates),
        delay_(delay) {}

  Array<Tensor> intermediates() const {
    Array<Tensor> r;
    r.push_back(gemm1_output);
    if (gemm1_output_scale.has_value()) r.push_back(gemm1_output_scale.value());
    r.push_back(gemm2_output);
    r.push_back(expanded_idx_to_permuted_idx);
    return r;
  }

 protected:
  void select_moe_tactic(int64_t& moe_tactic, int32_t effectiveTopK,
                         int32_t effectiveLocalExperts) override {
    if (mode_ != 0) {
      FusedMoeLauncher::select_moe_tactic(moe_tactic, effectiveTopK, effectiveLocalExperts);
      return;
    }
    int32_t const minSm = std::min(parts_[0].sm_count, parts_[1].sm_count);
    if (moe_tactic == -1) {
      // Only the no-tactic path enumerates the catalog; an explicit tactic is checked alone,
      // as the stock launcher does, so the per-call cost does not scale with the bundle.
      auto const valid = moe_runner->getValidPartitionConfigIndices(
          effectiveTopK, args->hidden_size, args->intermediate_size, effectiveLocalExperts,
          args->num_tokens, minSm);
      FLASHINFER_CHECK(!valid.empty(), "No split-N MoE config for tile_N=", tile_tokens_dim,
                       " H=", args->hidden_size, " I=", args->intermediate_size);
      moe_tactic = valid.front();
      return;
    }
    FLASHINFER_CHECK(moe_runner->isValidPartitionConfigIndex(
                         moe_tactic, effectiveTopK, args->hidden_size, args->intermediate_size,
                         effectiveLocalExperts, args->num_tokens, minSm),
                     "MoE tactic ", moe_tactic, " for tile_N=", tile_tokens_dim,
                     " is not valid for the split-N partitions (stale or mismatched tuning "
                     "record?)");
  }

  std::tuple<int32_t, int32_t> moe_workspace_sizes(int64_t moe_tactic) override {
    if (mode_ != 0) return FusedMoeLauncher::moe_workspace_sizes(moe_tactic);
    ws_sizes_ = moe_runner->getPartitionWorkspaceSizeInBytes(*args, moe_tactic);
    return ws_sizes_;
  }

  void run_moe(bool enable_pdl) override {
    int const device = hidden_states.device().device_id;
    cudaStream_t const caller = get_stream(hidden_states.device());
    if (poison_) {
      // Canary fill: any byte a slice fails to write, or writes outside its slice, differs
      // from the stock path run with the same fill.
      checkCuda(cudaMemsetAsync(gemm1_output.data_ptr(), 0xA5,
                                gemm1_output.numel() * gemm1_output.dtype().bits / 8, caller),
                "poison gemm1_output");
      if (gemm1_output_scale.has_value()) {
        checkCuda(cudaMemsetAsync(gemm1_output_scale.value().data_ptr(), 0x5A,
                                  gemm1_output_scale.value().numel(), caller),
                  "poison gemm1_output_scale");
      }
      checkCuda(cudaMemsetAsync(gemm2_output.data_ptr(), 0xC3,
                                gemm2_output.numel() * gemm2_output.dtype().bits / 8, caller),
                "poison gemm2_output");
    }
    if (mode_ != 0) {
      FusedMoeLauncher::run_moe(enable_pdl);
      return;
    }

    int32_t const H = args->hidden_size;
    int32_t const I = args->intermediate_size;
    LocalizedEvents& ev = localizedEvents(device);
    auto maybeDelay = [&](int p, int phase) {
      if (delay_.cycles > 0 && delay_.partition == p && delay_.phase == phase) {
        localizedDebugSpin<<<1, 1, 0, parts_[p].stream>>>(delay_.cycles);
      }
    };

    // Mutable GEMM scratch is per partition; routing buffers and gemm outputs are shared.
    std::array<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace, kLocalizedPartitions> ws{
        workspace, workspace};
    Tensor ws1_fc1, ws1_fc2;
    if (std::get<0>(ws_sizes_) > 0) {
      ws1_fc1 = alloc_tensor({std::get<0>(ws_sizes_)}, dl_int8, hidden_states.device());
      ws[1].bmm1_workspace = ws1_fc1.data_ptr();
    }
    if (std::get<1>(ws_sizes_) > 0) {
      ws1_fc2 = alloc_tensor({std::get<1>(ws_sizes_)}, dl_int8, hidden_states.device());
      ws[1].bmm2_workspace = ws1_fc2.data_ptr();
    }

    std::array<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs, kLocalizedPartitions> pa;
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      pa[p] = *args;
      pa[p].gemm1_weights = parts_[p].gemm1_weights.data_ptr();
      pa[p].gemm1_weights_scale = parts_[p].gemm1_weights_scale.data_ptr();
      pa[p].gemm2_weights = parts_[p].gemm2_weights.data_ptr();
      pa[p].gemm2_weights_scale = parts_[p].gemm2_weights_scale.data_ptr();
    }

    // Fork: both partitions start after routing on the caller stream.
    checkCuda(cudaEventRecord(ev.fork, caller), "record fork");
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      checkCuda(cudaStreamWaitEvent(parts_[p].stream, ev.fork, 0), "wait fork");
    }
    // FC1: partition p writes activated columns [p*I/2, (p+1)*I/2) of the shared output.
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      auto a = pa[p];
      a.intermediate_size = I / 2;
      maybeDelay(p, 1);
      moe_runner->runGemm1(a, ws[p], device, parts_[p].stream, moe_tactic, enable_pdl,
                           {I, static_cast<int64_t>(p) * (I / 2), parts_[p].sm_count});
      checkCuda(cudaEventRecord(ev.fc1[p], parts_[p].stream), "record fc1");
    }
    // Both FC1 halves must land before either FC2 half reads the full K = I.
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      checkCuda(cudaStreamWaitEvent(parts_[p].stream, ev.fc1[1 - p], 0), "wait fc1");
    }
    // FC2: partition p writes hidden columns [p*H/2, (p+1)*H/2).
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      auto a = pa[p];
      a.hidden_size = H / 2;
      maybeDelay(p, 2);
      moe_runner->runGemm2(a, ws[p], device, parts_[p].stream, moe_tactic, enable_pdl,
                           {H, static_cast<int64_t>(p) * (H / 2), parts_[p].sm_count});
      checkCuda(cudaEventRecord(ev.fc2[p], parts_[p].stream), "record fc2");
    }
    // Join back to the caller before finalize and before any buffer can be released.
    for (int p = 0; p < kLocalizedPartitions; ++p) {
      checkCuda(cudaStreamWaitEvent(caller, ev.fc2[p], 0), "wait fc2");
    }
    moe_runner->runFinalize(*args, workspace, caller, enable_pdl);
  }

 private:
  int64_t mode_;
  std::array<Partition, kLocalizedPartitions> parts_;
  bool poison_;
  DebugDelay delay_;
  std::tuple<int32_t, int32_t> ws_sizes_{0, 0};
};

// The trtllm_fp4_block_scale_moe arguments minus LoRA, fused shared experts, routing replay, DA
// and valid dims (all rejected by the localized path), with the four weight tensors being
// partition 0's shards, followed by partition 1's shards and the partition resources. In mode 1
// partition 0 holds the FULL weights and partition 1 is ignored. return_intermediates appends
// [gemm1_output, gemm1_output_scale, gemm2_output, expanded_idx_to_permuted_idx] after a canary
// fill. debug_delay_{partition,phase,cycles} is the test-only skew hook; 0 cycles in production.
// per_token_scales (last, so the earlier positions are unchanged) selects mode Q: FC1 applies the
// fp32 [T] token scale in its epilogue and FC2 stays static; mode F is not supported here.
Array<Tensor> trtllm_fp4_block_scale_moe_localized(
    int64_t routing_input_mode, Optional<TensorView> routing_logits, TensorView topk_ids,
    TensorView topk_weights, Optional<TensorView> routing_bias, TensorView hidden_states,
    Optional<TensorView> hidden_states_scale, TensorView gemm1_weights,
    TensorView gemm1_weights_scale, Optional<TensorView> gemm1_bias,
    Optional<TensorView> gemm1_alpha, Optional<TensorView> gemm1_beta,
    Optional<TensorView> gemm1_clamp_limit, TensorView gemm2_weights,
    TensorView gemm2_weights_scale, Optional<TensorView> gemm2_bias,
    Optional<TensorView> output1_scales_scalar, Optional<TensorView> output1_scales_gate_scalar,
    Optional<TensorView> output2_scales_scalar, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool do_finalize, bool enable_pdl, int64_t act_type,
    TensorView output, int64_t tile_n, int64_t config, bool norm_topk_prob,
    TensorView gemm1_weights_p1, TensorView gemm1_weights_scale_p1, TensorView gemm2_weights_p1,
    TensorView gemm2_weights_scale_p1, int64_t stream_p0, int64_t stream_p1, int64_t sm_p0,
    int64_t sm_p1, int64_t mode, bool return_intermediates, int64_t debug_delay_partition,
    int64_t debug_delay_phase, int64_t debug_delay_cycles, Optional<TensorView> per_token_scales) {
  TVM_FFI_ICHECK(mode == 0 || mode == 1) << "mode must be 0 (localized) or 1 (stock reference)";
  TVM_FFI_ICHECK(hidden_states.dtype() == dl_uint8 && hidden_states_scale.has_value() &&
                 hidden_states_scale.value().dtype() == dl_float8_e4m3fn)
      << "localized MoE admits NVFP4 activations only";
  TVM_FFI_ICHECK(isGatedActivation(static_cast<ActivationType>(act_type)))
      << "localized MoE requires a gated activation";
  TVM_FFI_ICHECK(!gemm1_bias.has_value() && !gemm2_bias.has_value())
      << "localized MoE does not admit bias";
  TVM_FFI_ICHECK(tile_n > 0)
      << "tile_n must be resolved by the caller (AutoTuner record or pinned default)";
  TVM_FFI_ICHECK(debug_delay_cycles >= 0 && debug_delay_phase >= 0 && debug_delay_phase <= 2)
      << "invalid debug delay";

  int const num_tokens = hidden_states.size(0);
  int const hidden_size = hidden_states.size(1) * 2;
  TVM_FFI_ICHECK(hidden_size % 256 == 0 && intermediate_size % 256 == 0)
      << "admission: H and I must be multiples of 256";
  if (per_token_scales.has_value()) {
    TVM_FFI_ICHECK(per_token_scales.value().dtype() == dl_float32 &&
                   per_token_scales.value().ndim() == 1 &&
                   per_token_scales.value().size(0) >= num_tokens)
        << "per_token_scales must be fp32 [>= num_tokens]";
  }
  // Shards carry half the FC1 rows and half the FC2 rows of the full tensors.
  int64_t const rows1 = mode == 0 ? intermediate_size : 2 * intermediate_size;
  int64_t const rows2 = mode == 0 ? hidden_size / 2 : hidden_size;
  auto check_w = [&](TensorView const& w, TensorView const& s, int64_t rows, int64_t k,
                     char const* name) {
    TVM_FFI_ICHECK(w.dtype() == dl_uint8 && w.ndim() == 3 && w.size(0) == local_num_experts &&
                   w.size(1) == rows && w.size(2) == k / 2)
        << name << " weights have the wrong shape for this partition";
    TVM_FFI_ICHECK(s.dtype() == dl_float8_e4m3fn && s.numel() == local_num_experts * rows * k / 16)
        << name << " block scales have the wrong size for this partition (NVFP4, vec 16)";
  };
  check_w(gemm1_weights, gemm1_weights_scale, rows1, hidden_size, "gemm1[p0]");
  check_w(gemm2_weights, gemm2_weights_scale, rows2, intermediate_size, "gemm2[p0]");
  if (mode == 0) {
    check_w(gemm1_weights_p1, gemm1_weights_scale_p1, rows1, hidden_size, "gemm1[p1]");
    check_w(gemm2_weights_p1, gemm2_weights_scale_p1, rows2, intermediate_size, "gemm2[p1]");
    TVM_FFI_ICHECK(stream_p0 != 0 && stream_p1 != 0 && sm_p0 > 0 && sm_p1 > 0)
        << "partition streams and SM counts are required";
  }

  auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
  args->num_tokens = num_tokens;
  args->num_experts = num_experts;
  args->hidden_size = hidden_size;
  // No valid dims here, so GEMM2's N is the full hidden size (see moe_hidden_size_output()).
  args->hidden_size_output =
      moe_hidden_size_output(output.size(1), hidden_size, Optional<int64_t>());
  args->top_k = top_k;
  args->num_fused_shared_experts = 0;
  args->n_group = n_group.value_or(0);
  args->topk_group = topk_group.value_or(0);
  args->local_expert_offset = local_expert_offset;
  args->local_num_experts = local_num_experts;
  args->intermediate_size = intermediate_size;
  args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
  args->do_finalize = do_finalize;
  args->output = output.data_ptr();
  args->output_scale = nullptr;

  using Launcher = FP4BlockScaleLocalizedLauncher;
  std::array<Launcher::Partition, kLocalizedPartitions> parts{
      Launcher::Partition{gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                          reinterpret_cast<cudaStream_t>(stream_p0), static_cast<int32_t>(sm_p0)},
      Launcher::Partition{gemm1_weights_p1, gemm1_weights_scale_p1, gemm2_weights_p1,
                          gemm2_weights_scale_p1, reinterpret_cast<cudaStream_t>(stream_p1),
                          static_cast<int32_t>(sm_p1)}};
  DebugDelay const delay{static_cast<int>(debug_delay_partition),
                         static_cast<int>(debug_delay_phase), debug_delay_cycles};
  Launcher launcher(
      mode, parts, return_intermediates, delay, static_cast<RoutingInputMode>(routing_input_mode),
      routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights,
      gemm1_weights_scale, gemm1_bias, gemm1_alpha, gemm1_beta, gemm1_clamp_limit, gemm2_weights,
      gemm2_weights_scale, gemm2_bias, output1_scales_scalar, output1_scales_gate_scalar,
      output2_scales_scalar, per_token_scales, topk_ids, topk_weights);
  launcher.init(std::move(args), tile_n, routing_method_type, /*use_shuffled_weight=*/true,
                /*weight_layout=*/0, static_cast<ActivationType>(act_type), btg::Dtype::E2m1,
                btg::Dtype::E2m1, static_cast<int64_t>(batchedGemm::gemm::BiasType::None),
                norm_topk_prob);
  launcher.set_per_token_scale_gemm2(false);
  Array<Tensor> result = launcher.run(config, enable_pdl).to_ffi();
  if (return_intermediates) {
    for (auto const& t : launcher.intermediates()) result.push_back(t);
  }
  return result;
}

// Partition-valid MoE configs for (tile_n, shape, SM counts, activation, FC1 per-token), for
// tactic selection in Python. A -1 terminator tells an empty set apart from a failed call.
Array<int64_t> trtllm_fp4_moe_localized_valid_configs(int64_t tile_n, int64_t top_k,
                                                      int64_t hidden_size,
                                                      int64_t intermediate_size,
                                                      int64_t num_local_experts, int64_t num_tokens,
                                                      int64_t sm_p0, int64_t sm_p1,
                                                      int64_t act_type, bool per_token) {
  // Same Runner options as the launcher: FC1 per-token in mode Q, FC2 never.
  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner runner(
      btg::Dtype::E2m1, btg::Dtype::E2m1, false, static_cast<int32_t>(tile_n),
      static_cast<ActivationType>(act_type), true, batchedGemm::gemm::MatrixLayout::MajorK,
      batchedGemm::gemm::BiasType::None, per_token, /*usePerTokenScalingGemm2=*/false);
  // The SM budget is passed as a scalar: a braced std::vector<int32_t> built here arrived at the
  // runner as {sm_p0, 0} in -O3 builds, which silently emptied the valid set.
  std::vector<int64_t> const valid = runner.getValidPartitionConfigIndices(
      static_cast<int32_t>(top_k), static_cast<int32_t>(hidden_size),
      static_cast<int32_t>(intermediate_size), static_cast<int32_t>(num_local_experts),
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(std::min(sm_p0, sm_p1)));
  Array<int64_t> out;
  for (int64_t c : valid) out.push_back(c);
  out.push_back(-1);
  return out;
}

// FC1 and FC2 kernel names of every MoE config of one tile, flattened as
// [fc1(0), fc2(0), fc1(1), fc2(1), ...] in config-index order. Default tactics are pinned by
// kernel name and resolved to an index through this table when the module is loaded.
Array<tvm::ffi::String> trtllm_fp4_moe_localized_config_kernels(int64_t tile_n, int64_t act_type,
                                                                bool per_token) {
  tensorrt_llm::kernels::trtllmgen_moe::MoE::Runner runner(
      btg::Dtype::E2m1, btg::Dtype::E2m1, false, static_cast<int32_t>(tile_n),
      static_cast<ActivationType>(act_type), true, batchedGemm::gemm::MatrixLayout::MajorK,
      batchedGemm::gemm::BiasType::None, per_token, /*usePerTokenScalingGemm2=*/false);
  Array<tvm::ffi::String> out;
  for (int64_t i = 0; i < runner.getNumConfigs(); ++i) {
    auto const c = runner.getConfigComponents(i);
    char const* fc1 = tensorrt_llm::kernels::trtllmGenBmmConfigName(c.gemm1Config);
    char const* fc2 = tensorrt_llm::kernels::trtllmGenBmmConfigName(c.gemm2Config);
    TVM_FFI_ICHECK(fc1 != nullptr && fc2 != nullptr)
        << "MoE config " << i << " refers to a BMM config outside the catalog";
    out.push_back(tvm::ffi::String(fc1));
    out.push_back(tvm::ffi::String(fc2));
  }
  return out;
}

#ifndef FLASHINFER_LOCALIZED_MOE_BUNDLE_MANIFEST_SHA256
#define FLASHINFER_LOCALIZED_MOE_BUNDLE_MANIFEST_SHA256 "unknown"
#endif

// Build identity: [bundle checksums.txt sha256, GEN export version]. The tuning cache key and
// the loader's pin check use it instead of hashing the .so file.
Array<tvm::ffi::String> trtllm_fp4_moe_localized_module_info() {
  Array<tvm::ffi::String> out;
  out.push_back(tvm::ffi::String(FLASHINFER_LOCALIZED_MOE_BUNDLE_MANIFEST_SHA256));
  out.push_back(tvm::ffi::String(tensorrt_llm::kernels::trtllmGenBmmExportVersion()));
  return out;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_moe_localized_module_info,
                              trtllm_fp4_moe_localized_module_info);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_block_scale_moe_localized,
                              trtllm_fp4_block_scale_moe_localized);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_moe_localized_valid_configs,
                              trtllm_fp4_moe_localized_valid_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_fp4_moe_localized_config_kernels,
                              trtllm_fp4_moe_localized_config_kernels);

}  // namespace flashinfer
