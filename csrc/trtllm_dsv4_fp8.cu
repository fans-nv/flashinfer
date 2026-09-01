/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The DSv4 sparse-MLA FP8-output planner and its execution entry point. See
// include/flashinfer/trtllm/fmha/dsv4Plan.h for the two-stage model and the output contract.

#include <cuda_runtime_api.h>
#include <flashinfer/allocator.h>
#include <flashinfer/exception.h>
#include <flashinfer/trtllm/common.h>
#include <flashinfer/trtllm/fmha/decoder_impl_common.h>
#include <flashinfer/trtllm/fmha/fmhaReduction.h>
#include <flashinfer/trtllm/fmha/fmhaRunnerParams.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <flashinfer/trtllm/fmha/dsv4PlanResolve.cuh>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "tvm/ffi/container/array.h"
#include "tvm/ffi/error.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;

namespace flashinfer {
namespace dsv4 {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Checked arithmetic: the extents below are host-known, but their products are large enough
// that a silent overflow would be a memory-safety bug.

namespace {

int64_t checkedMul(int64_t a, int64_t b, char const* what) {
  FLASHINFER_CHECK(a >= 0 && b >= 0, "negative extent computing", what);
  FLASHINFER_CHECK(a == 0 || b <= INT64_MAX / a, "64-bit overflow computing", what);
  return a * b;
}

int64_t alignUp(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

}  // namespace

std::string ProblemKey::describe() const {
  std::ostringstream os;
  os << "dev=" << deviceOrdinal << " ctx=" << contextId << " reg=" << registryGeneration
     << " dtypeQ=" << dtypeQ << " dtypeKv=" << dtypeKv << " dtypeO=" << dtypeO
     << " heads=" << numHeadsQ << "/" << numHeadsKv << " hd=" << headDimQk << "/" << headDimV
     << " page=" << numTokensPerPage << " batch=" << batchSize << " maxSeqLenQ=" << maxSeqLenQ
     << " topK=" << sparseTopK << " Q=" << sumOfSeqLensQ << " sm=" << multiProcessorCount
     << " win=" << attentionWindowSize << " mode=" << outputMode << " span=[" << span.tokenBase
     << "," << span.tokenBase + span.tokenCount << ")/" << span.totalTokens << "/"
     << span.tokenCapacity << " b=" << span.batchSize;
  return os.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Constant today -- the trtllm-gen kernel factory never unloads a module -- but carried in the
// key so that a future reload path can bump it and invalidate outstanding plans.

namespace {
std::atomic<uint64_t> gRegistryGeneration{1};
}  // namespace

uint64_t currentRegistryGeneration() { return gRegistryGeneration.load(); }

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

// Build the stage-1 selection input: exactly what an ordinary BF16 DSv4 sparse-MLA call
// constructs. The FP8 output request must not appear in it, and in particular must not pin
// mMultiCtasKvMode, the scheduler, or 2CTA.
TllmGenFmhaRunnerParams makeBaseRunnerParams(ProblemKey const& key) {
  TllmGenFmhaRunnerParams p{};
  p.mHeadDimQk = key.headDimQk;
  p.mHeadDimV = key.headDimV;
  p.mNumHeadsQ = key.numHeadsQ;
  p.mNumHeadsKv = key.numHeadsKv;
  p.mNumHeadsQPerKv = key.numHeadsQ / key.numHeadsKv;
  p.mBatchSize = key.batchSize;
  p.mMaxSeqLenKv = key.sparseTopK;
  p.mMaxNumPagesPerSeqKv = key.sparseTopK;
  p.mNumTokensPerPage = key.numTokensPerPage;
  p.mQkvLayout = QkvLayout::PagedKv;
  p.mMultiProcessorCount = key.multiProcessorCount;
  p.mChunkedAttentionSize = INT_MAX;
  p.mAttentionWindowSize = key.attentionWindowSize;
  p.mMaxSeqLenQ = key.maxSeqLenQ;
  p.mSumOfSeqLensQ = key.sumOfSeqLensQ;
  p.mUsesSharedPagedKvIdx = true;
  p.mSparseMlaType = TrtllmGenSparseMlaType::DynamicTokenSparse;
  p.mSparseMlaTopK = key.sparseTopK;
  p.mHasSlidingWindowKvPool = true;
  p.mMaskType = TrtllmGenAttentionMaskType::Dense;
  p.mKernelType = FmhaKernelType::Generation;
  p.mTileScheduler = TileScheduler::Static;
  p.mMultiCtasKvMode = true;
  p.mSkipsSoftmaxWhenPossible = false;
  p.mUsesDsv4Ue8m0ScaleO = false;  // Stage 1 selects on BF16; FP8 is decoration.
  return p;
}

// The region extents the reduction path addresses, derived from the frozen plan.
void computeWorkspaceLayout(SparseMlaPlan& plan) {
  ProblemKey const& key = plan.key;
  auto const& meta = plan.baseMeta;
  auto const& cta = plan.ctaLaunch;
  WorkspaceLayout layout{};

  int64_t offset = 0;
  auto place = [&offset](int64_t bytes, int64_t* outOffset, int64_t* outBytes) {
    *outOffset = offset;
    *outBytes = bytes;
    offset = alignUp(offset + bytes, WorkspaceLayout::kAlignment);
  };

  if (isMultiCtasKvEnabled(plan.selectParams.mMultiCtasKvMode)) {
    int32_t const numHeadsQPerKv = key.numHeadsQ / key.numHeadsKv;
    int32_t const numHeadsPerCta =
        meta.mGroupsHeadsQ ? std::min<int32_t>(numHeadsQPerKv, meta.mTileSizeQ) : 1;
    int32_t const numCtasForAllHeads = key.numHeadsQ / numHeadsPerCta;
    int32_t const headDimEff = meta.m2CtaMma ? meta.mHeadDimPerCtaV * 2 : meta.mHeadDimPerCtaV;
    int32_t const numHeadDimCtasV = meta.mHeadDimV / headDimEff;

    int64_t rows = checkedMul(key.batchSize, numCtasForAllHeads, "partial rows");
    rows = checkedMul(rows, numHeadDimCtasV, "partial rows");
    rows = checkedMul(rows, std::max(1, cta.mMaxNumCtasQ), "partial rows");
    rows = checkedMul(rows, std::max(1, cta.mMaxNumCtasKv), "partial rows");
    rows = checkedMul(rows, meta.mTileSizeQ, "partial rows");

    // The stock allocator gives the statistics region no explicit extent: partial O starts at
    // a fixed offset of multiProcessorCount * stepQ float2 elements and statistics take the
    // rest. Size both explicitly, never below that historical offset.
    int64_t const statsFloor = checkedMul(key.multiProcessorCount, meta.mStepQ, "stats floor");
    int64_t const statsElems = std::max(rows, statsFloor);
    place(checkedMul(statsElems, static_cast<int64_t>(sizeof(float) * 2), "partial stats bytes"),
          &layout.partialStatsOffset, &layout.partialStatsBytes);
    place(checkedMul(checkedMul(rows, headDimEff, "partial O elems"), 2, "partial O bytes"),
          &layout.partialOOffset, &layout.partialOBytes);
  }

  // The P3 merged-BF16 intermediate. It must keep the frozen base extent of Q rows and
  // batch-global row addressing so the unchanged base cubin's TMA descriptor and cumulative
  // offsets stay valid. P2 points ptrO here too, so that the main cubin never writing it in
  // separate-reducer mode is not load-bearing.
  if (plan.producer != Producer::Fusion) {
    int64_t bytes = checkedMul(key.sumOfSeqLensQ, key.numHeadsQ, "bf16 intermediate");
    bytes = checkedMul(bytes, key.headDimV, "bf16 intermediate");
    bytes = checkedMul(bytes, 2, "bf16 intermediate");
    place(bytes, &layout.bf16IntermediateOffset, &layout.bf16IntermediateBytes);
  }

  layout.totalBytes = offset;
  plan.workspace = layout;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

SparseMlaPlan planDsv4SparseMla(ProblemKey const& key) {
  // --- Validation of the launch span. Checked arithmetic, host-known values only. -----
  FLASHINFER_CHECK(key.numHeadsKv == 1, "DSv4 sparse MLA expects one KV head, got", key.numHeadsKv);
  FLASHINFER_CHECK(key.numHeadsQ % kHeadsPerGroup == 0, "The DSv4 output packs", kHeadsPerGroup,
                   "heads per group, got", key.numHeadsQ);
  FLASHINFER_CHECK(key.headDimQk == kHeadDim && key.headDimV == kHeadDim,
                   "The DSv4 output epilogue is defined for head dimension", kHeadDim);
  LaunchSpan const& span = key.span;
  FLASHINFER_CHECK(span.batchSize > 0, "batch_size must be positive.");
  FLASHINFER_CHECK(span.tokenBase >= 0, "token_base must be non-negative.");
  FLASHINFER_CHECK(span.tokenCount > 0, "token_count must be positive.");
  FLASHINFER_CHECK(span.totalTokens > 0, "total_tokens must be positive.");
  FLASHINFER_CHECK(static_cast<int64_t>(span.tokenBase) + span.tokenCount <= span.totalTokens,
                   "the launch span must fit inside the batch-global token range: token_base",
                   span.tokenBase, "+ token_count", span.tokenCount, "exceeds total_tokens",
                   span.totalTokens);
  FLASHINFER_CHECK(span.totalTokens <= span.tokenCapacity, "token_capacity", span.tokenCapacity,
                   "must cover total_tokens", span.totalTokens);
  FLASHINFER_CHECK(span.totalTokens <= key.sumOfSeqLensQ, "total_tokens", span.totalTokens,
                   "must not exceed the base query extent Q", key.sumOfSeqLensQ);
  FLASHINFER_CHECK(span.batchSize == key.batchSize,
                   "the span's batch size must match the problem's.");

  // --- Stage 1: the original selector, unchanged. --------------------------------------
  TllmGenFmhaRunnerParams const baseParams = makeBaseRunnerParams(key);
  TllmGenFmhaKernel const* baseKernels = getTllmFmhaKernels(
      static_cast<Data_type>(key.dtypeQ), static_cast<Data_type>(key.dtypeKv),
      static_cast<Data_type>(key.dtypeKv), Data_type::DATA_TYPE_BF16, getSMVersion());
  FLASHINFER_CHECK(baseKernels != nullptr, "trtllm-gen FMHA kernels are not available.");

  auto const [baseExists, baseInfo] = baseKernels->checkIfKernelExist(baseParams);
  // A missing base tactic is the pre-existing BF16 failure; the producer cascade cannot
  // recover from it, since P3 has no output to convert.
  FLASHINFER_CHECK(baseExists,
                   "Missing TRTLLM-GEN kernel (dsv4 sparse mla base): ", baseInfo.c_str());

  auto const resolved = baseKernels->resolvePlan(baseParams);

  SparseMlaPlan plan{baseParams};
  plan.key = key;
  plan.baseKernels = baseKernels;
  plan.baseMeta = resolved.kernelMeta;
  plan.baseFunc = resolved.func;
  plan.selectParams = resolved.selectKernelParams;
  plan.ctaLaunch = resolved.ctaLaunchParams;
  plan.baseKernelName = resolved.kernelMeta.mFuncName;

  if (key.outputMode == static_cast<int32_t>(OutputMode::Bf16)) {
    plan.producer = Producer::Standalone;
    plan.reason = "bf16:no_decoration";
    plan.producerKernelName = plan.baseKernelName;
    computeWorkspaceLayout(plan);
    return plan;
  }

  // --- Stage 2: decorate the resolved plan. Capability matching only. -----------------
  plan.producer = Producer::Standalone;
  plan.reason = "standalone:no_optimized_producer";

  // P1: an exact FP8-output twin of the base. The shipped fused cubin derives its FP8
  // value-group stride from mSumOfSeqLensQ, so it can only address a destination whose
  // physical extent equals the base query extent -- hence Q == L == T. A miss here is a
  // capability miss, not an error.
  bool const p1LayoutCompatible =
      (key.sumOfSeqLensQ == span.tokenCapacity) && (span.tokenCapacity == span.totalTokens);
  if (!p1LayoutCompatible) {
    plan.reason = "standalone:p1_layout_incompatible";
  } else {
    TllmGenFmhaKernel const* fp8Kernels = getTllmFmhaKernels(
        static_cast<Data_type>(key.dtypeQ), static_cast<Data_type>(key.dtypeKv),
        static_cast<Data_type>(key.dtypeKv), Data_type::DATA_TYPE_E4M3, getSMVersion());
    if (fp8Kernels != nullptr) {
      CUfunction twinFunc{};
      SparseMlaPlan::KernelMeta twinMeta{};
      std::string reason;
      if (fp8Kernels->tryFindExactFp8Twin(baseParams, plan.baseMeta, plan.selectParams,
                                          plan.ctaLaunch, &twinFunc, &twinMeta, &reason)) {
        plan.producer = Producer::Fusion;
        plan.twinKernels = fp8Kernels;
        plan.twinMeta = twinMeta;
        plan.twinFunc = twinFunc;
        plan.producerKernelName = twinMeta.mFuncName;
        plan.reason = reason;
      } else {
        plan.reason = reason;
      }
    } else {
      plan.reason = "standalone:no_fp8_registry";
    }
  }

  // P2: the base plan already launches a separate reduction kernel and a layout-compatible
  // FP8 specialization exists for its frozen reducer descriptor.
  if (plan.producer == Producer::Standalone &&
      isGmemReductionWithSeparateKernel(
          static_cast<MultiCtasKvMode>(plan.baseMeta.mMultiCtasKvMode)) &&
      tensorrt_llm::kernels::hasDsv4Fp8ReductionSpecialization(plan.baseMeta)) {
    plan.producer = Producer::Reduction;
    plan.reducerEffectiveHeadDim =
        plan.baseMeta.m2CtaMma ? plan.baseMeta.mHeadDimPerCtaV * 2 : plan.baseMeta.mHeadDimPerCtaV;
    plan.producerKernelName = std::string(plan.baseMeta.mFuncName) + "+fmhaReductionFp8";
    plan.reason = "reduction:exact_specialization";
  }

  if (plan.producer == Producer::Standalone) {
    plan.producerKernelName = std::string(plan.baseMeta.mFuncName) + "+dsv4InvRopeQuant";
  }

  computeWorkspaceLayout(plan);
  return plan;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

uint64_t currentContextId() {
  CUcontext ctx{nullptr};
  cuCtxGetCurrent(&ctx);
  return reinterpret_cast<uint64_t>(ctx);
}

int32_t currentDeviceOrdinal() {
  int device = -1;
  cudaGetDevice(&device);
  return device;
}

ProblemKey makeKey(int64_t batch_size, int64_t max_q_len, int64_t sum_seq_q, int64_t sparse_top_k,
                   int64_t num_qo_heads, int64_t sm_count, int64_t token_base, int64_t token_count,
                   int64_t total_tokens, int64_t token_capacity, int64_t output_mode) {
  ProblemKey key{};
  key.deviceOrdinal = currentDeviceOrdinal();
  key.contextId = currentContextId();
  key.registryGeneration = currentRegistryGeneration();
  key.dtypeQ = static_cast<int32_t>(Data_type::DATA_TYPE_E4M3);
  key.dtypeKv = static_cast<int32_t>(Data_type::DATA_TYPE_E4M3);
  key.dtypeO = static_cast<int32_t>(output_mode == static_cast<int64_t>(OutputMode::DeepGemmFp8)
                                        ? Data_type::DATA_TYPE_E4M3
                                        : Data_type::DATA_TYPE_BF16);
  key.numHeadsQ = static_cast<int32_t>(num_qo_heads);
  key.numHeadsKv = 1;
  key.headDimQk = kHeadDim;
  key.headDimV = kHeadDim;
  key.numTokensPerPage = 1;
  key.batchSize = static_cast<int32_t>(batch_size);
  key.maxSeqLenQ = static_cast<int32_t>(max_q_len);
  key.sparseTopK = static_cast<int32_t>(sparse_top_k);
  key.sumOfSeqLensQ = static_cast<int32_t>(sum_seq_q);
  key.multiProcessorCount = static_cast<int32_t>(sm_count);
  // The DSv4 sparse launcher pins window_left = 127.
  key.attentionWindowSize = 128;
  key.outputMode = static_cast<int32_t>(output_mode);
  key.span.batchSize = static_cast<int32_t>(batch_size);
  key.span.tokenBase = static_cast<int32_t>(token_base);
  key.span.tokenCount = static_cast<int32_t>(token_count);
  key.span.totalTokens = static_cast<int32_t>(total_tokens);
  key.span.tokenCapacity = static_cast<int32_t>(token_capacity);
  return key;
}

void checkBackingTensor(TensorView const& t, char const* name, DLDataType dtype,
                        std::vector<int64_t> const& shape, DLDevice device) {
  TVM_FFI_ICHECK_EQ(t.dtype(), dtype)
      << name << " must be the contiguous producer backing: wrong dtype";
  TVM_FFI_ICHECK_EQ(t.ndim(), static_cast<int>(shape.size()))
      << name << " must be the contiguous producer backing of rank " << shape.size()
      << ", not a logical consumer view";
  int64_t expectedStride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    TVM_FFI_ICHECK_EQ(t.size(i), shape[i])
        << name << " must be the contiguous producer backing, not a token slice; dim " << i
        << " has extent " << t.size(i) << ", expected " << shape[i];
    TVM_FFI_ICHECK_EQ(t.stride(i), expectedStride)
        << name << " must be the contiguous producer backing, not a view or a token slice; "
        << "dim " << i << " has stride " << t.stride(i) << ", expected " << expectedStride;
    expectedStride *= shape[i];
  }
  TVM_FFI_ICHECK_EQ(t.device().device_type, device.device_type) << name << " is on a wrong device";
  TVM_FFI_ICHECK_EQ(t.device().device_id, device.device_id) << name << " is on a wrong device";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % 256, 0u)
      << name << " must be 256-byte aligned";
}

bool rangesOverlap(void const* aBase, int64_t aBytes, void const* bBase, int64_t bBytes) {
  auto const a = reinterpret_cast<std::uintptr_t>(aBase);
  auto const b = reinterpret_cast<std::uintptr_t>(bBase);
  return a < b + static_cast<std::uintptr_t>(bBytes) && b < a + static_cast<std::uintptr_t>(aBytes);
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////
// FFI surface.

// The FP8-output execution body, shared by trtllm_dsv4_fp8_run and trtllm_dsv4_fp8_run_oneshot
// below. The output backings must be allocation bases, not logical consumer views or token
// slices of them.
void runResolvedPlan(std::shared_ptr<SparseMlaPlan> const& plan, TensorView query,
                     TensorView primary_kv_cache, TensorView sliding_window_kv_cache,
                     TensorView workspace_buffer, TensorView multi_ctas_kv_counter_buffer,
                     TensorView sparse_indices, TensorView seq_lens,
                     TensorView sparse_mla_top_k_lens, TensorView cum_seq_lens_q,
                     TensorView out_values_backing, TensorView out_scales_backing,
                     TensorView cos_sin_cache, double bmm1_scale, double bmm2_scale,
                     int64_t sm_count, bool enable_pdl, int64_t workspace_size,
                     Optional<TensorView> attention_sinks) {
  // Re-derive the problem key from the actual arguments and require it to equal the plan's, so
  // that a stale plan cannot survive a changed shape. A mismatch is invalid API usage.
  int64_t const sum_seq_q = query.size(0);
  int64_t const num_qo_heads = query.size(1);
  ProblemKey const observed = makeKey(
      plan->key.batchSize, plan->key.maxSeqLenQ, sum_seq_q, sparse_indices.size(-1), num_qo_heads,
      sm_count, plan->key.span.tokenBase, plan->key.span.tokenCount, plan->key.span.totalTokens,
      plan->key.span.tokenCapacity, static_cast<int64_t>(OutputMode::DeepGemmFp8));
  FLASHINFER_CHECK(observed == plan->key,
                   "the plan does not describe this call.\n  plan:", plan->key.describe().c_str(),
                   "\n  call:", observed.describe().c_str());

  // Reject a changed device, context, or registry generation before any write.
  FLASHINFER_CHECK(
      plan->baseKernels == getTllmFmhaKernels(static_cast<Data_type>(plan->key.dtypeQ),
                                              static_cast<Data_type>(plan->key.dtypeKv),
                                              static_cast<Data_type>(plan->key.dtypeKv),
                                              Data_type::DATA_TYPE_BF16, getSMVersion()),
      "the trtllm-gen registry changed since this plan was created.");

  ProblemKey const& key = plan->key;
  LaunchSpan const& span = key.span;
  int64_t const numGroups = key.numHeadsQ / kHeadsPerGroup;
  int64_t const scaleBufM = align4(span.tokenCapacity);

  TVM_FFI_ICHECK_EQ(query.ndim(), 3) << "query must have shape [Q, H, D]";
  TVM_FFI_ICHECK_EQ(query.dtype(), dl_float8_e4m3fn) << "query must be E4M3";
  TVM_FFI_ICHECK_EQ(primary_kv_cache.dtype(), dl_float8_e4m3fn) << "kv cache must be E4M3";
  TVM_FFI_ICHECK_EQ(seq_lens.size(0), key.batchSize);
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.dtype(), dl_int32) << "cum_seq_lens_q must be int32";
  TVM_FFI_ICHECK_EQ(cum_seq_lens_q.size(0), key.batchSize + 1)
      << "cum_seq_lens_q is a launch-local pointer to batch_size + 1 absolute offsets";

  checkBackingTensor(out_values_backing, "out_values_backing", dl_float8_e4m3fn,
                     {numGroups, span.tokenCapacity, kHeadsPerGroup, kHeadDim}, query.device());
  checkBackingTensor(out_scales_backing, "out_scales_backing", dl_int32,
                     {numGroups, kHeadsPerGroup, scaleBufM}, query.device());

  TVM_FFI_ICHECK_EQ(cos_sin_cache.dtype(), dl_float32) << "cos_sin_cache must be float32";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.ndim(), 2) << "cos_sin_cache must be [max_position, 64]";
  TVM_FFI_ICHECK_EQ(cos_sin_cache.size(1), kCosSinRowWidth)
      << "cos_sin_cache rows are cos(32) || sin(32)";
  TVM_FFI_ICHECK(cos_sin_cache.IsContiguous()) << "cos_sin_cache must be contiguous";

  FLASHINFER_CHECK(workspace_size >= plan->workspace.totalBytes, "workspace is too small: got",
                   static_cast<long long>(workspace_size), "bytes, the plan needs",
                   static_cast<long long>(plan->workspace.totalBytes));

  // The workspace must not alias either output backing.
  int64_t const valueBytes =
      numGroups * span.tokenCapacity * kHeadsPerGroup * kHeadDim;  // 1 byte per E4M3
  int64_t const scaleBytes = numGroups * kHeadsPerGroup * scaleBufM * 4;
  FLASHINFER_CHECK(!rangesOverlap(workspace_buffer.data_ptr(), plan->workspace.totalBytes,
                                  out_values_backing.data_ptr(), valueBytes) &&
                       !rangesOverlap(workspace_buffer.data_ptr(), plan->workspace.totalBytes,
                                      out_scales_backing.data_ptr(), scaleBytes),
                   "the workspace must not alias the output backings.");

  auto* workspaceBase = static_cast<char*>(workspace_buffer.data_ptr());
  auto const stream = get_stream(query.device());

  // --- Build the runner params: the frozen base call, plus destination overrides. ------
  TllmGenFmhaRunnerParams params = makeBaseRunnerParams(key);
  params.qPtr = query.data_ptr();
  params.kPtr = primary_kv_cache.data_ptr();
  params.vPtr = primary_kv_cache.data_ptr();
  params.slidingWindowKvPoolPtr = sliding_window_kv_cache.data_ptr();
  params.kvPageIdxPtr = static_cast<int*>(sparse_indices.data_ptr());
  params.seqLensKvPtr = static_cast<int*>(seq_lens.data_ptr());
  params.sparseMlaTopKLensPtr = static_cast<int*>(sparse_mla_top_k_lens.data_ptr());
  params.cumSeqLensQPtr = static_cast<int*>(cum_seq_lens_q.data_ptr());
  params.cumSeqLensKvPtr = nullptr;
  params.stream = stream;
  params.enable_pdl = enable_pdl;
  params.outputScale = bmm2_scale;
  params.scaleSoftmaxLog2 = bmm1_scale * M_LOG2E;
  params.mScaleSfKv = 1.0f;
  params.qStrideTokens = query.stride(0);
  params.qStrideHeads = query.stride(1);
  params.kStrideKeysValues = primary_kv_cache.stride(-2);
  params.kStrideHeads = primary_kv_cache.stride(-3);
  params.kStrideBatch = primary_kv_cache.stride(-2);
  params.vStrideKeysValues = params.kStrideKeysValues;
  params.vStrideHeads = params.kStrideHeads;
  params.vStrideBatch = params.kStrideBatch;
  params.mNumPagesInMemPool = primary_kv_cache.size(0) * primary_kv_cache.size(-2);
  params.multiCtasKvCounterPtr =
      reinterpret_cast<int32_t*>(multi_ctas_kv_counter_buffer.data_ptr());
  if (attention_sinks.has_value()) {
    params.ptrAttentionSinks = static_cast<float*>(attention_sinks.value().data_ptr());
  }

  TllmGenFmhaKernel::Dsv4Fp8Overrides overrides{};
  overrides.ptrDsv4OValues = out_values_backing.data_ptr();
  overrides.ptrDsv4OScales = out_scales_backing.data_ptr();
  overrides.ptrDsv4CosSin = static_cast<float const*>(cos_sin_cache.data_ptr());
  overrides.cosSinRows = static_cast<int32_t>(cos_sin_cache.size(0));
  overrides.tokenBase = span.tokenBase;
  overrides.tokenCount = span.tokenCount;
  overrides.totalTokens = span.totalTokens;
  overrides.tokenCapacity = span.tokenCapacity;
  overrides.scaleBufM = scaleBufM;
  if (plan->workspace.partialStatsBytes > 0) {
    overrides.ptrPartialStats = workspaceBase + plan->workspace.partialStatsOffset;
    overrides.ptrPartialO = workspaceBase + plan->workspace.partialOOffset;
  }
  params.multiCtasKvScratchPtr = overrides.ptrPartialStats;

  char* bf16Intermediate = nullptr;
  if (plan->workspace.bf16IntermediateBytes > 0) {
    bf16Intermediate = workspaceBase + plan->workspace.bf16IntermediateOffset;
  }

  switch (plan->producer) {
    case Producer::Fusion:
      // The twin addresses the value backing with a group stride derived from mSumOfSeqLensQ;
      // the plan already proved Q == L == T.
      params.oPtr = out_values_backing.data_ptr();
      params.mUsesDsv4Ue8m0ScaleO = true;
      params.dsv4InvRopeCosSinCachePtr = overrides.ptrDsv4CosSin;
      params.dsv4OScalePtr = overrides.ptrDsv4OScales;
      params.mDsv4ScaleBufM = scaleBufM;
      overrides.ptrO = out_values_backing.data_ptr();
      break;
    case Producer::Reduction:
      // The main cubin never writes ptrO in separate-reducer mode but still gets a real BF16
      // region. The reducer owns the final FP8 store, via the distinct ptrDsv4OValues.
      params.oPtr = bf16Intermediate;
      overrides.ptrO = bf16Intermediate;
      overrides.reducerWritesFp8 = true;
      break;
    case Producer::Standalone:
      params.oPtr = bf16Intermediate;
      overrides.ptrO = bf16Intermediate;
      break;
  }

  // Execute the exact plan. Selection is not invoked.
  if (plan->producer == Producer::Fusion) {
    TllmGenFmhaKernel::ResolvedPlan twinPlan{params};
    twinPlan.kernelMeta = plan->twinMeta;
    twinPlan.func = plan->twinFunc;
    twinPlan.selectKernelParams = plan->selectParams;
    twinPlan.ctaLaunchParams = plan->ctaLaunch;
    plan->twinKernels->runResolved(params, twinPlan, &overrides);
  } else {
    TllmGenFmhaKernel::ResolvedPlan basePlan{params};
    basePlan.kernelMeta = plan->baseMeta;
    basePlan.func = plan->baseFunc;
    basePlan.selectKernelParams = plan->selectParams;
    basePlan.ctaLaunchParams = plan->ctaLaunch;
    plan->baseKernels->runResolved(params, basePlan, &overrides);
  }

  if (plan->producer == Producer::Standalone) {
    launchDsv4InvRopeQuant(bf16Intermediate, out_values_backing.data_ptr(),
                           out_scales_backing.data_ptr(), overrides.ptrDsv4CosSin,
                           overrides.cosSinRows, static_cast<int32_t*>(cum_seq_lens_q.data_ptr()),
                           static_cast<int32_t*>(seq_lens.data_ptr()), key.numHeadsQ, span,
                           scaleBufM, stream);
  }
}

// Resolve and execute in one call; the plan is neither registered nor cached. Module loading is
// memoized by the trtllm-gen registry on kernel identity and the rest of planning is host
// arithmetic, so this is legal inside a CUDA-graph stream capture. Returns the producer that
// served the launch, as a diagnostic; nothing should branch on it.
int64_t trtllm_dsv4_fp8_run_oneshot(
    int64_t batch_size, int64_t max_q_len, int64_t token_base, int64_t token_count,
    int64_t total_tokens, int64_t token_capacity, TensorView query, TensorView primary_kv_cache,
    TensorView sliding_window_kv_cache, TensorView workspace_buffer,
    TensorView multi_ctas_kv_counter_buffer, TensorView sparse_indices, TensorView seq_lens,
    TensorView sparse_mla_top_k_lens, TensorView cum_seq_lens_q, TensorView out_values_backing,
    TensorView out_scales_backing, TensorView cos_sin_cache, double bmm1_scale, double bmm2_scale,
    int64_t sm_count, bool enable_pdl, int64_t workspace_size,
    Optional<TensorView> attention_sinks) {
  ProblemKey const key = makeKey(batch_size, max_q_len, query.size(0), sparse_indices.size(-1),
                                 query.size(1), sm_count, token_base, token_count, total_tokens,
                                 token_capacity, static_cast<int64_t>(OutputMode::DeepGemmFp8));
  auto const plan = std::make_shared<SparseMlaPlan>(planDsv4SparseMla(key));
  runResolvedPlan(plan, query, primary_kv_cache, sliding_window_kv_cache, workspace_buffer,
                  multi_ctas_kv_counter_buffer, sparse_indices, seq_lens, sparse_mla_top_k_lens,
                  cum_seq_lens_q, out_values_backing, out_scales_backing, cos_sin_cache, bmm1_scale,
                  bmm2_scale, sm_count, enable_pdl, workspace_size, attention_sinks);
  return static_cast<int64_t>(plan->producer);
}

}  // namespace dsv4
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_dsv4_fp8_run_oneshot,
                              flashinfer::dsv4::trtllm_dsv4_fp8_run_oneshot);
