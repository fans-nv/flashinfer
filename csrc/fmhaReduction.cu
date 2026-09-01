/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <float.h>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/common/cudaTypeUtils.cuh"
#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fmha/dsv4Epilogue.cuh"
#include "flashinfer/trtllm/fmha/fmhaReduction.h"
#include "flashinfer/trtllm/fmha/kernelUtils.h"
#include "flashinfer/utils.cuh"

namespace tensorrt_llm {
namespace kernels {

////////////////////////////////////////////////////////////////////////////////////////////////////

#define NumThreadsPerCta 512

template <int32_t TileSizePerCtaQ, int32_t HeadDimPerCta, bool IsE4m3Bmm, typename DtypeO,
          typename DtypePartialO, bool Dsv4Fp8Out = false>
__global__ void __launch_bounds__(NumThreadsPerCta, 2)
    fmhaReductionKernel(KernelParams const params, bool isTokenSparse, bool groupsTokensHeadsQ,
                        bool supportsVarSparseMlaTopKLens, int32_t numCtasForReduction,
                        int32_t numCtasForAllHeads, int32_t headDimV, int32_t numHeadDimCtasV,
                        Dsv4ReductionParams const dsv4Params) {
  // clang-format off
  // The shape of partialO buffer: [batchSize, numHeadCtas, numCtasQ, numCtasKv, TileSizePerCtaQ, headDimPerCta].
  // The shape of final O buffer: [batchSize, numCtasQ, numHeadsQ, headDim].
  // The shape of attentionSinks buffer: [numHeadsQ].
  // The shape of partialStats buffer: [batchSize, numHeadCtas, numCtasQ, numCtasKv, TileSizePerCtaQ], where each element is a float2 (max/sum).
  // The shape of softmaxStats buffer: [batchSize, numCtasQ, numHeadsQ], where each element is a float2 (max/sum).
  // Note that numValidRows includes both numValidTokens and numHeadsQPerKv if grouping headsQ.
  // clang-format on

  // The batchIdx.
  int32_t const batchIdx{static_cast<int32_t>(blockIdx.z)};
  // The headCtaIdxO.
  int32_t const headCtaIdxO{static_cast<int32_t>(blockIdx.y)};
  // The headDimCtaIdxV.
  int32_t const headDimCtaIdxV{static_cast<int32_t>(blockIdx.y % numHeadDimCtasV)};
  // The headGrpIdxO.
  int32_t const headGrpIdxO{static_cast<int32_t>(blockIdx.y / numHeadDimCtasV)};
  // The ctaIdxQ.
  int32_t const ctaIdxQ{static_cast<int32_t>(blockIdx.x % params.mMaxNumCtasQ)};
  // The ctaIdx for the reduction work.
  int32_t const ctaIdxForReduction{static_cast<int32_t>(blockIdx.x / params.mMaxNumCtasQ)};
  // The numHeadsQPerKvCta.
  int32_t const numHeadsQPerKvCta{min(params.mNumHeadsQPerKv, TileSizePerCtaQ)};
  // The headIdxO.
  int32_t const headIdxO{headGrpIdxO * numHeadsQPerKvCta};
  // The warpGrpThreadIdx.
  int32_t const warpGrpThreadIdx{static_cast<int32_t>(threadIdx.x)};

  // The seqOffsetQ in token units (cumSeqLensQ is already token-relative).
  // Fixed-Q batches are packed by mMaxSeqLenQ; grouped CTA coverage can include padded rows.
  int32_t const seqOffsetQ{params.ptrCumSeqLensQ == nullptr ? batchIdx * params.mMaxSeqLenQ
                                                            : params.ptrCumSeqLensQ[batchIdx]};
  // The seqLenQ.
  int32_t const seqLenQ{params.ptrCumSeqLensQ == nullptr
                            ? params.mMaxSeqLenQ
                            : (params.ptrCumSeqLensQ[batchIdx + 1] - seqOffsetQ)};
  // The number of validTokens.
  int32_t const numValidTokens{
      min(seqLenQ - ctaIdxQ * params.mNumTokensPerCtaQ, params.mNumTokensPerCtaQ)};
  // The number of validRows.
  int32_t const numValidRows{numValidTokens * numHeadsQPerKvCta};

  // Early exit if there are no valid tokens.
  if (numValidTokens <= 0) {
    return;
  }

  // The last tokenQ index (relative to the request) processed by this CTA.
  int32_t const lastTokenIdxQ{ctaIdxQ * params.mNumTokensPerCtaQ + numValidTokens - 1};

  // The actual number of seqLenKv. Block-sparse attention uses per-KV-head sequence lengths
  // laid out as [numHeadsKv, batchSize].
  int32_t seqLenKv;
  if (params.mUseBlockSparseAttention) {
    int32_t const headIdxKv{headIdxO / params.mNumHeadsQPerKv};
    seqLenKv = params.ptrSeqLensKv[headIdxKv * params.mBatchSize + batchIdx];
  } else {
    seqLenKv = params.ptrSeqLensKv[batchIdx];
  }
  // Consider the causal-mask speculative decoding. Use the per-batch seqLenQ (not mMaxSeqLenQ)
  // so variable-length batches get the correct KV extent; these agree when seqLenQ ==
  // mMaxSeqLenQ.
  seqLenKv = max(seqLenKv - seqLenQ + lastTokenIdxQ + 1, 0);
  // Consider sparseAttnTopK and variable sparse MLA topK lengths.
  if (supportsVarSparseMlaTopKLens) {
    seqLenKv = params.ptrSparseMlaTopKLens[seqOffsetQ + ctaIdxQ * params.mNumTokensPerCtaQ];
  } else if (isTokenSparse) {
    seqLenKv = min(seqLenKv, params.mSparseAttnTopK);
  }
  // The actual number of CtasKv (TileSizeKv is always 128 for now).
  int32_t numCtasKv{min((seqLenKv + 127) / 128, params.mMaxNumCtasKv)};

  // The tileIdx in the batch/head dimension.
  int64_t const batchHeadTileIdx{
      ((batchIdx * static_cast<int32_t>(gridDim.y) + headCtaIdxO) * params.mMaxNumCtasQ + ctaIdxQ)};

  // The offset of the partialStats buffer.
  int64_t const partialStatsOffset{batchHeadTileIdx * params.mMaxNumCtasKv * TileSizePerCtaQ};
  // The offset of the partialO buffer.
  int64_t const partialOOffset{partialStatsOffset * HeadDimPerCta};
  // The offset of the softmaxStats buffer.
  int64_t const softmaxStatsOffset{
      ((seqOffsetQ + ctaIdxQ * params.mNumTokensPerCtaQ) * numCtasForAllHeads + headGrpIdxO) *
      numHeadsQPerKvCta};
  // The offset of the O buffer.
  int64_t const oOffset{softmaxStatsOffset * headDimV + headDimCtaIdxV * HeadDimPerCta};

  // The partialStats pointer.
  float2* partialStatsPtr = reinterpret_cast<float2*>(params.ptrPartialStats) + partialStatsOffset;
  // The partialO pointer.
  DtypePartialO* partialOPtr =
      reinterpret_cast<DtypePartialO*>(params.ptrPartialO) + partialOOffset;
  // The softmaxStats pointer.
  float2* softmaxStatsPtr = reinterpret_cast<float2*>(params.ptrSoftmaxStats) + softmaxStatsOffset;
  // The O pointer.
  DtypeO* oPtr = reinterpret_cast<DtypeO*>(params.ptrO) + oOffset;
  // The attentionSinks pointer.
  float const* attentionSinksPtr =
      params.ptrAttentionSinks == nullptr ? nullptr : params.ptrAttentionSinks + headIdxO;

  // Whether to store the softmax stats.
  bool const storesSoftmaxStats{params.ptrSoftmaxStats != nullptr};

  // The softmaxScaleLog2. Prefer the device-side scale when supplied.
  float const softmaxScaleLog2 = params.ptrScaleSoftmaxLog2 != nullptr ? *params.ptrScaleSoftmaxLog2
                                                                       : params.mScaleSoftmaxLog2;

  int32_t constexpr NumBytesPerPartialElt{sizeof(DtypePartialO)};
  static_assert(NumBytesPerPartialElt == 2,
                "The data type of partialO should be either fp16 or bf16.");

  // The threads in the warp-group should load different values from one partial output
  // [numValidRows, headDim], and then iterate over partial outputs from different CTAs.
  int32_t constexpr NumEltsPer16BVec{16 / NumBytesPerPartialElt};
  static_assert((HeadDimPerCta * NumBytesPerPartialElt) % 16 == 0, "Not implemented");

  // The number of unrolled iterations to issue multiple LDGs.
  int32_t constexpr UnrollSize{4};

  // The number of processed rows in one slice where each CTA will process one slice.
  int32_t constexpr NumBytesPerHeadDim{HeadDimPerCta * NumBytesPerPartialElt};
  int32_t constexpr NumBytePerSlice{NumThreadsPerCta * 16};
  static_assert(NumBytePerSlice % NumBytesPerHeadDim == 0, "Not implemented");
  int32_t constexpr NumRowsPerSlice{NumBytePerSlice / NumBytesPerHeadDim};
  // The actual number of tensor slices for the reduction.
  int32_t numSlices{(numValidRows + NumRowsPerSlice - 1) / NumRowsPerSlice};

  // The number of slices that each CTA will process.
  int32_t numSlicesPerCta{(numSlices + numCtasForReduction - 1) / numCtasForReduction};
  // The start slice index for the current CTA.
  int32_t startSliceIdx{ctaIdxForReduction * numSlicesPerCta};
  // The end slice index for the current CTA.
  int32_t endSliceIdx{min(startSliceIdx + numSlicesPerCta, numSlices)};

  // The total number of rows in the partial buffers.
  int32_t numRowsInPartialBuffers{TileSizePerCtaQ};

  // DSv4 FP8 output (producer P2). Compile-time, so a plain reducer instantiation carries
  // neither this shared array nor the epilogue's registers. Guarded on HeadDimPerCta == 512
  // because one CTA must own a whole head -- all four quant blocks and the full RoPE range --
  // for the block amax and the packed scale word to be CTA-local.
  static constexpr bool kDsv4Enabled =
      Dsv4Fp8Out && HeadDimPerCta == flashinfer::dsv4::kHeadDim && NumEltsPer16BVec == 8;
  __shared__ uint32_t smemDsv4Exp[kDsv4Enabled ? NumRowsPerSlice : 1]
                                 [kDsv4Enabled ? flashinfer::dsv4::kBlocksPerHead : 1];

  // Iterate over different slices.
  // Split the reduction work across multiple CtasKv to reduce the latency.
  for (int32_t sliceIdx = startSliceIdx; sliceIdx < endSliceIdx; ++sliceIdx) {
    // The base offset that each thread points to.
    int32_t const baseOffset{warpGrpThreadIdx * NumEltsPer16BVec};
    // The index in the row dimension.
    int32_t const rowIdx{sliceIdx * NumRowsPerSlice + (baseOffset / HeadDimPerCta)};
    // Does this thread point to a valid row ?
    bool const isValidRow{rowIdx < numValidRows};
    int32_t validRowIdx{min(rowIdx, numValidRows - 1)};
    int32_t loadRowIdx{validRowIdx};
    // The index in the headDim dimension.
    int32_t const headDimIdx{baseOffset % HeadDimPerCta};
    // The memory load offset.
    int64_t const destMemOffset{loadRowIdx * HeadDimPerCta + headDimIdx};
    // The memory store offset.
    int64_t gmemStoreOffset{validRowIdx * headDimV + headDimIdx};
    // The local headIdxO.
    int32_t localHeadIdxO{validRowIdx};
    // The rowIdx of softmaxStats.
    int32_t softmaxStatsRowIdx{validRowIdx};
    // If grouping both tokens and headsQ, map validRowIdx into tokenIdx and headIdxInGrp.
    if (groupsTokensHeadsQ) {
      int32_t tokenIdx{validRowIdx / params.mNumHeadsQPerKv};
      int32_t headIdxInGrp{validRowIdx % params.mNumHeadsQPerKv};
      localHeadIdxO = headIdxInGrp;
      softmaxStatsRowIdx = tokenIdx * params.mNumHeadsQ + headIdxInGrp;
      gmemStoreOffset = int64_t(softmaxStatsRowIdx) * headDimV + headDimIdx;
    }

// Wait for the primary kernel to complete.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    // Add offset to the pointers.
    float2* localPartialStatsPtr = partialStatsPtr + loadRowIdx;
    DtypePartialO* localPartialOPtr = partialOPtr + destMemOffset;

    // Reduce max, sum and partialO vectors from different CtasKv.
    float sumVal{0.f};
    float oldMaxVal{-FLT_MAX}, maxVal{-FLT_MAX};
    float outputVals[NumEltsPer16BVec] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int32_t ii = 0; ii < numCtasKv; ii += UnrollSize) {
      // The partialStats array and partialO array.
      float2 partialStatsArray[UnrollSize];
      uint4 partialOArray[UnrollSize];
#pragma unroll
      for (int32_t jj = 0; jj < UnrollSize; ++jj) {
        int32_t ctaIdxKv = min(ii + jj, numCtasKv - 1);
        partialStatsArray[jj] = localPartialStatsPtr[ctaIdxKv * numRowsInPartialBuffers];
        partialOArray[jj] = *reinterpret_cast<uint4 const*>(
            localPartialOPtr + ctaIdxKv * numRowsInPartialBuffers * HeadDimPerCta);
      }
#pragma unroll
      for (int32_t jj = 0; jj < UnrollSize; ++jj) {
        // Whether the ctaIdxKv is valid.
        bool const isValidCtaIdxKv = (ii + jj) < numCtasKv;
        // The local max and sum values.
        auto partialStats = partialStatsArray[jj];
        float localMax = partialStats.x;
        float localSum = partialStats.y;
        // Update the max value.
        maxVal = fmaxf(maxVal, localMax);
        // Compute the correction scales.
        float corrScale0 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (oldMaxVal - maxVal)) : 1.f;
        float corrScale1 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (localMax - maxVal)) : 0.f;
        // Update the old max value.
        oldMaxVal = maxVal;
        // The partialO value.
        uint4 vec = partialOArray[jj];
        // Reduce sum and finalO.
        sumVal = sumVal * corrScale0 + localSum * corrScale1;
        convertToFloatAndAccumulate<DtypePartialO>(outputVals, vec, corrScale0, corrScale1);
      }
    }

    // Update the sums with the attention sink value.
    if (attentionSinksPtr != nullptr) {
      float attentionSinkVal =
          exp2f(attentionSinksPtr[localHeadIdxO] * M_LOG2E - maxVal * softmaxScaleLog2);
      // Multiply the attention sink value by 448.f if the MMA data type is e4m3 as the sum value
      // has also included the 448.f quantization scale.
      sumVal += IsE4m3Bmm ? attentionSinkVal * 448.f : attentionSinkVal;
    }

    // Stores the final softmax stats values to global memory if needed (Helix attention, which
    // splits seqLenKv across GPUs).
    if (storesSoftmaxStats && isValidRow && headDimIdx == 0) {
      // The softmaxScale.
      float softmaxScale = (softmaxScaleLog2 * (1.f / M_LOG2E));
      // The sumScale to unscale the 448.f quantization scale from P.
      float sumScale = IsE4m3Bmm ? (1.f / 448.f) : 1.f;
      // The final max and sum values.
      float2 stats{maxVal * softmaxScale, sumVal * sumScale};
      // Store the final max and sum values to global memory.
      reinterpret_cast<float2*>(softmaxStatsPtr)[softmaxStatsRowIdx] = stats;
    }

    // The final normalized scale.
    // If the output data type is e4m3, make sure that sumVal is divided by the quantization scale
    // (448.f), so 1.0f / (sumVal / 448.f) = 448.f / sumVal.
    float normalizedScale{IsE4m3Bmm ? (448.f / sumVal) : (1.0f / sumVal)};
    float2 normalizedScale2{normalizedScale, normalizedScale};

    // Apply the normalized scale to the reduced O values.
    for (int ii = 0; ii < NumEltsPer16BVec / 2; ++ii) {
      float2& f2 = reinterpret_cast<float2*>(outputVals)[ii];
      mul(f2, f2, normalizedScale2);
    }

    // ---- DSv4 inverse-RoPE + UE8M0 FP8 epilogue (producer P2) ------------------------
    // dsv4Epilogue.cuh's semantics applied to the merged, normalized FP32 values -- the same
    // precision the fused cubin quantizes from, so P1 and P2 agree on the same inputs.
    if constexpr (kDsv4Enabled) {
      namespace dsv4 = flashinfer::dsv4;

      // softmaxStats is indexed [token, head] over the batch-global token axis, and both
      // row-mapping modes above produce that flattened index once softmaxStatsOffset is added.
      int64_t const absStatsIdx{softmaxStatsOffset + softmaxStatsRowIdx};
      int32_t const tokenIdxGlobal{static_cast<int32_t>(absStatsIdx / params.mNumHeadsQ)};
      int32_t const headIdxGlobal{static_cast<int32_t>(absStatsIdx % params.mNumHeadsQ)};

      // Re-read: the local seqLenKv was adjusted for speculative decoding and clamped to topK.
      int32_t const cacheSeqLenKv{params.ptrSeqLensKv[batchIdx]};
      int32_t const position{
          dsv4::ropePosition(cacheSeqLenKv, seqLenQ, tokenIdxGlobal - seqOffsetQ)};

      // Inverse RoPE over the trailing kRopeDim dimensions; interleaved partners (j, j^1) both
      // live in this thread's 8 contiguous registers, so no shuffle is needed.
      //
      // The upper bound on the position is load-bearing: nothing on the host constrains
      // seq_lens_kv to the cos/sin cache extent, so a caller that raises the model length past
      // the cache it built (vLLM's VLLM_ALLOW_LONG_MAX_MODEL_LEN) would read off the end of it.
      // Out of range skips the rotation, as P3 does; the value is still quantized and stored.
      int32_t constexpr ropeStart{dsv4::kHeadDim - dsv4::kRopeDim};
      if (headDimIdx >= ropeStart && position >= 0 && position < dsv4Params.cosSinRows) {
        float const* cosSin{params.ptrDsv4InvRopeCosSinCache +
                            static_cast<int64_t>(position) * dsv4::kCosSinRowWidth};
#pragma unroll
        for (int32_t ii = 0; ii < NumEltsPer16BVec; ii += 2) {
          int32_t const ropeLocal{(headDimIdx + ii) - ropeStart};
          dsv4::inverseRopePair(outputVals[ii], outputVals[ii + 1], cosSin[ropeLocal >> 1],
                                cosSin[32 + (ropeLocal >> 1)]);
        }
      }

      // Per-128-element block amax. A block is 16 consecutive lanes (16 * 8 == 128) and never
      // straddles a warp, so four xor shuffles suffice; every lane of a block shares one row,
      // so a clamped invalid row cannot pollute a valid block's amax.
      float amax{0.f};
#pragma unroll
      for (int32_t ii = 0; ii < NumEltsPer16BVec; ++ii) {
        amax = fmaxf(amax, fabsf(outputVals[ii]));
      }
#pragma unroll
      for (int32_t off = 1; off < 16; off <<= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      }
      int32_t const expBiased{dsv4::ue8m0ExponentFromAmax(amax)};
      float const invScale{dsv4::reciprocalScaleFromExponent(expBiased)};

      int32_t const grpIdx{headIdxGlobal / dsv4::kHeadsPerGroup};
      int32_t const headInGrp{headIdxGlobal % dsv4::kHeadsPerGroup};
      // Only rows this launch owns are written, and addressed by the layout extent.
      bool const ownsRow{isValidRow && tokenIdxGlobal >= dsv4Params.tokenBase &&
                         tokenIdxGlobal < dsv4Params.tokenBase + dsv4Params.tokenCount};

      if (ownsRow) {
        __nv_fp8_e4m3* dstValues{reinterpret_cast<__nv_fp8_e4m3*>(dsv4Params.outValues) +
                                 dsv4::valueOffset(grpIdx, tokenIdxGlobal, headInGrp, headDimIdx,
                                                   dsv4Params.tokenCapacity)};
#pragma unroll
        for (int32_t ii = 0; ii < NumEltsPer16BVec; ++ii) {
          dstValues[ii] = dsv4::quantizeToE4m3(outputVals[ii] * invScale);
        }
      }

      // Pack the head's four block exponents into one INT32 word. The four block leaders
      // span two warps, so hand off through shared memory rather than a shuffle.
      int32_t const chunkIdx{headDimIdx / dsv4::kQuantBlock};
      int32_t const localRow{(warpGrpThreadIdx * NumEltsPer16BVec) / HeadDimPerCta};
      __syncthreads();
      if ((headDimIdx % dsv4::kQuantBlock) == 0) {
        smemDsv4Exp[localRow][chunkIdx] = static_cast<uint32_t>(expBiased) & 0xFFu;
      }
      __syncthreads();
      if (headDimIdx == 0 && ownsRow) {
        uint32_t const word{dsv4::packExponents(smemDsv4Exp[localRow][0], smemDsv4Exp[localRow][1],
                                                smemDsv4Exp[localRow][2],
                                                smemDsv4Exp[localRow][3])};
        reinterpret_cast<int32_t*>(params.ptrDsv4OScaleFp32)[dsv4::scaleOffset(
            grpIdx, headInGrp, tokenIdxGlobal, params.mDsv4ScaleBufM)] = static_cast<int32_t>(word);
      }
      continue;
    }

    // Convert the float values to DtypeO, and Store it to global memory.
    if (isValidRow) {
      convertAndStoreToGmem<DtypeO>(reinterpret_cast<char*>(oPtr + gmemStoreOffset), outputVals);
    }
  }

// Trigger the secondary kernel.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define SELECT_FMHA_REDUCTION_KERNEL(TileSizePerCtaQ, HeadDimPerCta)                            \
  if (kernelMeta.mDataTypeQ == DATA_TYPE_E4M3) {                                                \
    if (kernelMeta.mDataTypeO == DATA_TYPE_E4M3) {                                              \
      kernel = &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, true, __nv_fp8_e4m3, half>; \
    } else if (kernelMeta.mDataTypeO == DATA_TYPE_FP16) {                                       \
      kernel = &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, true, half, half>;          \
    } else if (kernelMeta.mDataTypeO == DATA_TYPE_BF16) {                                       \
      kernel = &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, true, __nv_bfloat16,        \
                                    __nv_bfloat16>;                                             \
    } else {                                                                                    \
      FLASHINFER_CHECK(false, "Not implemented");                                               \
    }                                                                                           \
  } else {                                                                                      \
    FLASHINFER_CHECK(kernelMeta.mDataTypeQ == kernelMeta.mDataTypeO, "Not implemented");        \
    if (kernelMeta.mDataTypeQ == DATA_TYPE_FP16) {                                              \
      kernel = &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, false, half, half>;         \
    } else if (kernelMeta.mDataTypeQ == DATA_TYPE_BF16) {                                       \
      kernel = &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, false, __nv_bfloat16,       \
                                    __nv_bfloat16>;                                             \
    } else {                                                                                    \
      FLASHINFER_CHECK(false, "Not implemented");                                               \
    }                                                                                           \
  }

#define SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(HeadDimPerCta) \
  if (kernelMeta.mTileSizeQ == 64) {                                      \
    SELECT_FMHA_REDUCTION_KERNEL(64, HeadDimPerCta);                      \
  } else if (kernelMeta.mTileSizeQ == 128) {                              \
    SELECT_FMHA_REDUCTION_KERNEL(128, HeadDimPerCta);                     \
  } else {                                                                \
    FLASHINFER_CHECK(false, "Not implemented");                           \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

// The effective head span one reducer CTA owns. 2CTA doubles it.
static int32_t effectiveHeadDimPerReducerCta(TllmGenFmhaKernelMetaInfo const& kernelMeta) {
  return kernelMeta.m2CtaMma ? kernelMeta.mHeadDimPerCtaV * 2 : kernelMeta.mHeadDimPerCtaV;
}

bool hasDsv4Fp8ReductionSpecialization(TllmGenFmhaKernelMetaInfo const& kernelMeta) {
  // Only the separate-kernel family runs a reducer that could own the final store.
  if (!isGmemReductionWithSeparateKernel(
          static_cast<MultiCtasKvMode>(kernelMeta.mMultiCtasKvMode))) {
    return false;
  }
  if (!isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType))) {
    return false;
  }
  if (kernelMeta.mTileSizeKv != 128 ||
      (kernelMeta.mTileSizeQ != 64 && kernelMeta.mTileSizeQ != 128)) {
    return false;
  }
  // Only an effective span of 512 is supported: one CTA then owns a whole head, so the block
  // amax and the packed scale word are CTA-local and the store is a full uint32 word. The
  // 128/256 variants would need partial-word ownership, which has no CUDA memory-model proof.
  if (effectiveHeadDimPerReducerCta(kernelMeta) != flashinfer::dsv4::kHeadDim) {
    return false;
  }
  // The supported domain: E4M3 query/KV, BF16 base output, DSv4 sparse MLA.
  if (kernelMeta.mDataTypeQ != DATA_TYPE_E4M3 || kernelMeta.mDataTypeO != DATA_TYPE_BF16) {
    return false;
  }
  if (kernelMeta.mHeadDimQk != flashinfer::dsv4::kHeadDim ||
      kernelMeta.mHeadDimV != flashinfer::dsv4::kHeadDim) {
    return false;
  }
  return isDynamicTokenSparseMla(static_cast<TrtllmGenSparseMlaType>(kernelMeta.mSparseAttn));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernelMeta, KernelParams const& params,
                      int32_t multiProcessorCount, bool enable_pdl, cudaStream_t stream,
                      Dsv4ReductionParams const* dsv4) {
  bool const dsv4Fp8Output{dsv4 != nullptr};
  Dsv4ReductionParams const dsv4Params{dsv4 != nullptr ? *dsv4 : Dsv4ReductionParams{}};
  // Skip the kernel if not using the separate reduction kernel.
  if (!isGmemReductionWithSeparateKernel(
          static_cast<MultiCtasKvMode>(kernelMeta.mMultiCtasKvMode))) {
    return;
  }

  // This should only be enabled when using keepsMmaAbForGeneration kernel.
  FLASHINFER_CHECK(
      isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType)),
      "Not implemented");
  // The reducer splits heads by mTileSizeQ where the main kernel's selector uses mStepQ
  // (computeCtaAndClusterConfig), so the grids agree only because every shipped separate-reducer
  // descriptor has mStepQ == mTileSizeQ. A cubin drop that broke it would fail as wrong numerics.
  FLASHINFER_CHECK(kernelMeta.mStepQ == kernelMeta.mTileSizeQ, "The separate-reducer descriptor",
                   (kernelMeta.mFuncName != nullptr ? kernelMeta.mFuncName : "<unnamed>"),
                   "has mStepQ", kernelMeta.mStepQ, "!= mTileSizeQ", kernelMeta.mTileSizeQ,
                   "-- the reduction grid is sized on the assumption that they are equal.");
  // The tileSizeQ should be 64 or 128 and tileSizeKv should be 128 for those kernels.
  FLASHINFER_CHECK((kernelMeta.mTileSizeQ == 64 || kernelMeta.mTileSizeQ == 128) &&
                       kernelMeta.mTileSizeKv == 128,
                   "Not implemented");

  // The headDimPerCtaV.
  int32_t const headDimPerCtaV = effectiveHeadDimPerReducerCta(kernelMeta);
  FLASHINFER_CHECK(headDimPerCtaV == 64 || headDimPerCtaV == 128 || headDimPerCtaV == 256 ||
                       headDimPerCtaV == 512,
                   "Not implemented");

  // Falling through here with no compiled specialization would write BF16 into an FP8
  // allocation.
  FLASHINFER_CHECK(!dsv4Fp8Output || hasDsv4Fp8ReductionSpecialization(kernelMeta),
                   "The DSv4 FP8-output reduction specialization does not cover this reducer "
                   "descriptor; the planner should have selected the standalone producer.");
  if (dsv4Fp8Output) {
    FLASHINFER_CHECK(dsv4Params.outValues != nullptr && params.ptrDsv4OScaleFp32 != nullptr &&
                         params.ptrDsv4InvRopeCosSinCache != nullptr &&
                         dsv4Params.tokenCapacity > 0 && params.mDsv4ScaleBufM > 0,
                     "The DSv4 FP8-output reduction requires a complete output descriptor.");
  }

  // The number of slices for the reduction work.
  int32_t const numSlices = (headDimPerCtaV * /* bytesPerPartialElt */ 2 * kernelMeta.mTileSizeQ) /
                            (NumThreadsPerCta * 16);
  // The number of heads computed by a single CTA.
  int numHeadsPerCta{1};
  if (kernelMeta.mGroupsHeadsQ) {
    numHeadsPerCta = std::min(params.mNumHeadsQPerKv, kernelMeta.mTileSizeQ);
  }
  // The number of Ctas for all heads.
  int32_t const numCtasForAllHeads{params.mNumHeadsQ / numHeadsPerCta};
  // The number of Ctas for headDim.
  int32_t const numHeadDimCtasV{kernelMeta.mHeadDimV / headDimPerCtaV};

  // The 512 threads will split the reduction work of TileSizePerCtaQ * HeadDimPerCta.
  dim3 blockDim(NumThreadsPerCta);
  dim3 gridDim;
  // Each CTA processes one tokenQ.
  gridDim.x = params.mMaxNumCtasQ;
  // The head dimension.
  gridDim.y = numCtasForAllHeads * numHeadDimCtasV;
  // The batch dimension.
  gridDim.z = params.mBatchSize;

  // The maximum number of Ctas for the reduction work.
  // This avoids having too many waves of CTAs which can have obvious launching overheads.
  int32_t const maxNumCtasForReduction{(multiProcessorCount * 2) /
                                       static_cast<int32_t>(gridDim.x * gridDim.y * gridDim.z)};
  // The number of Ctas for the reduction work.
  int32_t const numCtasForReduction{std::min(maxNumCtasForReduction, numSlices)};
  // A zero here would make gridDim.x zero below, so the reduction would silently not run. It
  // cannot happen: the selector only builds a separate reducer when the pre-split CTA product
  // divides into the SM count at least twice. Assert rather than clamp -- a std::max(1, ...)
  // would hide a reducer geometry that had diverged from the main kernel's.
  FLASHINFER_CHECK(numCtasForReduction > 0, "The reduction grid product",
                   static_cast<int32_t>(gridDim.x * gridDim.y * gridDim.z), "exceeds 2 * SM count",
                   multiProcessorCount * 2,
                   "-- the reducer's geometry has diverged from the main kernel's pre-split CTA "
                   "count.");
  // Launch more CTAs to split the reduction work if needed.
  gridDim.x *= numCtasForReduction;

  // The PDL attribute.
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  cudaLaunchConfig_t config;
  config.gridDim = gridDim;
  config.blockDim = blockDim;
  config.stream = stream;
  config.dynamicSmemBytes = 0;
  config.attrs = attribute;
  config.numAttrs = 1;

  // Select the kernel function pointer.
  void (*kernel)(KernelParams const, bool, bool, bool, int32_t, int32_t, int32_t, int32_t,
                 Dsv4ReductionParams const) = nullptr;
  if (dsv4Fp8Output) {
    // hasDsv4Fp8ReductionSpecialization has already pinned E4M3 query, BF16 base output and an
    // effective span of 512.
    if (kernelMeta.mTileSizeQ == 64) {
      kernel = &fmhaReductionKernel<64, flashinfer::dsv4::kHeadDim, true, __nv_bfloat16,
                                    __nv_bfloat16, true>;
    } else {
      kernel = &fmhaReductionKernel<128, flashinfer::dsv4::kHeadDim, true, __nv_bfloat16,
                                    __nv_bfloat16, true>;
    }
  } else if (headDimPerCtaV == 64) {
    SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(64);
  } else if (headDimPerCtaV == 128) {
    SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(128);
  } else if (headDimPerCtaV == 256) {
    SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(256);
  } else if (headDimPerCtaV == 512) {
    SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(512);
  }

  // Launch the kernel.
  bool const supportsVarSparseMlaTopKLens =
      isDynamicTokenSparseMla(static_cast<TrtllmGenSparseMlaType>(kernelMeta.mSparseAttn)) &&
      kernelMeta.mHeadDimQk == 512 && kernelMeta.mHeadDimV == 512;
  if (supportsVarSparseMlaTopKLens) {
    FLASHINFER_CHECK(params.ptrSparseMlaTopKLens != nullptr,
                     "Dynamic sparse MLA reduction requires sparseMlaTopkLengths.");
  }
  cudaLaunchKernelEx(&config, kernel, params, kernelMeta.mSparseAttn != 0,
                     kernelMeta.mGroupsTokensHeadsQ, supportsVarSparseMlaTopKLens,
                     numCtasForReduction, numCtasForAllHeads, kernelMeta.mHeadDimV, numHeadDimCtasV,
                     dsv4Params);
  cudaError_t err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess, "Failed to launch kernel: ", cudaGetErrorString(err));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernels
}  // namespace tensorrt_llm
