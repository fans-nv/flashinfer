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

// Producer P3: the standalone DSv4 inverse-RoPE + UE8M0 FP8 quantizer, the fallback that is
// always available because it only needs the merged BF16 output the base plan wrote to
// workspace. It is the only producer that must map a global row back to a request, since it
// runs outside any kernel that already owns a batchIdx.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <algorithm>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fmha/dsv4Epilogue.cuh"
#include "flashinfer/trtllm/fmha/dsv4Plan.h"

namespace flashinfer {
namespace dsv4 {

namespace {

// One CTA covers one (token, output group): 8 heads x 64 lanes, 8 values per lane.
constexpr int kLanesPerHead = kHeadDim / 8;                     // 64
constexpr int kThreadsPerCta = kLanesPerHead * kHeadsPerGroup;  // 512
constexpr int kLanesPerQuantBlock = kQuantBlock / 8;            // 16

// upper_bound(cum, cum + n + 1, row) - 1. Skips repeated offsets, so requests with a zero
// query length are transparently passed over.
__device__ __forceinline__ int32_t requestOfRow(int32_t const* cumSeqLensQ, int32_t batchSize,
                                                int32_t row) {
  int32_t lo = 0;
  int32_t hi = batchSize + 1;
  while (lo < hi) {
    int32_t const mid = lo + ((hi - lo) >> 1);
    if (cumSeqLensQ[mid] <= row) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo - 1;
}

__global__ void __launch_bounds__(kThreadsPerCta)
    dsv4InvRopeQuantKernel(__nv_bfloat16 const* __restrict__ bf16In, int32_t numHeadsQ,
                           __nv_fp8_e4m3* __restrict__ outValues, int32_t* __restrict__ outScales,
                           float const* __restrict__ cosSinCache, int32_t cosSinRows,
                           int32_t const* __restrict__ cumSeqLensQ,
                           int32_t const* __restrict__ seqLensKv, int32_t batchSize,
                           int32_t tokenBase, int32_t tokenCapacity, int64_t scaleBufM) {
  int32_t const tid = static_cast<int32_t>(threadIdx.x);
  int32_t const headInGroup = tid / kLanesPerHead;
  int32_t const lane = tid % kLanesPerHead;
  int32_t const dimBase = lane * 8;

  int32_t const group = static_cast<int32_t>(blockIdx.y);
  int32_t const globalRow = tokenBase + static_cast<int32_t>(blockIdx.x);
  int32_t const headIdxGlobal = group * kHeadsPerGroup + headInGroup;

  __shared__ int32_t smemPosition;
  __shared__ uint32_t smemExp[kHeadsPerGroup][kBlocksPerHead];

  if (tid == 0) {
    int32_t const b = requestOfRow(cumSeqLensQ, batchSize, globalRow);
    // A row outside every request interval means the span and the cumulative offsets disagree;
    // encode it as an out-of-range position so the RoPE guard below skips the cache read
    // instead of forming an out-of-bounds address.
    int32_t position = -1;
    if (b >= 0 && b < batchSize) {
      int32_t const start = cumSeqLensQ[b];
      int32_t const end = cumSeqLensQ[b + 1];
      if (start <= globalRow && globalRow < end) {
        position = ropePosition(seqLensKv[b], end - start, globalRow - start);
      }
    }
    smemPosition = position;
  }
  __syncthreads();
  int32_t const position = smemPosition;

  // The intermediate has the same row/head strides as the ordinary BF16 destination.
  int64_t const srcOffset =
      (static_cast<int64_t>(globalRow) * numHeadsQ + headIdxGlobal) * kHeadDim + dimBase;
  uint4 const packed = *reinterpret_cast<uint4 const*>(bf16In + srcOffset);
  __nv_bfloat16 const* srcVals = reinterpret_cast<__nv_bfloat16 const*>(&packed);
  float vals[8];
#pragma unroll
  for (int32_t ii = 0; ii < 8; ++ii) {
    vals[ii] = __bfloat162float(srcVals[ii]);
  }

  // Inverse RoPE over the trailing kRopeDim dimensions. Interleaved partners are (j, j^1)
  // and both live in this lane's registers.
  constexpr int32_t kRopeStart = kHeadDim - kRopeDim;
  if (dimBase >= kRopeStart && position >= 0 && position < cosSinRows) {
    float const* cosSin = cosSinCache + static_cast<int64_t>(position) * kCosSinRowWidth;
#pragma unroll
    for (int32_t ii = 0; ii < 8; ii += 2) {
      int32_t const ropeLocal = (dimBase + ii) - kRopeStart;
      inverseRopePair(vals[ii], vals[ii + 1], cosSin[ropeLocal >> 1],
                      cosSin[(kRopeDim / 2) + (ropeLocal >> 1)]);
    }
  }

  // Per-128-element block amax: exactly 16 consecutive lanes, always inside one warp.
  float amax = 0.f;
#pragma unroll
  for (int32_t ii = 0; ii < 8; ++ii) {
    amax = fmaxf(amax, fabsf(vals[ii]));
  }
#pragma unroll
  for (int32_t off = 1; off < kLanesPerQuantBlock; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  }
  int32_t const expBiased = ue8m0ExponentFromAmax(amax);
  float const invScale = reciprocalScaleFromExponent(expBiased);

  __nv_fp8_e4m3* dst =
      outValues + valueOffset(group, globalRow, headInGroup, dimBase, tokenCapacity);
#pragma unroll
  for (int32_t ii = 0; ii < 8; ++ii) {
    dst[ii] = quantizeToE4m3(vals[ii] * invScale);
  }

  if ((lane % kLanesPerQuantBlock) == 0) {
    smemExp[headInGroup][lane / kLanesPerQuantBlock] = static_cast<uint32_t>(expBiased) & 0xFFu;
  }
  __syncthreads();
  if (lane == 0) {
    uint32_t const word = packExponents(smemExp[headInGroup][0], smemExp[headInGroup][1],
                                        smemExp[headInGroup][2], smemExp[headInGroup][3]);
    outScales[scaleOffset(group, headInGroup, globalRow, scaleBufM)] = static_cast<int32_t>(word);
  }
}

}  // namespace

void launchDsv4InvRopeQuant(void const* bf16In, void* outValues, void* outScales,
                            float const* cosSinCache, int32_t cosSinRows,
                            int32_t const* cumSeqLensQ, int32_t const* seqLensKv, int32_t numHeadsQ,
                            LaunchSpan const& span, int64_t scaleBufM, cudaStream_t stream) {
  FLASHINFER_CHECK(numHeadsQ % kHeadsPerGroup == 0, "DSv4 standalone epilogue packs",
                   kHeadsPerGroup, "heads per output group, got", numHeadsQ);
  FLASHINFER_CHECK(span.tokenCount > 0, "DSv4 standalone epilogue requires token_count > 0.");

  dim3 const gridDim(static_cast<unsigned int>(span.tokenCount),
                     static_cast<unsigned int>(numHeadsQ / kHeadsPerGroup));
  dsv4InvRopeQuantKernel<<<gridDim, kThreadsPerCta, 0, stream>>>(
      reinterpret_cast<__nv_bfloat16 const*>(bf16In), numHeadsQ,
      reinterpret_cast<__nv_fp8_e4m3*>(outValues), reinterpret_cast<int32_t*>(outScales),
      cosSinCache, cosSinRows, cumSeqLensQ, seqLensKv, span.batchSize, span.tokenBase,
      span.tokenCapacity, scaleBufM);
  cudaError_t const err = cudaGetLastError();
  FLASHINFER_CHECK(err == cudaSuccess,
                   "Failed to launch dsv4InvRopeQuantKernel: ", cudaGetErrorString(err));
}

}  // namespace dsv4
}  // namespace flashinfer
