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

// DSv4 output epilogue (inverse RoPE, UE8M0 block scales, E4M3); must match the fused cubin.

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace flashinfer {
namespace dsv4 {

static constexpr int kHeadDim = 512;
static constexpr int kRopeDim = 64;  // trailing interleaved pairs
static constexpr int kQuantBlock = 128;
static constexpr int kHeadsPerGroup = 8;
static constexpr int kCosSinRowWidth = 64;  // cos(32) || sin(32)
static constexpr float kE4m3Max = 448.f;
static constexpr float kAmaxFloor = 1e-10f;

// Plain multiply-add so nvcc contracts to FMA.
__device__ __forceinline__ void inverseRopePair(float& even, float& odd, float cosVal,
                                                float sinVal) {
  float const e = even;
  float const o = odd;
  even = e * cosVal + o * sinVal;
  odd = o * cosVal - e * sinVal;
}

// ceil(log2(max(amax, floor) / 448)) + 127. Not ceilf(log2f(amax * (1/448))): 1/448 is inexact
// in binary32, so exact power-of-two quotients (e.g. amax 3.5) would round one exponent high.
__device__ __forceinline__ int32_t ue8m0ExponentFromAmax(float amax) {
  amax = fmaxf(amax, kAmaxFloor);
  int32_t e = ilogbf(amax) - 8;  // 448 = 1.75 * 2^8: never high, at most one low
  if (ldexpf(kE4m3Max, e) < amax) {
    ++e;
  }
  return min(max(e + 127, 0), 255);
}

__device__ __forceinline__ float reciprocalScaleFromExponent(int32_t biasedExp) {
  return __frcp_rn(__int_as_float(biasedExp << 23));
}

__device__ __forceinline__ __nv_fp8_e4m3 quantizeToE4m3(float scaled) {
  return static_cast<__nv_fp8_e4m3>(fminf(fmaxf(scaled, -kE4m3Max), kE4m3Max));
}

// Element offset into E4M3 values [numGroups, numTokens, 8, kHeadDim].
__device__ __forceinline__ int64_t valueOffset(int32_t group, int32_t token, int32_t headInGroup,
                                               int32_t dim, int32_t numTokens) {
  return (((static_cast<int64_t>(group) * numTokens + token) * kHeadsPerGroup) + headInGroup) *
             kHeadDim +
         dim;
}

// Byte offset of a block's exponent in INT32 scales [numGroups, 8, scaleBufM]; block 0 is the LSB.
__device__ __forceinline__ int64_t scaleByteOffset(int32_t group, int32_t headInGroup,
                                                   int32_t token, int32_t block,
                                                   int64_t scaleBufM) {
  return ((static_cast<int64_t>(group) * kHeadsPerGroup + headInGroup) * scaleBufM + token) *
             sizeof(int32_t) +
         block;
}

}  // namespace dsv4
}  // namespace flashinfer
