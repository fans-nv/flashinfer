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

// Canonical device-side semantics of the DeepSeek-V4 attention output epilogue: inverse
// RoPE, per-128-element UE8M0 block scaling, E4M3FN conversion, and the packed scale-word
// encoding.
//
// Every producer of the DeepGEMM-ready FP8 pair must agree with this file: the shipped
// RopeQuantUe8m0Sf cubin (P1), the FP8-output specialization of the trtllm-gen separate
// reduction kernel (P2), the standalone csrc/dsv4_inv_rope_quant.cu kernel (P3), and vLLM's
// Triton `fused_inv_rope_fp8_quant`.
//
// P1 and P2 quantize the merged FP32 accumulators, one rounding earlier than the Triton
// reference, so a block's amax can differ from it by one BF16 step.

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace flashinfer {
namespace dsv4 {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Static model invariants of the DSv4 FP8-output domain.

static constexpr int kHeadDim = 512;
// Trailing dimensions carrying interleaved RoPE pairs: [kHeadDim - kRopeDim, kHeadDim).
static constexpr int kRopeDim = 64;
// Contiguous values covered by one UE8M0 exponent.
static constexpr int kQuantBlock = 128;
static constexpr int kBlocksPerHead = kHeadDim / kQuantBlock;
// Heads packed into one output group; one INT32 scale word carries one head's exponents.
static constexpr int kHeadsPerGroup = 8;
// Row width of the cos/sin cache: cos(32) || sin(32), FP32.
static constexpr int kCosSinRowWidth = 64;
static constexpr float kE4m3Max = 448.f;
// Floor applied to a block's amax so an all-zero block still has a defined exponent.
static constexpr float kAmaxFloor = 1e-10f;

////////////////////////////////////////////////////////////////////////////////////////////////////

// Inverse rotation of one interleaved RoPE pair. Written as plain multiply-add so nvcc
// contracts it into FMAs, which is load-bearing for bit-exactness: vLLM's Triton quantizer
// contracts the same expression, and non-contracting intrinsics here introduce
// one-E4M3-step differences against it.
__device__ __forceinline__ void inverseRopePair(float& even, float& odd, float cosVal,
                                                float sinVal) {
  float const e = even;
  float const o = odd;
  even = e * cosVal + o * sinVal;
  odd = o * cosVal - e * sinVal;
}

// Biased UE8M0 exponent of a block whose absolute maximum is `amax`:
//   stored = ceil(log2(max(amax, kAmaxFloor) / 448)) + 127
// The floor makes an all-zero block encode as 0x55, a valid scale distinct from the zero
// sentinel of the scale-layout padding tail.
//
// Do not rewrite the ceiling as `ceilf(log2f(amax * (1/448)))`: 1/448 is not representable in
// binary32, so an amax whose true quotient is an exact power of two (3.5 -> 2^-7) lands just
// above the integer and rounds to the wrong exponent. Finding the smallest e with
// 448 * 2^e >= amax directly is exact at every boundary.
__device__ __forceinline__ int32_t ue8m0ExponentFromAmax(float amax) {
  amax = fmaxf(amax, kAmaxFloor);
  // 448 == 2^8 * 1.75, so this guess is never high and never more than one step low:
  // amax >= 2^k implies 448 * 2^(k-9) = 0.875 * 2^k < amax.
  int32_t e = ilogbf(amax) - 8;
  if (ldexpf(kE4m3Max, e) < amax) {
    ++e;
  }
  return min(max(e + 127, 0), 255);
}

// Multiplicative inverse of the scale `biasedExp` encodes; exact, the scale is a power of two.
__device__ __forceinline__ float reciprocalScaleFromExponent(int32_t biasedExp) {
  return __frcp_rn(__int_as_float(biasedExp << 23));
}

// Quantize one value that has already been multiplied by `invScale`.
__device__ __forceinline__ __nv_fp8_e4m3 quantizeToE4m3(float scaled) {
  return static_cast<__nv_fp8_e4m3>(fminf(fmaxf(scaled, -kE4m3Max), kE4m3Max));
}

// Pack one head's four block exponents into the INT32 word DeepGEMM reads, block 0 in the LSB.
__device__ __forceinline__ uint32_t packExponents(uint32_t e0, uint32_t e1, uint32_t e2,
                                                  uint32_t e3) {
  return (e0 & 0xFFu) | ((e1 & 0xFFu) << 8) | ((e2 & 0xFFu) << 16) | ((e3 & 0xFFu) << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Addressing. Both faces use the physical layout extent L, never the logical row count T.

// Element offset into the value backing [G, L, 8, kHeadDim] E4M3FN.
__device__ __forceinline__ int64_t valueOffset(int32_t group, int32_t token, int32_t headInGroup,
                                               int32_t dim, int32_t tokenCapacity) {
  return (((static_cast<int64_t>(group) * tokenCapacity + token) * kHeadsPerGroup) + headInGroup) *
             kHeadDim +
         dim;
}

// Element offset into the scale backing [G, 8, align4(L)] INT32.
__device__ __forceinline__ int64_t scaleOffset(int32_t group, int32_t headInGroup, int32_t token,
                                               int64_t scaleBufM) {
  return (static_cast<int64_t>(group) * kHeadsPerGroup + headInGroup) * scaleBufM + token;
}

// The position fed to the inverse-RoPE cache for a token, per the rule the shipped cubin
// applies: position = seq_lens_kv[b] - seq_len_q[b] + local_token.
__device__ __forceinline__ int32_t ropePosition(int32_t cacheSeqLenKv, int32_t seqLenQ,
                                                int32_t localToken) {
  return cacheSeqLenKv - seqLenQ + localToken;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dsv4
}  // namespace flashinfer
