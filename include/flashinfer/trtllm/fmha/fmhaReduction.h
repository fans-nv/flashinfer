/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "flashInferMetaInfo.h"
#include "fmhaRunnerParams.h"
#include "kernelParams.h"

namespace tensorrt_llm {
namespace kernels {

////////////////////////////////////////////////////////////////////////////////////////////////////

// A non-null `dsv4` selects the compile-time FP8-output specialization (producer P2).
// It is an explicit plan decision, never inferred from the base params:
// a runtime gate would still cost the plain reducer registers, shared memory and code
// size even when the branch is false.
//
// The specialization exists only for an effective head span of 512. The planner is
// responsible for never setting this flag for a reducer descriptor it does not cover;
// this function re-checks and reports a hard error rather than silently writing BF16
// into an FP8 allocation.
// The DSv4 FP8-output destination and launch span. Read only by the reduction kernel, so
// it is passed as an argument rather than added to KernelParams, whose layout is the
// prebuilt cubins' ABI.
struct Dsv4ReductionParams {
  // FP8 value destination, kept distinct from ptrO so the plain BF16 store cannot be
  // reached with an FP8 allocation underneath it.
  void* outValues{nullptr};
  // Addressing uses the physical extent tokenCapacity, never the logical row count.
  int32_t tokenBase{0};
  int32_t tokenCount{0};
  int32_t tokenCapacity{0};
  // Row count of the inverse-RoPE cos/sin cache, for the position upper bound.
  int32_t cosSinRows{0};
};

void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernelMeta, KernelParams const& params,
                      int32_t multiProcessorCount, bool enable_pdl, cudaStream_t stream,
                      Dsv4ReductionParams const* dsv4 = nullptr);

// Whether a compile-time FP8-output specialization exists for this reducer descriptor.
// Host-side, pure, and the single source of P2 eligibility for both the planner and the
// launcher.
bool hasDsv4Fp8ReductionSpecialization(TllmGenFmhaKernelMetaInfo const& kernelMeta);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernels
}  // namespace tensorrt_llm
