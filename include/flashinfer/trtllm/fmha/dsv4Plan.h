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

// Host-side types for the DeepSeek-V4 sparse-MLA FP8-output planner.
//
//   original BF16 selector, run once -> immutable resolved plan -> FP8 output decoration
//
// A missing fused twin or reduction specialization is a capability miss that continues the
// producer cascade, never a dispatch error.

#pragma once

#include <cstdint>
#include <string>

#include "flashinfer/trtllm/fmha/dsv4Epilogue.cuh"

// Deliberately does not include fmhaKernels.cuh, so that consumers needing only the launch
// span and the output contract do not have to compile the selector. SparseMlaPlan, which does
// need it, lives in dsv4PlanResolve.cuh.

namespace flashinfer {
namespace dsv4 {

enum class OutputMode : int32_t {
  Bf16 = 0,
  // The DeepGEMM-ready pair: E4M3FN values plus INT32-packed UE8M0 block scales.
  DeepGemmFp8 = 1,
};

enum class Producer : int32_t {
  // Exact schedule twin with the RopeQuantUe8m0Sf epilogue writes the final FP8 pair.
  Fusion = 0,
  // The base plan's already-selected separate reducer writes the final FP8 pair.
  Reduction = 1,
  // The base plan writes merged BF16 to workspace; a standalone kernel writes the pair.
  Standalone = 2,
};

inline char const* toString(Producer p) {
  switch (p) {
    case Producer::Fusion:
      return "fusion";
    case Producer::Reduction:
      return "reduction";
    case Producer::Standalone:
      return "standalone";
  }
  return "unknown";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The rows of the batch-global destination that one launch owns. Several serialized launches
// may tile one destination with different producers; their spans must be disjoint and tile
// [0, totalTokens) without gaps.
struct LaunchSpan {
  int32_t batchSize{0};
  // First batch-global token row owned by this launch, and the number of rows.
  int32_t tokenBase{0};
  int32_t tokenCount{0};
  // Batch-global logical row count tiled by all real launches.
  int32_t totalTokens{0};
  // Physical layout extent and FP8 value-group stride; > totalTokens only under graph padding.
  int32_t tokenCapacity{0};

  bool operator==(LaunchSpan const& o) const {
    return batchSize == o.batchSize && tokenBase == o.tokenBase && tokenCount == o.tokenCount &&
           totalTokens == o.totalTokens && tokenCapacity == o.tokenCapacity;
  }
};

// align(x, 4) -- the token extent of the scale layout.
inline int64_t align4(int64_t x) { return (x + 3) & ~static_cast<int64_t>(3); }

////////////////////////////////////////////////////////////////////////////////////////////////////

// Every host-visible value that affects stage-1 selection, launch geometry, workspace, or
// output layout. Tensor contents and addresses are not key fields; execution validates them.
struct ProblemKey {
  // Device / context / registry identity.
  int32_t deviceOrdinal{-1};
  uint64_t contextId{0};
  uint64_t registryGeneration{0};

  // Dtypes and static dimensions.
  int32_t dtypeQ{0};
  int32_t dtypeKv{0};
  int32_t dtypeO{0};
  int32_t numHeadsQ{0};
  int32_t numHeadsKv{0};
  int32_t headDimQk{0};
  int32_t headDimV{0};
  int32_t numTokensPerPage{0};

  // Stage-1 selector inputs.
  int32_t batchSize{0};
  int32_t maxSeqLenQ{0};
  int32_t sparseTopK{0};
  int32_t sumOfSeqLensQ{0};  // Q: the full-width token extent of the base attention call.
  int32_t multiProcessorCount{0};
  int32_t attentionWindowSize{0};

  // Output contract.
  int32_t outputMode{static_cast<int32_t>(OutputMode::Bf16)};
  LaunchSpan span{};

  bool operator==(ProblemKey const& o) const {
    return deviceOrdinal == o.deviceOrdinal && contextId == o.contextId &&
           registryGeneration == o.registryGeneration && dtypeQ == o.dtypeQ &&
           dtypeKv == o.dtypeKv && dtypeO == o.dtypeO && numHeadsQ == o.numHeadsQ &&
           numHeadsKv == o.numHeadsKv && headDimQk == o.headDimQk && headDimV == o.headDimV &&
           numTokensPerPage == o.numTokensPerPage && batchSize == o.batchSize &&
           maxSeqLenQ == o.maxSeqLenQ && sparseTopK == o.sparseTopK &&
           sumOfSeqLensQ == o.sumOfSeqLensQ && multiProcessorCount == o.multiProcessorCount &&
           attentionWindowSize == o.attentionWindowSize && outputMode == o.outputMode &&
           span == o.span;
  }
  bool operator!=(ProblemKey const& o) const { return !(*this == o); }

  std::string describe() const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Workspace regions, all of which are live simultaneously.
struct WorkspaceLayout {
  static constexpr int64_t kAlignment = 256;

  // Partial softmax statistics, float2 elements.
  int64_t partialStatsOffset{0};
  int64_t partialStatsBytes{0};
  // Partial O, 16-bit elements.
  int64_t partialOOffset{0};
  int64_t partialOBytes{0};
  // The P3 merged-BF16 intermediate, [Q, numHeadsQ, headDimV] BF16.
  int64_t bf16IntermediateOffset{0};
  int64_t bf16IntermediateBytes{0};

  int64_t totalBytes{0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Producer P3, implemented in csrc/dsv4_inv_rope_quant.cu.

// Read the launch's own rows of the merged BF16 intermediate and write the corresponding
// batch-global FP8 rows. `cumSeqLensQ` is launch-local but its values are absolute
// batch-global token offsets; `seqLensKv` is the launch-local [batchSize] array.
void launchDsv4InvRopeQuant(void const* bf16In, void* outValues, void* outScales,
                            float const* cosSinCache, int32_t cosSinRows,
                            int32_t const* cumSeqLensQ, int32_t const* seqLensKv, int32_t numHeadsQ,
                            LaunchSpan const& span, int64_t scaleBufM, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dsv4
}  // namespace flashinfer
