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

// The immutable plan value and the planner declaration: the only part of the DSv4 planner
// that needs the trtllm-gen selector's types.

#pragma once

#include <type_traits>
#include <utility>  // std::declval

#include "flashinfer/trtllm/fmha/dsv4Plan.h"
#include "flashinfer/trtllm/fmha/fmhaKernels.cuh"
#include "flashinfer/trtllm/fmha/fmhaRunnerParams.h"

namespace flashinfer {
namespace dsv4 {

////////////////////////////////////////////////////////////////////////////////////////////////////

// The immutable result of planning. Execution consumes this exact value and never re-selects.
struct SparseMlaPlan {
  using KernelMeta = tensorrt_llm::kernels::TllmGenFmhaKernelMetaInfo;

  ProblemKey key{};

  // Stage 1: the original BF16 selection outcome, verbatim.
  TllmGenFmhaKernel const* baseKernels{nullptr};
  KernelMeta baseMeta{};
  CUfunction baseFunc{};
  TllmGenSelectKernelParams selectParams;
  TllmGenFmhaKernel::CtaLaunchParams ctaLaunch{};

  // Stage 2: the attached output producer.
  Producer producer{Producer::Standalone};
  // P1 only: the exact fused twin.
  TllmGenFmhaKernel const* twinKernels{nullptr};
  KernelMeta twinMeta{};
  CUfunction twinFunc{};
  // P2 only: the effective head span of the frozen reducer descriptor.
  int32_t reducerEffectiveHeadDim{0};

  WorkspaceLayout workspace{};

  // Diagnostics. Never a dispatch input.
  std::string reason;
  std::string baseKernelName;
  std::string producerKernelName;

  // SelectKernelParams has no default constructor; seed it from a zeroed runner params.
  explicit SparseMlaPlan(TllmGenFmhaRunnerParams const& seed) : selectParams(seed) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// Resolve the attention plan and attach an output producer. Host-side only: no device reads
// and no stream synchronization.
SparseMlaPlan planDsv4SparseMla(ProblemKey const& key);

uint64_t currentRegistryGeneration();

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dsv4
}  // namespace flashinfer
