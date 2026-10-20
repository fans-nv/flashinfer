/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */
// Test-only placement probes for the localized MoE tests (not part of any production module):
// the locality domain backing a device address, and the %smid of blocks launched on a stream.
// Plain C ABI, built with nvcc by the tests and loaded through ctypes.
#include <cuda.h>
#include <cuda_runtime_api.h>

#if CUDA_VERSION < 13040
#error "the localized MoE probes need CUDA >= 13.4 headers (locality-domain pointer attribute)"
#endif

namespace {
__global__ void smidProbeKernel(int* out, int spin) {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  // Brief busy-wait so blocks spread across the partition instead of reusing one SM.
  long long start = clock64();
  while (clock64() - start < spin) {
  }
  if (threadIdx.x == 0) out[blockIdx.x] = static_cast<int>(smid);
}
}  // namespace

extern "C" {

// Locality-domain ordinal of the allocation containing ptr (-1 if not localized); returns
// the CUresult.
int localized_pointer_domain(void const* ptr, int* domain) {
  int d = -2;
  CUresult r = cuPointerGetAttribute(&d, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                                     reinterpret_cast<CUdeviceptr>(ptr));
  *domain = d;
  return r;
}

// Writes %smid of each block to out[nblocks] (device memory), launched on stream.
int localized_smid_probe(void* stream, int* out, int nblocks, int spinCycles) {
  smidProbeKernel<<<nblocks, 32, 0, static_cast<cudaStream_t>(stream)>>>(out, spinCycles);
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
