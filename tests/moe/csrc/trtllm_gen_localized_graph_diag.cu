/*
 * Copyright (c) 2026 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */
// Test-only graph inspection for the localized MoE tests (not part of any production module):
// per kernel node its context (primary, or which partition's green context, matched through the
// partition streams' contexts), that context's SM resource where the driver reports one, and the
// PDL attribute; per edge its dependency type.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

namespace {
using CtxGetDevResourceFn = CUresult(CUDAAPI*)(CUcontext, CUdevResource*, CUdevResourceType);
// Resolved at driver API version 13.4 (the first with locality domains), so the CUdevResource
// layout (with localityDomainId) matches these headers.
CtxGetDevResourceFn ctxGetDevResource() {
  static CtxGetDevResourceFn fn = [] {
    void* address{};
    CUdriverProcAddressQueryResult status{};
    cuGetProcAddress("cuCtxGetDevResource", &address, 13040, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    return reinterpret_cast<CtxGetDevResourceFn>(address);
  }();
  return fn;
}
}  // namespace

extern "C" {

static void appendf(std::string& s, char const* fmt, ...) __attribute__((format(printf, 2, 3)));
static void appendf(std::string& s, char const* fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  s += tmp;
}

// Dumps nodes (type; for kernels: name, node ctx and that ctx's SM resource) and edges
// (from_port, to_port, type) of a CUgraph into buf. Returns CUresult of the first failure.
int localized_graph_dump(void* graphPtr, void* stream0, void* stream1, char* buf, int buflen) {
  CUgraph g = static_cast<CUgraph>(graphPtr);
  std::string out;
  size_t n = 0;
  CUresult r = cuGraphGetNodes(g, nullptr, &n);
  if (r != CUDA_SUCCESS) return r;
  std::vector<CUgraphNode> nodes(n);
  r = cuGraphGetNodes(g, nodes.data(), &n);
  if (r != CUDA_SUCCESS) return r;
  CUcontext primary{};
  cuCtxGetCurrent(&primary);
  // cuCtxGetDevResource rejects green-context nodes (CUDA_ERROR_INVALID_CONTEXT), so partition
  // membership is established by the context identity of each partition's stream instead.
  CUcontext partCtx[2]{};
  cuStreamGetCtx(static_cast<CUstream>(stream0), &partCtx[0]);
  cuStreamGetCtx(static_cast<CUstream>(stream1), &partCtx[1]);
  for (size_t i = 0; i < n; ++i) {
    CUgraphNodeType t{};
    cuGraphNodeGetType(nodes[i], &t);
    appendf(out, "node %zu type %d", i, static_cast<int>(t));
    if (t == CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS kp{};
      CUresult kr = cuGraphKernelNodeGetParams(nodes[i], &kp);
      char const* name = "?";
      if (kr == CUDA_SUCCESS && kp.func != nullptr)
        cuFuncGetName(&name, kp.func);
      else if (kr == CUDA_SUCCESS && kp.kern != nullptr)
        cuKernelGetName(&name, kp.kern);
      int part = -1;
      for (int q = 0; q < 2; ++q)
        if (kp.ctx != nullptr && kp.ctx == partCtx[q]) part = q;
      int sm = -1, dom = -1, rc = -1;
      if (kp.ctx != nullptr) {
        CUdevResource res{};
        rc = static_cast<int>(ctxGetDevResource()(kp.ctx, &res, CU_DEV_RESOURCE_TYPE_SM));
        if (rc == CUDA_SUCCESS) {
          sm = res.sm.smCount;
          dom = res.sm.localityDomainId;
        }
      }
      // Programmatic-launch attribute recorded on the node, if any.
      CUkernelNodeAttrValue av{};
      int pss = -1;
      if (cuGraphKernelNodeGetAttribute(
              nodes[i], CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, &av) == CUDA_SUCCESS)
        pss = av.programmaticStreamSerializationAllowed;
      appendf(out,
              " kr %d grid %u,%u,%u ctx %p is_primary %d partition %d ctx_rc %d ctx_sm %d "
              "ctx_domain %d pss %d name %.200s",
              static_cast<int>(kr), kp.gridDimX, kp.gridDimY, kp.gridDimZ,
              static_cast<void*>(kp.ctx), kp.ctx == primary ? 1 : 0, part, rc, sm, dom, pss, name);
    }
    out += "\n";
  }
  size_t ne = 0;
  r = cuGraphGetEdges(g, nullptr, nullptr, nullptr, &ne);
  if (r != CUDA_SUCCESS) return r;
  std::vector<CUgraphNode> from(ne), to(ne);
  std::vector<CUgraphEdgeData> ed(ne);
  r = cuGraphGetEdges(g, from.data(), to.data(), ed.data(), &ne);
  if (r != CUDA_SUCCESS) return r;
  auto idx = [&](CUgraphNode x) {
    for (size_t i = 0; i < n; ++i)
      if (nodes[i] == x) return static_cast<long>(i);
    return -1L;
  };
  for (size_t e = 0; e < ne; ++e) {
    appendf(out, "edge %ld -> %ld from_port %u to_port %u type %u\n", idx(from[e]), idx(to[e]),
            ed[e].from_port, ed[e].to_port, ed[e].type);
  }
  snprintf(buf, buflen, "%s", out.c_str());
  return 0;
}

// Returns the graph currently being captured on stream (for inspecting a capture in flight).
int localized_capturing_graph(void* stream, void** graphOut) {
  cudaStreamCaptureStatus st{};
  cudaGraph_t g{};
  cudaError_t e = cudaStreamGetCaptureInfo(static_cast<cudaStream_t>(stream), &st, nullptr, &g,
                                           nullptr, nullptr, nullptr);
  *graphOut = g;
  return e != cudaSuccess ? 1000 + e : static_cast<int>(st);
}

}  // extern "C"
