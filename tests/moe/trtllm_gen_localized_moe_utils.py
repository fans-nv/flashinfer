"""Fixtures for the localized (split-N) MoE tests: locality resources from PyTorch, test-only
placement probes, NVFP4 weights, shards cut by the slice-then-shuffle rule, routes, and
bit-level comparisons."""

import ctypes
import hashlib
import importlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import torch

from flashinfer import fp4_quantize
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.quantization.fp4_quantization import block_scale_interleave

H, I, TOPK, E = 6144, 1536, 4, 128
RENORMALIZE = 1
MINIMAX2 = (
    7  # RoutingMethodType.MiniMax2: sigmoid + bias -> top-k -> scaled sum-normalize
)
ROUTED_SCALE = 2.0
SWIGLU = 3
FP4_MAX = 448.0 * 6.0
SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA = 1.702, 7.0, 1.0


def check_accuracy(actual, expected, percent_threshold=0.97):
    """Same rule as check_accuracy in tests/moe/utils.py (the NVFP4 MoE tolerance)."""
    actual, expected = actual.float(), expected.float()
    output_scale = max(expected.std().item(), 0.01)
    atol = max(0.05, 1.5 * output_scale)
    abs_diff = (actual - expected).abs()
    within = (abs_diff < atol) | (abs_diff / (expected.abs() + 1e-8) < 0.5)
    pct = within.float().mean().item()
    return pct >= percent_threshold, pct, atol


# ---- locality resources (owned by the framework in production; built here for the tests) -----
@dataclass
class LocalityResources:
    """One green context, stream and domain-local pool per locality domain (strict split)."""

    device: torch.device
    api: str
    domain_ids: List[int]
    sm_counts: List[int]
    contexts: List[Any] = field(repr=False)
    streams: List[torch.cuda.Stream] = field(repr=False)
    pools: List[Any] = field(repr=False)

    def localized_kwargs(self, shards):
        return dict(
            localized_weights=shards,
            localized_streams=self.streams,
            localized_sm_count=list(self.sm_counts),
        )


def make_locality_resources(
    device: torch.device, num_domains: int = 2
) -> LocalityResources:
    """PyTorch's locality API, as a serving framework uses it: the upstream SMPartition +
    LocalizedAllocator form first, then the older GreenContext(locality_domain_id=) +
    LocalizedMemPool form. Raises RuntimeError when neither is usable."""
    green = importlib.import_module("torch.cuda.green_contexts")
    idx = device.index if device.index is not None else torch.cuda.current_device()
    for name in ("get_num_locality_domains", "is_localization_supported"):
        if not hasattr(green, name):
            raise RuntimeError(f"torch.cuda.green_contexts has no {name}")
    if not green.is_localization_supported(idx):
        raise RuntimeError(f"is_localization_supported({idx}) is False")
    if green.get_num_locality_domains(idx) < num_domains:
        raise RuntimeError(
            f"device {idx} has fewer than {num_domains} locality domains"
        )
    if hasattr(green, "SMPartition") and hasattr(torch.cuda, "LocalizedAllocator"):
        parts, _remainder = green.SMPartition.from_device(idx).split(
            num_sms=0, locality_domain_ids=list(range(num_domains)), backfill=False
        )
        contexts = [green.GreenContext(sm_partition=p) for p in parts]
        with torch.cuda.device(idx):
            pools = [
                torch.cuda.MemPool(
                    allocator=torch.cuda.LocalizedAllocator(i, device=idx).allocator()
                )
                for i in range(num_domains)
            ]
        api = "sm_partition"
    elif hasattr(green, "GreenContext") and hasattr(torch.cuda, "LocalizedMemPool"):
        contexts = [
            green.GreenContext(locality_domain_id=i, device_id=idx)
            for i in range(num_domains)
        ]
        pools = [torch.cuda.LocalizedMemPool(i, device=idx) for i in range(num_domains)]
        api = "locality_domain_id"
    else:
        raise RuntimeError(
            "torch has no locality-domain green context / localized pool API"
        )
    sm_counts = [getattr(c, "sm_count", None) for c in contexts]
    if any(n is None for n in sm_counts):
        raise RuntimeError("this torch's GreenContext does not report sm_count")
    return LocalityResources(
        device=torch.device("cuda", idx),
        api=api,
        domain_ids=list(range(num_domains)),
        sm_counts=sm_counts,
        contexts=contexts,
        streams=[c.Stream() for c in contexts],
        pools=pools,
    )


# ---- test-only native helpers (placement probes, graph inspection) ----------------------------
_GENCODE = [
    "-gencode=arch=compute_100,code=sm_100",
    "-gencode=arch=compute_100,code=compute_100",
]


def build_test_library(src: Path, stem: str) -> Path:
    """nvcc-builds a small C-ABI CUDA source into a cached .so, keyed by source and flags."""
    data = Path(src).read_bytes()
    key = hashlib.sha256(data + " ".join(_GENCODE).encode()).hexdigest()[:16]
    base = Path(os.environ.get("FLASHINFER_WORKSPACE_BASE", Path.home()))
    out_dir = base / ".cache" / "flashinfer_localized_moe_test_libs" / f"{stem}_{key}"
    so = out_dir / f"lib{stem}.so"
    if so.exists():
        return so
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".tmp.{os.getpid()}.so"
    nvcc = os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc")
    cmd = [
        nvcc,
        "-O2",
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        *_GENCODE,
        str(src),
        "-o",
        str(tmp),
        "-lcuda",
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp, so)
    return so


class Probes:
    """Pointer-to-domain and %smid placement probes (tests/moe/csrc/trtllm_gen_localized_probes.cu)."""

    def __init__(self):
        so = build_test_library(
            Path(__file__).parent / "csrc" / "trtllm_gen_localized_probes.cu",
            "trtllm_gen_localized_probes",
        )
        lib = ctypes.CDLL(str(so))
        lib.localized_pointer_domain.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.localized_smid_probe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib = lib

    def pointer_domain_addr(self, addr: int) -> int:
        """Locality-domain ordinal of the allocation containing addr, -1 if not localized."""
        d = ctypes.c_int()
        rc = self._lib.localized_pointer_domain(ctypes.c_void_p(addr), ctypes.byref(d))
        if rc != 0:
            raise RuntimeError(f"cuPointerGetAttribute failed ({rc})")
        return d.value

    def pointer_domain(self, t: torch.Tensor) -> int:
        return self.pointer_domain_addr(t.data_ptr())

    def smid_probe(
        self, stream: torch.cuda.Stream, out: torch.Tensor, spin_cycles=0
    ) -> None:
        """One block per element of int32 `out` on stream; each block writes its %smid."""
        rc = self._lib.localized_smid_probe(
            ctypes.c_void_p(stream.cuda_stream),
            ctypes.c_void_p(out.data_ptr()),
            out.numel(),
            spin_cycles,
        )
        if rc != 0:
            raise RuntimeError(f"smid probe launch failed ({rc})")

    def check_placement(self, shards, domain_ids) -> None:
        """Raises unless every shard's first and last byte live in its partition's domain."""
        for p, shard in enumerate(shards):
            for name, t in shard.items():
                # Base and last byte: a shard spanning two pool segments needs both checked.
                last = t.data_ptr() + t.numel() * t.element_size() - 1
                for where, got in (
                    ("base", self.pointer_domain(t)),
                    ("last byte", self.pointer_domain_addr(last)),
                ):
                    if got != domain_ids[p]:
                        raise RuntimeError(
                            f"shard {p} {name} {where} is in domain {got}, "
                            f"expected {domain_ids[p]}"
                        )


def quant_rows(w, gs):
    q, sf = fp4_quantize(w, gs, 16, False, False)
    return q.view(torch.uint8), sf.view(torch.uint8).reshape(
        w.shape[0], w.shape[1] // 16
    )


def shuffle_fc1(q, s, cache):
    """trtllm-gen shuffle of one expert's gated FC1 ([up | gate] rows) and its block scales."""
    p = _maybe_get_cached_w3_w1_permute_indices(cache, q, 128, is_gated_act_gemm=True)
    ps = _maybe_get_cached_w3_w1_permute_indices(
        cache, s, 128, num_elts_per_sf=16, is_gated_act_gemm=True
    )
    return q[p].contiguous(), block_scale_interleave(s[ps].contiguous())


def shuffle_fc2(q, s, cache):
    p = get_w2_permute_indices_with_cache(cache, q, 128)
    ps = get_w2_permute_indices_with_cache(cache, s, 128, num_elts_per_sf=16)
    return q[p].contiguous(), block_scale_interleave(s[ps].contiguous())


def _stack(w1s, w1ss, w2s, w2ss, rows1, rows2, k1, k2):
    E_ = len(w1s)
    return dict(
        gemm1_weights=torch.stack(w1s),
        gemm1_weights_scale=torch.stack(w1ss)
        .view(torch.float8_e4m3fn)
        .reshape(E_, rows1, k1 // 16),
        gemm2_weights=torch.stack(w2s),
        gemm2_weights_scale=torch.stack(w2ss)
        .view(torch.float8_e4m3fn)
        .reshape(E_, rows2, k2 // 16),
    )


def build_weights(E, dev, seed, I=I):
    """Returns (full shuffled weights, logical quantized weights, gs1, gs2). The logical form
    is per expert (q1 [2I, H/2], s1 [2I, H/16], q2 [H, I/2], s2 [H, I/16]), unshuffled."""
    g = torch.Generator(device=dev).manual_seed(seed)
    cache = {}
    w1s, w1ss, w2s, w2ss, gs1, gs2, logical = [], [], [], [], [], [], []
    for _ in range(E):
        w1 = (torch.randn(2 * I, H, generator=g, device=dev) * 0.03).bfloat16()
        w2 = (torch.randn(H, I, generator=g, device=dev) * 0.03).bfloat16()
        g1 = FP4_MAX / w1.float().abs().max()
        g2 = FP4_MAX / w2.float().abs().max()
        q1, s1 = quant_rows(w1, g1.reshape(1))
        q2, s2 = quant_rows(w2, g2.reshape(1))
        logical.append((q1, s1, q2, s2))
        a, b = shuffle_fc1(q1, s1, cache)
        c, d = shuffle_fc2(q2, s2, cache)
        w1s.append(a)
        w1ss.append(b)
        w2s.append(c)
        w2ss.append(d)
        gs1.append(g1)
        gs2.append(g2)
    full = _stack(w1s, w1ss, w2s, w2ss, 2 * I, H, H, I)
    return full, logical, torch.stack(gs1).float(), torch.stack(gs2).float()


def make_inputs(T, E, dev, seed, skew):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(T, H, generator=g, device=dev).bfloat16()
    logits = torch.randn(T, E, generator=g, device=dev).float()
    if skew == "hot4":
        logits[:, :4] += (
            8.0  # every token to experts 0..3; the other E-4 experts are empty
        )
    elif skew == "hot1":
        logits[:, 5] += (
            8.0  # one expert receives every token; three others share the rest
        )
    hs_gs = (FP4_MAX / x.float().abs().max().clamp(min=1e-6)).reshape(1)
    xq, xs = fp4_quantize(x, hs_gs, 16, False, False)
    return (
        xq.view(torch.uint8),
        xs.view(torch.float8_e4m3fn).reshape(T, H // 16),
        logits,
        hs_gs,
    )


def scalars(gs1, gs2, hs_gs, fc2_input_scale=None):
    """fc2_input_scale: a static FC2 activation global scale (mode Q uses 16, i.e. a 1/16
    decode scale) instead of the clamp-derived one."""
    c_gsf = (
        FP4_MAX / (SWIGLU_LIMIT * (SWIGLU_LIMIT + 1.0))  # |h| <= 7 * 8 under the clamp
        if fc2_input_scale is None
        else fc2_input_scale
    )
    g1_alpha = 1.0 / (gs1 * hs_gs)
    return dict(
        output1_scale_scalar=(c_gsf * g1_alpha).float().contiguous(),
        output1_scale_gate_scalar=g1_alpha.float().contiguous(),
        output2_scale_scalar=(1.0 / (c_gsf * gs2)).float().contiguous(),
        gemm1_alpha=torch.full_like(gs1, SWIGLU_ALPHA),
        gemm1_beta=(SWIGLU_BETA / g1_alpha)
        .float()
        .contiguous(),  # raw-accumulator units
        gemm1_clamp_limit=(SWIGLU_LIMIT / g1_alpha).float().contiguous(),
    )


def sf_offsets(rows, n_sf_cols, tile):
    """Byte offsets of the gemm1 output block scales for the given rows (all scale columns).
    R8c4 below tile 128, R128c4 at tile >= 128 (runner.cu convention)."""
    c4 = (n_sf_cols + 3) // 4 * 4
    r = rows.view(-1, 1).long()
    c = torch.arange(n_sf_cols, device=rows.device).view(1, -1)
    if tile >= 128:
        tile_id = (r // 128) * (c4 // 4) + c // 4
        off = tile_id * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    else:
        tile_id = (r // 8) * (c4 // 4) + c // 4
        off = tile_id * 32 + (r % 8) * 4 + c % 4
    return off.reshape(-1)


def masked_compare(ru_i, rs_i, tile, I=I):
    """Intermediates per expanded (token, slot) index: row perm_u[i] of the localized buffers
    against row perm_s[i] of the reference (the routing kernel's permutation order is not
    deterministic across calls for larger T, so buffers are compared through each run's own
    map). Bit-exact expected. Padding rows are characterized by how many still hold the canary."""
    g1u, sfu, g2u, permu = ru_i
    g1s, sfs, g2s, perms = rs_i
    res = dict(perm_bitexact=bits_equal(permu, perms))
    pu, ps = permu.reshape(-1).long(), perms.reshape(-1).long()
    ok = (pu >= 0) & (ps >= 0)
    res["valid_mask_equal"] = bool(torch.equal(pu >= 0, ps >= 0))
    pu, ps = pu[ok], ps[ok]
    res["valid_rows"] = int(pu.numel())
    res["gemm1_valid_rows_bitexact"] = bits_equal(g1u[pu], g1s[ps])
    res["gemm2_valid_rows_bitexact"] = bits_equal(g2u[pu], g2s[ps])
    fu8, fs8 = sfu.view(torch.uint8).reshape(-1), sfs.view(torch.uint8).reshape(-1)
    res["gemm1_scale_valid_rows_bitexact"] = bool(
        torch.equal(
            fu8[sf_offsets(pu, I // 16, tile)], fs8[sf_offsets(ps, I // 16, tile)]
        )
    )
    for name, bu, bs, sent in (("gemm1", g1u, g1s, 0xA5), ("gemm2", g2u, g2s, 0xC3)):
        cu = cs = 0
        for buf, perm, key in ((bu, pu, "u"), (bs, ps, "s")):
            pad = torch.ones(buf.shape[0], dtype=torch.bool, device=buf.device)
            pad[perm] = False
            rows = buf[pad].contiguous().view(torch.uint8)
            n = int((rows == sent).all(dim=1).sum()) if rows.numel() else 0
            if key == "u":
                cu, padn = n, int(pad.sum())
            else:
                cs = n
        res[f"{name}_padding_rows"] = padn
        res[f"{name}_padding_canary_rows_localized_vs_ref"] = [cu, cs]
    return res


def bits_equal(a, b):
    if a.shape != b.shape:
        return False
    return bool(
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def routes(T, dev, seed, skew):
    g = torch.Generator(device=dev).manual_seed(seed)
    logits = torch.randn(T, E, generator=g, device=dev)
    bias = torch.randn(E, generator=g, device=dev) * 0.1
    if skew == "hot4":
        bias[:4] += 10.0
    elif skew == "hot1":
        bias[5] += 10.0
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias, TOPK, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True) * 2.0
    return ids.to(torch.int32).contiguous(), w.float().contiguous()


def pack(ids, w):
    """Packed top-k: expert id in the high 16 bits, bf16 weight bits in the low 16."""
    wb = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return ((ids.to(torch.int32) << 16) | wb).contiguous()


def make_shards(logical, res, I=I, swap=False, pools=True):
    """Shard p is cut from the logical weights (FC1 rows [p*I/2, (p+1)*I/2) of
    both gated halves, FC2 rows [p*H/2, (p+1)*H/2) with the full K) and then shuffled as a
    standalone tensor, allocated in partition p's pool. swap gives partition p the other
    partition's shard (negative control)."""
    cache = {}
    out = []
    for p in range(2):
        q = 1 - p if swap else p
        rows1 = slice(q * I // 2, (q + 1) * I // 2)
        rows2 = slice(q * H // 2, (q + 1) * H // 2)
        w1s, w1ss, w2s, w2ss = [], [], [], []
        for q1, s1, q2, s2 in logical:
            a, b = shuffle_fc1(
                torch.cat([q1[:I][rows1], q1[I:][rows1]]),
                torch.cat([s1[:I][rows1], s1[I:][rows1]]),
                cache,
            )
            c, d = shuffle_fc2(q2[rows2].contiguous(), s2[rows2].contiguous(), cache)
            w1s.append(a)
            w1ss.append(b)
            w2s.append(c)
            w2ss.append(d)
        shard = _stack(w1s, w1ss, w2s, w2ss, I, H // 2, H, I)
        if pools:
            with torch.cuda.use_mem_pool(res.pools[p]):
                placed = {k: torch.empty_like(v) for k, v in shard.items()}
        else:
            placed = {k: torch.empty_like(v) for k, v in shard.items()}
        for k, v in shard.items():
            placed[k].copy_(v)
        out.append(placed)
    return out


def full_shards(full):
    """Mode-1 reference: the full weights passed as shard 0."""
    return [dict(full)]


def routed_kwargs(xq, xs, sc, do_finalize, E=E, I=I):
    return dict(
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs,
        gemm1_bias=None,
        gemm1_alpha=sc["gemm1_alpha"],
        gemm1_beta=sc["gemm1_beta"],
        gemm1_clamp_limit=sc["gemm1_clamp_limit"],
        gemm2_bias=None,
        output1_scale_scalar=sc["output1_scale_scalar"],
        output1_scale_gate_scalar=sc["output1_scale_gate_scalar"],
        output2_scale_scalar=sc["output2_scale_scalar"],
        num_experts=E,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=do_finalize,
        activation_type=SWIGLU,
    )


def monolithic_kwargs(logits, xq, xs, sc, do_finalize, E=E, I=I):
    return dict(
        routing_logits=logits,
        routing_bias=None,
        hidden_states=xq,
        hidden_states_scale=xs,
        gemm1_bias=None,
        gemm1_alpha=sc["gemm1_alpha"],
        gemm1_beta=sc["gemm1_beta"],
        gemm1_clamp_limit=sc["gemm1_clamp_limit"],
        gemm2_bias=None,
        output1_scale_scalar=sc["output1_scale_scalar"],
        output1_scale_gate_scalar=sc["output1_scale_gate_scalar"],
        output2_scale_scalar=sc["output2_scale_scalar"],
        num_experts=E,
        top_k=TOPK,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=RENORMALIZE,
        do_finalize=do_finalize,
        activation_type=SWIGLU,
    )


def minimax2_routing(T, dev, seed, skew):
    """MiniMax2 router inputs for the monolithic op: fp32 logits and fp32 correction bias."""
    g = torch.Generator(device=dev).manual_seed(seed)
    logits = torch.randn(T, E, generator=g, device=dev).float()
    bias = torch.randn(E, generator=g, device=dev).float() * 0.1
    if skew == "hot4":
        bias[:4] += 10.0
    return logits.contiguous(), bias.contiguous()


def minimax2_kwargs(logits, bias, xq, xs, sc, do_finalize, E=E, I=I):
    kw = monolithic_kwargs(logits, xq, xs, sc, do_finalize, E=E, I=I)
    kw.update(
        routing_bias=bias,
        routing_method_type=MINIMAX2,
        routed_scaling_factor=ROUTED_SCALE,
    )
    return kw
