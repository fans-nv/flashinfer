"""GPU tests for the localized (split-N) trtllm-gen NVFP4 MoE on SM107.

Needs one SM107 GPU with two locality domains, PyTorch's locality-domain API (SMPartition +
LocalizedAllocator, or GreenContext(locality_domain_id=) + LocalizedMemPool), the split-N BMM
package (installed, or staged from FLASHINFER_LOCALIZED_MOE_BMM_BUNDLE) and nvcc for the
test-only probes; everything skips elsewhere.

The green-context streams and the domain-local pools are created through PyTorch, as the
serving framework does, and passed to the stock ops as localized_streams / localized_weights.
Bit-exactness is against the same module's stock single-stream path (mode 1, full weights, same
tactic), which runs the same cubins; the installed stock op is compared within the FlashInfer
NVFP4 MoE tolerance.
"""

import ctypes
import subprocess
import warnings
from pathlib import Path

import pytest
import torch

from flashinfer.utils import get_compute_capability

if not torch.cuda.is_available() or get_compute_capability(torch.device("cuda")) != (
    10,
    7,
):
    pytest.skip("localized trtllm-gen MoE is SM107 only", allow_module_level=True)

from flashinfer.artifacts import has_trtllm_gen_bmm_split_n  # noqa: E402
from flashinfer.fused_moe import (  # noqa: E402
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.trtllm_localized import moe as M  # noqa: E402
from flashinfer.fused_moe.trtllm_localized import tuning as TU  # noqa: E402

from . import trtllm_gen_localized_moe_utils as U  # noqa: E402

H, I, TOPK, E = U.H, U.I, U.TOPK, U.E


def _unavailable(reason):
    pytest.skip(reason, allow_module_level=True)


DEV = torch.device("cuda", torch.cuda.current_device())
try:
    RES = U.make_locality_resources(DEV)
except Exception as e:  # noqa: BLE001
    _unavailable(f"no locality resources from PyTorch: {e}")
if not has_trtllm_gen_bmm_split_n():
    _unavailable("split-N BMM package unavailable")
# A build, pin or ABI failure must fail the suite, not skip it.
M.get_module()
try:
    PROBES = U.Probes()
except Exception as e:  # noqa: BLE001
    _unavailable(f"cannot build the placement probes: {e}")
TOTAL_SMS = torch.cuda.get_device_properties(DEV).multi_processor_count


def _placeholders():
    """Storage-released full-width weights: the localized path must not read them."""
    t = torch.empty(0, dtype=torch.uint8, device=DEV)
    return dict(
        gemm1_weights=t, gemm1_weights_scale=t, gemm2_weights=t, gemm2_weights_scale=t
    )


@pytest.fixture(scope="module")
def weights():
    full, logical, gs1, gs2 = U.build_weights(E, DEV, seed=7)
    return full, gs1, gs2, U.make_shards(logical, RES), logical


def _routed(T, seed, skew, gs1, gs2, fin, fmt="packed"):
    xq, xs, _, hs_gs = U.make_inputs(T, E, DEV, seed, "none")
    ids, w = U.routes(T, DEV, seed + 1, skew)
    topk = U.pack(ids, w) if fmt == "packed" else (ids, w.to(torch.bfloat16))
    return topk, U.routed_kwargs(xq, xs, U.scalars(gs1, gs2, hs_gs), fin)


def routed(topk, kw, shards, **extra):
    """The public routed op with localized_* arguments."""
    return trtllm_fp4_block_scale_routed_moe(
        topk, **kw, **_placeholders(), **RES.localized_kwargs(shards), **extra
    )


def monolithic(kw, shards, **extra):
    return trtllm_fp4_block_scale_moe(
        **kw, **_placeholders(), **RES.localized_kwargs(shards), **extra
    )


def internal(routing, kw, shards, **extra):
    """The localized body with its private hooks (mode, intermediates, tactic, debug delay).
    routing is packed ids, an (ids, weights) tuple, or None for the monolithic kw."""
    kw = dict(kw)
    if routing is None:
        args = dict(
            routing_input_mode=0,
            routing_logits=kw.pop("routing_logits"),
            topk_ids=None,
            topk_weights=None,
        )
    elif isinstance(routing, tuple):
        args = dict(
            routing_input_mode=2,
            routing_logits=None,
            topk_ids=routing[0],
            topk_weights=routing[1],
        )
    else:
        args = dict(
            routing_input_mode=1,
            routing_logits=None,
            topk_ids=routing,
            topk_weights=None,
        )
    kw.pop("tune_max_num_tokens", None)
    return M.trtllm_fp4_block_scale_moe_localized(
        **args, **kw, **RES.localized_kwargs(shards), **extra
    )


def _call_pair(routing, kw, shards, full, tactic, fin, T):
    """Runs localized (mode 0) and reference (mode 1) with a canary-guarded output."""
    outs = []
    for mode, sh in ((0, shards), (1, U.full_shards(full))):
        output = None
        if fin:
            big = torch.full((T + 2, H), 12345.0, dtype=torch.bfloat16, device=DEV)
            output = big[1 : T + 1]
        r = internal(
            routing,
            kw,
            sh,
            output=output,
            tactic=tactic,
            _mode=mode,
            _return_intermediates=True,
        )
        torch.cuda.synchronize()
        if fin:
            assert (big[0] == 12345.0).all() and (big[T + 1] == 12345.0).all(), (
                "output overrun"
            )
        outs.append(r)
    return outs


def _assert_same(ru, rs, fin, tile):
    if fin:
        assert U.bits_equal(ru[0], rs[0])
        n = 1
    else:  # unfinalized: compare through each run's own permutation map
        assert U.bits_equal(
            ru[0][ru[2].reshape(-1).long()], rs[0][rs[2].reshape(-1).long()]
        )
        assert U.bits_equal(ru[1], rs[1])
        n = 3
    mc = U.masked_compare(ru[n : n + 4], rs[n : n + 4], tile)
    assert mc["valid_mask_equal"] and mc["valid_rows"] > 0
    assert mc["gemm1_valid_rows_bitexact"] and mc["gemm1_scale_valid_rows_bitexact"]
    assert mc["gemm2_valid_rows_bitexact"]


def _tactic(T):
    return M.default_tactic(T, TOPK, H, I, E, RES.sm_counts)


# ---- T1: topology and placement --------------------------------------------------------------
def test_partition_topology_and_smid_placement():
    assert len(RES.streams) == 2 and all(n > 0 for n in RES.sm_counts)
    assert sum(RES.sm_counts) <= TOTAL_SMS
    sets = []
    for p in range(2):
        n = 8 * RES.sm_counts[p]
        out = torch.full((n,), -1, dtype=torch.int32, device=DEV)
        PROBES.smid_probe(RES.streams[p], out, spin_cycles=200000)
        torch.cuda.synchronize()
        s = set(out.tolist())
        assert -1 not in s and 0.9 * RES.sm_counts[p] <= len(s) <= RES.sm_counts[p]
        sets.append(s)
    assert not (sets[0] & sets[1]), "partitions share SMs"


def test_pool_pointer_domain():
    for p in range(2):
        with torch.cuda.use_mem_pool(RES.pools[p]):
            t = torch.empty(3 << 20, dtype=torch.uint8, device=DEV)
        last = t.data_ptr() + t.numel() - 1
        assert (
            PROBES.pointer_domain(t)
            == PROBES.pointer_domain_addr(last)
            == RES.domain_ids[p]
        )
    ordinary = torch.empty(16, dtype=torch.uint8, device=DEV)
    assert PROBES.pointer_domain(ordinary) == -1  # negative control


def test_check_placement(weights):
    full, _, _, shards, logical = weights
    PROBES.check_placement(shards, RES.domain_ids)
    with pytest.raises(RuntimeError, match="domain"):
        PROBES.check_placement(U.make_shards(logical, RES, pools=False), RES.domain_ids)
    with pytest.raises(RuntimeError, match="domain"):
        PROBES.check_placement(shards[::-1], RES.domain_ids)


def test_shards_match_row_slices_of_the_full_shuffle(weights):
    """Slice-then-shuffle equals the earlier contract (rows of the shuffled full tensors),
    including the interleaved block scales."""
    full, _, _, shards, _ = weights
    for p, shard in enumerate(shards):
        for k, rows in (
            ("gemm1_weights", I),
            ("gemm1_weights_scale", I),
            ("gemm2_weights", H // 2),
            ("gemm2_weights_scale", H // 2),
        ):
            ref = full[k][:, p * rows : (p + 1) * rows]
            assert U.bits_equal(shard[k], ref.contiguous()), (p, k)


# ---- T2: bit-exact vs same-module stock (mode 1) -----------------------------------------------
@pytest.mark.parametrize("T", [1, 4, 8, 64, 257, 1024, 4096])
@pytest.mark.parametrize("fin", [True, False])
@pytest.mark.parametrize("skew", ["none", "hot4"])
@pytest.mark.parametrize("fmt", ["packed", "unpacked"])
def test_routed_bitexact_vs_mode1(weights, T, fin, skew, fmt):
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(T, 1000 + T, skew, gs1, gs2, fin, fmt)
    tactic = _tactic(T)
    ru, rs = _call_pair(topk, kw, shards, full, tactic, fin, T)
    _assert_same(ru, rs, fin, tactic[0])


@pytest.mark.parametrize("T", [4, 257, 1024])
@pytest.mark.parametrize("fin", [True, False])
def test_monolithic_bitexact_vs_mode1(weights, T, fin):
    full, gs1, gs2, shards, _ = weights
    xq, xs, logits, hs_gs = U.make_inputs(T, E, DEV, 2000 + T, "none")
    kw = U.monolithic_kwargs(logits, xq, xs, U.scalars(gs1, gs2, hs_gs), fin)
    tactic = _tactic(T)
    ru, rs = _call_pair(None, kw, shards, full, tactic, fin, T)
    _assert_same(ru, rs, fin, tactic[0])


@pytest.mark.parametrize("T", [1, 4, 8, 64, 257, 1024, 4096])
@pytest.mark.parametrize("fin", [True, False])
@pytest.mark.parametrize("skew", ["none", "hot4"])
def test_monolithic_minimax2_bitexact_vs_mode1(weights, T, fin, skew):
    """Routing inside the launcher with MiniMax2 (sigmoid + correction bias,
    routed_scaling_factor 2.0), then the split-N fork/join."""
    full, gs1, gs2, shards, _ = weights
    xq, xs, _, hs_gs = U.make_inputs(T, E, DEV, 2500 + T, "none")
    logits, bias = U.minimax2_routing(T, DEV, 2600 + T, skew)
    kw = U.minimax2_kwargs(logits, bias, xq, xs, U.scalars(gs1, gs2, hs_gs), fin)
    tactic = _tactic(T)
    ru, rs = _call_pair(None, kw, shards, full, tactic, fin, T)
    _assert_same(ru, rs, fin, tactic[0])
    if not fin:  # routed weights carry the 2.0 scale: each token's weights sum to ~2
        w = ru[1].float().sum(-1)
        assert torch.allclose(w, torch.full_like(w, U.ROUTED_SCALE), atol=0.05)


@pytest.mark.parametrize("T", [4, 1024])
def test_monolithic_minimax2_vs_installed_stock(weights, T):
    full, gs1, gs2, shards, _ = weights
    xq, xs, _, hs_gs = U.make_inputs(T, E, DEV, 2700 + T, "none")
    logits, bias = U.minimax2_routing(T, DEV, 2800 + T, "none")
    kw = U.minimax2_kwargs(logits, bias, xq, xs, U.scalars(gs1, gs2, hs_gs), True)
    loc = monolithic(kw, shards)[0]
    ref = trtllm_fp4_block_scale_moe(**kw, **full, output=None)[0]
    torch.cuda.synchronize()
    assert ref.float().abs().sum() > 0
    ok, pct, _ = U.check_accuracy(loc, ref)
    assert ok, f"only {pct:.4f} within the NVFP4 MoE tolerance"


def test_monolithic_minimax2_graph_replay(weights):
    full, gs1, gs2, shards, _ = weights
    T = 8
    xq, xs, _, hs_gs = U.make_inputs(T, E, DEV, 2900, "none")
    logits, bias = U.minimax2_routing(T, DEV, 2901, "none")
    kw = U.minimax2_kwargs(logits, bias, xq, xs, U.scalars(gs1, gs2, hs_gs), True)
    tactic = _tactic(T)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    internal(None, kw, shards, output=out, tactic=tactic)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        internal(None, kw, shards, output=out, tactic=tactic)
    for rep in range(3):
        xq2, xs2, _, _ = U.make_inputs(T, E, DEV, 2910 + rep, "none")
        l2, b2 = U.minimax2_routing(T, DEV, 2920 + rep, "hot4" if rep == 2 else "none")
        kw["hidden_states"].copy_(xq2)
        kw["hidden_states_scale"].copy_(xs2)
        kw["routing_logits"].copy_(l2)
        kw["routing_bias"].copy_(b2)
        graph.replay()
        torch.cuda.synchronize()
        ref = internal(None, kw, U.full_shards(full), tactic=tactic, _mode=1)[0]
        torch.cuda.synchronize()
        assert U.bits_equal(out, ref), f"replay {rep}"


@pytest.mark.parametrize("which", ["fc1", "fc2"])
def test_negative_control_swapped_shards(weights, which):
    """Red-first: partition p running the other partition's shard must change the output."""
    full, gs1, gs2, shards, logical = weights
    swapped = U.make_shards(logical, RES, swap=True)
    keys = (
        ("gemm1_weights", "gemm1_weights_scale")
        if which == "fc1"
        else ("gemm2_weights", "gemm2_weights_scale")
    )
    bad = [
        {k: (swapped[p][k] if k in keys else shards[p][k]) for k in M.SHARD_KEYS}
        for p in range(2)
    ]
    T = 64
    topk, kw = _routed(T, 77, "none", gs1, gs2, True)
    tactic = _tactic(T)
    good = internal(topk, kw, shards, tactic=tactic)[0]
    wrong = internal(topk, kw, bad, tactic=tactic)[0]
    torch.cuda.synchronize()
    assert (good != wrong).float().mean().item() > 0.5


def test_eplb_expert_rearrangement_of_shards(weights):
    """EPLB permutes dim 0 (experts); the same permutation on the shards and on the full
    weights must keep localized and reference bit-exact."""
    full, gs1, gs2, shards, _ = weights
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(11)).to(DEV)
    moved = []
    for p, shard in enumerate(shards):
        with torch.cuda.use_mem_pool(RES.pools[p]):
            moved.append({k: v[perm].contiguous() for k, v in shard.items()})
    full_moved = {k: v[perm].contiguous() for k, v in full.items()}
    PROBES.check_placement(moved, RES.domain_ids)
    T = 257
    topk, kw = _routed(T, 7000, "none", gs1, gs2, True)
    for k in (
        "gemm1_alpha",
        "gemm1_beta",
        "gemm1_clamp_limit",
        "output1_scale_scalar",
        "output1_scale_gate_scalar",
        "output2_scale_scalar",
    ):
        kw[k] = kw[k][perm].contiguous()
    tactic = _tactic(T)
    ru, rs = _call_pair(topk, kw, moved, full_moved, tactic, True, T)
    _assert_same(ru, rs, True, tactic[0])


# ---- T3: vs the installed stock routed op ------------------------------------------------------
@pytest.mark.parametrize("T", [1, 64, 1024])
def test_routed_vs_installed_stock(weights, T):
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(T, 3000 + T, "none", gs1, gs2, True)
    loc = routed(topk, kw, shards)[0]
    ref = trtllm_fp4_block_scale_routed_moe(
        topk, **kw, **full, output=torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    )[0]
    torch.cuda.synchronize()
    assert ref.float().abs().sum() > 0
    ok, pct, _ = U.check_accuracy(loc, ref)
    assert ok, f"only {pct:.4f} within the NVFP4 MoE tolerance"


# ---- T4: every partition-valid tactic ----------------------------------------------------------
def _runner(T_mode=1):
    return TU.LocalizedFp4MoeRunner(
        routing_input_mode=T_mode,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        num_local_experts=E,
        top_k=TOPK,
        activation_type=U.SWIGLU,
        do_finalize=True,
        enable_pdl=True,
        sm_counts=tuple(RES.sm_counts),
        signature=("test",),
    )


@pytest.mark.parametrize("T", [4, 257, 1024])
def test_every_valid_tactic_bitexact(weights, T):
    full, gs1, gs2, shards, _ = weights
    tactics = _runner().valid_tactics(T)
    # The split-N package has ~480 partition-valid tactics per T; a small set means the wrong
    # package or a validity regression.
    assert len(tactics) >= 100, tactics
    topk, kw = _routed(T, 4000 + T, "none", gs1, gs2, True)
    fails = []
    for tactic in tactics:
        ru = internal(topk, kw, shards, tactic=tactic)[0]
        rs = internal(topk, kw, U.full_shards(full), tactic=tactic, _mode=1)[0]
        torch.cuda.synchronize()
        if not (U.bits_equal(ru, rs) and rs.float().abs().sum() > 0):
            fails.append(tactic)
    assert not fails, f"{len(fails)}/{len(tactics)} tactics differ: {fails[:10]}"


# ---- T5/T6: graph replay, placement in the graph, PDL skew ------------------------------------
def _diag_lib():
    try:
        so = U.build_test_library(
            Path(__file__).parent / "csrc" / "trtllm_gen_localized_graph_diag.cu",
            "trtllm_gen_localized_graph_diag",
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot build the graph diag library: {e}")
    lib = ctypes.CDLL(str(so))
    lib.localized_graph_dump.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.localized_capturing_graph.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    return lib


def _dump_capturing(lib):
    gp = ctypes.c_void_p()
    st = lib.localized_capturing_graph(
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.byref(gp)
    )
    assert st == 1, f"not capturing ({st})"  # cudaStreamCaptureStatusActive
    buf = ctypes.create_string_buffer(1 << 20)
    assert (
        lib.localized_graph_dump(
            gp,
            ctypes.c_void_p(RES.streams[0].cuda_stream),
            ctypes.c_void_p(RES.streams[1].cuda_stream),
            buf,
            1 << 20,
        )
        == 0
    )
    return buf.value.decode().splitlines()


def _fields(line):
    toks = line.split()
    return {
        toks[i]: toks[i + 1] for i in range(0, len(toks) - 1) if not toks[i].isdigit()
    }


@pytest.mark.parametrize("T", [1, 4, 64, 257, 1024])
def test_graph_replay_and_placement(weights, T):
    """Capture with PyTorch-created green-context streams; the 4 GEMM nodes land on
    the two green contexts, routing/finalize on the primary context, >= 8 programmatic edges,
    3 replays bit-exact."""
    full, gs1, gs2, shards, _ = weights
    lib = _diag_lib()
    topk, kw = _routed(T, 5000 + T, "none", gs1, gs2, True)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    routed(
        topk, kw, shards, output=out, enable_pdl=True
    )  # eager first call creates events
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed(topk, kw, shards, output=out, enable_pdl=True)
        lines = _dump_capturing(lib)
    kernels = [_fields(x) for x in lines if x.startswith("node") and " name " in x]
    # Green contexts are identified through the partition streams' contexts.
    green = [k for k in kernels if k["partition"] != "-1"]
    assert len(green) == 4, lines  # FC1 and FC2 on each partition, nothing else
    assert sorted(int(k["partition"]) for k in green) == [0, 0, 1, 1]
    assert all(k["is_primary"] == "0" and k["name"].startswith("bmm_") for k in green)
    rest = [k for k in kernels if k not in green]
    assert rest and all(
        k["is_primary"] == "1" and int(k["ctx_sm"]) == TOTAL_SMS for k in rest
    )
    programmatic = [x for x in lines if x.startswith("edge") and x.endswith("type 1")]
    assert len(programmatic) >= 8, lines
    for rep in range(3):
        xq2, xs2, _, _ = U.make_inputs(T, E, DEV, 5100 + rep, "none")
        ids2, w2 = U.routes(T, DEV, 5200 + rep, "hot4" if rep == 2 else "none")
        kw["hidden_states"].copy_(xq2)
        kw["hidden_states_scale"].copy_(xs2)
        topk.copy_(U.pack(ids2, w2))
        graph.replay()
        torch.cuda.synchronize()
        eager = routed(topk, kw, shards, enable_pdl=True)[0]
        torch.cuda.synchronize()
        assert U.bits_equal(out, eager), f"replay {rep}"


@pytest.mark.parametrize(
    "delay", [(0, 1), (1, 1), (0, 2), (1, 2)], ids=["p0fc1", "p1fc1", "p0fc2", "p1fc2"]
)
def test_pdl_skew_graph_bitexact(weights, delay):
    """A long spin before one partition's FC1 or FC2 must not let the other partition (or
    finalize) read partial data through PDL."""
    full, gs1, gs2, shards, _ = weights
    lib = _diag_lib()
    T, cycles, reps = 4, 20_000_000, 10
    topk, kw = _routed(T, 6000, "none", gs1, gs2, True)
    tactic = _tactic(T)
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    dd = (delay[0], delay[1], cycles)
    internal(
        topk, kw, shards, output=out, tactic=tactic, _debug_delay=dd, enable_pdl=True
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        internal(
            topk,
            kw,
            shards,
            output=out,
            tactic=tactic,
            enable_pdl=True,
            _debug_delay=dd,
        )
        lines = _dump_capturing(lib)
    spins = [x for x in lines if "localizedDebugSpin" in x]
    assert len(spins) == 1, (
        f"the skew kernel is not in the graph (vacuous test): {lines}"
    )
    assert _fields(spins[0])["partition"] == str(delay[0])
    # The spin must precede the delayed GEMM on its partition's stream.
    spin_idx = spins[0].split()[1]
    assert any(x.startswith(f"edge {spin_idx} -> ") for x in lines), lines
    for rep in range(reps):
        xq2, xs2, _, _ = U.make_inputs(T, E, DEV, 6100 + rep, "none")
        ids2, w2 = U.routes(T, DEV, 6200 + rep, "hot4" if rep % 3 == 2 else "none")
        kw["hidden_states"].copy_(xq2)
        kw["hidden_states_scale"].copy_(xs2)
        topk.copy_(U.pack(ids2, w2))
        graph.replay()
        torch.cuda.synchronize()
        ref = internal(topk, kw, U.full_shards(full), tactic=tactic, _mode=1)[0]
        torch.cuda.synchronize()
        assert U.bits_equal(out, ref), f"replay {rep}"


def test_debug_delay_launches_eagerly(weights):
    """The skew hook really runs: a long spin before one partition's FC1 delays the whole op."""
    import time

    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(4, 6400, "none", gs1, gs2, True)
    tactic = _tactic(4)

    def timed(dd):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        internal(topk, kw, shards, tactic=tactic, _debug_delay=dd)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    timed(None)
    base = min(timed(None) for _ in range(3))
    for dd in ((0, 1, 500_000_000), (1, 2, 500_000_000)):
        assert timed(dd) > max(0.05, 20 * base), dd


def test_debug_delay_off_launches_nothing(weights):
    full, gs1, gs2, shards, _ = weights
    lib = _diag_lib()
    topk, kw = _routed(4, 6500, "none", gs1, gs2, True)
    routed(topk, kw, shards)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed(topk, kw, shards)
        lines = _dump_capturing(lib)
    assert not [x for x in lines if "localizedDebugSpin" in x]


# ---- T8: tactic resolution on the GPU ----------------------------------------------------------
def test_pinned_defaults_resolve_by_name():
    """The pins are kernel names. Their config indices are module-bound: they were measured
    as (8, 52), (16, 55), (32, 27) and (128, 28); a different index here means the runner's config filtering
    changed, which tuning records follow through catalog_fingerprint."""
    mod = M.get_module()
    for pin in M.DEFAULT_TACTIC_PINS:
        tile, cfg = M._resolved_pins[pin.label]
        names = list(mod.trtllm_fp4_moe_localized_config_kernels(tile, U.SWIGLU, False))
        assert (names[2 * cfg], names[2 * cfg + 1]) == (pin.fc1, pin.fc2)
    for T, label in ((16, "decode"), (128, "small"), (257, "mid"), (2048, "prefill")):
        pin = M._resolved_pins[label]
        assert pin in _runner().valid_tactics(T), (T, pin)
        assert _tactic(T) == pin
    measured = {
        "decode": (8, 52),
        "small": (16, 55),
        "mid": (32, 27),
        "prefill": (128, 28),
    }
    if M._resolved_pins != measured:
        warnings.warn(
            f"pinned config indices moved: {M._resolved_pins} vs measured {measured}",
            stacklevel=1,
        )


def test_autotune_resolution_and_tuned_graph(weights, monkeypatch):
    from flashinfer.autotuner import AutoTuner, autotune

    full, gs1, gs2, shards, _ = weights
    op = TU.custom_op(1, False)
    AutoTuner.get().clear_cache()
    TU.last_resolution.clear()
    tune_max = 64

    def run(T, seed, out=None):
        topk, kw = _routed(T, seed, "none", gs1, gs2, True)
        kw["tune_max_num_tokens"] = tune_max
        return routed(topk, kw, shards, output=out)[0]

    run(4, 8000)
    assert TU.last_resolution[(op, 4)] == ("pinned", M._resolved_pins["decode"])
    # Pins win over tuning records in production; let records through to test the tuned path.
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    with autotune(True):
        for T in (1, 4, 16, 64):
            run(T, 8100 + T)
    torch.cuda.synchronize()
    for T in (1, 4, 16, 64):
        run(T, 8200 + T)
        path, tactic = TU.last_resolution[(op, T)]
        assert path == "tuned", (T, path)
        assert tactic in _runner().valid_tactics(T)
    # A tuned op captured into a graph replays bit-exact against eager.
    T = 4
    topk, kw = _routed(T, 8300, "none", gs1, gs2, True)
    kw["tune_max_num_tokens"] = tune_max
    out = torch.empty(T, H, dtype=torch.bfloat16, device=DEV)
    routed(topk, kw, shards, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed(topk, kw, shards, output=out)
    graph.replay()
    torch.cuda.synchronize()
    eager = routed(topk, kw, shards)[0]
    torch.cuda.synchronize()
    assert U.bits_equal(out, eager)
    AutoTuner.get().clear_cache()


# ---- T9: admission failures on the GPU ---------------------------------------------------------
def test_admission_wrong_shard_shape(weights):
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(4, 9000, "none", gs1, gs2, True)
    bad = [{**s, "gemm2_weights": full["gemm2_weights"]} for s in shards]
    with pytest.raises(Exception, match="wrong shape"):
        internal(topk, kw, bad, tactic=_tactic(4))


def test_admission_invalid_explicit_tactic(weights):
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(4, 9100, "none", gs1, gs2, True)
    with pytest.raises(ValueError, match="partition-valid"):
        internal(topk, kw, shards, tactic=(8, 10_000))


def test_admission_stream_on_other_device(weights):
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second device")
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(4, 9150, "none", gs1, gs2, True)
    other = torch.cuda.Stream(device=torch.device("cuda", (DEV.index + 1) % 2))
    with pytest.raises(ValueError, match="localized_streams"):
        trtllm_fp4_block_scale_routed_moe(
            topk,
            **kw,
            **_placeholders(),
            localized_weights=shards,
            localized_streams=[RES.streams[0], other],
            localized_sm_count=RES.sm_counts,
        )


def test_native_rejects_invalid_tactic_single_check(weights):
    """The native launcher validates the one requested tactic (no catalog scan); an invalid one
    still fails loudly when the Python guard is bypassed."""
    full, gs1, gs2, shards, _ = weights
    topk, kw = _routed(4, 9200, "none", gs1, gs2, True)
    mod = M.get_module()
    xq, xs = kw["hidden_states"], kw["hidden_states_scale"]
    out = torch.empty(4, H, dtype=torch.bfloat16, device=DEV)
    s0, s1 = shards
    weights_ = torch.empty(4, TOPK, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(Exception, match="not valid for the split-N partitions"):
        mod.trtllm_fp4_block_scale_moe_localized(
            1,
            None,
            topk,
            weights_,
            None,
            xq,
            xs,
            s0["gemm1_weights"],
            s0["gemm1_weights_scale"],
            None,
            kw["gemm1_alpha"],
            kw["gemm1_beta"],
            kw["gemm1_clamp_limit"],
            s0["gemm2_weights"],
            s0["gemm2_weights_scale"],
            None,
            kw["output1_scale_scalar"],
            kw["output1_scale_gate_scalar"],
            kw["output2_scale_scalar"],
            E,
            TOPK,
            0,
            0,
            I,
            0,
            E,
            None,
            1,
            True,
            True,
            U.SWIGLU,
            out,
            8,
            10_000,
            True,
            s1["gemm1_weights"],
            s1["gemm1_weights_scale"],
            s1["gemm2_weights"],
            s1["gemm2_weights_scale"],
            RES.streams[0].cuda_stream,
            RES.streams[1].cuda_stream,
            RES.sm_counts[0],
            RES.sm_counts[1],
            0,
            False,
            -1,
            0,
            0,
            None,
        )


# ---- T10: valid-config table -------------------------------------------------------------------
def test_valid_config_table():
    for tile in (8, 32, 64, 128):
        cfgs = set()
        for T in (4, 1024):
            cfgs |= set(M._valid_configs(tile, TOPK, H, I, E, T, *RES.sm_counts))
        assert cfgs, f"TP2 tile {tile} has no partition-valid config"
    tp4 = [
        M._valid_configs(t, TOPK, H, I // 2, E, 1024, *RES.sm_counts)
        for t in (8, 32, 64, 128)
    ]
    assert any(tp4), "TP4 (I=768) has no partition-valid config at any tile"


def test_module_identity_is_pinned():
    info = tuple(str(x) for x in M.get_module().trtllm_fp4_moe_localized_module_info())
    assert info == M.MODULE_INFO


def test_module_exports_only_ffi_and_cubin_hooks():
    """The split-N module shares C++ symbol names with the stock one; the version script must
    keep them local so both load in one process."""
    allowed = {"FlashInferSetCubinCallback", "FlashInferSetCurrentCubin"}
    for path in M.loaded_module_paths():
        out = subprocess.run(
            ["nm", "-D", "--defined-only", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        names = [line.split()[-1] for line in out.splitlines() if line.strip()]
        bad = [n for n in names if n not in allowed and not n.startswith("__tvm_ffi_")]
        assert names and not bad, bad[:10]
