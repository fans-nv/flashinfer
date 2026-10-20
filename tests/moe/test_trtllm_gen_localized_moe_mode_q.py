"""GPU tests for mode Q (MiniMax-M3.1) on the localized and stock trtllm-gen NVFP4 MoE, SM107.

Mode Q: FC1 input is NVFP4 with a per-token fp32 decode scale s_t = row_amax / 2688
(nvfp4_quantize(..., per_token_activation=True)). FC1 applies s_t to the accumulator before the
clamp and SwiGLU-OAI and writes NVFP4 with the static output1 scale; FC2 is static (one 1/16
decode scale folded into output1/output2 scales). Needs what test_trtllm_gen_localized_moe.py
needs; the mode-Q tests also need a split-N package with the per-token FC1 kernels and skip
without one.

- The M3 catalog carries no per-token kernel (runner filter) when the package has plain twins.
- Red-first wiring check: with s_t = 1 a per-token FC1 kernel equals its plain twin bit-exactly.
- Localized mode Q equals the stock mode-Q path (same module, single stream, full weights,
  same tactic) bit-exactly, intermediates included, up to T = 16384 (prefill tile, several
  tiles per CTA).
- Both match a torch reference within the NVFP4 MoE tolerance; the same reference without s_t
  does not (negative control).
The installed stock module ships no mode-Q FC1 kernels, so "stock mode Q" is the split-N
module's own single-stream path.
"""

import warnings

import pytest
import torch

from flashinfer.utils import get_compute_capability

if not torch.cuda.is_available() or get_compute_capability(torch.device("cuda")) != (
    10,
    7,
):
    pytest.skip("localized trtllm-gen MoE is SM107 only", allow_module_level=True)

from flashinfer import nvfp4_quantize  # noqa: E402
from flashinfer.artifacts import has_trtllm_gen_bmm_split_n  # noqa: E402
from flashinfer.fused_moe.trtllm_localized import moe as M  # noqa: E402
from flashinfer.tllm_enums import SfLayout  # noqa: E402

from . import trtllm_gen_localized_moe_utils as U  # noqa: E402

H, I, TOPK, E = U.H, U.I, U.TOPK, U.E
DEV = torch.device("cuda", torch.cuda.current_device())
FC2_INPUT_SCALE = 16.0  # the M3.1 / MegaMoE static FC2 decode scale is 1/16

try:
    RES = U.make_locality_resources(DEV)
except Exception as e:  # noqa: BLE001
    pytest.skip(f"no locality resources from PyTorch: {e}", allow_module_level=True)
if not has_trtllm_gen_bmm_split_n():
    pytest.skip("split-N BMM package unavailable", allow_module_level=True)
# A build, pin or ABI failure must fail the suite, not skip it.
MOD = M.get_module()


def _pairs(tile, per_token):
    try:
        names = [
            str(n)
            for n in MOD.trtllm_fp4_moe_localized_config_kernels(
                tile, U.SWIGLU, per_token
            )
        ]
    except RuntimeError:  # no kernel of this tile in this mode
        return []
    return list(zip(names[0::2], names[1::2], strict=True))


HAS_MODE_Q = any(_pairs(t, True) for t in M._TILES)
needs_mode_q = pytest.mark.skipif(
    not HAS_MODE_Q, reason="split-N package has no per-token FC1 kernels"
)


@pytest.fixture(scope="module")
def weights():
    full, logical, gs1, gs2 = U.build_weights(E, DEV, seed=11)
    return full, logical, gs1, gs2, U.make_shards(logical, RES)


def _inputs(T, seed, spread=2.0):
    """bf16 rows with very different magnitudes (so s_t matters), quantized per token."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows = torch.exp(spread * torch.randn(T, 1, generator=g, device=DEV))
    x = (torch.randn(T, H, generator=g, device=DEV) * rows).bfloat16()
    xq, xs, st = nvfp4_quantize(
        x,
        1.0 / U.FP4_MAX,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        nvfp4_4over6=None,
    )
    ids, w = U.routes(T, DEV, seed + 1, "none")
    return (
        xq.view(torch.uint8).reshape(T, H // 2),
        xs.view(torch.float8_e4m3fn).reshape(T, H // 16),
        st.float().contiguous(),
        ids,
        w.to(torch.bfloat16),
    )


def _kwargs(xq, xs, gs1, gs2, fin):
    # No activation global scale in mode Q: s_t carries it, so hs_gs = 1.
    sc = U.scalars(gs1, gs2, torch.ones(1, device=DEV), FC2_INPUT_SCALE)
    kw = U.routed_kwargs(xq, xs, sc, fin)
    kw["hidden_states_scale_layout"] = SfLayout.layout_linear
    return kw


def _run(ids, w, kw, shards, tactic, mode, st, T, fin):
    output = torch.empty(T, H, dtype=torch.bfloat16, device=DEV) if fin else None
    r = M.trtllm_fp4_block_scale_moe_localized(
        routing_input_mode=2,
        routing_logits=None,
        topk_ids=ids,
        topk_weights=w,
        **kw,
        **RES.localized_kwargs(shards),
        output=output,
        tactic=tactic,
        _mode=mode,
        _return_intermediates=True,
        per_token_scale=st,
        per_token_scale_gemm2=st is None,
    )
    torch.cuda.synchronize()
    return r


def _assert_same(ru, rs, fin, tile):
    if fin:
        assert U.bits_equal(ru[0], rs[0])
        n = 1
    else:
        assert U.bits_equal(
            ru[0][ru[2].reshape(-1).long()], rs[0][rs[2].reshape(-1).long()]
        )
        n = 3
    mc = U.masked_compare(ru[n : n + 4], rs[n : n + 4], tile)
    assert mc["valid_mask_equal"] and mc["valid_rows"] > 0
    assert mc["gemm1_valid_rows_bitexact"] and mc["gemm1_scale_valid_rows_bitexact"]
    assert mc["gemm2_valid_rows_bitexact"]


def _tactic_q(T):
    return M.default_tactic(T, TOPK, H, I, E, RES.sm_counts, per_token=True)


# ---- the runner filter: per-token kernels only when asked for ---------------------------------
def test_m3_catalog_has_no_per_token_kernels():
    tiles = [t for t in M._TILES if _pairs(t, False)]
    assert tiles
    for tile in tiles:
        m3 = _pairs(tile, False)
        assert not [p for p in m3 if "tok" in p[0] or "tok" in p[1]], tile
        q = _pairs(tile, True)
        assert all("_Bb16tokFp32_" in a and "tok" not in b for a, b in q), tile
        # Every per-token FC1 is the twin of a plain FC1 of the same tile.
        assert {a for a, _ in q} <= {M.per_token_fc1_name(a) for a, _ in m3}, tile


@needs_mode_q
def test_per_token_pins_resolve_by_name():
    pins = M._per_token_pins()
    assert set(pins) == {p.label for p in M.PER_TOKEN_TACTIC_PINS}, pins
    for pin in M.PER_TOKEN_TACTIC_PINS:
        tile, cfg = pins[pin.label]
        assert _pairs(tile, True)[cfg] == (pin.fc1, pin.fc2)
    # Measured on the v5 package (checksums.txt 092fe73e...); a move means the filtering changed.
    measured = {
        "decode": (8, 67),
        "small": (16, 71),
        "mid": (32, 139),
        "prefill": (128, 24),
    }
    if pins != measured:
        warnings.warn(
            f"per-token pin config indices moved: {pins} vs measured {measured}",
            stacklevel=1,
        )


# ---- red-first wiring: s_t = 1 makes a per-token FC1 equal its plain twin -----------------------
@needs_mode_q
@pytest.mark.parametrize("T", [4, 257, 2048])
def test_unit_token_scale_equals_the_plain_twin(weights, T):
    """The same quantized input through M3 (plain FC1) and mode Q with s_t = 1 (tok FC1, same
    FC2) must agree bit for bit; a mis-bound token-scale pointer or a wrong kernel breaks it."""
    full, _, gs1, gs2, shards = weights
    xq, xs, _, ids, w = _inputs(T, 4000 + T)
    kw = _kwargs(xq, xs, gs1, gs2, True)
    tq = _tactic_q(T)
    fc1_tok, fc2 = _pairs(tq[0], True)[tq[1]]
    twin = _pairs(tq[0], False).index((fc1_tok.replace("tokFp32", "", 1), fc2))
    ones = torch.ones(T, dtype=torch.float32, device=DEV)
    rq = _run(ids, w, kw, shards, tq, 0, ones, T, True)
    rm3 = _run(ids, w, kw, shards, (tq[0], twin), 0, None, T, True)
    assert U.bits_equal(rq[0], rm3[0])
    # And s_t is really read: scaling it changes the result.
    rq2 = _run(ids, w, kw, shards, tq, 0, ones * 2, T, True)
    assert not U.bits_equal(rq2[0], rq[0])


# ---- localized mode Q == stock mode Q -----------------------------------------------------------
@needs_mode_q
@pytest.mark.parametrize("T", [1, 4, 64, 257, 1024, 4096, 16384])
@pytest.mark.parametrize("fin", [True, False])
def test_localized_equals_stock_mode_q(weights, T, fin):
    full, _, gs1, gs2, shards = weights
    xq, xs, st, ids, w = _inputs(T, 5000 + T)
    kw = _kwargs(xq, xs, gs1, gs2, fin)
    tactic = _tactic_q(T)
    ru = _run(ids, w, kw, shards, tactic, 0, st, T, fin)
    rs = _run(ids, w, kw, U.full_shards(full), tactic, 1, st, T, fin)
    _assert_same(ru, rs, fin, tactic[0])


# ---- torch reference ----------------------------------------------------------------------------
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequant(q, sf, rows, cols):
    """NVFP4 (low nibble first, E4M3 block scales per 16) to float, without a global scale."""
    q = q.reshape(rows, cols // 2)
    nib = torch.stack([q & 0xF, q >> 4], dim=-1).reshape(rows, cols).long()
    grid = torch.tensor(_E2M1, device=q.device)
    val = grid[nib & 7] * torch.where(nib >= 8, -1.0, 1.0)
    s = sf.reshape(rows, cols // 16).view(torch.float8_e4m3fn).float()
    return val * s.repeat_interleave(16, dim=1)


def _fake_quant(x, gs):
    """Round-trip through NVFP4 with global scale gs (E4M3 block scale amax/6 per 16)."""
    n, k = x.shape
    xb = x.reshape(n, k // 16, 16) * gs
    sf = (xb.abs().amax(-1, keepdim=True) / 6.0).to(torch.float8_e4m3fn).float()
    scaled = torch.where(sf > 0, xb / sf.clamp(min=1e-30), torch.zeros_like(xb))
    grid = torch.tensor(_E2M1, device=x.device)
    idx = (scaled.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    q = grid[idx] * torch.sign(scaled)
    return (q * sf / gs).reshape(n, k)


def _reference(xq, xs, st, ids, w, logical, gs1, gs2):
    T = xq.shape[0]
    x = _dequant(xq, xs.view(torch.uint8), T, H) * st.view(-1, 1)
    out = torch.zeros(T, H, device=DEV)
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        q1, s1, q2, s2 = logical[e]
        h = x[tok] @ (_dequant(q1, s1, 2 * I, H) / gs1[e]).T
        up = h[:, :I].clamp(-U.SWIGLU_LIMIT, U.SWIGLU_LIMIT)
        gate = h[:, I:].clamp(max=U.SWIGLU_LIMIT)
        a = gate * torch.sigmoid(U.SWIGLU_ALPHA * gate) * (up + U.SWIGLU_BETA)
        a = _fake_quant(a, FC2_INPUT_SCALE)
        y = a @ (_dequant(q2, s2, H, I) / gs2[e]).T
        out.index_add_(0, tok, y * w[tok, slot].float().view(-1, 1))
    return out


@needs_mode_q
@pytest.mark.parametrize("T", [4, 1024])
def test_mode_q_matches_torch_reference(weights, T):
    full, logical, gs1, gs2, shards = weights
    xq, xs, st, ids, w = _inputs(T, 6000 + T)
    kw = _kwargs(xq, xs, gs1, gs2, True)
    tactic = _tactic_q(T)
    loc = _run(ids, w, kw, shards, tactic, 0, st, T, True)[0]
    stock = _run(ids, w, kw, U.full_shards(full), tactic, 1, st, T, True)[0]
    ref = _reference(xq, xs, st, ids, w, logical, gs1, gs2)
    assert ref.abs().sum() > 0
    for got in (loc, stock):
        ok, pct, _ = U.check_accuracy(got, ref)
        assert ok, f"only {pct:.4f} within the NVFP4 MoE tolerance"
    # Negative control: dropping s_t from the reference must be detectable.
    wrong = _reference(xq, xs, torch.ones_like(st), ids, w, logical, gs1, gs2)
    ok, pct, _ = U.check_accuracy(loc, wrong)
    assert not ok, f"tolerance cannot see a missing token scale ({pct:.4f})"
