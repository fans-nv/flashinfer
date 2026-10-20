"""Host-only tests (CPU, no GPU, no native module) for the localized trtllm-gen NVFP4 MoE:
the localized_* arguments of the stock FP4 MoE ops, AutoTuner integration, pinned default
tactics, split-N package staging, module identity and admission.

For tuning, the native module is replaced by a fake that records the launched (tile, config)
and the weight tensors it received; profiling is replaced by a deterministic latency table, so
the real AutoTuner search / cache / bucketing code runs unchanged.
"""

import collections
import hashlib
import inspect
import types

import pytest
import torch

from flashinfer import artifacts as A
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import core as C
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.fused_moe.trtllm_localized import moe as M
from flashinfer.fused_moe.trtllm_localized import tuning as TU
from flashinfer.tllm_enums import SfLayout

H, I, E, TOPK = 6144, 1536, 128, 4
# Partition-valid configs per tile. (8, 2) is what the first-valid rule picks at decode sizes
# (the old fallback); (8, 52), (16, 55), (32, 27) and (128, 28) are the pinned defaults.
VALID = {8: [2, 3, 52], 16: [0, 55], 32: [0, 1, 27], 64: [1], 128: [0, 1, 28]}
INFO = M.MODULE_INFO
FAST = {(32, 1): 1.0}  # fake latency: (32, 1) wins, everything else 5.0
PINS = {p.label: p for p in M.DEFAULT_TACTIC_PINS}
DECODE_PIN, SMALL_PIN, MID_PIN, PREFILL_PIN = (8, 52), (16, 55), (32, 27), (128, 28)
# The fake per-token catalog holds the mode-Q pins at other (partition-valid) indices, so a
# launch tells which table resolved it.
PT_PINS = {"decode": (8, 3), "small": (16, 0), "mid": (32, 1), "prefill": (128, 1)}


def _catalog(tile, n=64, per_token=False):
    """[fc1(0), fc2(0), fc1(1), fc2(1), ...] with the pinned pairs at their measured indices; with
    per_token, the catalog a runner with FC1 per-token scaling sees (tok FC1 names, pins at
    PT_PINS)."""
    if per_token:
        pairs = [(f"fc1tok_t{tile}_c{i}", f"fc2_t{tile}_c{i}") for i in range(n)]
        for pin in M.PER_TOKEN_TACTIC_PINS:
            if pin.tile == tile:
                pairs[PT_PINS[pin.label][1]] = (pin.fc1, pin.fc2)
        return [name for pair in pairs for name in pair]
    pairs = [(f"fc1_t{tile}_c{i}", f"fc2_t{tile}_c{i}") for i in range(n)]
    if tile == 8:
        pairs[2] = ("fc1_schedS", "fc2_t8")
        pairs[52] = (PINS["decode"].fc1, PINS["decode"].fc2)
    if tile == 16:
        pairs[55] = (PINS["small"].fc1, PINS["small"].fc2)
    if tile == 32:
        pairs[27] = (PINS["mid"].fc1, PINS["mid"].fc2)
    if tile == 128:
        pairs[28] = (PINS["prefill"].fc1, PINS["prefill"].fc2)
    return [name for pair in pairs for name in pair]


class FakeModule:
    def __init__(self, catalog=_catalog, per_token_catalog=None):
        self.launches = []
        self.catalog = catalog
        self.per_token_catalog = per_token_catalog or (
            lambda tile: _catalog(tile, per_token=True)
        )
        self.valid_queries = []

    def trtllm_fp4_moe_localized_module_info(self):
        return list(INFO)

    def trtllm_fp4_moe_localized_valid_configs(
        self, tile, top_k, h, i, e, t, sm0, sm1, act, per_token
    ):
        self.valid_queries.append(per_token)
        if tile not in VALID:
            raise RuntimeError("no kernel of this tile")
        return VALID[tile] + [-1]

    def trtllm_fp4_moe_localized_config_kernels(self, tile, act, per_token):
        if tile not in VALID:
            raise RuntimeError("no kernel of this tile")
        return self.per_token_catalog(tile) if per_token else self.catalog(tile)

    def trtllm_fp4_block_scale_moe_localized(self, *args):
        self.launches.append(
            dict(
                tile=args[32],
                config=args[33],
                w=[args[i] for i in (7, 8, 13, 14, 35, 36, 37, 38)],
                n=args[5].shape[0],
                streams=args[39:41],
                sms=args[41:43],
                delay=args[45:48],
                per_token_scale=args[48],
                args=args,
            )
        )
        return [torch.empty(0), torch.empty(0), torch.empty(0)]


def fake_stream(handle, device="cpu"):
    return types.SimpleNamespace(cuda_stream=handle, device=torch.device(device))


STREAMS = [fake_stream(11), fake_stream(22)]


@pytest.fixture
def env(monkeypatch):
    fake = FakeModule()
    monkeypatch.setattr(M, "get_module", lambda: fake)
    monkeypatch.setattr(M, "module_info", lambda: INFO)
    monkeypatch.setattr(M, "_resolved_pins", M.resolve_pins(fake))
    monkeypatch.setattr(M, "_resolved_per_token_pins", None)
    M._valid_configs.cache_clear()
    M.catalog_fingerprint.cache_clear()
    M._tile_probe.clear()
    TU._runners.clear()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    TU.last_resolution.clear()

    def fake_profile(self, runner, inputs, tactic, tuning_config, **kw):
        return FAST.get(tuple(tactic), 5.0) if tactic != -1 else 9.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    # Host tests use CPU tensors whether or not a GPU is visible: never ask about stream capture,
    # and skip the cold-L2 profile arena (it requires CUDA inputs; timing is faked anyway).
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        AutoTuner,
        "_prepare_input_tensors_with_batches",
        lambda self, inputs, cfg: [inputs],
    )
    yield fake
    tuner.clear_cache()
    M._valid_configs.cache_clear()
    M.catalog_fingerprint.cache_clear()
    M._tile_probe.clear()
    TU._runners.clear()


def shards(e=E, i=I, h=H):
    w = lambda *s: torch.zeros(*s, dtype=torch.uint8)  # noqa: E731
    return [
        dict(
            gemm1_weights=w(e, i, h // 2),
            gemm1_weights_scale=w(e, i, h // 16),
            gemm2_weights=w(e, h // 2, i // 2),
            gemm2_weights_scale=w(e, h // 2, i // 16),
        )
        for _ in range(2)
    ]


def placeholders():
    """Storage-released full-width weights: the localized path must not read them."""
    t = torch.empty(0, dtype=torch.uint8)
    return t, t, t, t


def loc_kwargs(sh, **kw):
    return dict(
        localized_weights=sh, localized_streams=STREAMS, localized_sm_count=100, **kw
    )


def call(sh, T, packed=None, **kw):
    if packed is None:
        packed = torch.zeros(T, TOPK, dtype=torch.int32)
    hs = torch.zeros(T, H // 2, dtype=torch.uint8)
    hss = torch.zeros(T, H // 16, dtype=torch.uint8)
    one = torch.ones(E)
    w1, w1s, w2, w2s = placeholders()
    opts = dict(
        routing_method_type=1,
        do_finalize=True,
        enable_pdl=True,
        tune_max_num_tokens=16,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    opts.update(loc_kwargs(sh))
    opts.update(kw)
    return trtllm_fp4_block_scale_routed_moe(
        packed,
        None,
        hs,
        hss,
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        0,
        0,
        I,
        0,
        E,
        None,
        **opts,
    )


def call_internal(sh, T, **kw):
    """The localized body with its private hooks (explicit tactic, debug delay)."""
    one = torch.ones(E)
    base = dict(
        routing_input_mode=1,
        routing_logits=None,
        topk_ids=torch.zeros(T, TOPK, dtype=torch.int32),
        topk_weights=None,
        routing_bias=None,
        hidden_states=torch.zeros(T, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(T, H // 16, dtype=torch.uint8),
        gemm1_bias=None,
        gemm1_alpha=one,
        gemm1_beta=one,
        gemm1_clamp_limit=one,
        gemm2_bias=None,
        output1_scale_scalar=one,
        output1_scale_gate_scalar=one,
        output2_scale_scalar=one,
        num_experts=E,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        localized_weights=sh,
        localized_streams=STREAMS,
        localized_sm_count=100,
        routing_method_type=1,
        enable_pdl=True,
    )
    base.update(kw)
    return M.trtllm_fp4_block_scale_moe_localized(**base)


def launched(fake):
    return fake.launches[-1]["tile"], fake.launches[-1]["config"]


def runner(**kw):
    base = dict(
        routing_input_mode=1,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        num_local_experts=E,
        top_k=TOPK,
        activation_type=3,
        do_finalize=True,
        enable_pdl=True,
        sm_counts=(100, 100),
        signature=M.localization_signature(shards(), [100, 100], None),
    )
    base.update(kw)
    return TU.LocalizedFp4MoeRunner(**base)


def test_valid_set_is_partition_valid_set(env):
    assert runner().valid_tactics(4) == [
        (t, c) for t in (8, 16, 32, 64, 128) for c in VALID[t]
    ]


# ---- default tactic without a tuning record (autotune off) ---------------------------------
@pytest.mark.parametrize(
    "T,expect",
    [
        (1, DECODE_PIN),
        (4, DECODE_PIN),
        (16, DECODE_PIN),
        (48, DECODE_PIN),
        (49, SMALL_PIN),
        (256, SMALL_PIN),
        (257, MID_PIN),
        (1024, MID_PIN),
        (1536, MID_PIN),
        (1537, PREFILL_PIN),
        (4095, PREFILL_PIN),
        (4096, PREFILL_PIN),
        (16384, PREFILL_PIN),
    ],
)
def test_autotune_off_uses_pinned_default_not_first_valid(env, T, expect):
    """Red-first against the old fallback: it launched the first valid config, (8, 2) at
    decode sizes and (128, 0) at prefill sizes."""
    call(shards(), T)
    assert launched(env) == expect
    assert launched(env) != M.choose_tactic(T, TOPK, H, I, E, (100, 100))
    assert TU.last_resolution[(TU.custom_op(1, False), T)] == ("pinned", expect)


@pytest.mark.parametrize("T", [49, 257, 1537])
def test_pin_not_valid_keeps_first_valid(env, monkeypatch, T):
    for tile in (16, 32, 128):
        monkeypatch.setitem(VALID, tile, VALID[tile][:-1])
    M._valid_configs.cache_clear()
    call(shards(), T)
    assert launched(env) == M.choose_tactic(T, TOPK, H, I, E, (100, 100))


def test_pins_cover_every_swiglu_token_count(env):
    for T in range(1, 20000):
        assert M.pinned_default(T, act_type=3) is not None, T


def test_pin_not_valid_for_shape_falls_back_with_warning(env, monkeypatch, caplog):
    monkeypatch.setitem(VALID, 8, [2, 3])  # the decode pin is not partition-valid here
    M._valid_configs.cache_clear()
    with caplog.at_level("WARNING", logger="flashinfer.fused_moe.trtllm_localized"):
        call(shards(), 4)
    assert launched(env) == (8, 2)
    assert any("not partition-valid" in r.message for r in caplog.records)


def test_pins_apply_to_swiglu_only(env):
    assert M.pinned_default(4, act_type=3) == ("decode", DECODE_PIN)
    assert (
        M.pinned_default(4, act_type=4) is None
    )  # Geglu: the pins were measured for SwiGLU


def test_tuning_picks_fastest_valid_and_is_cached(env, monkeypatch):
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    sh = shards()
    with autotune(True):
        call(sh, 4)
    # Every tactic the tuner launched was partition-valid; profiling saw the real shards.
    valid = {(t, c) for t in VALID for c in VALID[t]}
    expect_w = [sh[0][k] for k in M.SHARD_KEYS] + [sh[1][k] for k in M.SHARD_KEYS]
    for L in env.launches:
        assert (L["tile"], L["config"]) in valid
        assert all(a is b for a, b in zip(L["w"], expect_w, strict=True))
        assert L["streams"] == (11, 22) and L["sms"] == (100, 100)
    env.launches.clear()
    for T in (4, 3):  # outside autotune: cache hit, same stock bucketing
        call(sh, T)
        assert launched(env) == (32, 1)
        assert TU.last_resolution[(TU.custom_op(1, False), T)] == ("tuned", (32, 1))


def test_red_first_invalid_tactic_is_rejected(env, monkeypatch):
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    sh = shards()
    bad = (16, 7)
    FAST[bad] = 0.1  # the invalid tactic would win the profile
    orig = TU.LocalizedFp4MoeRunner.get_valid_tactics
    monkeypatch.setattr(
        TU.LocalizedFp4MoeRunner,
        "get_valid_tactics",
        lambda self, inputs, profile, **kw: orig(self, inputs, profile) + [bad],
    )
    try:
        # Red: with the guard also fooled, the invalid tactic reaches the launch.
        orig_valid = TU.LocalizedFp4MoeRunner.is_valid_tactic
        monkeypatch.setattr(
            TU.LocalizedFp4MoeRunner,
            "is_valid_tactic",
            lambda self, t, n: t == bad or orig_valid(self, t, n),
        )
        with autotune(True):
            call(sh, 4)
        call(sh, 4)
        assert launched(env) == bad
        # Green: the real guard rejects the cached invalid tactic and uses the default.
        monkeypatch.setattr(TU.LocalizedFp4MoeRunner, "is_valid_tactic", orig_valid)
        env.launches.clear()
        call(sh, 4)
        assert launched(env) == DECODE_PIN
        assert TU.last_resolution[(TU.custom_op(1, False), 4)][0] == "rejected"
    finally:
        FAST.pop(bad, None)


@pytest.mark.parametrize(
    "T,pin",
    [
        (4, DECODE_PIN),
        (48, DECODE_PIN),
        (128, SMALL_PIN),
        (1024, MID_PIN),
        (2048, PREFILL_PIN),
    ],
)
def test_pins_beat_tuned_records(env, monkeypatch, T, pin):
    """The isolated profile picked a slower decode FC1 in serving; pins measured in-chain win."""
    sh = shards()
    with autotune(True):
        call(sh, T)
    env.launches.clear()
    call(sh, T)
    assert launched(env) == pin
    assert TU.last_resolution[(TU.custom_op(1, False), T)] == ("pinned", pin)
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    with autotune(True):
        call(sh, T)
    env.launches.clear()
    call(sh, T)
    assert TU.last_resolution[(TU.custom_op(1, False), T)][0] == "tuned"
    assert launched(env) != pin


def test_explicit_invalid_tactic_raises_before_launch(env):
    with pytest.raises(ValueError, match="partition-valid"):
        call_internal(shards(), 4, tactic=(8, 0))
    assert not env.launches


def test_cache_identity_distinct_from_stock_and_localization(env):
    r0 = runner()
    variants = [
        runner(do_finalize=False),
        runner(enable_pdl=False),
        runner(routing_input_mode=2),
        runner(signature=M.localization_signature(shards(), [98, 100], None)),
        runner(signature=M.localization_signature(shards(i=768), [100, 100], None)),
        runner(signature=M.localization_signature(shards(), [100, 100], object())),
    ]
    ids = {r0.identity()} | {v.identity() for v in variants}
    assert len(ids) == 1 + len(variants)
    assert r0.identity()[-2] == INFO  # the module's split-N package is part of the key
    assert r0.identity()[-1] == M.catalog_fingerprint(3)
    assert r0.signature[0] == "localized"
    # Op names follow #5276: the stock op name plus the localization policy.
    assert r0.op == "flashinfer::trtllm_fp4_block_scale_routed_moe::Localized"
    assert runner(allow_nonlocalized=True).op.endswith("::AdaptiveLocalized")
    assert TU.custom_op(0, False) == "flashinfer::trtllm_fp4_block_scale_moe::Localized"
    cfg = r0.tuning_config(
        [
            None,
            None,
            torch.zeros(4, TOPK, dtype=torch.int32),
            None,
            torch.zeros(4, H // 2),
            torch.zeros(4, H // 16),
        ],
        8192,
    )
    assert cfg.use_cold_l2_cache and cfg.use_cuda_graph  # the stock FP4 MoE protocol


def test_cache_identity_follows_the_config_index_map(env, monkeypatch):
    """Same package, different (tile, index) -> kernel map (e.g. runner filtering drift):
    tuning records must not carry over."""
    before = runner().identity()
    shifted = FakeModule(lambda tile: _catalog(tile)[2:] + _catalog(tile)[:2])
    monkeypatch.setattr(M, "get_module", lambda: shifted)
    M.catalog_fingerprint.cache_clear()
    after = runner().identity()
    assert before[:-1] == after[:-1] and before[-1] != after[-1]


def test_debug_delay_is_off_by_default_and_forwarded(env):
    call_internal(shards(), 4, tactic=(8, 3))
    assert tuple(env.launches[-1]["delay"]) == (-1, 0, 0)
    call_internal(shards(), 4, tactic=(8, 3), _debug_delay=(1, 2, 5))
    assert tuple(env.launches[-1]["delay"]) == (1, 2, 5)


def test_runner_and_identity_are_reused_across_calls(env, monkeypatch):
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    sh = shards()
    call(sh, 4)
    call(sh, 5)
    assert len(TU._runners) == 1
    (r,) = TU._runners.values()
    assert r._identity is not None and len(r._tuning_configs) == 1


def test_streams_and_sm_counts_reach_the_launcher(env):
    call(shards(), 4, localized_sm_count=[96, 100])
    assert env.launches[-1]["streams"] == (11, 22)
    assert env.launches[-1]["sms"] == (96, 100)
    # The memset stream is accepted and unused; allow_nonlocalized only renames the op.
    call(
        shards(),
        4,
        localized_memset_stream=fake_stream(33),
        localized_allow_nonlocalized=True,
        autotune_use_cuda_graph=True,
    )
    assert env.launches[-1]["streams"] == (11, 22)
    assert (TU.custom_op(1, True), 4) in TU.last_resolution


# ---- admission (checked before any module or GPU work) -------------------------------------
def test_admission_shape_bias_and_unsupported_features(env):
    sh = shards()
    packed = torch.zeros(4, TOPK, dtype=torch.int32)
    one = torch.ones(E)
    w1, w1s, w2, w2s = placeholders()

    def op(H_=H, I_=I, b1=None, ids=packed, **kw):
        return trtllm_fp4_block_scale_routed_moe(
            ids,
            None,
            torch.zeros(4, H_ // 2, dtype=torch.uint8),
            torch.zeros(4, H_ // 16, dtype=torch.uint8),
            w1,
            w1s,
            b1,
            one,
            one,
            one,
            w2,
            w2s,
            None,
            one,
            one,
            one,
            E,
            TOPK,
            0,
            0,
            I_,
            0,
            E,
            None,
            **{**loc_kwargs(sh), **kw},
        )

    with pytest.raises(ValueError, match="multiples of 256"):
        op(H_=6144 + 128)
    with pytest.raises(ValueError, match="multiples of 256"):
        op(I_=1536 + 128)
    with pytest.raises(NotImplementedError):
        op(b1=one)
    with pytest.raises(NotImplementedError):
        op(per_token_scale=torch.ones(4))
    with pytest.raises(TypeError):
        op(ids=packed.to(torch.int64))
    with pytest.raises(ValueError, match="exactly two weight shards"):
        op(localized_weights=sh[:1])
    with pytest.raises(ValueError, match="stream per domain"):
        op(localized_streams=STREAMS[:1])
    with pytest.raises(ValueError, match="localized_sm_count"):
        op(localized_sm_count=None)
    with pytest.raises(ValueError, match="localized_sm_count"):
        op(localized_sm_count=[100, 0])
    with pytest.raises(ValueError, match="keys"):
        op(localized_weights=[{**sh[0], "w1_weight": sh[0]["gemm1_weights"]}, sh[1]])
    with pytest.raises(ValueError, match="localized_streams\\[1\\] is on"):
        op(localized_streams=[STREAMS[0], fake_stream(22, "cuda:7")])
    with pytest.raises(NotImplementedError, match="NVFP4"):
        trtllm_fp4_block_scale_routed_moe(
            packed,
            None,
            torch.zeros(4, H, dtype=torch.bfloat16),
            None,
            w1,
            w1s,
            None,
            one,
            one,
            one,
            w2,
            w2s,
            None,
            one,
            one,
            one,
            E,
            TOPK,
            0,
            0,
            I,
            0,
            E,
            None,
            **loc_kwargs(sh),
        )
    assert not env.launches


def test_admission_shard_device_mismatch(env):
    sh = shards()
    sh[1] = {k: v.to("meta") for k, v in sh[1].items()}
    with pytest.raises(ValueError, match="localized_weights\\[1\\]"):
        call(sh, 4)
    assert not env.launches


def test_monolithic_rejects_routing_replay_and_fused_shared(env):
    one = torch.ones(E)
    w1, w1s, w2, w2s = placeholders()
    args = (
        torch.zeros(4, E),
        None,
        torch.zeros(4, H // 2, dtype=torch.uint8),
        torch.zeros(4, H // 16, dtype=torch.uint8),
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        None,
        None,
        I,
        0,
        E,
        2.0,
    )
    with pytest.raises(NotImplementedError, match="routing replay"):
        trtllm_fp4_block_scale_moe(
            *args,
            routing_method_type=7,
            routing_replay_out=torch.empty(4, TOPK, dtype=torch.int16),
            **loc_kwargs(shards()),
        )
    with pytest.raises(NotImplementedError):
        trtllm_fp4_block_scale_moe(
            *args,
            routing_method_type=7,
            num_fused_shared_experts=1,
            **loc_kwargs(shards()),
        )
    assert not env.launches


def test_monolithic_minimax2_forwards_routing_and_scale(env):
    one = torch.ones(E)
    bias = torch.zeros(E)
    w1, w1s, w2, w2s = placeholders()
    trtllm_fp4_block_scale_moe(
        torch.zeros(4, E),
        bias,
        torch.zeros(4, H // 2, dtype=torch.uint8),
        torch.zeros(4, H // 16, dtype=torch.uint8),
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        None,
        None,
        I,
        0,
        E,
        2.0,
        routing_method_type=7,
        enable_pdl=True,
        **loc_kwargs(shards()),
    )
    a = env.launches[-1]["args"]
    # routing_input_mode 0 (from logits), bias passed through, scale 2.0, MiniMax2 = 7.
    assert a[0] == 0 and a[4] is bias and a[26] == 2.0 and a[27] == 7
    assert launched(env) == DECODE_PIN


def test_stock_args_accepted_only_in_noop_form(env):
    from flashinfer.tllm_enums import SfLayout

    sh = shards()
    with pytest.raises(NotImplementedError, match="valid"):
        call(sh, 4, valid_hidden_size=H - 256)
    with pytest.raises(NotImplementedError, match="valid"):
        call(sh, 4, valid_intermediate_size=I - 256)
    with pytest.raises(NotImplementedError, match="lora"):
        call(sh, 4, gemm1_lora_delta=torch.zeros(4, 2 * I))
    with pytest.raises(NotImplementedError):
        call(sh, 4, num_fused_shared_experts=1)
    with pytest.raises(NotImplementedError, match="hidden_states_scale_layout"):
        call(sh, 4, hidden_states_scale_layout=SfLayout.layout_128x4)
    assert not env.launches
    call(
        sh,
        4,
        valid_hidden_size=H,
        valid_intermediate_size=I,
        hidden_states_scale_layout=SfLayout.layout_linear,
    )
    assert len(env.launches) == 1


def test_routed_tuple_fp32_weights_is_unpacked_mode(env):
    """Frameworks pass (ids int32, weights fp32) tuples to the routed op."""
    one = torch.ones(E)
    ids = torch.zeros(4, TOPK, dtype=torch.int32)
    w = torch.ones(4, TOPK, dtype=torch.float32)
    w1, w1s, w2, w2s = placeholders()
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, w),
        routing_bias=None,
        hidden_states=torch.zeros(4, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(4, H // 16, dtype=torch.uint8),
        gemm1_weights=w1,
        gemm1_weights_scale=w1s,
        gemm1_bias=None,
        gemm1_alpha=one,
        gemm1_beta=one,
        gemm1_clamp_limit=one,
        gemm2_weights=w2,
        gemm2_weights_scale=w2s,
        gemm2_bias=None,
        output1_scale_scalar=one,
        output1_scale_gate_scalar=one,
        output2_scale_scalar=one,
        num_experts=E,
        top_k=TOPK,
        n_group=0,
        topk_group=0,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=1,
        do_finalize=True,
        output=torch.empty(4, H, dtype=torch.bfloat16),
        tune_max_num_tokens=16,
        **loc_kwargs(shards()),
    )
    a = env.launches[-1]["args"]
    assert (
        a[0] == 2 and a[2] is ids and a[3] is w
    )  # UnpackedPrecomputed, weights not cast


def test_pdl_follows_the_stock_default(env):
    call(shards(), 4)
    assert env.launches[-1]["args"][29] is True
    call(shards(), 4, enable_pdl=False)
    assert env.launches[-1]["args"][29] is False
    call(shards(), 4, enable_pdl=None)  # device_support_pdl: False off-GPU
    assert env.launches[-1]["args"][29] is False


@pytest.mark.parametrize(
    "cc, expected", [((10, 7), False), ((10, 0), True), ((9, 0), True), ((8, 0), False)]
)
def test_default_pdl_honours_the_sm107_moe_gate(monkeypatch, cc, expected):
    from flashinfer.fused_moe import core

    core._device_support_moe_pdl.cache_clear()
    monkeypatch.setattr(core, "get_compute_capability", lambda d: cc)
    monkeypatch.setattr(core, "device_support_pdl", lambda d: cc[0] >= 9)
    try:
        assert M.default_enable_pdl(torch.device("cuda", 0)) is expected
        assert M.default_enable_pdl(torch.device("cpu")) is False
    finally:
        core._device_support_moe_pdl.cache_clear()


# ---- the stock path is untouched -------------------------------------------------------------
class _StockRecorder:
    def __init__(self):
        self.calls = []

    def trtllm_fp4_block_scale_moe(self, *args):
        self.calls.append(args)
        return ["stock"]


def test_new_arguments_are_keyword_only_and_default_off():
    for fn in (trtllm_fp4_block_scale_moe, trtllm_fp4_block_scale_routed_moe):
        params = inspect.signature(inspect.unwrap(fn)).parameters
        expect = dict(
            localized_weights=None,
            localized_streams=None,
            localized_sm_count=None,
            localized_memset_stream=None,
            localized_allow_nonlocalized=False,
            autotune_use_cuda_graph=False,
        )
        assert params["per_token_scale_gemm2"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["per_token_scale_gemm2"].default is True
        for name, default in expect.items():
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert params[name].default is default
        # Every pre-existing parameter stays positional, in its old place.
        assert list(params)[-len(expect) :] == list(expect)


def test_stock_call_without_localized_args_is_unchanged(monkeypatch):
    rec = _StockRecorder()
    monkeypatch.setattr(C, "get_trtllm_moe_sm100_module", lambda: rec)

    def no_localized(*a, **k):
        raise AssertionError("the localized module was touched by a stock call")

    monkeypatch.setattr(M, "get_module", no_localized)
    T = 4
    one = torch.ones(E)
    logits = torch.zeros(T, E)
    hs = torch.zeros(T, H // 2, dtype=torch.uint8)
    hss = torch.zeros(T, H // 16, dtype=torch.uint8)
    w1 = torch.zeros(E, 2 * I, H // 2, dtype=torch.uint8)
    w1s = torch.zeros(E, 2 * I, H // 16, dtype=torch.uint8)
    w2 = torch.zeros(E, H, I // 2, dtype=torch.uint8)
    w2s = torch.zeros(E, H, I // 16, dtype=torch.uint8)
    assert trtllm_fp4_block_scale_moe(
        logits,
        None,
        hs,
        hss,
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        None,
        None,
        I,
        0,
        E,
        None,
        routing_method_type=1,
    ) == ["stock"]
    expect_mono = (
        C.RoutingInputMode.FromLogits,
        logits,
        None,
        None,
        None,
        hs,
        hss,
        w1,
        w1s,
        None,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        None,
        E,
        TOPK,
        0,
        None,
        None,
        I,
        0,
        E,
        None,
        1,
        True,
        None,
        3,
        None,
        8192,
        True,
        None,
        None,
        None,
        None,
        True,
    )
    assert len(rec.calls[-1]) == len(expect_mono)
    assert all(
        a is b or a == b for a, b in zip(rec.calls[-1], expect_mono, strict=True)
    )
    packed = torch.zeros(T, TOPK, dtype=torch.int32)
    assert trtllm_fp4_block_scale_routed_moe(
        packed,
        None,
        hs,
        hss,
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        0,
        0,
        I,
        0,
        E,
        None,
        routing_method_type=1,
    ) == ["stock"]
    expect_routed = (
        C.RoutingInputMode.PackedPrecomputed,
        None,
        packed,
        None,
        None,
        hs,
        hss,
        w1,
        w1s,
        None,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        None,
        E,
        TOPK,
        0,
        0,
        0,
        I,
        0,
        E,
        None,
        1,
        True,
        None,
        3,
        None,
        8192,
        True,
        None,
        None,
        None,
        None,
        True,
    )
    assert len(rec.calls[-1]) == len(expect_routed)
    assert all(
        a is b or a == b for a, b in zip(rec.calls[-1], expect_routed, strict=True)
    )


# ---- shard layout --------------------------------------------------------------------------------
def test_slice_then_shuffle_equals_row_slice_of_stock_shuffle():
    """Shards cut from the logical [up | gate] weights and shuffled on their own hold the same
    rows as the old contract (row slices of the stock-shuffled tensor), for weights and the
    pre-interleave scale rows."""
    cache = {}
    g = torch.Generator().manual_seed(0)
    i_, h_ = 256, 128
    w13 = torch.randint(0, 255, (2 * i_, h_ // 2), generator=g, dtype=torch.uint8)
    w13_sf = torch.randint(0, 255, (2 * i_, h_ // 16), generator=g, dtype=torch.uint8)
    w2 = torch.randint(0, 255, (h_, i_ // 2), generator=g, dtype=torch.uint8)

    def shuffle13(w, sf):
        kw = dict(num_elts_per_sf=16) if sf else {}
        return w[C._maybe_get_cached_w3_w1_permute_indices(cache, w, 128, **kw)]

    def shuffle2(w):
        return w[C.get_w2_permute_indices_with_cache(cache, w, 128)]

    for p in range(2):
        rows = slice(p * i_ // 2, (p + 1) * i_ // 2)
        for full, sf in ((w13, False), (w13_sf, True)):
            up, gate = full[:i_], full[i_:]
            shard = shuffle13(torch.cat([up[rows], gate[rows]]), sf)
            assert torch.equal(shard, shuffle13(full, sf)[p * i_ : (p + 1) * i_])
        hrows = slice(p * h_ // 2, (p + 1) * h_ // 2)
        assert torch.equal(shuffle2(w2[hrows]), shuffle2(w2)[hrows])


# ---- pinned default resolution at module load ------------------------------------------------
def test_resolve_pins_finds_the_measured_indices():
    assert M.resolve_pins(FakeModule()) == {
        "decode": DECODE_PIN,
        "small": SMALL_PIN,
        "mid": MID_PIN,
        "prefill": PREFILL_PIN,
    }


def _load_with(monkeypatch, mod):
    from flashinfer.jit import cubin_loader

    monkeypatch.setattr(M, "_module", None)
    monkeypatch.setattr(M, "_module_paths", ())
    monkeypatch.setattr(M, "_resolved_pins", {})
    monkeypatch.setattr(M, "_resolved_per_token_pins", None)
    monkeypatch.setattr(M, "_load_native", lambda: (mod, ("x.so",)))
    monkeypatch.setattr(cubin_loader, "setup_cubin_loader", lambda p: None)
    return M.get_module()


def test_get_module_resolves_pins(monkeypatch):
    fake = FakeModule()
    assert _load_with(monkeypatch, fake) is fake
    assert M._resolved_pins == {
        "decode": DECODE_PIN,
        "small": SMALL_PIN,
        "mid": MID_PIN,
        "prefill": PREFILL_PIN,
    }
    assert M.loaded_module_paths() == ("x.so",)


@pytest.mark.parametrize("label", ["decode", "small", "mid", "prefill"])
def test_missing_pinned_name_fails_the_load(monkeypatch, label):
    pin = PINS[label]

    def renamed(tile):
        names = _catalog(tile)
        return [n + "_x" if n == pin.fc2 else n for n in names]

    with pytest.raises(RuntimeError, match=f"pinned default '{label}'.*not in"):
        _load_with(monkeypatch, FakeModule(renamed))
    assert M._module is None and M._resolved_pins == {}


def test_pinned_tile_without_kernels_fails_the_load(monkeypatch):
    fake = FakeModule()
    fake.trtllm_fp4_moe_localized_config_kernels = lambda tile, act, per_token: (
        _ for _ in ()
    ).throw(RuntimeError("no kernel of this tile"))
    with pytest.raises(RuntimeError, match="cannot construct"):
        _load_with(monkeypatch, fake)
    assert M._module is None


def test_get_module_refuses_wrong_package(monkeypatch):
    class Mod(FakeModule):
        def trtllm_fp4_moe_localized_module_info(self):
            return ["0" * 64, INFO[1]]

    with pytest.raises(RuntimeError, match="expected"):
        _load_with(monkeypatch, Mod())
    with pytest.raises(RuntimeError, match="not a localized MoE module"):
        _load_with(monkeypatch, object())
    assert M._module is None


def test_prebuilt_module_override(monkeypatch):
    import tvm_ffi

    fake = FakeModule()
    monkeypatch.setenv(M.SO_ENV, "/prebuilt/localized.so")
    monkeypatch.setattr(tvm_ffi, "load_module", lambda p: fake)
    assert M._load_native() == (fake, ("/prebuilt/localized.so",))


# ---- the module build follows the stock trtllm-gen MoE JIT spec -------------------------------
def test_jit_spec_is_the_split_n_variant_of_the_stock_module(monkeypatch, tmp_path):
    from flashinfer.jit import fused_moe as JF

    captured = {}
    monkeypatch.setattr(JF, "stage_trtllm_gen_bmm_split_n", lambda: False)
    monkeypatch.setattr(
        JF,
        "_trtllm_gen_bmm_include_root",
        lambda path, sha, name, archs: (
            captured.update(root=(path, sha, name, archs)) or tmp_path
        ),
    )
    monkeypatch.setattr(
        JF.current_compilation_context,
        "get_nvcc_flags_list",
        lambda **kw: ["-gencode=arch=compute_100f,code=sm_100f"],
    )
    monkeypatch.setattr(
        JF, "gen_jit_spec", lambda name, sources, **kw: (name, sources, kw)
    )
    name, sources, kw = JF.gen_trtllm_gen_fused_moe_localized_module()
    assert name == "fused_moe_trtllm_sm107_localized"
    assert captured["root"][:3] == (
        A.ArtifactPath.TRTLLM_GEN_BMM_SPLIT_N,
        A.CheckSumHash.TRTLLM_GEN_BMM_SPLIT_N,
        name,
    )
    stock, _, _ = JF.trtllm_gen_fused_moe_build_inputs(
        "fused_moe_trtllm_sm107", True, nvcc_flags=[]
    )
    assert [s.name for s in sources] == [
        "trtllm_fused_moe_localized_launcher.cu"
        if s.name.endswith("_launcher.cu")
        else s.name
        for s in stock
    ]
    flags = kw["extra_cuda_cflags"]
    assert "-DTLLM_GEN_BMM_SPLIT_N_OUTPUT" in flags
    assert "-DTLLM_RUBIN_FEATURES" in flags
    assert any(
        A.ArtifactPath.TRTLLM_GEN_BMM_SPLIT_N in f and "CUBIN_PATH" in f for f in flags
    )
    assert any(INFO[0] in f for f in flags) and any(INFO[1] in f for f in flags)
    (ld,) = kw["extra_ldflags"]
    assert ld.startswith("-Wl,--version-script=")
    assert ld.endswith("trtllm_fused_moe_localized_exports.map")


# ---- split-N package staging ---------------------------------------------------------------------
def _fake_package(tmp_path, extra=b""):
    src = tmp_path / "package"
    files = {
        "include/flashinferMetaInfo.h": b'#define TLLM_GEN_EXPORT_VERSION "7.0.4.0.5.1"\n',
        "include/trtllmGen_bmm_export/KernelParams.h": b"// params\n" + extra,
        "Bmm_x_sm107a.cubin": b"cubin",
    }
    lines = []
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    manifest = ("\n".join(lines) + "\n").encode()
    (src / "checksums.txt").write_bytes(manifest)
    return src, hashlib.sha256(manifest).hexdigest(), files


@pytest.fixture
def cubin_dir(monkeypatch, tmp_path):
    d = tmp_path / "cubins"
    monkeypatch.setattr(A, "FLASHINFER_CUBIN_DIR", d)
    monkeypatch.setattr(A, "_staged_split_n_bundles", set())
    monkeypatch.delenv(A.TRTLLM_GEN_BMM_SPLIT_N_BUNDLE_ENV, raising=False)
    return d


def test_split_n_package_unavailable_by_default(cubin_dir):
    assert A.stage_trtllm_gen_bmm_split_n() is False
    assert A.get_trtllm_gen_bmm_split_n_file_list() == []
    assert not A.has_trtllm_gen_bmm_split_n()
    assert A.CheckSumHash.TRTLLM_GEN_BMM_SPLIT_N == (
        "092fe73ea347403156a93c6b2b23a835364802ad9bb2398b6accd671ffb6a8fd"
    )


def test_split_n_package_refuses_unpinned_manifest(cubin_dir, tmp_path):
    src, _, _ = _fake_package(tmp_path)
    with pytest.raises(RuntimeError, match="expected"):
        A.stage_trtllm_gen_bmm_split_n(src)
    assert not cubin_dir.exists()


def test_split_n_package_staged_from_env(cubin_dir, tmp_path, monkeypatch):
    src, sha, files = _fake_package(tmp_path)
    monkeypatch.setattr(A.CheckSumHash, "TRTLLM_GEN_BMM_SPLIT_N", sha)
    monkeypatch.setenv(A.TRTLLM_GEN_BMM_SPLIT_N_BUNDLE_ENV, str(src))
    listed = A.get_trtllm_gen_bmm_split_n_file_list()
    sub = A.ArtifactPath.TRTLLM_GEN_BMM_SPLIT_N
    assert dict(listed) == {
        sub + "checksums.txt": sha,
        **{sub + n: hashlib.sha256(d).hexdigest() for n, d in files.items()},
    }
    for name, digest in listed:
        assert hashlib.sha256((cubin_dir / name).read_bytes()).hexdigest() == digest
    # Once staged, the package stays available without the override.
    monkeypatch.delenv(A.TRTLLM_GEN_BMM_SPLIT_N_BUNDLE_ENV)
    assert A.has_trtllm_gen_bmm_split_n()


def test_split_n_package_tampered_or_missing_file(cubin_dir, tmp_path, monkeypatch):
    src, sha, _ = _fake_package(tmp_path)
    monkeypatch.setattr(A.CheckSumHash, "TRTLLM_GEN_BMM_SPLIT_N", sha)
    (src / "Bmm_x_sm107a.cubin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="does not match"):
        A.stage_trtllm_gen_bmm_split_n(src)
    (src / "Bmm_x_sm107a.cubin").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        A.stage_trtllm_gen_bmm_split_n(src)
    # No manifest was written, so the partial copy is not taken as a package.
    assert not A.has_trtllm_gen_bmm_split_n()


def _offline_repository(monkeypatch):
    """The public cubin repository with no files, so only the split-N tail is listed."""
    monkeypatch.setattr(
        A, "get_checksums", lambda dirs: collections.defaultdict(lambda: "0" * 64)
    )
    monkeypatch.setattr(A, "get_available_cubin_files", lambda url: [])
    monkeypatch.setattr(A, "get_available_header_files", lambda url: [])
    monkeypatch.setattr(
        A.CheckSumHash, "map_checksums", collections.defaultdict(lambda: "0" * 64)
    )


def test_cubin_file_list_keeps_a_staged_split_n_package(
    cubin_dir, tmp_path, monkeypatch
):
    _offline_repository(monkeypatch)
    src, sha, files = _fake_package(tmp_path)
    monkeypatch.setattr(A.CheckSumHash, "TRTLLM_GEN_BMM_SPLIT_N", sha)
    monkeypatch.setenv(A.TRTLLM_GEN_BMM_SPLIT_N_BUNDLE_ENV, str(src))
    listed = dict(A.get_subdir_file_list())
    sub = A.ArtifactPath.TRTLLM_GEN_BMM_SPLIT_N
    assert listed[sub + "checksums.txt"] == sha
    for n, d in files.items():
        assert listed[sub + n] == hashlib.sha256(d).hexdigest()


def test_cubin_file_list_strict_flag_requires_the_split_n_package(
    cubin_dir, monkeypatch
):
    _offline_repository(monkeypatch)
    sub = A.ArtifactPath.TRTLLM_GEN_BMM_SPLIT_N
    assert not any(n.startswith(sub) for n, _ in A.get_subdir_file_list())
    monkeypatch.setenv(A.TRTLLM_GEN_BMM_SPLIT_N_REQUIRE_ENV, "1")
    with pytest.raises(RuntimeError, match="not staged"):
        list(A.get_subdir_file_list())


def test_stock_trtllm_gen_moe_modules_are_renamed(monkeypatch):
    from flashinfer.jit import fused_moe as JF

    monkeypatch.setattr(
        JF,
        "trtllm_gen_fused_moe_build_inputs",
        lambda name, rubin, **kw: ([], [], []),
    )
    monkeypatch.setattr(JF, "gen_jit_spec", lambda name, sources, **kw: name)
    # A 0.7.2 jit-cache ships fused_moe_trtllm_sm10x built from the old FFI.
    assert JF.gen_trtllm_gen_fused_moe_sm100_module(True) == (
        "fused_moe_trtllm_sm107_m31loc"
    )
    assert JF.gen_trtllm_gen_fused_moe_sm100_module(False) == (
        "fused_moe_trtllm_sm100_m31loc"
    )


# ---- mode Q (MiniMax-M3.1: per-token FC1, static FC2) ------------------------------------------
def _pts(T):
    return torch.rand(T, dtype=torch.float32) + 0.5


def test_per_token_fc1_name_inserts_tok_into_the_b_scale_tag():
    assert M.per_token_fc1_name(PINS["decode"].fc1) == (
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16tokFp32_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_"
        "TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm107a"
    )
    with pytest.raises(ValueError, match="_Bb16_"):
        M.per_token_fc1_name("bmm_E2m1_E2m1E2m1_Fp32_Ab16_Cb16_t128x8x512_sm107a")


def test_per_token_pin_table_parallels_the_m3_table():
    assert len(M.PER_TOKEN_TACTIC_PINS) == len(M.DEFAULT_TACTIC_PINS) == 4
    for q, f in zip(M.PER_TOKEN_TACTIC_PINS, M.DEFAULT_TACTIC_PINS, strict=True):
        assert (q.label, q.min_tokens, q.max_tokens, q.tile, q.fc2) == (
            f.label,
            f.min_tokens,
            f.max_tokens,
            f.tile,
            f.fc2,
        )
        assert q.fc1.count("_Bb16tokFp32_") == 1 and "tok" not in q.fc2
        assert q.fc1.replace("tokFp32", "", 1) == f.fc1


def test_mode_q_admission(env):
    sh = shards()
    # The stock default per_token_scale_gemm2=True is mode F, which stays rejected.
    with pytest.raises(NotImplementedError, match="mode Q"):
        call(sh, 4, per_token_scale=_pts(4))
    for bad in (_pts(4).double(), _pts(3), _pts(8).view(2, 4), _pts(4).to("meta")):
        with pytest.raises(ValueError, match="per_token_scale"):
            call(sh, 4, per_token_scale=bad, per_token_scale_gemm2=False)
    assert not env.launches


@pytest.mark.parametrize(
    "T,label", [(4, "decode"), (128, "small"), (1024, "mid"), (2048, "prefill")]
)
def test_mode_q_forwards_scale_and_uses_per_token_pins(env, T, label):
    s = _pts(T)
    call(shards(), T, per_token_scale=s, per_token_scale_gemm2=False)
    L = env.launches[-1]
    assert L["per_token_scale"] is s
    assert (L["tile"], L["config"]) == PT_PINS[label] == M._per_token_pins()[label]
    assert env.valid_queries and all(env.valid_queries)
    assert TU.last_resolution[(TU.custom_op(1, False), T)] == ("pinned", PT_PINS[label])


def test_mode_q_monolithic_forwards_scale(env):
    one = torch.ones(E)
    w1, w1s, w2, w2s = placeholders()
    s = _pts(4)
    trtllm_fp4_block_scale_moe(
        torch.zeros(4, E),
        None,
        torch.zeros(4, H // 2, dtype=torch.uint8),
        torch.zeros(4, H // 16, dtype=torch.uint8),
        w1,
        w1s,
        None,
        one,
        one,
        one,
        w2,
        w2s,
        None,
        one,
        one,
        one,
        E,
        TOPK,
        None,
        None,
        I,
        0,
        E,
        2.0,
        routing_method_type=7,
        enable_pdl=True,
        hidden_states_scale_layout=SfLayout.layout_linear,
        per_token_scale=s,
        per_token_scale_gemm2=False,
        **loc_kwargs(shards()),
    )
    assert env.launches[-1]["per_token_scale"] is s
    assert launched(env) == PT_PINS["decode"]


def test_m3_call_stays_off_the_per_token_catalog(env):
    call(shards(), 4)
    assert env.launches[-1]["per_token_scale"] is None
    assert env.valid_queries and not any(env.valid_queries)
    assert launched(env) == DECODE_PIN
    # A mode flag without a per-token scale is the M3 call.
    call(shards(), 4, per_token_scale_gemm2=False)
    assert env.launches[-1]["per_token_scale"] is None and not any(env.valid_queries)


def test_mode_q_tuning_records_are_separate_from_m3(env, monkeypatch):
    monkeypatch.setattr(TU, "PINS_OVERRIDE_TUNING", False)
    sh = shards()
    with autotune(True):
        call(sh, 4)
    call(sh, 4)
    assert TU.last_resolution[(TU.custom_op(1, False), 4)] == ("tuned", (32, 1))
    env.launches.clear()
    # The M3 record must not serve a mode-Q call: its FC1 kernel has no per-token scaling.
    call(sh, 4, per_token_scale=_pts(4), per_token_scale_gemm2=False)
    assert TU.last_resolution[(TU.custom_op(1, False), 4)] == (
        "default",
        PT_PINS["decode"],
    )
    with autotune(True):
        call(sh, 4, per_token_scale=_pts(4), per_token_scale_gemm2=False)
    # Profiling launches carry a per-token scale (the tuner synthesizes it as ones).
    assert env.launches and all(L["per_token_scale"] is not None for L in env.launches)
    call(sh, 5, per_token_scale=_pts(5), per_token_scale_gemm2=False)
    assert TU.last_resolution[(TU.custom_op(1, False), 5)] == ("tuned", (32, 1))
    assert len(TU._runners) == 2


def test_mode_q_cache_identity(env):
    m3, q = runner(), runner(per_token=True)
    assert m3.identity() != q.identity()
    assert q.identity()[-1] == "per_token_fc1"
    assert q.identity()[-2] == M.catalog_fingerprint(3, True)
    assert M.catalog_fingerprint(3, True) != M.catalog_fingerprint(3)
    # The M3 key keeps its pre-mode-Q form, so existing tuning records stay valid.
    assert m3.identity()[-2:] == (INFO, M.catalog_fingerprint(3))
    assert q.default_tactic(4) == PT_PINS["decode"]
    assert m3.default_tactic(4) == DECODE_PIN


def test_bundle_without_a_per_token_pin_warns_once_and_falls_back(
    env, monkeypatch, caplog
):
    """A bundle without the per-token decode FC1 (an M3-only package): mode Q loads, warns once and takes
    the first valid tactic; M3 is unaffected."""
    env.per_token_catalog = lambda tile: [
        n.replace("tokFp32", "tokFp32_absent")
        if tile == 8 and n == M.PER_TOKEN_TACTIC_PINS[0].fc1
        else n
        for n in _catalog(tile, per_token=True)
    ]
    with caplog.at_level("WARNING", logger="flashinfer.fused_moe.trtllm_localized"):
        for _ in range(3):
            call(shards(), 4, per_token_scale=_pts(4), per_token_scale_gemm2=False)
    warned = [
        r for r in caplog.records if "per-token pinned default 'decode'" in r.message
    ]
    assert len(warned) == 1
    assert M.pinned_default(4, per_token=True) is None
    first_valid = M.choose_tactic(4, TOPK, H, I, E, (100, 100), per_token=True)
    assert launched(env) == first_valid == (8, 2)
    assert TU.last_resolution[(TU.custom_op(1, False), 4)] == ("default", first_valid)
    assert M.pinned_default(128, per_token=True) == ("small", PT_PINS["small"])
    call(shards(), 4)
    assert launched(env) == DECODE_PIN


def test_bundle_without_per_token_kernels_still_loads(monkeypatch, caplog):
    def no_tok(tile):
        raise RuntimeError("No kernel found for the given options")

    fake = FakeModule(per_token_catalog=no_tok)
    # Per-token pins are not resolved at load.
    assert _load_with(monkeypatch, fake) is fake
    assert M._resolved_per_token_pins is None
    with caplog.at_level("WARNING", logger="flashinfer.fused_moe.trtllm_localized"):
        assert M._per_token_pins() == {}
    assert len([r for r in caplog.records if "per-token pinned" in r.message]) == 4
    assert M.pinned_default(4, per_token=True) is None
    assert M.pinned_default(4) == ("decode", DECODE_PIN)


# ---- mode Q on the stock op -------------------------------------------------------------------
def test_stock_mode_q_reaches_the_native_op(monkeypatch):
    rec = _StockRecorder()
    monkeypatch.setattr(C, "get_trtllm_moe_sm100_module", lambda: rec)
    T = 4
    one = torch.ones(E)
    s = _pts(T)
    w = torch.zeros(1, dtype=torch.uint8)
    args = [torch.zeros(T, TOPK, dtype=torch.int32), None]
    args += [torch.zeros(T, H // 2, dtype=torch.uint8)]
    args += [torch.zeros(T, H // 16, dtype=torch.uint8), w, w, None, one, one, one]
    args += [w, w, None, one, one, one, E, TOPK, 0, 0, I, 0, E, None]
    for gemm2, want in ((False, False), (True, True)):
        trtllm_fp4_block_scale_routed_moe(
            *args, per_token_scale=s, per_token_scale_gemm2=gemm2
        )
        assert rec.calls[-1][20] is s and rec.calls[-1][-1] is want
    trtllm_fp4_block_scale_routed_moe(*args)
    assert rec.calls[-1][20] is None and rec.calls[-1][-1] is True


class _NativeRecorder:
    def __init__(self):
        self.valid_calls, self.fp4_calls = [], []

    def trtllm_get_valid_moe_configs(self, *args):
        self.valid_calls.append(args)
        return [[8, 0]]

    def trtllm_fp4_block_scale_moe(self, *args):
        self.fp4_calls.append(args)
        return [args[34]]


def _trt_runner(op, **kw):
    from flashinfer.fused_moe.backends.trtllm.sm100_runner import MoERunner
    from flashinfer.tllm_enums import DtypeTrtllmGen, Fp8QuantizationType

    return MoERunner(
        op,
        top_k=TOPK,
        num_local_experts=E,
        dtype_act=DtypeTrtllmGen.E2m1,
        dtype_weights=DtypeTrtllmGen.E2m1,
        fp8_quantization_type=Fp8QuantizationType.NoneFp8,
        hidden_size=H,
        intermediate_size=I,
        use_shuffled_weight=True,
        **kw,
    )


def test_stock_runner_keys_queries_and_launches_mode_q():
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs

    op = _NativeRecorder()
    m3 = _trt_runner(op)
    f = _trt_runner(op, use_per_token_scaling=True)
    q = _trt_runner(op, use_per_token_scaling=True, per_token_scale_gemm2=False)
    # Mode F and M3 keys keep their form; mode Q adds a marker; the flag alone is a no-op.
    assert q.get_cache_key_extras([]) == f.get_cache_key_extras([]) + (
        "per_token_fc1_only",
    )
    assert _trt_runner(op, per_token_scale_gemm2=False).get_cache_key_extras(
        []
    ) == m3.get_cache_key_extras([])
    assert hash(q) != hash(f)

    T = 4
    s = _pts(T)
    inputs = MoeRunnerInputs(
        output=torch.empty(T, H, dtype=torch.bfloat16),
        routing_logits=None,
        topk_ids=torch.zeros(T, TOPK, dtype=torch.int32),
        expert_weights=torch.zeros(T, TOPK, dtype=torch.bfloat16),
        hidden_states=torch.zeros(T, H // 2, dtype=torch.uint8),
        hidden_states_scale=torch.zeros(T, H // 16, dtype=torch.uint8),
        gemm1_lora_delta=None,
        per_token_scale=s,
    ).to_list()
    for r, gemm2 in ((q, False), (f, True), (m3, True)):
        r.get_valid_tactics(inputs, None)
        assert op.valid_calls[-1][11] is r.use_per_token_scaling
        assert op.valid_calls[-1][-1] is gemm2
    one = torch.ones(E)
    kw = {
        k: None
        for k in (
            "routing_bias",
            "gemm1_bias",
            "gemm2_bias",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
        )
    }
    kw.update(
        gemm1_weights=one,
        gemm1_weights_scale=one,
        gemm1_alpha=one,
        gemm1_beta=one,
        gemm1_clamp_limit=one,
        gemm2_weights=one,
        gemm2_weights_scale=one,
        output1_scale_scalar=one,
        output1_scale_gate_scalar=one,
        output2_scale_scalar=one,
        num_experts=E,
        local_expert_offset=0,
        routing_method_type=1,
        do_finalize=True,
        enable_pdl=True,
        routing_input_mode=1,
    )
    q.forward(inputs, tactic=[8, 0], **kw)
    assert op.fp4_calls[-1][20] is s and op.fp4_calls[-1][-1] is False
    f.forward(inputs, tactic=[8, 0], **kw)
    assert op.fp4_calls[-1][-1] is True
