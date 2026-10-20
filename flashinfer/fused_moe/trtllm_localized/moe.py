"""Localized trtllm-gen NVFP4 MoE: a fork/join of split-N GEMM halves over two locality domains.

Reached through ``flashinfer.fused_moe.trtllm_fp4_block_scale_moe`` and
``trtllm_fp4_block_scale_routed_moe`` when ``localized_weights`` is given (the argument names
follow the CuTe DSL localization of #5276). Per call, on the caller's stream s (native side,
csrc/trtllm_fused_moe_localized_launcher.cu):
  1. routing + permutation once on s (full device)
  2. fork: record on s; both ``localized_streams`` wait
  3. FC1 on localized_streams[p]: shard p's gemm1_weights, activated columns [p*I/2, (p+1)*I/2)
     of the shared gemm1 output, slice {rowWidthC=I, columnOffsetC=p*I/2, smCount[p]}
  4. cross-join: each partition waits for the other's FC1 (FC2 reads the full K = I)
  5. FC2 on localized_streams[p]: shard p's gemm2_weights, hidden columns [p*H/2, (p+1)*H/2),
     slice {rowWidthC=H, columnOffsetC=p*H/2, smCount[p]}
  6. join to s; finalize on s when do_finalize, else return the unfinalized triple
The caller owns the green contexts, their streams and the domain-local memory the shards live
in; this module creates neither.

Per-token activation scales are admitted in mode Q only (``per_token_scale_gemm2=False``, the
MiniMax-M3.1 recipe): both FC1 halves apply the same fp32 [T] token scale in their epilogue and
write NVFP4 directly; FC2 is static. Mode F (FC1 BF16 output re-quantized per row, per-token
FC2) is rejected. Mode Q needs the FC1 kernels with per-token B scaling (``Bb16tokFp32``).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ...artifacts import CheckSumHash
from ...jit.fused_moe import TRTLLM_GEN_BMM_SPLIT_N_EXPORT_VERSION
from ..utils import check_localized_args

logger = logging.getLogger("flashinfer.fused_moe.trtllm_localized")

# Overrides the JIT / jit-cache module with a prebuilt one; it must report the pinned package.
SO_ENV = "FLASHINFER_LOCALIZED_MOE_SO"
MODULE_INFO = (
    CheckSumHash.TRTLLM_GEN_BMM_SPLIT_N,
    TRTLLM_GEN_BMM_SPLIT_N_EXPORT_VERSION,
)
# Keys of each localized_weights entry: the stock op's own weight argument names.
SHARD_KEYS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)
NUM_PARTITIONS = 2
# Tile sizes probed for partition-valid configs; tiles absent from the bundle yield no configs.
_TILES = (8, 16, 32, 64, 128)
_SWIGLU = 3  # ActivationType.Swiglu


@dataclass(frozen=True)
class PinnedDefault:
    """A default tactic used when no tuning record exists, pinned by its FC1/FC2 kernel names.

    Applies to SwiGLU calls with ``min_tokens <= T <= max_tokens`` (None = unbounded). The
    names are resolved to the module's config index when the module is loaded; a name missing
    from the catalog fails the load.
    """

    label: str
    min_tokens: int
    max_tokens: Optional[int]
    tile: int
    fc1: str
    fc2: str

    def covers(self, num_tokens: int) -> bool:
        return self.min_tokens <= num_tokens and (
            self.max_tokens is None or num_tokens <= self.max_tokens
        )


# Min-max-regret picks on the split-N package across uniform, zipf and round-robin routing;
# together they cover every SwiGLU T.
DEFAULT_TACTIC_PINS: Tuple[PinnedDefault, ...] = (
    PinnedDefault(
        "decode",
        1,
        48,
        8,
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm107a",
        "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x8x512_s5_et128x8_m128x8x64_c1x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm107a",
    ),
    PinnedDefault(
        "small",
        49,
        256,
        16,
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x16x512_s5_et128x16_m128x16x64_c1x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm107a",
        "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x16x256u2_s9_et128x16_m128x16x64_c1x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm107a",
    ),
    PinnedDefault(
        "mid",
        257,
        1536,
        32,
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm107a",
        "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x32x512_s5_et128x32_m256x32x64_c2x1x1_rM_TN_"
        "transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm107a",
    ),
    PinnedDefault(
        "prefill",
        1537,
        None,
        128,
        "bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x128x512_s3x3x3x3x1x3_et128x32_m256x128x64_"
        "c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_fCp_fLoadSf_bN_ldgsts_ldgstsSf_rgTma_clmp_"
        "swiGlu_lbW4_lsfbW4_dynB_sm107a",
        "bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x128x256u2_s6_et128x128_m256x128x64_c2x1x1_rM_"
        "TN_transOut_schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm107a",
    ),
)


# trtllm-gen names a kernel with FP32 per-token B scaling by inserting "tok<Dtype>" into its
# B scale tag (Gemm/Utils.h); tokens are operand B, so only the B tag changes.
_B_SCALE_TAG = "_Bb16_"
_B_SCALE_TAG_PER_TOKEN = "_Bb16tokFp32_"


def per_token_fc1_name(name: str) -> str:
    """The mode-Q (FP32 per-token B scaling) variant of an FC1 kernel name."""
    if name.count(_B_SCALE_TAG) != 1:
        raise ValueError(
            f"expected exactly one '{_B_SCALE_TAG}' in FC1 kernel name {name}"
        )
    return name.replace(_B_SCALE_TAG, _B_SCALE_TAG_PER_TOKEN)


# Mode Q pins: the same buckets and tiles, FC1 swapped for its per-token variant, FC2 unchanged
# (FC2 is static in mode Q). On the pinned package they resolve to (8, 67), (16, 71), (32, 139)
# and (128, 24); max regret is <= 5% for every T >= 2 (5.5% at T = 1).
PER_TOKEN_TACTIC_PINS: Tuple[PinnedDefault, ...] = tuple(
    dataclasses.replace(pin, fc1=per_token_fc1_name(pin.fc1))
    for pin in DEFAULT_TACTIC_PINS
)


_module_lock = threading.Lock()
_module = None
_module_paths: Tuple[str, ...] = ()
# label -> (tile, config index) of each DEFAULT_TACTIC_PINS entry in the loaded module.
_resolved_pins: Dict[str, Tuple[int, int]] = {}
# Same for PER_TOKEN_TACTIC_PINS, resolved on first mode-Q use; None until then.
_resolved_per_token_pins: Optional[Dict[str, Tuple[int, int]]] = None


def loaded_module_paths() -> Tuple[str, ...]:
    """Library paths of the loaded module (empty before get_module())."""
    return _module_paths


def module_info() -> Tuple[str, str]:
    """(split-N package checksums.txt sha256, GEN export version) of the loaded module."""
    get_module()
    return MODULE_INFO


def resolve_pins(
    mod,
    pins: Sequence[PinnedDefault] = DEFAULT_TACTIC_PINS,
    per_token: bool = False,
    strict: bool = True,
) -> Dict[str, Tuple[int, int]]:
    """Maps each pinned (FC1, FC2) kernel-name pair to its (tile, config index) in ``mod``.

    ``strict`` raises on a pin the catalog lacks; otherwise the pin is left out with a warning,
    and the token counts it covers are tuned or take the first valid tactic.
    """
    resolved = {}
    tables: Dict[int, List[str]] = {}
    for pin in pins:
        if pin.tile not in tables:
            try:
                tables[pin.tile] = [
                    str(n)
                    for n in mod.trtllm_fp4_moe_localized_config_kernels(
                        pin.tile, _SWIGLU, per_token
                    )
                ]
            except Exception as e:  # noqa: BLE001 - the runner refuses a tile with no kernel
                if strict:
                    raise RuntimeError(
                        f"pinned default '{pin.label}' needs tile {pin.tile}, which the "
                        f"localized MoE module cannot construct: {e}"
                    ) from e
                tables[pin.tile] = []
        names = tables[pin.tile]
        pairs = list(zip(names[0::2], names[1::2], strict=True))
        hits = [i for i, pair in enumerate(pairs) if pair == (pin.fc1, pin.fc2)]
        if not hits:
            if strict:
                raise RuntimeError(
                    f"pinned default '{pin.label}' (tile {pin.tile}) is not in the localized MoE "
                    f"module's catalog: FC1 {pin.fc1} / FC2 {pin.fc2}"
                )
            logger.warning(
                "[localized MoE] %spinned default '%s' (tile %d) is not in the module's catalog "
                "(FC1 %s / FC2 %s); its token counts are tuned or take the first valid tactic",
                "per-token " if per_token else "",
                pin.label,
                pin.tile,
                pin.fc1,
                pin.fc2,
            )
            continue
        resolved[pin.label] = (pin.tile, hits[0])
    return resolved


def _per_token_pins() -> Dict[str, Tuple[int, int]]:
    """PER_TOKEN_TACTIC_PINS resolved against the loaded module, once (a bundle without the
    per-token FC1 kernels warns once and leaves them unpinned)."""
    global _resolved_per_token_pins
    mod = get_module()
    if _resolved_per_token_pins is None:
        with _module_lock:
            if _resolved_per_token_pins is None:
                _resolved_per_token_pins = resolve_pins(
                    mod, PER_TOKEN_TACTIC_PINS, per_token=True, strict=False
                )
    return _resolved_per_token_pins


def default_enable_pdl(device: torch.device) -> bool:
    """enable_pdl=None: the stock MoE PDL gate (off on SM107); an explicit True is honoured."""
    if device.type != "cuda":
        return False
    from ..core import _device_support_moe_pdl

    return _device_support_moe_pdl(device)


def _load_native() -> Tuple[Any, Tuple[str, ...]]:
    """The module from ``FLASHINFER_LOCALIZED_MOE_SO`` if set, else from the jit-cache or JIT."""
    path = os.environ.get(SO_ENV)
    if path:
        import tvm_ffi

        return tvm_ffi.load_module(path), (os.path.abspath(path),)
    from ...jit.fused_moe import gen_trtllm_gen_fused_moe_localized_module

    spec = gen_trtllm_gen_fused_moe_localized_module()
    mod = spec.build_and_load()
    return mod, tuple(str(p) for p in spec.get_library_paths())


def get_module():
    """Loads the split-N MoE module once per process.

    The module must report the pinned split-N package and export version, and must carry
    every pinned default tactic.
    """
    global _module, _module_paths, _resolved_pins, _resolved_per_token_pins
    if _module is not None:
        return _module
    with _module_lock:
        if _module is not None:
            return _module
        from ...jit.cubin_loader import setup_cubin_loader

        mod, paths = _load_native()
        try:
            info = tuple(str(x) for x in mod.trtllm_fp4_moe_localized_module_info())
        except Exception as e:  # noqa: BLE001 - a missing entry point surfaces differently per tvm_ffi version
            raise RuntimeError(
                f"{paths} is not a localized MoE module (no module_info entry point)"
            ) from e
        if info != MODULE_INFO:
            raise RuntimeError(
                f"{paths} was built against split-N package {info[0]} / export {info[1]}; "
                f"expected {MODULE_INFO[0]} / {MODULE_INFO[1]}"
            )
        pins = resolve_pins(mod)
        for path in paths:
            setup_cubin_loader(path)
        _module, _module_paths, _resolved_pins = mod, paths, pins
        _resolved_per_token_pins = None
        return _module


@functools.lru_cache(maxsize=4096)
def _valid_configs(
    tile, top_k, H, I, E, T, sm0, sm1, act_type=_SWIGLU, per_token=False
) -> Tuple[int, ...]:
    try:
        r = list(
            get_module().trtllm_fp4_moe_localized_valid_configs(
                tile, top_k, H, I, E, T, sm0, sm1, act_type, per_token
            )
        )
    except RuntimeError:
        # The runner refuses to construct for a tile the bundle has no kernel for; any other
        # failure propagates.
        if tile not in _tiles_with_kernels(
            top_k, H, I, E, sm0, sm1, act_type, per_token
        ):
            return ()
        raise
    end = r.index(-1)
    return tuple(r[:end])


_tile_probe: dict = {}


def _tiles_with_kernels(
    top_k, H, I, E, sm0, sm1, act_type, per_token=False
) -> Tuple[int, ...]:
    """Tiles whose runner constructs at all (probed at T=1); cached per shape and mode."""
    key = (top_k, H, I, E, sm0, sm1, act_type, per_token)
    if key not in _tile_probe:
        ok = []
        for tile in _TILES:
            try:
                get_module().trtllm_fp4_moe_localized_valid_configs(
                    tile, top_k, H, I, E, 1, sm0, sm1, act_type, per_token
                )
                ok.append(tile)
            except RuntimeError:
                pass
        _tile_probe[key] = tuple(ok)
    return _tile_probe[key]


@functools.lru_cache(maxsize=None)
def catalog_fingerprint(act_type: int = _SWIGLU, per_token: bool = False) -> str:
    """sha256 of the (tile, config index) -> (FC1, FC2) kernel-name map of the loaded module.

    The config index of a kernel pair depends on the runner's filtering as well as on the
    bundle, so tuning records are keyed on this map rather than on the bundle alone.
    """
    mod = get_module()
    h = hashlib.sha256()
    for tile in _TILES:
        try:
            names = [
                str(n)
                for n in mod.trtllm_fp4_moe_localized_config_kernels(
                    tile, act_type, per_token
                )
            ]
        except RuntimeError:  # no kernel of this tile
            names = []
        h.update(f"{tile}:{len(names)}\0".encode())
        h.update("\0".join(names).encode())
    return h.hexdigest()


def choose_tactic(
    num_tokens, top_k, H, I, E, sm_counts, act_type=_SWIGLU, per_token=False
) -> Tuple[int, int]:
    """First valid (tile, config): the tile closest to the mean tokens per expert that has a
    partition-valid config, and its first valid config (no timing)."""
    target = max(1, num_tokens * top_k // max(E, 1))
    for tile in sorted(
        _TILES, key=lambda t: (abs(t.bit_length() - target.bit_length()), t)
    ):
        cfgs = _valid_configs(
            tile,
            top_k,
            H,
            I,
            E,
            num_tokens,
            sm_counts[0],
            sm_counts[1],
            act_type,
            per_token,
        )
        if cfgs:
            return tile, cfgs[0]
    raise RuntimeError(f"no partition-valid MoE tactic for T={num_tokens} H={H} I={I}")


def pinned_default(
    num_tokens: int, act_type: int = _SWIGLU, per_token: bool = False
) -> Optional[Tuple[str, Tuple[int, int]]]:
    """(label, (tile, config)) of the pin covering num_tokens, or None (also for a mode-Q pin
    the module's catalog lacks)."""
    if act_type != _SWIGLU:
        return None
    if per_token:
        pins, resolved = PER_TOKEN_TACTIC_PINS, _per_token_pins()
    else:
        get_module()
        pins, resolved = DEFAULT_TACTIC_PINS, _resolved_pins
    for pin in pins:
        if pin.covers(num_tokens):
            tactic = resolved.get(pin.label)
            return None if tactic is None else (pin.label, tactic)
    return None


_warned_invalid_pins: set = set()


def default_tactic(
    num_tokens, top_k, H, I, E, sm_counts, act_type=_SWIGLU, per_token=False
) -> Tuple[int, int]:
    """The tactic used without a tuning record: the pinned default for this token count when
    it is partition-valid for the shape, else the first valid tactic."""
    pin = pinned_default(num_tokens, act_type, per_token)
    if pin is not None:
        label, tactic = pin
        if tactic[1] in _valid_configs(
            tactic[0],
            top_k,
            H,
            I,
            E,
            num_tokens,
            sm_counts[0],
            sm_counts[1],
            act_type,
            per_token,
        ):
            return tactic
        key = (label, per_token, top_k, H, I, E, tuple(sm_counts))
        if key not in _warned_invalid_pins:
            _warned_invalid_pins.add(key)
            logger.warning(
                "[localized MoE] pinned default '%s' %s is not partition-valid for T=%d H=%d "
                "I=%d E=%d SMs %s; using the first valid tactic",
                label,
                tactic,
                num_tokens,
                H,
                I,
                E,
                tuple(sm_counts),
            )
    return choose_tactic(num_tokens, top_k, H, I, E, sm_counts, act_type, per_token)


def _sm_counts(localized_sm_count) -> List[int]:
    if isinstance(localized_sm_count, int):
        counts = [localized_sm_count] * NUM_PARTITIONS
    else:
        counts = [int(x) for x in localized_sm_count]
    if len(counts) != NUM_PARTITIONS or any(c <= 0 for c in counts):
        raise ValueError(
            f"localized_sm_count must be a positive int or {NUM_PARTITIONS} positive ints, got "
            f"{localized_sm_count}"
        )
    return counts


def _check_localized_args(
    hidden_states: torch.Tensor,
    localized_weights,
    localized_streams,
    localized_sm_count,
) -> None:
    check_localized_args(localized_weights, localized_streams, localized_sm_count)
    device = hidden_states.device
    for p, shard in enumerate(localized_weights):
        if set(shard) != set(SHARD_KEYS):
            raise ValueError(
                f"localized_weights[{p}] must have exactly the keys {SHARD_KEYS}, got "
                f"{sorted(shard)}"
            )
        for k in SHARD_KEYS:
            if shard[k].device != device:
                raise ValueError(
                    f"localized_weights[{p}]['{k}'] is on {shard[k].device}, hidden_states on "
                    f"{device}"
                )
        if torch.device(localized_streams[p].device) != device:
            raise ValueError(
                f"localized_streams[{p}] is on {localized_streams[p].device}, hidden_states on "
                f"{device}"
            )
    if device.type == "cuda" and torch.cuda.get_device_capability(device) != (10, 7):
        raise NotImplementedError("locality-domain localization is Rubin (SM107) only")
    if hidden_states.dtype != torch.uint8:
        raise NotImplementedError(
            "locality-domain localization of the trtllm-gen MoE admits NVFP4 activations only"
        )


def _check_stock_only_args(
    hidden_size: int,
    intermediate_size: int,
    hidden_states_scale,
    valid_hidden_size: Optional[int],
    valid_intermediate_size: Optional[int],
    hidden_states_scale_layout,
) -> None:
    """Stock arguments the localized path accepts only in their no-op form."""
    if valid_hidden_size not in (None, hidden_size) or valid_intermediate_size not in (
        None,
        intermediate_size,
    ):
        raise NotImplementedError(
            "valid (padded) dims are not supported by the localized MoE"
        )
    if hidden_states_scale is not None:
        from ..shared.validation import resolve_moe_act_sf_layout

        # Same contract as the stock op: None infers linear (deprecated), anything else raises.
        resolve_moe_act_sf_layout(hidden_states_scale_layout)


def localization_signature(
    localized_weights, sm_counts: Sequence[int], memset_stream
) -> Tuple[Any, ...]:
    """Tuner cache-key component, in the form of #5276's ``_localization_signature``."""
    shard_shapes = tuple(
        (tuple(shard["gemm1_weights"].shape), tuple(shard["gemm2_weights"].shape))
        for shard in localized_weights
    )
    return ("localized", tuple(sm_counts), shard_shapes, memset_stream is not None)


def trtllm_fp4_block_scale_moe_localized(
    *,
    routing_input_mode: int,
    routing_logits: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    localized_weights: Sequence[Dict[str, torch.Tensor]],
    localized_streams: Optional[Sequence[torch.cuda.Stream]],
    localized_sm_count,
    localized_memset_stream: Optional[torch.cuda.Stream] = None,
    localized_allow_nonlocalized: bool = False,
    autotune_use_cuda_graph: bool = False,
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = _SWIGLU,
    per_token_scale: Optional[torch.Tensor] = None,
    per_token_scale_gemm2: bool = True,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    num_fused_shared_experts: Optional[int] = None,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    valid_hidden_size: Optional[int] = None,
    valid_intermediate_size: Optional[int] = None,
    hidden_states_scale_layout=None,
    tactic: Optional[Tuple[int, int]] = None,
    _mode: int = 0,
    _return_intermediates: bool = False,
    _debug_delay: Optional[Tuple[int, int, int]] = None,
) -> List[torch.Tensor]:
    """The localized body of the stock FP4 MoE ops; returns what they return.

    ``localized_memset_stream`` is accepted and unused (finalize writes every output element).
    ``localized_allow_nonlocalized`` is accepted; the tuner considers split-N tactics only.
    ``autotune_use_cuda_graph`` is accepted; profiling always replays under a CUDA graph.
    ``per_token_scale`` is accepted in mode Q only (``per_token_scale_gemm2=False``).

    Private, for tests and kernel A/B tooling only (not a stable API): ``tactic`` forces a
    (tile, config) that must be partition-valid; ``_mode=1`` runs the stock single-stream path
    inside the same module with the FULL weights passed as localized_weights[0] (the bit-exact
    reference); ``_return_intermediates`` appends [gemm1_output, gemm1_output_scale,
    gemm2_output, expanded_idx_to_permuted_idx] after a canary fill; ``_debug_delay=(partition,
    phase 1=FC1 / 2=FC2, cycles)`` spins on that partition's stream before that GEMM.
    """
    del localized_memset_stream, autotune_use_cuda_graph
    routing_input_mode = int(routing_input_mode)
    per_token = per_token_scale is not None
    if per_token and per_token_scale_gemm2:
        raise NotImplementedError(
            "the localized MoE supports per_token_scale only in mode Q "
            "(per_token_scale_gemm2=False: per-token FC1, static FC2); mode F (per-row FC2 "
            "requantization) is not supported"
        )
    if routing_replay_out is not None or num_fused_shared_experts:
        raise NotImplementedError(
            "routing replay and fused shared experts are not supported by the localized MoE"
        )
    if gemm1_bias is not None or gemm2_bias is not None:
        raise NotImplementedError("bias is not supported by the localized MoE")
    if gemm1_lora_delta is not None:
        raise NotImplementedError(
            "gemm1_lora_delta is not supported by the localized MoE"
        )
    if topk_ids is not None and topk_ids.dtype != torch.int32:
        raise TypeError("topk_ids must be int32")
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1] * 2
    if per_token and (
        per_token_scale.dtype != torch.float32
        or per_token_scale.dim() != 1
        or per_token_scale.shape[0] < num_tokens
        or per_token_scale.device != hidden_states.device
    ):
        raise ValueError(
            f"per_token_scale must be a float32 [>= num_tokens] tensor on "
            f"{hidden_states.device}, got {per_token_scale.dtype} "
            f"{tuple(per_token_scale.shape)} on {per_token_scale.device}"
        )
    if hidden_size % 256 or intermediate_size % 256:
        raise ValueError(
            f"localized MoE needs H and I to be multiples of 256 "
            f"(H={hidden_size}, I={intermediate_size})"
        )
    _check_stock_only_args(
        hidden_size,
        intermediate_size,
        hidden_states_scale,
        valid_hidden_size,
        valid_intermediate_size,
        hidden_states_scale_layout,
    )
    if _mode == 1:
        # Reference: full weights as shard 0; partition 1 and the streams are unused.
        shards = [localized_weights[0], localized_weights[0]]
        streams = [0, 0]
        sms = [1, 1]
    else:
        _check_localized_args(
            hidden_states, localized_weights, localized_streams, localized_sm_count
        )
        shards = list(localized_weights)
        streams = [s.cuda_stream for s in localized_streams]
        sms = _sm_counts(localized_sm_count)
    if (
        hidden_states_scale is not None
        and hidden_states_scale.dim() == 1
        and num_tokens > 0
    ):
        # The flat linear scale the stock op also accepts; the tuner buckets dim 0 as tokens.
        hidden_states_scale = hidden_states_scale.view(num_tokens, -1)
    device = hidden_states.device
    mod = get_module()
    if enable_pdl is None:
        enable_pdl = default_enable_pdl(device)
    if output is None:
        output = torch.empty(
            num_tokens,
            hidden_size if do_finalize else 0,
            dtype=torch.bfloat16,
            device=device,
        )
    if _mode == 0 and tactic is None:
        from . import tuning

        runner = tuning.get_runner(
            routing_input_mode=routing_input_mode,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=local_num_experts,
            top_k=top_k,
            activation_type=activation_type,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            sm_counts=tuple(sms),
            signature=localization_signature(shards, sms, None),
            allow_nonlocalized=localized_allow_nonlocalized,
            per_token=per_token,
        )
        inputs = [
            output,
            routing_logits,
            topk_ids,
            topk_weights,
            hidden_states,
            hidden_states_scale,
        ]
        if per_token:  # appended only in mode Q, so M3 tuning keys are unchanged
            inputs.append(per_token_scale)
        tactic = tuning.select_tactic(
            runner,
            inputs,
            dict(
                routing_bias=routing_bias,
                localized_weights=shards,
                localized_streams=localized_streams,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
                output1_scale_scalar=output1_scale_scalar,
                output1_scale_gate_scalar=output1_scale_gate_scalar,
                output2_scale_scalar=output2_scale_scalar,
                n_group=n_group,
                topk_group=topk_group,
                local_expert_offset=local_expert_offset,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=routing_method_type,
                norm_topk_prob=norm_topk_prob,
                hidden_states_scale_layout=hidden_states_scale_layout,
            ),
            tune_max_num_tokens,
        )
    elif _mode == 0:
        tactic = (int(tactic[0]), int(tactic[1]))
        valid = _valid_configs(
            tactic[0],
            top_k,
            hidden_size,
            intermediate_size,
            local_num_experts,
            num_tokens,
            sms[0],
            sms[1],
            activation_type,
            per_token,
        )
        if tactic[1] not in valid:
            raise ValueError(
                f"tactic {tactic} is not in the partition-valid set for T={num_tokens} "
                f"(SMs {tuple(sms)})"
            )
    elif tactic is None:
        raise ValueError("the stock reference (_mode=1) needs an explicit tactic")
    if routing_input_mode == 2:  # UnpackedPrecomputed: both are inputs
        if topk_ids is None or topk_weights is None:
            raise ValueError(
                "unpacked precomputed routing needs topk_ids and topk_weights"
            )
    else:  # FromLogits / Packed: ids (logits mode) or packed ids (input); weights are outputs
        if topk_ids is None:
            topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
        topk_weights = torch.empty(
            num_tokens, top_k, dtype=torch.bfloat16, device=device
        )
    tile, config = tactic
    s0, s1 = shards
    delay = (
        tuple(int(x) for x in _debug_delay) if _debug_delay is not None else (-1, 0, 0)
    )
    res = mod.trtllm_fp4_block_scale_moe_localized(
        routing_input_mode,
        routing_logits,
        topk_ids,
        topk_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        s0["gemm1_weights"],
        s0["gemm1_weights_scale"],
        None,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        s0["gemm2_weights"],
        s0["gemm2_weights_scale"],
        None,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        activation_type,
        output,
        tile,
        config,
        norm_topk_prob,
        s1["gemm1_weights"],
        s1["gemm1_weights_scale"],
        s1["gemm2_weights"],
        s1["gemm2_weights_scale"],
        streams[0],
        streams[1],
        sms[0],
        sms[1],
        _mode,
        _return_intermediates,
        *delay,
        per_token_scale,
    )

    def conv(t):
        return t if isinstance(t, torch.Tensor) else torch.from_dlpack(t)

    # Like the stock wrapper: slot 0 of the finalized result and slot 1 of the unfinalized
    # result are placeholders; the real tensors are `output` and `topk_weights`.
    if do_finalize:
        out = [output]
        extra = [conv(t) for t in res[1:]]
    else:
        out = [conv(res[0]), topk_weights, conv(res[2])]
        extra = [conv(t) for t in res[3:]]
    return out + extra if _return_intermediates else out
