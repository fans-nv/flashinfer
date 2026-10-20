"""AutoTuner integration for the localized trtllm-gen NVFP4 MoE (routed and monolithic).

Tactic selection goes through FlashInfer's own AutoTuner, like the stock FP4 MoE: a
TunableRunner whose tactics are (tile, config) pairs, profiled under autotune(True) and looked up
from the cache otherwise.

- Tactics are the partition-valid set only: configs valid for both half shapes
  (FC1 N = 2 * I/2, K = H; FC2 N = H/2, K = I) at the per-partition SM count
  (MoE::Runner::getValidPartitionConfigIndices), for every tile the module carries. With
  localized_allow_nonlocalized=True the full-width tactics are not added yet (v1); only the op
  name records the policy, so later full-width records cannot be confused with these.
- Identity: op names of the form ``<stock op>::Localized`` / ``::AdaptiveLocalized`` (#5276's
  naming), plus cache-key extras covering finalize mode, PDL, routing input mode, problem shape,
  the localization signature (#5276's form: per-partition SM counts, shard shapes, and a
  memset-stream flag, always False here since the stream is unused), the module build identity
  (bundle manifest sha256 + GEN export version, reported by the module itself) and a
  fingerprint of its (tile, config index) -> kernel-name map. Mode Q (per-token FC1) runners
  add a marker and fingerprint the per-token catalog, so M3 and M3.1 records never mix. Token
  bucketing and initializers are the stock FP4 MoE ones (per-token scales profile as ones).
- Protocol: use_cold_l2_cache=True and use_cuda_graph=True, the stock FP4 MoE settings. Only
  the dynamic inputs are ever re-synthesized for profiling; the shards are keyword arguments,
  so they stay in their domain-local memory during profiling.
- Pinned token counts (moe.DEFAULT_TACTIC_PINS) always use the pin, tuned or not. No cached
  record: the first valid tactic. A cached tactic outside the partition-valid set is rejected
  and the default is used. Each path is logged once per (op, T, path, tactic).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    autotuner_initializer_rand,
    autotuner_initializer_randn,
)
from ..shared.tuning import moe_topk_ids_init
from ..utils import get_hybrid_num_tokens_buckets, make_hybrid_bucket_mapper
from . import moe as _moe

logger = logging.getLogger("flashinfer.fused_moe.trtllm_localized")

_STOCK_OP = {
    0: "flashinfer::trtllm_fp4_block_scale_moe",  # routing from logits
    1: "flashinfer::trtllm_fp4_block_scale_routed_moe",  # packed top-k
    2: "flashinfer::trtllm_fp4_block_scale_routed_moe",  # unpacked top-k
}
PROTOCOL = "trtllm-localized-tune-v1"
# Pins were measured in-chain, where isolated profiling picks slower decode tactics.
PINS_OVERRIDE_TUNING = True


def custom_op(routing_input_mode: int, allow_nonlocalized: bool) -> str:
    policy = "AdaptiveLocalized" if allow_nonlocalized else "Localized"
    return f"{_STOCK_OP[routing_input_mode]}::{policy}"


# Dynamic-input order (a prefix of the stock MoeRunnerInputs order, fused_moe/shared/inputs.py),
# plus the per-token scale, which is present only in mode Q.
F_OUTPUT, F_LOGITS, F_TOPK_IDS, F_EXPERT_W, F_HIDDEN, F_HIDDEN_SCALE, F_PER_TOKEN = (
    range(7)
)

# Last resolution per (op, num_tokens): ("pinned" | "tuned" | "default" | "rejected", tactic). Both
# tables are cleared when they reach _MAX_RECORDS (prefill T varies per call).
last_resolution: Dict[Tuple[str, int], Tuple[str, Tuple[int, int]]] = {}
_logged: set = set()
_MAX_RECORDS = 4096


def module_identity() -> Tuple[str, str]:
    """(bundle manifest sha256, GEN export version) of the loaded module; part of every cache
    key, so records follow the kernel catalog rather than the .so file."""
    return _moe.module_info()


class LocalizedFp4MoeRunner(TunableRunner):
    """Tactic = (tile, config) from the partition-valid set; -1 = the default tactic."""

    def __init__(
        self,
        *,
        routing_input_mode: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        activation_type: int,
        do_finalize: bool,
        enable_pdl: bool,
        sm_counts: Tuple[int, ...],
        signature: Tuple[Any, ...],
        allow_nonlocalized: bool = False,
        per_token: bool = False,
    ):
        self.routing_input_mode = routing_input_mode
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.activation_type = activation_type
        self.do_finalize = bool(do_finalize)
        self.enable_pdl = bool(enable_pdl)
        self.sm_counts = tuple(int(x) for x in sm_counts)
        self.signature = signature
        self.allow_nonlocalized = bool(allow_nonlocalized)
        self.per_token = bool(per_token)
        self.op = custom_op(routing_input_mode, self.allow_nonlocalized)
        self._identity: Optional[Tuple[Any, ...]] = None
        self._tuning_configs: Dict[Tuple[int, Tuple[bool, ...]], TuningConfig] = {}
        self._valid_cache: Dict[int, Tuple[List[Tuple[int, int]], frozenset]] = {}

    # ---- identity -------------------------------------------------------------------
    def identity(self) -> Tuple[Any, ...]:
        if self._identity is None:
            self._identity = (
                PROTOCOL,
                self.do_finalize,
                self.enable_pdl,
                self.routing_input_mode,
                self.hidden_size,
                self.intermediate_size,
                self.num_experts,
                self.num_local_experts,
                self.top_k,
                self.activation_type,
                self.signature,
                module_identity(),
                _moe.catalog_fingerprint(self.activation_type, self.per_token),
            ) + (("per_token_fc1",) if self.per_token else ())
        return self._identity

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> Tuple[Any, ...]:
        return self.identity()

    def __hash__(self) -> int:
        return hash(self.identity())

    # ---- tactics --------------------------------------------------------------------
    def _valid(self, num_tokens: int) -> Tuple[List[Tuple[int, int]], frozenset]:
        hit = self._valid_cache.get(num_tokens)
        if hit is None:
            out = []
            for tile in _moe._TILES:
                for cfg in _moe._valid_configs(
                    tile,
                    self.top_k,
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_local_experts,
                    num_tokens,
                    self.sm_counts[0],
                    self.sm_counts[1],
                    self.activation_type,
                    self.per_token,
                ):
                    out.append((tile, cfg))
            if len(self._valid_cache) >= _MAX_RECORDS:
                self._valid_cache.clear()
            hit = self._valid_cache[num_tokens] = (out, frozenset(out))
        return hit

    def valid_tactics(self, num_tokens: int) -> List[Tuple[int, int]]:
        return list(self._valid(num_tokens)[0])

    def is_valid_tactic(self, tactic: Tuple[int, int], num_tokens: int) -> bool:
        return tactic in self._valid(num_tokens)[1]

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile, **kwargs
    ) -> List[Tuple[int, int]]:
        return self.valid_tactics(inputs[F_HIDDEN].shape[0])

    def default_tactic(self, num_tokens: int) -> Tuple[int, int]:
        return _moe.default_tactic(
            num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts,
            self.sm_counts,
            self.activation_type,
            self.per_token,
        )

    def pinned_tactic(self, num_tokens: int) -> Optional[Tuple[int, int]]:
        """The pinned default for this token count if it is partition-valid, else None."""
        pin = _moe.pinned_default(num_tokens, self.activation_type, self.per_token)
        if pin is not None and self.is_valid_tactic(pin[1], num_tokens):
            return pin[1]
        return None

    def check_tactic(self, tactic, num_tokens: int) -> Tuple[int, int]:
        t = (int(tactic[0]), int(tactic[1]))
        if not self.is_valid_tactic(t, num_tokens):
            raise ValueError(
                f"tactic {t} is not in the partition-valid set for T={num_tokens} "
                f"(SMs {self.sm_counts}): {self.valid_tactics(num_tokens)}"
            )
        return t

    # ---- tuning config ----------------------------------------------------------------
    def tuning_config(
        self, inputs: List[Optional[torch.Tensor]], tune_max_num_tokens: int
    ) -> TuningConfig:
        key = (tune_max_num_tokens, tuple(t is not None for t in inputs))
        cfg = self._tuning_configs.get(key)
        if cfg is None:
            cfg = self._tuning_configs[key] = self._build_tuning_config(
                inputs, tune_max_num_tokens
            )
        return cfg

    def _build_tuning_config(
        self, inputs: List[Optional[torch.Tensor]], tune_max_num_tokens: int
    ) -> TuningConfig:
        spec = {
            F_OUTPUT: autotuner_initializer_empty,
            F_HIDDEN: autotuner_initializer_randn,
            F_HIDDEN_SCALE: autotuner_initializer_ones,
        }
        if inputs[F_LOGITS] is not None:
            spec[F_LOGITS] = autotuner_initializer_rand
        if inputs[F_TOPK_IDS] is not None:
            spec[F_TOPK_IDS] = moe_topk_ids_init(
                self.num_experts, packed=self.routing_input_mode != 2
            )
        if inputs[F_EXPERT_W] is not None:
            spec[F_EXPERT_W] = autotuner_initializer_ones
        if len(inputs) > F_PER_TOKEN and inputs[F_PER_TOKEN] is not None:
            spec[F_PER_TOKEN] = autotuner_initializer_ones
        idx = tuple(sorted(spec))
        return TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    idx,
                    tuple(0 for _ in idx),
                    get_hybrid_num_tokens_buckets(tune_max_num_tokens, 1),
                    make_hybrid_bucket_mapper(tune_max_num_tokens),
                ),
            ),
            tensor_initializers=tuple((i, spec[i]) for i in idx),
            use_cold_l2_cache=True,
            use_cuda_graph=True,
        )

    # ---- execution ----------------------------------------------------------------------
    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> Any:
        num_tokens = inputs[F_HIDDEN].shape[0]
        t = (
            self.default_tactic(num_tokens)
            if tactic == -1
            else self.check_tactic(tactic, num_tokens)
        )
        return _moe.trtllm_fp4_block_scale_moe_localized(
            routing_input_mode=self.routing_input_mode,
            routing_logits=inputs[F_LOGITS],
            topk_ids=inputs[F_TOPK_IDS],
            topk_weights=inputs[F_EXPERT_W],
            routing_bias=kwargs["routing_bias"],
            hidden_states=inputs[F_HIDDEN],
            hidden_states_scale=inputs[F_HIDDEN_SCALE],
            gemm1_bias=None,
            gemm1_alpha=kwargs["gemm1_alpha"],
            gemm1_beta=kwargs["gemm1_beta"],
            gemm1_clamp_limit=kwargs["gemm1_clamp_limit"],
            gemm2_bias=None,
            output1_scale_scalar=kwargs["output1_scale_scalar"],
            output1_scale_gate_scalar=kwargs["output1_scale_gate_scalar"],
            output2_scale_scalar=kwargs["output2_scale_scalar"],
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=kwargs["n_group"],
            topk_group=kwargs["topk_group"],
            intermediate_size=self.intermediate_size,
            local_expert_offset=kwargs["local_expert_offset"],
            local_num_experts=self.num_local_experts,
            routed_scaling_factor=kwargs["routed_scaling_factor"],
            localized_weights=kwargs["localized_weights"],
            localized_streams=kwargs["localized_streams"],
            localized_sm_count=list(self.sm_counts),
            routing_method_type=kwargs["routing_method_type"],
            do_finalize=self.do_finalize,
            enable_pdl=self.enable_pdl,
            activation_type=self.activation_type,
            output=inputs[F_OUTPUT],
            norm_topk_prob=kwargs["norm_topk_prob"],
            hidden_states_scale_layout=kwargs["hidden_states_scale_layout"],
            per_token_scale=inputs[F_PER_TOKEN] if len(inputs) > F_PER_TOKEN else None,
            per_token_scale_gemm2=False,
            tactic=t,
        )


_runners: Dict[Tuple[Any, ...], LocalizedFp4MoeRunner] = {}


def get_runner(**kw) -> LocalizedFp4MoeRunner:
    """One runner per (problem, mode, localization): its identity and tuning configs are
    computed once instead of per call."""
    key = tuple(sorted(kw.items()))
    r = _runners.get(key)
    if r is None:
        if len(_runners) >= _MAX_RECORDS:
            _runners.clear()
        r = _runners[key] = LocalizedFp4MoeRunner(**kw)
    return r


def _log_once(op: str, num_tokens: int, path: str, tactic, detail: str = "") -> None:
    if len(last_resolution) >= _MAX_RECORDS or len(_logged) >= _MAX_RECORDS:
        last_resolution.clear()
        _logged.clear()
    last_resolution[(op, num_tokens)] = (path, tuple(tactic))
    key = (op, num_tokens, path, tuple(tactic))
    if key in _logged:
        return
    _logged.add(key)
    logger.info(
        "[localized MoE] %s T=%d tactic=%s via %s%s",
        op,
        num_tokens,
        tuple(tactic),
        path,
        detail,
    )


def select_tactic(
    runner: LocalizedFp4MoeRunner,
    inputs: List[Optional[torch.Tensor]],
    static_kwargs: Dict[str, Any],
    tune_max_num_tokens: int,
) -> Tuple[int, int]:
    """The pinned default where one covers the token count; otherwise profiles under
    autotune(True), else a cache lookup, else the default when no record."""
    op = runner.op
    num_tokens = inputs[F_HIDDEN].shape[0]
    if PINS_OVERRIDE_TUNING:
        pin = runner.pinned_tactic(num_tokens)
        if pin is not None:
            _log_once(op, num_tokens, "pinned", pin)
            return pin
    cfg = runner.tuning_config(inputs, tune_max_num_tokens)
    _, tactic = AutoTuner.get().choose_one(op, [runner], cfg, inputs, **static_kwargs)
    if tactic != -1 and tactic is not None:
        t = (int(tactic[0]), int(tactic[1]))
        if runner.is_valid_tactic(t, num_tokens):
            _log_once(op, num_tokens, "tuned", t)
            return t
        fb = runner.default_tactic(num_tokens)
        logger.error(
            "[localized MoE] %s T=%d: cached tactic %s is not partition-valid; rejected, "
            "using default %s",
            op,
            num_tokens,
            t,
            fb,
        )
        _log_once(op, num_tokens, "rejected", fb, f" (rejected {t})")
        return fb
    fb = runner.default_tactic(num_tokens)
    _log_once(op, num_tokens, "default", fb, " (no autotuner record)")
    return fb
