"""DeepSeek-V4 sparse-MLA attention with a DeepGEMM-ready FP8 output.

Attention kernel selection runs unchanged; an output producer is then attached to whatever
plan it chose -- the fused twin, the FP8 reduction specialization, or a standalone
quantizer. Which one served a launch is returned for tracing and nothing else.

FlashInfer never allocates the output. The caller owns both destinations and all layout
arithmetic::

    values = torch.empty((G, L, 8, 512), dtype=torch.float8_e4m3fn, device=dev)
    scales = torch.empty((G, 8, dsv4_fp8_scale_buf_m(L)), dtype=torch.int32, device=dev)
    trtllm_batch_decode_sparse_mla_dsv4_fp8(query, ..., values, scales, ...,
                                            token_capacity=L, ...)
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

__all__ = [
    "DSV4_COS_SIN_ROW_WIDTH",
    "DSV4_HEAD_DIM",
    "DSV4_HEADS_PER_GROUP",
    "DSV4_QUANT_BLOCK",
    "DSV4_SCALE_TOKEN_ALIGNMENT",
    "dsv4_fp8_scale_buf_m",
    "has_dsv4_fp8_output_abi",
    "make_cum_seq_lens_q",
    "trtllm_batch_decode_sparse_mla_dsv4_fp8",
]

DSV4_HEAD_DIM = 512
DSV4_HEADS_PER_GROUP = 8
DSV4_QUANT_BLOCK = 128
DSV4_COS_SIN_ROW_WIDTH = 64

# DeepGEMM's `get_tma_aligned_size(m, 4)` demands this of the MN-major scale backing's
# innermost (token) axis.
DSV4_SCALE_TOKEN_ALIGNMENT = 4

_PRODUCER_NAMES = ("fusion", "reduction", "standalone")


def dsv4_fp8_scale_buf_m(token_capacity: int) -> int:
    """Token extent of the scale backing: ``token_capacity`` rounded up to 4."""
    return (token_capacity + 3) & ~3


@functools.cache
def _module():
    from ._core import get_trtllm_gen_fmha_module

    op = get_trtllm_gen_fmha_module()
    if getattr(op, "trtllm_dsv4_fp8_run_oneshot", None) is None:
        raise RuntimeError(
            "trtllm_dsv4_fp8_run_oneshot is not available. Rebuild FlashInfer with "
            "csrc/trtllm_dsv4_fp8.cu."
        )
    return op


def has_dsv4_fp8_output_abi() -> bool:
    """Whether this build exposes the DSv4 FP8-output entry point.

    A symbol probe only; it says nothing about device or problem support.
    """
    try:
        _module()
        return True
    except (ImportError, RuntimeError):
        return False


def make_cum_seq_lens_q(
    q_lens: torch.Tensor, token_base: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Build the int32 ``[batch_size + 1]`` offsets the entry point expects.

    Values are batch-global and start at ``token_base``.
    """
    if device is None:
        device = q_lens.device
    out = torch.zeros(q_lens.numel() + 1, dtype=torch.int32, device=device)
    out[0] = token_base
    out[1:] = token_base + torch.cumsum(q_lens.to(device=device, dtype=torch.int32), 0)
    return out


def trtllm_batch_decode_sparse_mla_dsv4_fp8(
    query: torch.Tensor,
    primary_kv_cache: torch.Tensor,
    sliding_window_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    out_values_backing: torch.Tensor,
    out_scales_backing: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    *,
    max_q_len: int,
    token_base: int,
    token_count: int,
    total_tokens: int,
    token_capacity: int,
    enable_pdl: Optional[bool] = None,
    sinks: Optional[torch.Tensor] = None,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
) -> str:
    r"""DSv4 sparse MLA decode writing a DeepGEMM-ready FP8 pair.

    Parameters
    ----------
    query : torch.Tensor
        ``[sum_seq_q, num_qo_heads, 512]``, ``float8_e4m3fn``.
    primary_kv_cache, sliding_window_kv_cache : torch.Tensor
        Paged KV caches, ``float8_e4m3fn``, in the DSv4 sparse-MLA layout.
    workspace_buffer : torch.Tensor
        Scratch. Every producer except the fused twin needs a merged BF16 intermediate of
        ``sum_seq_q * num_qo_heads * 512 * 2`` bytes, so size for that worst case.
    sparse_indices, seq_lens, sparse_topk_lens : torch.Tensor
        Sparse-MLA metadata, as for :func:`trtllm_batch_decode_sparse_mla_dsv4`.
    cum_seq_lens_q : torch.Tensor
        int32 ``[batch_size + 1]``, batch-global, starting at ``token_base``. See
        :func:`make_cum_seq_lens_q`. ``batch_size`` is derived from its length.
    out_values_backing : torch.Tensor
        ``float8_e4m3fn`` ``[G, token_capacity, 8, 512]`` with ``G = num_qo_heads // 8``.
        Must be the allocation base: a token slice of a larger destination has a matching
        ``numel`` and the wrong group stride, and is rejected.
    out_scales_backing : torch.Tensor
        int32 ``[G, 8, dsv4_fp8_scale_buf_m(token_capacity)]``, MN-major. Each word packs
        one head's four block-128 UE8M0 exponents, block 0 in the LSB. Allocation base, as
        above.
    cos_sin_cache : torch.Tensor
        float32 ``[max_position, 64]``, each row ``cos(32) || sin(32)``, contiguous.
    bmm1_scale, bmm2_scale : float
        Attention scales.
    max_q_len : int
        Longest query in the batch.
    token_base, token_count, total_tokens, token_capacity : int
        The launch's span into the destination. ``token_capacity`` is the physical extent
        and the FP8 value-group stride; it exceeds ``total_tokens`` only under graph
        padding. Rows ``[total_tokens, token_capacity)`` are consumed by the GEMM at
        ``M = token_capacity`` and are the caller's to initialize -- uninitialized E4M3 is
        typically ``0x7F`` (NaN).
    enable_pdl : bool, optional
        Defaults to :func:`device_support_pdl`, as the BF16 sibling does.
    sinks : torch.Tensor, optional
        Attention sinks.
    multi_ctas_kv_counter_buffer : torch.Tensor, optional
        Reuse a held counter buffer; the kernel self-resets it. Allocated here if omitted.

    Returns
    -------
    str
        The producer that served the launch: ``"fusion"``, ``"reduction"`` or
        ``"standalone"``. Tracing only -- branching on it in model code recreates the
        kernel-selection heuristic this entry point exists to hide.
    """
    from ..utils import (
        _get_trtllm_gen_multi_ctas_kv_counter_buffer,
        device_support_pdl,
        get_device_sm_count,
    )

    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    batch_size = cum_seq_lens_q.numel() - 1
    sm_count = get_device_sm_count(query.device)
    if multi_ctas_kv_counter_buffer is None:
        multi_ctas_kv_counter_buffer = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
            batch_size, query.size(1), sm_count, query.device
        )
    producer = _module().trtllm_dsv4_fp8_run_oneshot(
        int(batch_size),
        int(max_q_len),
        int(token_base),
        int(token_count),
        int(total_tokens),
        int(token_capacity),
        query,
        primary_kv_cache,
        sliding_window_kv_cache,
        workspace_buffer,
        multi_ctas_kv_counter_buffer,
        sparse_indices,
        seq_lens,
        sparse_topk_lens,
        cum_seq_lens_q,
        out_values_backing,
        out_scales_backing,
        cos_sin_cache,
        float(bmm1_scale),
        float(bmm2_scale),
        int(sm_count),
        bool(enable_pdl),
        workspace_buffer.numel() * workspace_buffer.element_size(),
        sinks,
    )
    return _PRODUCER_NAMES[int(producer)]
