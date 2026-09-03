"""DeepSeek-V4 sparse-MLA decode with a fused FP8 + UE8M0 output (DeepGEMM layout)."""

from __future__ import annotations

import torch

from ..utils import check_shape_dtype_device, device_support_pdl, get_device_sm_count
from ._core import get_trtllm_gen_fmha_module

__all__ = [
    "dsv4_fp8_output_available",
    "dsv4_fp8_scale_buf_m",
    "trtllm_batch_decode_sparse_mla_dsv4_fp8",
]


def dsv4_fp8_scale_buf_m(sum_seq_q: int) -> int:
    """Token extent of the scale layout: ``sum_seq_q`` rounded up to 4 (DeepGEMM TMA alignment)."""
    return (sum_seq_q + 3) // 4 * 4


def dsv4_fp8_output_available(
    *,
    device: torch.device,
    batch_size: int,
    max_q_len: int,
    sum_seq_q: int,
    sparse_top_k: int,
    num_qo_heads: int,
) -> bool:
    """Whether :func:`trtllm_batch_decode_sparse_mla_dsv4_fp8` has a kernel for this problem.

    The fused kernel exists only for the 2-CTA schedule, which single-token decode at small
    batch does not select. Ask before allocating the destinations; on ``False`` use
    :func:`~flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` and quantize its BF16 output.
    """
    # The kernel registry is keyed on the current device.
    with torch.cuda.device(device):
        return bool(
            get_trtllm_gen_fmha_module().trtllm_dsv4_fp8_output_available(
                batch_size,
                max_q_len,
                sum_seq_q,
                sparse_top_k,
                num_qo_heads,
                get_device_sm_count(device),
            )
        )


def trtllm_batch_decode_sparse_mla_dsv4_fp8(
    query: torch.Tensor,
    primary_kv_cache: torch.Tensor,
    sliding_window_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    out_values: torch.Tensor,
    out_scales: torch.Tensor,
    bmm1_scale: float,
    bmm2_scale: float,
    *,
    max_q_len: int,
    enable_pdl: bool | None = None,
    sinks: torch.Tensor | None = None,
) -> None:
    r"""DSv4 sparse-MLA decode writing E4M3 values and packed UE8M0 scales in place.

    Inputs follow :func:`~flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4` (FP8 query
    and KV pools, contiguous HND pools). Raises when no fused kernel serves the problem;
    ask :func:`dsv4_fp8_output_available` first.

    Parameters
    ----------
    cos_sin_cache : torch.Tensor
        float32 ``[max_position, 64]``, each row ``cos(32) || sin(32)``. The inverse
        rotation of the trailing 64 lanes uses position
        ``seq_lens[b] - seq_len_q[b] + local_token``.
    out_values : torch.Tensor
        float8_e4m3fn ``[num_qo_heads // 8, sum_seq_q, 8, 512]``, contiguous.
    out_scales : torch.Tensor
        int32 ``[num_qo_heads // 8, 8, dsv4_fp8_scale_buf_m(sum_seq_q)]``, MN-major,
        contiguous. Each word packs one head's four block-128 UE8M0 exponents, block 0 in
        the least significant byte; a block dequantizes by ``2 ** (e - 127)``. Columns past
        ``sum_seq_q`` are not written.
    sinks : torch.Tensor, optional
        Attention sink logits, float32 ``[num_qo_heads]``.
    """
    sum_seq_q, num_qo_heads = query.size(0), query.size(1)
    if num_qo_heads % 8:
        raise ValueError(f"num_qo_heads must be a multiple of 8, got {num_qo_heads}")
    n_groups = num_qo_heads // 8
    check_shape_dtype_device(
        out_values,
        (n_groups, sum_seq_q, 8, 512),
        torch.float8_e4m3fn,
        query.device,
        "out_values",
    )
    check_shape_dtype_device(
        out_scales,
        (n_groups, 8, dsv4_fp8_scale_buf_m(sum_seq_q)),
        torch.int32,
        query.device,
        "out_scales",
    )
    if not (out_values.is_contiguous() and out_scales.is_contiguous()):
        raise ValueError("out_values and out_scales must be contiguous")
    if enable_pdl is None:
        enable_pdl = device_support_pdl(query.device)
    get_trtllm_gen_fmha_module().trtllm_paged_attention_decode_sparse_mla_dsv4_fp8(
        out_values,
        out_scales,
        query,
        primary_kv_cache,
        sliding_window_kv_cache,
        sparse_indices.reshape(sum_seq_q, -1).contiguous(),
        seq_lens.contiguous(),
        sparse_topk_lens.contiguous(),
        cum_seq_lens_q.contiguous(),
        cos_sin_cache.contiguous(),
        float(bmm1_scale),
        float(bmm2_scale),
        cum_seq_lens_q.numel() - 1,
        int(max_q_len),
        get_device_sm_count(query.device),
        bool(enable_pdl),
        sinks,
    )
