"""Input-validation guards for the DSv4 fused UE8M0 FP8-output epilogue.

These are argument-contract tests: every case must fail before any kernel is
launched, so none of them needs a correct DSv4 problem, only enough of one to
reach the guard under test. The numerical behaviour of the epilogue is covered
by the DSv4 sparse-MLA suites; what is covered here is the failure modes that
are silent if the guard is missing.

The backend cases matter most. ``trtllm_batch_decode_sparse_mla_dsv4`` grew a
``backend`` selector (auto / trtllm-gen / cute-dsl / sparse) after the fused
epilogue was written, and the fused destination only exists in the TRTLLM-GEN
cubins. Without an explicit refusal the cute-dsl and sparse paths accept
``out_block_scale``, return BF16 in ``out``, and never write the scale tensor --
so the caller hands an uninitialized UE8M0 exponent to DeepGEMM and gets
plausible-looking garbage rather than an error.
"""

import pytest
import torch

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.utils import get_compute_capability

HEAD_DIM = 512
NUM_QO_HEADS = 128
HEADS_PER_GROUP = 8


def _skip_unless_sm100_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DSv4 sparse MLA tests")
    cc = get_compute_capability(torch.device("cuda"))
    if cc not in ((10, 0), (10, 3)):
        pytest.skip(f"DSv4 fused FP8 output requires SM100/SM103, got SM{cc[0]}{cc[1]}")


def _minimal_fused_kwargs(device: torch.device, num_tokens: int = 4) -> dict:
    """Just enough to reach the guards. These buffers are never read."""
    n_groups = NUM_QO_HEADS // HEADS_PER_GROUP
    scale_buf_m = (num_tokens + 3) // 4 * 4
    return dict(
        query=torch.zeros(
            num_tokens, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        ),
        # Required positionally, but the guards under test fire before either is
        # read, so their contents are irrelevant.
        swa_kv_cache=torch.zeros(
            1, 1, HEAD_DIM, dtype=torch.float8_e4m3fn, device=device
        ),
        workspace_buffer=torch.zeros(1024, dtype=torch.uint8, device=device),
        out=torch.zeros(
            n_groups,
            num_tokens,
            HEADS_PER_GROUP,
            HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        out_block_scale=torch.zeros(
            n_groups, HEADS_PER_GROUP, scale_buf_m, dtype=torch.int32, device=device
        ),
        cos_sin_cache=torch.zeros(num_tokens, 64, dtype=torch.float32, device=device),
        scale_buf_m=scale_buf_m,
    )


@pytest.mark.parametrize("backend", ["cute-dsl", "sparse"])
def test_fused_output_rejected_on_non_trtllm_gen_backend(backend: str) -> None:
    """The fused destination must be refused, not silently ignored.

    ``sparse`` additionally fails its own SM120 arch check on this hardware; both
    are acceptable refusals, so the assertion only requires that *some*
    ValueError is raised and that nothing is written to the scale tensor.
    """
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    kwargs = _minimal_fused_kwargs(device)
    scales = kwargs["out_block_scale"]
    scales.fill_(0x5A5A5A5A)
    sentinel = scales.clone()

    with pytest.raises(ValueError):
        trtllm_batch_decode_sparse_mla_dsv4(backend=backend, **kwargs)

    # The refusal must happen before anything touches the destination.
    torch.testing.assert_close(scales, sentinel, rtol=0, atol=0)


def test_fused_output_accepted_on_trtllm_gen_backend_reaches_later_validation() -> None:
    """backend='trtllm-gen' must NOT be rejected by the backend guard itself.

    A negative control for the test above: if the guard were written as
    ``backend != 'auto'`` or similar it would also refuse the one backend that
    does support the epilogue, and the tests above would still pass. The call
    is expected to fail on a *different*, later validation (the required sparse
    MLA tensors are absent here) -- what must not appear is the backend refusal.
    """
    _skip_unless_sm100_family()
    device = torch.device("cuda")

    with pytest.raises((ValueError, TypeError, RuntimeError)) as excinfo:
        trtllm_batch_decode_sparse_mla_dsv4(
            backend="trtllm-gen", **_minimal_fused_kwargs(device)
        )
    assert "requires backend='trtllm-gen'" not in str(excinfo.value)
