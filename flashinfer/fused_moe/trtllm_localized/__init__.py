"""Locality-domain localization of the trtllm-gen NVFP4 MoE (split-N across two locality
domains on SM107).

Not a public API on its own: the stock ops ``flashinfer.fused_moe.trtllm_fp4_block_scale_moe``
and ``trtllm_fp4_block_scale_routed_moe`` dispatch here when ``localized_weights`` is given.
moe.py holds the op body and the module loader, tuning.py the AutoTuner integration; the module
is built by ``flashinfer.jit.fused_moe.gen_trtllm_gen_fused_moe_localized_module``.
"""
