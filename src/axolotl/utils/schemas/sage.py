"""SageAttention-3 FP4 (sage_fp4) inference-backend config schema.

Knobs for the opt-in ``attn_implementation: sage_fp4`` path (the SageAttention-3
FP4 microscaling attention forward). Independent of the ``nvfp4_training`` block —
this configures attention only, for inference/eval and serving Attn-QAT models.
"""

from pydantic import BaseModel, Field


class SageAttentionConfig(BaseModel):
    """SageAttention-3 FP4 attention settings (sage_fp4)."""

    per_block_mean: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Subtract a per-128-block query mean before FP4 "
            "quantization (Sage-3 smoothing). On by default; matches the kernel's "
            "default and improves accuracy on outlier-heavy activations."
        },
    )
    allow_fallback: bool = Field(
        default=True,
        json_schema_extra={
            "description": "When an input can't use the FP4 kernel (per-key padding, "
            "sliding window, custom softmax scale, or head_dim>=256) fall back to a "
            "high-precision kernel for a correct result (default): flash attention for "
            "dense causal/full masks (covers head_dim<=256, so Qwen3.5 causal prefill), "
            "SDPA only for masks flash can't represent. Set false to RAISE instead — "
            "useful when benchmarking to guarantee every call runs the FP4 kernel "
            "rather than silently degrading to high precision."
        },
    )
