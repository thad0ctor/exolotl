"""NVFP4-GEMM training config schema.

Real low-precision COMPUTE (FP4 forward/backward GEMMs on Blackwell), distinct
from the fake-quant QAT/PTQ `quantization:` block.
"""

import warnings
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LMHeadDistillationConfig(BaseModel):
    """KL-distillation aux-loss for the FROZEN FP4 lm_head (accuracy recovery).

    When ``quantize_lm_head`` freezes the lm_head on its FP4 grid, the head adds
    ~9.5% uniform e2m1 mantissa noise (~21% argmax flips, all near-ties; +~0.04
    nats CE on an 8B head). The frozen weight cannot move off its grid, but the
    trainable body CAN adapt to it: an additive KL term against the ORIGINAL bf16
    head (retained as a frozen reference) teaches the body to produce hidden
    states whose FP4 logits match the bf16-head logits.

    Total loss = CE_student + lambda * T^2 * KL(softmax(z_teacher/T) || softmax(z_student/T)),
    where z_student are the FP4 head's logits and z_teacher = hidden @ W_bf16.T.

    The FP4 head's SPEED is the reason it exists, so every field here is a lever to
    bound the per-step teacher cost: ``top_k`` bounds the KL math, ``cadence``
    amortizes the (dominant) teacher matmul, and the teacher logits are always
    computed in vocab tiles (peak memory never the full [tokens, vocab]).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Enable the bf16-head KL-distillation aux-loss. OFF by "
            "default (opt-in). Requires nvfp4_training.quantize_lm_head: the teacher "
            "is the original bf16 head retained at swap time, and the student logits "
            "are the FP4 head's logits."
        },
    )
    lambda_: float = Field(
        default=1.0,
        ge=0.0,
        alias="lambda",
        json_schema_extra={
            "description": "Weight on the KL aux-loss (the `lambda` key in YAML). "
            "total = CE + lambda * T^2 * KL. 0.0 disables the term (same as "
            "enabled: false)."
        },
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        json_schema_extra={
            "description": "Softmax temperature T for both teacher and student in "
            "the KL term. The gradient is scaled by T^2 (Hinton distillation) so "
            "lambda stays comparable across T. Default 1.0 (match the argmax "
            "distribution directly)."
        },
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": "SPEED lever. If set, distill only the top-k teacher "
            "logits plus a single aggregated tail bucket (the remaining vocab mass "
            "as one outcome), bounding the KL math to k+1 terms. Null = full-vocab "
            "KL. Note the teacher matmul itself (over the full vocab) is the dominant "
            "cost; top_k bounds the softmax/KL work, not the matmul."
        },
    )
    cadence: int = Field(
        default=1,
        ge=1,
        json_schema_extra={
            "description": "SPEED lever. Apply the distillation loss only every N "
            "optimizer steps, amortizing the (dominant) teacher forward. 1 = every "
            "step. On off-steps the loss is the plain FP4 CE (zero teacher cost)."
        },
    )
    teacher: Literal["live", "precomputed"] = Field(
        default="live",
        json_schema_extra={
            "description": "'live' (default): recompute teacher logits each applied "
            "step as hidden @ W_bf16.T (tiled). 'precomputed': read offline-cached "
            "top-k teacher logits per token (near-zero per-step teacher cost) — the "
            "speed-optimal option. The precomputed path is a Phase-1 design stub "
            "(interface present, raises if selected without a cache)."
        },
    )
    teacher_vocab_block: int = Field(
        default=8192,
        ge=256,
        json_schema_extra={
            "description": "Vocab tile width for the tiled teacher/student logit "
            "computation (peak teacher memory is one [tokens, block] tile, never the "
            "full [tokens, vocab]). Tunable; not load-bearing for correctness."
        },
    )

    @model_validator(mode="after")
    def _check(self):
        if self.enabled and self.teacher == "precomputed":
            # Surface the stub clearly rather than failing deep in the trainer.
            import warnings

            warnings.warn(
                "lm_head_distillation.teacher='precomputed' is a Phase-1 design "
                "stub (no cache builder yet); the live teacher will be used at "
                "runtime. Set teacher: live to silence this.",
                stacklevel=2,
            )
        return self


class LMHeadResidualConfig(BaseModel):
    """Low-rank bf16 residual correction for the FROZEN FP4 lm_head (Phase 2).

    Stores ``A [V,k]`` / ``B [k,H]`` factors of the quant error
    ``E = W_bf16 - dequant(Q(W))`` computed ONCE at load (after the FP4 quant), so
    the head logits become ``Q(W) @ h + A @ (B @ h)``. The factors are threaded
    into every logit path (direct GEMM, fused CE tiles, distillation student
    logits), so the residual is applied consistently.

    The quant error is ~9.5% UNIFORM e2m1 rounding noise, so a PLAIN SVD of E is
    nearly full-rank and recovers almost nothing (measured: top-64 singular
    directions hold only ~2-10% of E's energy). The ACTIVATION-weighted (L2QER)
    construction weights E by the calibration activation second moment before the
    SVD, concentrating the *logit-relevant* error into a low-rank subspace
    (measured: top-16 directions hold ~56-74% of the weighted energy), and is the
    construction that actually shrinks the logit/CE gap — on a Qwen3-8B head it
    cut KL(bf16||fp4) by ~57% at rank 16 and closed the CE gap. Plain SVD is kept
    only as an ablation/diagnostic. OFF by default; opt-in, gated to
    quantize_lm_head.

    SPEED: this is a PERMANENT cost (train + inference). The cost is dominated by
    the [tokens, vocab] projection and is nearly rank-INDEPENDENT, so prefer the
    smallest rank that captures the gain (16 on these heads). Fused into the CE
    tile loop it adds ~10-15% to the head forward; as a bolt-on after a
    materialized GEMM it is ~36%. Not the "<1-2%" a small adapter would cost — the
    large vocab makes any second projection back to V expensive.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Enable the low-rank bf16 residual correction on the "
            "frozen FP4 lm_head. OFF by default (opt-in). Requires "
            "nvfp4_training.quantize_lm_head (the bf16 teacher weight is retained "
            "at swap to compute the quant error E)."
        },
    )
    rank: int = Field(
        default=32,
        ge=1,
        json_schema_extra={
            "description": "Rank k of the residual A[V,k]/B[k,H]. The cost is nearly "
            "rank-independent (the [tokens, vocab] projection dominates), so prefer "
            "the smallest k that captures the gain — 16 sufficed on Qwen3 0.6B/8B "
            "heads. Clamped to min(V,H)."
        },
    )
    calibration: Literal["activation", "svd"] = Field(
        default="activation",
        json_schema_extra={
            "description": "'activation' (default, L2QER): weight the quant error by "
            "the calibration activation second moment before the SVD, so the rank-k "
            "captures the directions that matter for the LOGITS — the only "
            "construction that meaningfully helps given the uniform error. 'svd': "
            "plain truncated SVD of the raw quant error (ablation; recovers little "
            "because the e2m1 error is near full-rank)."
        },
    )
    calib_tokens: int = Field(
        default=512,
        ge=16,
        json_schema_extra={
            "description": "Number of real tokens to forward once at load to capture "
            "the calibration hidden states for the activation weighting (only the "
            "per-hidden-dim second moment is used). Ignored when calibration='svd'."
        },
    )


class NVFP4AttentionBackwardConfig(BaseModel):
    """Native-NVFP4 attention training (backward) settings (expert recipe)."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Use the native NVFP4 autograd attention path while "
            "training. Validated for convergence but can be slower than bf16 at "
            "short sequence lengths, so it stays explicitly opt-in."
        },
    )
    rtn_grad_packs: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Deterministic round-to-nearest for the measured-safe "
            "gradient packs (softmax P / transposed dO for dV, dS for dQ), leaving "
            "the dK and dPt packs governed by stochastic_rounding. Faster in "
            "microbenchmarks; convergence validation still required."
        },
    )
    save_packs: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Save the forward pass's deterministic Q/K/V NVFP4 packs "
            "(+ transposed backward layouts) and reuse them in backward — trades "
            "activation memory for less backward pack-prep work. The compile custom "
            "op applies this only on short/mid static sequence lengths where the "
            "forward dual-pack cost measured worthwhile."
        },
    )
    dkdv_scratch_bf16: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store the per-query-head dK/dV scratch buffers in bf16 "
            "before GQA reduction instead of fp32 (less scratch traffic; bit-exact "
            "vs the fp32-then-cast path)."
        },
    )
    grad_dots: Literal["bf16", "fp4_rownorm", "fp8_rownorm", "fp4_legacy"] | None = (
        Field(
            default=None,
            json_schema_extra={
                "description": "Attention-backward grad-GEMM mode (the kernel's "
                "backward_grad_dots). 'bf16': FP4 S/dP recomputes + bf16 grad "
                "GEMMs with exact P/dS operands — the most accurate mode and the "
                "fastest below ~4k sequence (grad cosine vs bf16 SDPA ~0.991, "
                "~4x less backward scratch than legacy); needs the saved "
                "high-precision q/k/v, so it is incompatible with save_packs. "
                "'fp4_rownorm': all-FP4 grad GEMMs with per-row RMS-normalized "
                "gradient packs (underflow-proof e4m3 scales) — matches the "
                "bf16/hp backward on training quality (memorization 0.233 vs "
                "0.236, grad cos 0.97-0.98) and overtakes it in speed from ~4k "
                "sequence up: backward 1.12x hp / 1.13x FA2 at s8192 d256. RTN "
                "only (the SR grad knobs are no-ops — measured strictly worse "
                "under rownorm). 'fp8_rownorm': MXFP8 (e4m3 + e8m0 group-32) "
                "gradient-side GEMM operands; the accuracy-leaning fp4-family "
                "option (best memorization 0.230, grad cos 0.978-0.992, 1.08x hp "
                "at s8192 d256); RTN only, incompatible with save_packs. "
                "'fp4_legacy': the original all-FP4 backward (plain grad packs; "
                "honors rtn_grad_packs / stochastic-rounding) — kept for "
                "compatibility, rownorm dominates it on accuracy. null (default) "
                "= kernel AUTO: 'bf16' below an effective sequence of 4096 (the "
                "MEAN SAMPLE LENGTH for packed/varlen batches — the fp4 arms "
                "only beat the hp backward from ~4k up), else 'fp4_rownorm'. "
                "All modes compose with packed (varlen) batches."
            },
        )
    )
    bf16_grad_dots: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "DEPRECATED alias for grad_dots: true -> 'bf16', "
            "false -> 'fp4_legacy'. Setting both grad_dots and bf16_grad_dots "
            "is an error. Use grad_dots."
        },
    )
    compile_custom_op: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Route the native NVFP4 flash attention through an opaque "
            "torch custom op (torch.compile escape hatch). Tri-state: None "
            "auto-enables it whenever torch_compile is on; True/False force it. "
            "The op reuses forward LSE in backward and honors save_packs conditionally "
            "by carrying deterministic forward FP4 packs across the opaque boundary "
            "only for short/mid static sequence lengths."
        },
    )

    @model_validator(mode="after")
    def _check_grad_dots(self):
        if self.bf16_grad_dots is not None:
            if self.grad_dots is not None:
                raise ValueError(
                    "nvfp4_training.attention.backward: set either grad_dots or "
                    "the deprecated bf16_grad_dots alias, not both "
                    f"(got grad_dots={self.grad_dots!r}, "
                    f"bf16_grad_dots={self.bf16_grad_dots})."
                )
            warnings.warn(
                "nvfp4_training.attention.backward.bf16_grad_dots is deprecated; "
                "use grad_dots: 'bf16' (true) / 'fp4_legacy' (false) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.grad_dots = "bf16" if self.bf16_grad_dots else "fp4_legacy"
        if self.grad_dots in ("bf16", "fp8_rownorm") and self.save_packs:
            raise ValueError(
                f"nvfp4_training.attention.backward.grad_dots: '{self.grad_dots}' "
                "is incompatible with save_packs: true (the 'bf16' backward needs "
                "the saved high-precision q/k/v and 'fp8_rownorm' repacks Q^T/K^T "
                "as MXFP8 — both conflict with the saved forward FP4 pack set). "
                "Set save_packs: false or use grad_dots: 'fp4_rownorm' / "
                "'fp4_legacy'."
            )
        return self


class NVFP4AttentionConfig(BaseModel):
    """Native-NVFP4 full-attention path (model-agnostic; applied where the "
    architecture supports it). Replaces the flat ``qwen3_5_native_attention*`` flags."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch full softmax-attention layers to the native NVFP4 "
            "attention path on dense causal/full batches. Falls back to the model's "
            "configured attention for unsupported masks/cache states."
        },
    )
    fuse_vproj: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Run v_proj as a native NVFP4 GEMM with key-axis pack "
            "epilogue on inference/cache-free prefill (~1.4-1.5x V producer; v_proj "
            "goes FP4). Default (null): ON for inference, OFF for training."
        },
    )
    fp4_projections: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Run q/k/o_proj as native NVFP4 GEMMs (q/k share one "
            "activation pack) on inference prefill. Parity-affecting (not bit-exact); "
            "speed-neutral on hybrid models, a per-layer win on dense models. OFF by "
            "default; plain-nn.Linear only."
        },
    )
    packed_min_sample_len: int | None = Field(
        default=None,
        ge=0,
        json_schema_extra={
            "description": "Packed-batch (multipack) perf gate: packs whose MEAN "
            "sample length (pack tokens / number of samples) is below this keep "
            "the model's original FA2-varlen attention; at/above it they use FP4 "
            "varlen attention. Default None = AUTO per patch: 0 (gate OFF) for "
            "the Qwen3.5 text patch — with the fused RoPE+quant producers "
            "feeding the training forward pre-packed q/k and the varlen-tuned "
            "forward tiles, FP4 varlen beats FA2-varlen at every packed mean "
            "sample length measured (packed 8192, RTX PRO 6000; op-level "
            "rope+attn at mean 256: grad fwd 1.22x / fwd+bwd 1.22x at d256 "
            "h16/4 and 1.51x / 1.32x at d128 h32/8; >=1.0x at means 128-2048) — "
            "and 1024 for the Qwen3-VL patch, whose packed path does not yet "
            "use the pre-packed producers and measured 8-27% slower than "
            "FA2-varlen at head_dim 128 (its value there is numerical "
            "consistency with the FP4 forward, not speed). Set 0 to force FP4 "
            "varlen everywhere, a positive threshold to re-route short-mean "
            "packs to FA2, a very large value to never use FP4 for packed "
            "batches."
        },
    )
    backward: NVFP4AttentionBackwardConfig = Field(
        default_factory=NVFP4AttentionBackwardConfig,
        json_schema_extra={"description": "Native-NVFP4 attention training settings."},
    )
    allow_full_model: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Allow native NVFP4 attention on NON-hybrid models where "
            "EVERY layer is full softmax attention (e.g. Qwen3-VL). FP4 attention is "
            "~0.97 cosine per layer; on a hybrid model only a few layers are affected, "
            "but on a full-attention model the per-layer error compounds over all "
            "layers (~0.97^N) and destroys the forward (Qwen3-VL: step-1 loss ~5.3 vs "
            "~1.4). Refused by default for such models. Set True only if you have "
            "verified it is acceptable for your model."
        },
    )

    @model_validator(mode="after")
    def _check_requires(self):
        if not self.enabled:
            if self.backward.enabled:
                raise ValueError(
                    "nvfp4_training.attention.backward.enabled requires "
                    "attention.enabled: true."
                )
            if self.fuse_vproj:
                raise ValueError(
                    "nvfp4_training.attention.fuse_vproj requires attention.enabled: true."
                )
            if self.compile_custom_op_set():
                raise ValueError(
                    "nvfp4_training.attention.backward.compile_custom_op requires "
                    "attention.enabled: true."
                )
        if not self.backward.enabled:
            for sub in (
                "rtn_grad_packs",
                "save_packs",
                "dkdv_scratch_bf16",
                "grad_dots",
            ):
                if getattr(self.backward, sub):
                    raise ValueError(
                        f"nvfp4_training.attention.backward.{sub} requires "
                        "attention.backward.enabled: true."
                    )
        return self

    def compile_custom_op_set(self) -> bool:
        return bool(self.backward.compile_custom_op)

class BaseResidualConfig(BaseModel):
    """Low-rank bf16 L2QER residual on EVERY frozen FP4 base linear (LoRA/QLoRA).

    The generalization of ``lm_head_residual`` from the head to all frozen base
    linears: at swap time, while the bf16 master ``W`` and the packed ``Q(W)``
    coexist, the quant error ``E = W - dequant(Q(W))`` is factored
    activation-weighted (L2QER) into ``A [out,k] @ B [k,in]`` and the corrected
    base computes ``y = Q(W) x + A (B x)``. The weighting ``g = sqrt(mean_t
    x_t^2)`` per input channel comes from ONE forward of the still-bf16 model
    over a short slice of the training data, hooked on every linear about to be
    swapped — run BEFORE the quantize swap (the QLoRA storage mode discards the
    master right after packing).

    DEFAULT ON (deliberate product decision): the e2m1 base error is a pure
    accuracy loss the frozen base can never train away, the correction is exact
    in dgrad, and the memory cost is negligible (rank-16 A/B add ~(in+out)*k*2
    bytes per layer). MEASURED accuracy: logit KL(bf16||fp4) -27% at rank 16 on
    a fully FP4-swapped Qwen3-0.6B body. MEASURED speed: on the fused LoRA
    kernel path (lora_qkv/o/mlp_kernel — the production training path) the
    residual is FOLDED into the adapter GEMMs (it has exactly the LoRA shape;
    the concatenated factors are version-cached per module), so the marginal
    cost is ~0-3% wall / <=1% GPU-side: a lone rank-16-LoRA'd base linear's
    fwd+bwd lands within noise of no-residual (the old bolt-on was +12-18%),
    and the 0.6B whole-model LoRA fwd+bwd measures -0.4% eager / +0.1-2.7%
    compiled wall (GPU time +0.4% / +0.9%) vs the old bolt-on's +16-22%.
    Plain module forwards (no adapter riding the fused kernels) keep a
    bolt-on, now a single fused addmm per direction (~half the old bolt-on's
    memory traffic, though that path stays launch-bound — ~+20% whole-model
    without any adapters). Set ``enabled: false`` for
    pure-throughput runs where the FP4 base accuracy gap does not matter. It
    silently applies whenever frozen FP4 base linears exist (LoRA/QLoRA
    storage/compute modes); full fine-tune (trainable weights — the residual
    would go stale every optimizer step) and the hp base mode are skipped with
    a debug log, and the lm_head keeps its own ``lm_head_residual`` block.

    If the activation calibration cannot be captured at load, the residual is
    DISABLED with a loud warning — it does NOT silently degrade to plain SVD,
    which is measurably useless (the e2m1 error is near full-rank; see
    LMHeadResidualConfig / nvfp4_residual.py for the measurements).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Enable the low-rank L2QER residual on every frozen FP4 "
            "base linear. ON by default (product decision: it activates "
            "automatically whenever frozen FP4 base linears exist — LoRA/QLoRA "
            "with base_mode storage/compute). Measured: KL(bf16||fp4) -27% at "
            "rank 16 on a Qwen3-0.6B body, for ~0-3% whole-model fwd+bwd cost "
            "on the fused LoRA kernel path (the residual factors are "
            "concatenated into the LoRA adapter GEMMs and version-cached, so "
            "the marginal cost is just the extra rank; eager and compiled). Plain "
            "module-forward (no-adapter) paths pay a single fused addmm bolt-on "
            "instead. Set false for pure-throughput runs. Full fine-tune and "
            "hp-mode bases are never corrected (trainable/hp master — the "
            "residual would go stale)."
        },
    )
    rank: int = Field(
        default=16,
        ge=1,
        json_schema_extra={
            "description": "Rank k of the per-layer residual A[out,k]/B[k,in]. 16 "
            "captures nearly all of the recoverable (activation-weighted) error on "
            "real heads/bodies; the per-layer cost is two thin matmuls and "
            "~(in+out)*k*2 bytes of bf16 buffers. Clamped to min(out,in)."
        },
    )
    calibration: Literal["activation", "svd"] = Field(
        default="activation",
        json_schema_extra={
            "description": "'activation' (default, L2QER): weight the quant error "
            "by the per-input-channel activation second moment (captured in one "
            "pre-swap forward over calib_tokens training tokens) before the SVD — "
            "the only construction that meaningfully helps. If that calibration "
            "cannot be captured, the residual is DISABLED with a warning (no "
            "silent plain-SVD fallback). 'svd': plain truncated SVD of the raw "
            "error (ablation only; recovers little)."
        },
    )
    calib_tokens: int = Field(
        default=512,
        ge=16,
        json_schema_extra={
            "description": "Number of real tokens forwarded once at load (before "
            "the quantize swap) to capture per-input-channel activation second "
            "moments on every to-be-swapped linear. Only the [in_features] running "
            "mean of x^2 is kept per layer (tiny). Ignored for calibration='svd'."
        },
    )


class NVFP4TrainingConfig(BaseModel):
    """NVFP4-GEMM training settings (module-swap on eligible nn.Linear)."""

    # Reject unknown keys so stale or misspelled options fail loudly instead of
    # silently no-opping.
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable NVFP4-GEMM training (real FP4 forward/backward GEMMs). "
            "Blackwell-only. The speedup only materializes under torch.compile."
        },
    )
    backend: Literal["native", "te"] = Field(
        default="native",
        json_schema_extra={
            "description": "FP4 compute backend. 'native' (default): our "
            "torch._scaled_mm module-swap; runs the full SR+RHT convergence recipe "
            "on any FP4-capable GPU (sm_100/sm_120), no extra build. 'te': NVIDIA "
            "Transformer Engine NVFP4BlockScaling; faster hand-tuned kernels, but "
            "needs `pip install axolotl[transformer-engine]` (source build) and on "
            "consumer Blackwell (sm_120) its RHT/SR kernels do not run — TE there is "
            "recipe-less (convergence unproven). Supports FFT and LoRA/QLoRA, but on "
            "adapter paths it keeps a high-precision base (ignores "
            "base_mode/quantize_base, so no FP4-storage saving) and is incompatible "
            "with the fused LoRA kernels."
        },
    )
    stochastic_rounding: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Stochastic rounding on gradient operands (NVFP4Recipe)."
        },
    )
    hadamard: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Random Hadamard transform on wgrad inputs (NVFP4Recipe)."
        },
    )
    quantize_base: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Adapter modes only. When false (LoRA + FP4 compute): the "
            "frozen base weight stays high-precision and only the GEMM operands are "
            "FP4 — a throughput win, no memory saving. When true (NVFP4-QLoRA): the "
            "frozen base weight is stored packed in FP4 (~3.5x weight memory saving), "
            "replacing bnb NF4 storage. This is the FP4 equivalent of QLoRA, so it "
            "conflicts with load_in_4bit/load_in_8bit (drop those). `adapter: qlora` "
            "implies this (qlora == quantized base intent)."
        },
    )
    base_mode: Literal["compute", "storage", "hp"] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Adapter modes only; overrides quantize_base when set. "
            "'compute' (recommended): the frozen base is pre-quantized once into two "
            "NVFP4 layouts so fprop+dgrad run as pure FP4 GEMMs with no per-step base "
            "quant prologue — fastest base compute, ~1.75x weight memory. 'storage' "
            "(== NVFP4-QLoRA): base packed in FP4, ~3.5x memory, backward dequants to "
            "bf16 (max memory, modest speed). 'hp': base kept high-precision, "
            "re-quantized each step (FP4 GEMM, no memory win)."
        },
    )
    exclude_modules: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"],
        json_schema_extra={
            "description": "Module name fragments kept in high precision (not swapped)."
        },
    )
    quantize_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Remove lm_head from the high-precision exclusion and "
            "quantize the output projection to NVFP4 (memory + GEMM win on a large "
            "[vocab, hidden] tensor), at a small convergence cost (the NVFP4 paper "
            "keeps lm_head bf16). OFF by default. With UNtied embeddings only the "
            "lm_head is quantized (embed_tokens stays excluded unless "
            "quantize_embeddings is also set). With TIED embeddings (the frozen "
            "shared weight) the shared weight is quantized ONCE and routed to both "
            "the embedding lookup and the lm_head GEMM. A TRAINABLE tied weight "
            "still RAISES (FP4-storing it would corrupt training)."
        },
    )
    fused_fp4_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Requires quantize_lm_head. Fuse the FP4 lm_head "
            "projection with the cross-entropy loss so the full [batch*seq, vocab] "
            "logit tensor is never materialized (~1 GiB at vocab 152k / seq 8k): "
            "the loss is computed by tiling over the vocab, dequantizing each FP4 "
            "weight tile on read, and accumulating logsumexp in fp32 (like "
            "cut_cross_entropy, but reading the NVFP4-packed weight directly). "
            "This is a MEMORY win (no logit materialization), not an FP4-GEMM "
            "throughput win — the per-tile matmul runs in bf16/fp32, so the lm_head "
            "does not hit FP4 tensor cores. Frozen lm_head only (returns dL/dhidden, "
            "no weight grad). Falls back to the materialized path if the lm_head "
            "store isn't row-sliceable (MSLK-swizzled scales) or carries a bias. "
            "OFF by default."
        },
    )
    fused_ce_vocab_block: int = Field(
        default=4096,
        gt=0,
        json_schema_extra={
            "description": "Vocab-tile width for fused_fp4_cross_entropy. The fused "
            "CE streams the vocabulary in [tokens, fused_ce_vocab_block] tiles; this "
            "is a pure speed<->VRAM dial and is loss-invariant (the tile loop is a "
            "reduction split, so loss/grad are bit-stable across block widths). "
            "4096 (default) is balanced — lighter than a bf16 head's transient logit "
            "tile while still faster. 8192 is max throughput (~+0.4% speed for ~+2 "
            "GiB peak at long seq); wider buys nothing. Overridable at runtime via "
            "the AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK env var (env wins over this "
            "config). Only affects fused_fp4_cross_entropy."
        },
    )
    bf16_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy by tiling over the vocab in "
            "bf16, avoiding full [batch*seq, vocab] logits materialization (~1 GiB "
            "at vocab 152k / seq 8k) and the matching logits-gradient GEMM. The "
            "per-tile lm_head matmul runs in plain bf16 (bit-for-bit the same "
            "arithmetic as the materialized hidden @ W.t()); logsumexp/softmax and "
            "the dL/dhidden accumulation are kept in fp32. This is the exact tiled "
            "CE gradient (no low-probability vocab filtering), so it is "
            "convergence-safe under NVFP4 stochastic-rounding grads where the fused "
            "cut_cross_entropy / Liger paths collapsed. Returns dL/dhidden only (no "
            "lm_head weight grad). Incompatible with quantize_lm_head, "
            "fused_fp4_cross_entropy, and the FP8 cross-entropy patch. This is a "
            "MEMORY/backward-traffic win, not an FP4 tensor-core throughput win. "
            "OFF by default."
        },
    )
    fp8_lm_head: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen lm_head to use torch FP8 scaled "
            "matmul in eval/no-grad forward only. Training forwards still use the "
            "original high-precision Linear. This is for eval/scoring/logprob "
            "throughput; greedy generation can diverge after the first changed "
            "argmax token. OFF by default."
        },
    )
    fp8_lm_head_cross_entropy: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch a plain frozen bias-free nn.Linear lm_head "
            "training forward to compute cross-entropy with FP8 scaled-matmul vocab "
            "tiles, avoiding full [batch*seq, vocab] logits materialization. "
            "Returns dL/dhidden only (no lm_head weight grad); backward uses FP8 "
            "scaled matmul against a prepacked dgrad weight layout. Incompatible "
            "with quantize_lm_head, "
            "fused_fp4_cross_entropy, and other cross-entropy optimization patches. "
            "OFF by default."
        },
    )
    fp8_lm_head_granularity: Literal["tensorwise", "rowwise"] = Field(
        default="rowwise",
        json_schema_extra={
            "description": "Scaling granularity for fp8_lm_head. Rowwise keeps one "
            "scale per vocab row and had the best Qwen3.5 real-hidden-state argmax "
            "parity in the validation sweep; it is also the validated FP8 CE "
            "training default."
        },
    )
    fused_fp4_cross_entropy_fp4_matmul: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Experimental. When fused_fp4_cross_entropy is enabled, "
            "use per-tile FP4 torch._scaled_mm for lm_head logits before the tiled "
            "online logsumexp. This hits Blackwell FP4 tensor cores for the matmul, "
            "but still materializes each [tokens, vocab_block] tile and still uses "
            "a dequantized weight tile for dhidden in backward. Early microbenchmarks "
            "show it is faster than the memory-only FP4 CE path but slower than "
            "materialized bf16/Liger CE; keep this off unless benchmarking the "
            "next native CE-epilogue design."
        },
    )
    lm_head_distillation: LMHeadDistillationConfig = Field(
        default_factory=LMHeadDistillationConfig,
        json_schema_extra={
            "description": "KL-distillation aux-loss against the original bf16 head, "
            "to recover the accuracy the frozen FP4 lm_head loses. Only activates "
            "with quantize_lm_head. OFF by default. See LMHeadDistillationConfig."
        },
    )
    lm_head_residual: LMHeadResidualConfig = Field(
        default_factory=LMHeadResidualConfig,
        json_schema_extra={
            "description": "Low-rank bf16 residual correction A@B on the frozen FP4 "
            "lm_head (W ~= Q(W) + A B), computed once at load. Activation-weighted "
            "(L2QER) by default; recovers ~57% of the FP4-head KL gap at rank 16 on "
            "a Qwen3-8B head. Only activates with quantize_lm_head. OFF by default. "
            "PERMANENT inference cost (~10-15% of the head forward when fused). See "
            "LMHeadResidualConfig."
        },
    )
    base_residual: BaseResidualConfig = Field(
        default_factory=BaseResidualConfig,
        json_schema_extra={
            "description": "Low-rank bf16 L2QER residual A@B on EVERY frozen FP4 "
            "base linear (W ~= Q(W) + A B), built at swap time from the bf16 "
            "master and a one-shot activation calibration. ON by default "
            "(measured: KL -27% at rank 16 on a Qwen3-0.6B body for +16-22% "
            "whole-model fwd+bwd at that scale; exact dgrad, no wgrad); applies "
            "to LoRA/QLoRA storage/compute base modes only — full-FT/hp bases "
            "and the lm_head (own lm_head_residual block) are excluded. Disabled "
            "with a loud warning if the activation calibration cannot be "
            "captured. See BaseResidualConfig."
        },
    )
    quantize_embeddings: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store the input token embedding (embed_tokens) packed "
            "in FP4 and dequantize on lookup (W4A16; the lookup gathers rows, no "
            "activation quant). FROZEN only — a trainable embedding is skipped with "
            "a warning (no FP4 master for gradients). Hidden dim must be %16. OFF by "
            "default."
        },
    )
    quantize_vision_tower: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Multimodal only. Swap the FROZEN nn.Linear layers under "
            "the vision encoder (attn qkv/proj, mlp fc1/fc2) to the NVFP4 frozen "
            "base. The vision merger and patch-embed projection are left untouched, "
            "as are %32-ineligible dims. Warns if no vision tower is found (text "
            "model). OFF by default."
        },
    )
    fuse_rmsnorm: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Default off (throughput benefit is within noise; under "
            "torch.compile the reuse cache is bypassed). Fuse decoder RMSNorm with "
            "the NVFP4 activation quant in one Triton kernel so the qkv/gate-up base "
            "linears reuse the norm's pre-quantized activation. Auto-detects the "
            "gamma convention (plain `weight` vs zero-centered `1 + weight`) per norm "
            "and verifies each swap against the original before committing (reverts "
            "on mismatch)."
        },
    )
    fuse_qk_norm_rope: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Experimental, Qwen only. Monkeypatch qwen3/qwen3_5/"
            "qwen3_5_moe attention to fuse head-dim q_norm/k_norm with RoPE using "
            "the Gemma4 fused RMSNorm+RoPE Triton kernel. This removes the separate "
            "q/k RMSNorm and rotate_half/apply_rope materializations but leaves the "
            "attention matmul itself unchanged. OFF by default."
        },
    )
    shared_lora_base_fprop: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Experimental, LoRA/QLoRA native backend only. For fused "
            "LoRA QKV/QK projections whose frozen base layers are pre-quantized NVFP4 "
            "compute/storage bases, quantize and pack the shared activation once and "
            "reuse it across the base fprops. Omitted/null uses the "
            "AXOLOTL_NVFP4_SHARED_BASE_FPROP environment fallback (default off); "
            "set true or false in YAML to make the choice explicit. On Qwen3.5-9B "
            "with only 8 full-attention layers this was below whole-step noise, but "
            "models with more full-attention LoRA layers may benefit."
        },
    )
    fla_causal_conv_compile_boundary: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Qwen3.5/MoE sample-packing only. Run FLA varlen "
            "causal_conv1d behind a torch.compile boundary so packed cu_seqlens "
            "length changes do not trigger repeated Dynamo recompiles. Trades graph "
            "coverage for steadier steps."
        },
    )
    skip_first_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the first N transformer blocks in high precision. "
            "The NVFP4 paper keeps ~15% of linear layers in bf16 (embeddings/lm_head "
            "plus the first ~2 and last ~8 blocks of a 12B model, weighted to the "
            "tail) for convergence; for a model with L blocks a reasonable default is "
            "skip_first_n_blocks=1 and skip_last_n_blocks~=round(0.13*L)."
        },
    )
    skip_last_n_blocks: int = Field(
        default=0,
        ge=0,
        json_schema_extra={
            "description": "Keep the last N transformer blocks in high precision. "
            "See skip_first_n_blocks for the ~15% high-precision policy from the "
            "NVFP4 training paper (the tail blocks matter most)."
        },
    )
    save_nvfp4: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Store eligible weights NVFP4-packed (qdata + scales) in "
            "a torch.save sidecar (nvfp4_packed.pt) on every checkpoint/final save, "
            "for ~3.5x smaller weight files. safetensors cannot serialize the FP4 "
            "tensor subclass, so the packed weights go to a sidecar and the bf16 "
            "weights are dropped from the safetensors shard. For FROZEN weights "
            "(LoRA/QLoRA base, lm_head, embeddings) the FP4 IS the faithful stored "
            "form — load-back is bit-exact. For FFT (NVFP4Linear keeps a bf16 master) "
            "this is LOSSY for resume: only the FP4 packing is kept, no bf16 master, "
            "so a save_nvfp4 FFT checkpoint is for storage/inference export, not exact "
            "resume. OFF by default (bf16 save, unchanged)."
        },
    )
    attention: NVFP4AttentionConfig = Field(
        default_factory=NVFP4AttentionConfig,
        json_schema_extra={
            "description": "Native-NVFP4 full-attention path settings (model-agnostic; "
            "applied where the architecture supports it). Replaces the deprecated flat "
            "qwen3_5_native_attention* flags."
        },
    )
    linear_attn: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch linear-attention (e.g. GatedDeltaNet) large "
            "projection GEMMs to native NVFP4 in no-grad forward/eval. Training "
            "forwards fall back to the original implementation."
        },
    )
    mlp: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Patch dense SwiGLU MLP GEMMs to native NVFP4 in no-grad "
            "forward/eval. Training forwards fall back to the original implementation."
        },
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_attention_flags(cls, data):
        """Map deprecated flat ``qwen3_5_*`` attention flags onto the nested schema."""
        if not isinstance(data, dict):
            return data
        attn_map = {
            "qwen3_5_native_attention": ("enabled",),
            "qwen3_5_fuse_vproj": ("fuse_vproj",),
            "qwen3_5_native_attention_backward": ("backward", "enabled"),
            "qwen3_5_native_attention_backward_rtn_grad_packs": (
                "backward",
                "rtn_grad_packs",
            ),
            "qwen3_5_native_attention_save_backward_packs": ("backward", "save_packs"),
            "qwen3_5_native_attention_dkdv_scratch_bf16": (
                "backward",
                "dkdv_scratch_bf16",
            ),
            "qwen3_5_native_attention_compile_custom_op": (
                "backward",
                "compile_custom_op",
            ),
        }
        top_map = {
            "qwen3_5_native_linear_attn": "linear_attn",
            "qwen3_5_native_mlp": "mlp",
            "qwen3_5_fla_causal_conv_compile_boundary": "fla_causal_conv_compile_boundary",
        }
        used: list[str] = []
        attn = dict(data.get("attention") or {})
        bwd = dict(attn.get("backward") or {})
        for old, path in attn_map.items():
            if old not in data:
                continue
            used.append(old)
            val = data.pop(old)
            if len(path) == 1:
                attn.setdefault(path[0], val)  # explicit nested value wins
            else:
                bwd.setdefault(path[1], val)
        if bwd:
            attn["backward"] = bwd
        if attn:
            data["attention"] = attn
        for old, new in top_map.items():
            if old in data:
                used.append(old)
                data.setdefault(new, data.pop(old))
        if used:
            warnings.warn(
                "nvfp4_training: flat attention flags "
                f"{sorted(used)} are deprecated; use the nested "
                "nvfp4_training.attention.* schema instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data
