"""Schema, gate, and (GPU-gated) end-to-end tests for nvfp4_training."""

import pytest

import axolotl.utils.nvfp4_training as nvfp4_mod
from axolotl.utils.schemas.config import (
    AxolotlConfigWCapabilities,
    AxolotlInputConfig,
)

BASE = {
    "base_model": "Qwen/Qwen2.5-0.5B",
    "datasets": [{"path": "tatsu-lab/alpaca", "type": "alpaca"}],
    "learning_rate": 1e-5,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "num_epochs": 1,
}

# Capability gates live on AxolotlConfigWCapabilities; supply the two capability
# blocks so the gate validator runs without touching real hardware.
CAPS = {
    "capabilities": {
        "bf16": True,
        "fp8": True,
        "n_gpu": 1,
        "compute_capability": "sm_120",
    },
    "env_capabilities": {"torch_version": "2.8.0"},
}


def _supported(monkeypatch, ok, reason=""):
    monkeypatch.setattr(nvfp4_mod, "nvfp4_supported", lambda: (ok, reason))


def test_schema_accepts_valid_nvfp4_config(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "stochastic_rounding": True,
            "hadamard": True,
            "exclude_modules": ["lm_head", "embed_tokens"],
            "skip_first_n_blocks": 1,
            "skip_last_n_blocks": 1,
        },
    )
    assert cfg.nvfp4_training.enabled is True
    assert cfg.nvfp4_training.skip_first_n_blocks == 1


def test_schema_backend_defaults_native_and_accepts_te(monkeypatch):
    _supported(monkeypatch, True)
    assert (
        AxolotlInputConfig(
            **BASE, nvfp4_training={"enabled": True}
        ).nvfp4_training.backend
        == "native"
    )
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True, "backend": "te"})
    assert cfg.nvfp4_training.backend == "te"


def test_schema_accepts_shared_lora_base_fprop(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={"enabled": True, "shared_lora_base_fprop": True},
    )
    assert cfg.nvfp4_training.shared_lora_base_fprop is True

    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True})
    assert cfg.nvfp4_training.shared_lora_base_fprop is None


def test_schema_accepts_fp8_lm_head_eval_knobs(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "fp8_lm_head": True,
            "fp8_lm_head_cross_entropy": True,
            "fp8_lm_head_granularity": "rowwise",
        },
    )
    assert cfg.nvfp4_training.fp8_lm_head is True
    assert cfg.nvfp4_training.fp8_lm_head_cross_entropy is True
    assert cfg.nvfp4_training.fp8_lm_head_granularity == "rowwise"


def test_schema_refuses_fp8_lm_head_ce_with_other_ce(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Only one cross entropy optimization"):
        AxolotlInputConfig(
            **BASE,
            cut_cross_entropy=True,
            nvfp4_training={
                "enabled": True,
                "fp8_lm_head_cross_entropy": True,
            },
        )


def test_disabled_fp8_lm_head_ce_does_not_conflict(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        chunked_cross_entropy=True,
        nvfp4_training={
            "enabled": False,
            "fp8_lm_head_cross_entropy": True,
        },
    )
    assert cfg.chunked_cross_entropy is True
    assert cfg.nvfp4_training.enabled is False


def test_gate_refuses_fp8_lm_head_ce_with_quantized_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="fp8_lm_head_cross_entropy"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": True,
                "fp8_lm_head_cross_entropy": True,
            },
        )


def test_schema_accepts_bf16_lm_head_ce(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={"enabled": True, "bf16_lm_head_cross_entropy": True},
    )
    assert cfg.nvfp4_training.bf16_lm_head_cross_entropy is True


def test_schema_refuses_bf16_lm_head_ce_with_other_ce(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Only one cross entropy optimization"):
        AxolotlInputConfig(
            **BASE,
            cut_cross_entropy=True,
            nvfp4_training={"enabled": True, "bf16_lm_head_cross_entropy": True},
        )


def test_gate_refuses_bf16_lm_head_ce_with_quantized_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="bf16_lm_head_cross_entropy"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": True,
                "bf16_lm_head_cross_entropy": True,
            },
        )


def test_schema_lm_head_distillation_defaults_off(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True})
    distill = cfg.nvfp4_training.lm_head_distillation
    assert distill.enabled is False
    assert distill.cadence == 1
    assert distill.temperature == 1.0
    assert distill.teacher == "live"


def test_schema_lm_head_distillation_parses_lambda_alias(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "quantize_lm_head": True,
            "lm_head_distillation": {
                "enabled": True,
                "lambda": 2.0,
                "temperature": 1.5,
                "top_k": 128,
                "cadence": 4,
                "teacher": "live",
            },
        },
    )
    distill = cfg.nvfp4_training.lm_head_distillation
    assert distill.lambda_ == 2.0
    assert distill.top_k == 128
    assert distill.cadence == 4


def test_gate_refuses_lm_head_distillation_without_quantize_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="lm_head_distillation requires"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": False,
                "lm_head_distillation": {"enabled": True},
            },
        )


def test_gate_allows_lm_head_distillation_with_quantize_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        nvfp4_training={
            "enabled": True,
            "quantize_lm_head": True,
            "fused_fp4_cross_entropy": True,
            "lm_head_distillation": {"enabled": True, "lambda": 1.0, "top_k": 64},
        },
    )
    assert cfg.nvfp4_training.lm_head_distillation.enabled is True


def test_schema_lm_head_residual_defaults_off(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True})
    res = cfg.nvfp4_training.lm_head_residual
    assert res.enabled is False
    assert res.rank == 32
    assert res.calibration == "activation"


def test_gate_refuses_lm_head_residual_without_quantize_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="lm_head_residual requires"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "quantize_lm_head": False,
                "lm_head_residual": {"enabled": True},
            },
        )


def test_gate_allows_lm_head_residual_with_quantize_lm_head(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        nvfp4_training={
            "enabled": True,
            "quantize_lm_head": True,
            "fused_fp4_cross_entropy": True,
            "lm_head_residual": {
                "enabled": True,
                "rank": 16,
                "calibration": "activation",
            },
        },
    )
    assert cfg.nvfp4_training.lm_head_residual.enabled is True
    assert cfg.nvfp4_training.lm_head_residual.rank == 16


def test_schema_accepts_qwen3_5_native_switches(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        model_config_type="qwen3_5",
        nvfp4_training={
            "enabled": True,
            "attention": {
                "enabled": True,
                "fuse_vproj": True,
                "fp4_projections": True,
                "packed_min_sample_len": 256,
                "backward": {
                    "enabled": True,
                    "rtn_grad_packs": True,
                    "save_packs": True,
                    "dkdv_scratch_bf16": True,
                },
            },
            "linear_attn": True,
            "mlp": True,
            "fla_causal_conv_compile_boundary": True,
        },
    )
    a = cfg.nvfp4_training.attention
    assert a.enabled is True and a.fuse_vproj is True and a.fp4_projections is True
    assert a.packed_min_sample_len == 256
    assert a.backward.enabled is True and a.backward.rtn_grad_packs is True
    assert a.backward.save_packs is True and a.backward.dkdv_scratch_bf16 is True
    assert cfg.nvfp4_training.linear_attn is True and cfg.nvfp4_training.mlp is True
    assert cfg.nvfp4_training.fla_causal_conv_compile_boundary is True


def test_schema_packed_min_sample_len_default_and_bounds(monkeypatch):
    """Packed perf gate knob: defaults to None = AUTO per patch (0/gate-off for
    the Qwen3.5 text patch, where the pre-packed training forward beats
    FA2-varlen at every measured packed mean; 1024 for the VL patch, whose
    packed path lacks the pre-packed producers and measured slower at d128),
    accepts a positive threshold, rejects negatives."""
    _supported(monkeypatch, True)
    base = {**BASE, "model_config_type": "qwen3_5"}
    cfg = AxolotlInputConfig(
        **base, nvfp4_training={"enabled": True, "attention": {"enabled": True}}
    )
    assert cfg.nvfp4_training.attention.packed_min_sample_len is None
    cfg0 = AxolotlInputConfig(
        **base,
        nvfp4_training={
            "enabled": True,
            "attention": {"enabled": True, "packed_min_sample_len": 1024},
        },
    )
    assert cfg0.nvfp4_training.attention.packed_min_sample_len == 1024
    with pytest.raises(ValueError):
        AxolotlInputConfig(
            **base,
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "packed_min_sample_len": -1},
            },
        )


def test_schema_migrates_legacy_attention_flags(monkeypatch):
    _supported(monkeypatch, True)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "qwen3_5_native_attention": True,
                "qwen3_5_native_attention_backward": True,
                "qwen3_5_native_attention_save_backward_packs": True,
                "qwen3_5_native_mlp": True,
                "qwen3_5_fla_causal_conv_compile_boundary": True,
            },
        )
    a = cfg.nvfp4_training.attention
    assert a.enabled is True and a.backward.enabled is True
    assert a.backward.save_packs is True
    assert cfg.nvfp4_training.mlp is True
    assert cfg.nvfp4_training.fla_causal_conv_compile_boundary is True
    assert any("deprecated" in str(x.message) for x in w)


def test_gate_refuses_qwen3_5_switch_on_other_model(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="model_config_type"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen2",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True},
            },
        )


@pytest.mark.parametrize("mct", ["qwen3_vl", "qwen3_5", "qwen3_5_moe"])
def test_gate_allows_nvfp4_attention_on_supported_models(monkeypatch, mct):
    # The native NVFP4 attention path (attention.enabled + backward.*) is allowed
    # on Qwen3.5/MoE AND Qwen3-VL (each has its own forward patch).
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        model_config_type=mct,
        nvfp4_training={
            "enabled": True,
            "attention": {
                "enabled": True,
                "backward": {"enabled": True, "save_packs": True},
            },
        },
    )
    assert cfg.nvfp4_training.attention.enabled is True


def test_gate_refuses_nvfp4_attention_on_unsupported_model(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="nvfp4_training.attention requires"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="llama",
            nvfp4_training={"enabled": True, "attention": {"enabled": True}},
        )


@pytest.mark.parametrize(
    "attn_extra",
    [{"fuse_vproj": True}, {"fp4_projections": True}],
)
def test_gate_refuses_qwen3_5_only_attn_flags_on_qwen3_vl(monkeypatch, attn_extra):
    # fuse_vproj / fp4_projections are Qwen3.5/MoE-only fast paths; not on Qwen3-VL.
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Qwen3.5/MoE-only"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_vl",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, **attn_extra},
            },
        )


@pytest.mark.parametrize("flag", ["linear_attn", "mlp"])
def test_gate_refuses_qwen3_5_only_module_flags_on_qwen3_vl(monkeypatch, flag):
    # DeltaNet linear_attn / Qwen3.5 MLP are Qwen3.5/MoE-only; not on Qwen3-VL.
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="Qwen3.5/MoE-only"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_vl",
            nvfp4_training={"enabled": True, flag: True},
        )
def test_schema_base_residual_defaults_on(monkeypatch):
    """base_residual is the DEFAULT-ON product decision: a bare nvfp4_training
    block already carries enabled=True, rank 16, activation calibration."""
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(**BASE, nvfp4_training={"enabled": True})
    res = cfg.nvfp4_training.base_residual
    assert res.enabled is True
    assert res.rank == 16
    assert res.calibration == "activation"
    assert res.calib_tokens == 512


def test_schema_base_residual_accepts_overrides(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        nvfp4_training={
            "enabled": True,
            "base_residual": {
                "enabled": False,
                "rank": 32,
                "calibration": "svd",
                "calib_tokens": 128,
            },
        },
    )
    res = cfg.nvfp4_training.base_residual
    assert res.enabled is False
    assert res.rank == 32
    assert res.calibration == "svd"
    assert res.calib_tokens == 128


def test_schema_base_residual_rejects_bad_values(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError):
        AxolotlInputConfig(
            **BASE,
            nvfp4_training={"enabled": True, "base_residual": {"rank": 0}},
        )
    with pytest.raises(ValueError):
        AxolotlInputConfig(
            **BASE,
            nvfp4_training={
                "enabled": True,
                "base_residual": {"calibration": "magic"},
            },
        )
    with pytest.raises(ValueError):  # extra=forbid catches stale keys
        AxolotlInputConfig(
            **BASE,
            nvfp4_training={"enabled": True, "base_residual": {"ranks": 16}},
        )


def test_gate_allows_base_residual_with_lora(monkeypatch):
    """Default-on must not break the capability gate for adapter runs."""
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, adapter="lora", nvfp4_training={"enabled": True}
    )
    assert cfg.nvfp4_training.base_residual.enabled is True
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        adapter="lora",
        nvfp4_training={
            "enabled": True,
            "base_mode": "storage",
            "base_residual": {"rank": 8},
        },
    )
    assert cfg.nvfp4_training.base_residual.rank == 8


def test_gate_refuses_unsupported_hardware(monkeypatch):
    _supported(monkeypatch, False, "no Blackwell here")
    with pytest.raises(ValueError, match="no Blackwell here"):
        AxolotlConfigWCapabilities(**BASE, **CAPS, nvfp4_training={"enabled": True})


def test_gate_allows_lora(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE, **CAPS, adapter="lora", nvfp4_training={"enabled": True}
    )
    assert cfg.adapter == "lora"
    assert cfg.nvfp4_training.quantize_base is False


def test_gate_allows_lora_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        adapter="lora",
        nvfp4_training={"enabled": True, "quantize_base": True},
    )
    assert cfg.nvfp4_training.quantize_base is True


def test_gate_refuses_unsupported_adapter(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="lora, qlora"):
        AxolotlConfigWCapabilities(
            **BASE, **CAPS, adapter="llama-adapter", nvfp4_training={"enabled": True}
        )


def test_gate_refuses_load_in_4bit_with_quantize_base(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="load_in_4bit"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            adapter="lora",
            load_in_4bit=True,
            nvfp4_training={"enabled": True, "quantize_base": True},
        )


def test_gate_refuses_deepspeed(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="DeepSpeed"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            deepspeed="deepspeed_configs/zero3.json",
            nvfp4_training={"enabled": True},
        )


def test_gate_refuses_fp16(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match="does not support fp16"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            fp16=True,
            bf16=False,
            nvfp4_training={"enabled": True},
        )


def test_disabled_nvfp4_skips_gate(monkeypatch):
    _supported(monkeypatch, False, "should not be raised")
    cfg = AxolotlConfigWCapabilities(**BASE, **CAPS, nvfp4_training={"enabled": False})
    assert cfg.nvfp4_training.enabled is False


def test_gate_refuses_qwen3_5_backward_without_attention(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"requires attention\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            nvfp4_training={
                "enabled": True,
                "attention": {"backward": {"enabled": True}},
            },
        )


def test_gate_refuses_qwen3_5_rtn_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"requires attention\.backward\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "backward": {"rtn_grad_packs": True}},
            },
        )


def test_gate_refuses_qwen3_5_saved_packs_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"requires attention\.backward\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "backward": {"save_packs": True}},
            },
        )


def test_gate_refuses_qwen3_5_dkdv_scratch_bf16_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"requires attention\.backward\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "backward": {"dkdv_scratch_bf16": True}},
            },
        )


@pytest.mark.parametrize(
    "mode", ["bf16", "fp4_rownorm", "fp8_rownorm", "fp4_legacy", None]
)
def test_schema_accepts_grad_dots_modes(monkeypatch, mode):
    """grad_dots accepts the four kernel modes; None (default) = kernel AUTO
    ('bf16' below an effective sequence of 4096 — the mean sample length for
    packed batches — else 'fp4_rownorm')."""
    _supported(monkeypatch, True)
    cfg = AxolotlInputConfig(
        **BASE,
        model_config_type="qwen3_5",
        nvfp4_training={
            "enabled": True,
            "attention": {
                "enabled": True,
                "backward": (
                    {"enabled": True, "grad_dots": mode}
                    if mode is not None
                    else {"enabled": True}
                ),
            },
        },
    )
    assert cfg.nvfp4_training.attention.backward.grad_dots == mode


def test_schema_refuses_unknown_grad_dots_mode(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"grad_dots"):
        AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {"enabled": True, "grad_dots": "fp16"},
                },
            },
        )


def test_schema_accepts_bf16_grad_dots_alias(monkeypatch):
    """The DEPRECATED boolean alias maps onto grad_dots: True -> 'bf16',
    False -> 'fp4_legacy'."""
    _supported(monkeypatch, True)
    with pytest.warns(DeprecationWarning, match=r"bf16_grad_dots"):
        cfg = AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {"enabled": True, "bf16_grad_dots": True},
                },
            },
        )
    assert cfg.nvfp4_training.attention.backward.grad_dots == "bf16"
    with pytest.warns(DeprecationWarning, match=r"bf16_grad_dots"):
        cfg2 = AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {"enabled": True, "bf16_grad_dots": False},
                },
            },
        )
    assert cfg2.nvfp4_training.attention.backward.grad_dots == "fp4_legacy"


def test_schema_refuses_grad_dots_and_alias_both_set(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"grad_dots.*bf16_grad_dots"):
        AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {
                        "enabled": True,
                        "grad_dots": "bf16",
                        "bf16_grad_dots": True,
                    },
                },
            },
        )


@pytest.mark.parametrize(
    "backward_extra",
    [
        {"grad_dots": "bf16"},  # hp backward needs the saved hp q/k/v
        {"grad_dots": "fp8_rownorm"},  # MXFP8 Q^T/K^T repack needs hp q/k
        {"bf16_grad_dots": True},  # alias maps to "bf16" -> same conflict
    ],
)
def test_schema_refuses_grad_dots_with_save_packs(monkeypatch, backward_extra):
    # 'bf16' needs the saved HP q/k/v and 'fp8_rownorm' repacks Q^T/K^T as
    # MXFP8; both conflict with the packs-only saved set save_packs keeps.
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"grad_dots.*save_packs"):
        AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {
                        "enabled": True,
                        "save_packs": True,
                        **backward_extra,
                    },
                },
            },
        )


def test_schema_allows_rownorm_and_legacy_grad_dots_with_save_packs(monkeypatch):
    # fp4_rownorm / fp4_legacy only change the GRADIENT-side packing; they
    # compose with the saved forward FP4 pack set.
    _supported(monkeypatch, True)
    for mode in ("fp4_rownorm", "fp4_legacy"):
        cfg = AxolotlInputConfig(
            **BASE,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {
                    "enabled": True,
                    "backward": {
                        "enabled": True,
                        "save_packs": True,
                        "grad_dots": mode,
                    },
                },
            },
        )
        assert cfg.nvfp4_training.attention.backward.grad_dots == mode


def test_gate_refuses_grad_dots_without_backward(monkeypatch):
    _supported(monkeypatch, True)
    with pytest.raises(ValueError, match=r"requires attention\.backward\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "backward": {"grad_dots": "bf16"}},
            },
        )
    # The deprecated alias hits the same gate (it maps onto grad_dots).
    with pytest.raises(ValueError, match=r"requires attention\.backward\.enabled"):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"enabled": True, "backward": {"bf16_grad_dots": True}},
            },
        )


def test_qwen3_5_compile_custom_op_allowed_with_backward(monkeypatch):
    # The compile custom op is now a differentiable, training-compatible op
    # (register_autograd), so it is ALLOWED together with native_attention_backward
    # (the prior inference-only rejection gate was removed).
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        model_config_type="qwen3_5",
        nvfp4_training={
            "enabled": True,
            "attention": {
                "enabled": True,
                "backward": {"enabled": True, "compile_custom_op": True},
            },
        },
    )
    assert cfg.nvfp4_training.attention.backward.compile_custom_op is True
    assert cfg.nvfp4_training.attention.backward.enabled is True


def test_qwen3_5_compile_custom_op_requires_native_attention(monkeypatch):
    # The remaining dependency gate: compile_custom_op needs attention.enabled on.
    _supported(monkeypatch, True)
    with pytest.raises(
        ValueError, match=r"compile_custom_op requires attention\.enabled"
    ):
        AxolotlConfigWCapabilities(
            **BASE,
            **CAPS,
            model_config_type="qwen3_5",
            nvfp4_training={
                "enabled": True,
                "attention": {"backward": {"compile_custom_op": True}},
            },
        )


def test_qwen3_5_compile_custom_op_autoenabled_under_torch_compile(monkeypatch):
    # Tri-state default (None): under torch_compile the bare tl.dot_scaled flash
    # kernel raises an Inductor CompilationError and silently falls back to eager,
    # so the opaque custom op is auto-enabled.
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        model_config_type="qwen3_5",
        torch_compile=True,
        nvfp4_training={
            "enabled": True,
            "attention": {"enabled": True, "backward": {"enabled": True}},
        },
    )
    assert cfg.nvfp4_training.attention.backward.compile_custom_op is True


def test_qwen3_5_compile_custom_op_default_off_without_torch_compile(monkeypatch):
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        model_config_type="qwen3_5",
        nvfp4_training={
            "enabled": True,
            "attention": {"enabled": True, "backward": {"enabled": True}},
        },
    )
    assert cfg.nvfp4_training.attention.backward.compile_custom_op is False


def test_qwen3_5_compile_custom_op_explicit_optout_under_torch_compile(monkeypatch):
    # An explicit False must survive auto-resolution (opt-out wins).
    _supported(monkeypatch, True)
    cfg = AxolotlConfigWCapabilities(
        **BASE,
        **CAPS,
        model_config_type="qwen3_5",
        torch_compile=True,
        nvfp4_training={
            "enabled": True,
            "attention": {
                "enabled": True,
                "backward": {"enabled": True, "compile_custom_op": False},
            },
        },
    )
    assert cfg.nvfp4_training.attention.backward.compile_custom_op is False


def _tiny_lora_model():
    """A 2-layer toy model wrapped with a PEFT LoRA adapter (CPU-friendly)."""
    from peft import LoraConfig, get_peft_model
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

        def forward(self, x):
            for blk in self.layers:
                x = blk.v_proj(blk.q_proj(x))
            return x

    base = Net()
    lcfg = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], lora_alpha=8)
    return get_peft_model(base, lcfg)


def _patch_manager(cfg_dict):
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    return PatchManager(DictDefault(cfg_dict), model_config=DictDefault({}))


# --- Residual calibration plumbing (regression: the standard `axolotl train`
# path must thread the prepared train dataset to the PatchManager so the NVFP4
# residual calibration resolves REAL input_ids at swap time; previously neither
# `PatchManager.train_dataset` nor `cfg._train_dataset` was ever populated and
# the base residual silently DISABLED / the lm_head residual fell back to
# plain SVD). ---


def test_calib_input_ids_resolve_from_threaded_train_dataset():
    """PatchManager built with a prepared dataset must calibrate from its first
    sample, not the deterministic arange fallback."""
    import torch
    from datasets import Dataset
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils.dict import DictDefault

    sample = list(range(1000, 1064))
    ds = Dataset.from_dict({"input_ids": [sample, list(range(7))]})
    pm = PatchManager(
        DictDefault({}), model_config=DictDefault({}), train_dataset=ds
    )

    class _Model:
        # Tiny vocab: the arange fallback would emit ids < 8, so any value from
        # `sample` (>= 1000) proves the real dataset path was taken.
        config = DictDefault({"vocab_size": 8})

    ids = pm._nvfp4_calib_input_ids(_Model(), 32, torch.device("cpu"))
    assert ids is not None
    assert ids.shape == (1, 32)
    assert ids[0].tolist() == sample[:32]


def test_calib_input_ids_fallback_without_dataset():
    """No dataset threaded -> the deterministic in-vocab range stays the last
    resort (never None for a model with a vocab)."""
    import torch
    from axolotl.utils.dict import DictDefault

    pm = _patch_manager({})

    class _Model:
        config = DictDefault({"vocab_size": 8})

    ids = pm._nvfp4_calib_input_ids(_Model(), 16, torch.device("cpu"))
    assert ids is not None
    assert ids.max().item() < 8  # in-vocab arange, not garbage


def test_model_loader_threads_train_dataset_to_patch_manager(monkeypatch):
    """ModelLoader(train_dataset=...) must hand the dataset to its PatchManager."""
    from datasets import Dataset

    import axolotl.loaders.model as model_mod
    from axolotl.utils.dict import DictDefault

    monkeypatch.setattr(
        model_mod, "load_model_config", lambda cfg: DictDefault({})
    )
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4]]})
    loader = model_mod.ModelLoader(
        DictDefault({"base_model": "dummy"}), tokenizer=None, train_dataset=ds
    )
    assert loader.patch_manager.train_dataset is ds


def test_setup_model_and_trainer_threads_prepared_dataset(monkeypatch):
    """The trainer-like flow: setup_model_and_trainer must pass
    dataset_meta.train_dataset into setup_model_and_tokenizer (the prepared
    dataset exists BEFORE the model load / NVFP4 swap in the standard path)."""
    from datasets import Dataset

    import axolotl.train as train_mod
    from axolotl.common.datasets import TrainDatasetMeta
    from axolotl.utils.dict import DictDefault

    captured = {}

    class _Stop(Exception):
        pass

    def fake_setup(cfg, train_dataset=None):
        captured["train_dataset"] = train_dataset
        raise _Stop

    monkeypatch.setattr(train_mod, "setup_model_and_tokenizer", fake_setup)
    ds = Dataset.from_dict({"input_ids": [[5, 6, 7]]})
    meta = TrainDatasetMeta(train_dataset=ds)
    with pytest.raises(_Stop):
        train_mod.setup_model_and_trainer(DictDefault({}), meta)
    assert captured["train_dataset"] is ds


def test_plain_lora_module_forwards_masks_and_restores_fused_routing():
    """The calibration contextmanager must reroute every fused-LoRA helper
    through plain module calls (so the residual-calibration nn.Module hooks
    fire) and restore the fused routing verbatim on exit."""
    import types

    import torch
    import torch.nn as nn

    from axolotl.kernels.lora import apply_lora_mlp_swiglu
    from axolotl.monkeypatch.lora_kernels import plain_lora_module_forwards

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 4, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)
            self.o_proj = nn.Linear(4, 4, bias=False)

    class GDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(4, 4, bias=False)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(4, 4, bias=False)
            self.up_proj = nn.Linear(4, 4, bias=False)
            self.down_proj = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()
            self.gdn = GDN()
            self.mlp = MLP()

    model = Model()

    def fused_marker(self, x):  # stands in for the Triton-kernel helpers
        return "FUSED"

    model.attn.apply_qkv = types.MethodType(fused_marker, model.attn)
    model.attn.apply_o = types.MethodType(fused_marker, model.attn)
    model.attn._lora_batch_kernel = True
    model.gdn.apply_in_proj_qkv = types.MethodType(fused_marker, model.gdn)
    model.mlp.forward = types.MethodType(apply_lora_mlp_swiglu, model.mlp)

    fired = []
    handles = [
        m.register_forward_pre_hook(lambda *_a, _n=n: fired.append(_n))
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
    ]
    x = torch.randn(2, 4)
    with plain_lora_module_forwards(model):
        assert not hasattr(model.attn, "_lora_batch_kernel")
        assert not hasattr(model.gdn, "apply_in_proj_qkv")
        assert "forward" not in model.mlp.__dict__
        # apply_qkv/apply_o stay CALLABLE (the source-rewritten attention
        # forward calls them unconditionally) but route via the modules.
        model.attn.apply_qkv(x)
        model.attn.apply_o(x)
        model.mlp(x)
        model.gdn.in_proj_qkv(x)  # what _la_proj_fwd falls back to

    for handle in handles:
        handle.remove()
    assert set(fired) >= {
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "gdn.in_proj_qkv",
    }
    # Fused routing restored verbatim.
    assert model.attn.apply_qkv(x) == "FUSED"
    assert model.attn.apply_o(x) == "FUSED"
    assert model.attn._lora_batch_kernel is True
    assert model.gdn.apply_in_proj_qkv(x) == "FUSED"
    assert model.mlp.__dict__["forward"].__func__ is apply_lora_mlp_swiglu


def test_capture_calib_hidden_uses_lm_head_pre_hook():
    """lm_head calibration capture must take the head's INPUT via a forward
    pre-hook — robust to model-output wrappers without ``last_hidden_state``
    (regression: Qwen3_5CausalLMOutputWithPast broke the output spelunking)."""
    import torch.nn as nn

    from axolotl.utils.dict import DictDefault

    class Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = DictDefault({"vocab_size": 16})
            self.emb = nn.Embedding(16, 4)
            self.lm_head = nn.Linear(4, 16, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids):
            h = self.emb(input_ids)
            logits = self.lm_head(h)

            class _WrapperOutput:  # no last_hidden_state, not a tuple/list
                pass

            out = _WrapperOutput()
            out.logits = logits
            return out

    pm = _patch_manager({})
    h = pm._nvfp4_capture_calib_hidden(Wrapped(), 8)
    assert h is not None
    assert h.shape == (8, 4)


def test_apply_qwen3_5_native_attention_forwards_saved_pack_flag(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.monkeypatch.attention import nvfp4_flash_attn

    captured = {}

    def fake_patch(_model, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(nvfp4_flash_attn, "patch_qwen3_5_nvfp4_attention", fake_patch)
    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "nvfp4_training": {
                "enabled": True,
                "stochastic_rounding": True,
                "attention": {
                    "enabled": True,
                    "backward": {
                        "enabled": True,
                        "save_packs": True,
                        "dkdv_scratch_bf16": True,
                    },
                },
            },
        }
    )

    pm._apply_qwen3_5_native_nvfp4_patches(object())

    assert captured["save_backward_packs"] is True
    assert captured["dkdv_scratch_bf16"] is True
    assert captured["train_backward"] is True


def test_apply_qwen3_vl_routes_to_vl_attention_not_qwen3_5(monkeypatch):
    # qwen3_vl must dispatch to the VL patch (standard attn + mRoPE), NOT the
    # Qwen3.5 gated patch, and forward the backward/SR knobs. fuse_vproj/
    # fp4_projections are Qwen3.5-only and must not be passed to the VL patch.
    _supported(monkeypatch, True)
    from axolotl.monkeypatch.attention import nvfp4_flash_attn, nvfp4_flash_attn_vl

    vl_captured = {}

    def fake_vl(_model, **kwargs):
        vl_captured.update(kwargs)
        return 36

    def fail_qwen35(*_a, **_k):  # pragma: no cover - must not be called
        raise AssertionError("qwen3_vl must not call the Qwen3.5 attention patch")

    monkeypatch.setattr(
        nvfp4_flash_attn_vl, "patch_qwen3_vl_nvfp4_attention", fake_vl
    )
    monkeypatch.setattr(
        nvfp4_flash_attn, "patch_qwen3_5_nvfp4_attention", fail_qwen35
    )
    pm = _patch_manager(
        {
            "model_config_type": "qwen3_vl",
            "nvfp4_training": {
                "enabled": True,
                "stochastic_rounding": True,
                "attention": {
                    "enabled": True,
                    # VL native attention is refused by default (degrades non-hybrid
                    # models); the override is required to exercise the routing.
                    "allow_full_model": True,
                    "backward": {
                        "enabled": True,
                        "rtn_grad_packs": True,
                        "save_packs": True,
                        "dkdv_scratch_bf16": True,
                    },
                },
            },
        }
    )

    pm._apply_qwen3_5_native_nvfp4_patches(object())

    assert vl_captured["train_backward"] is True
    assert vl_captured["save_backward_packs"] is True
    assert vl_captured["dkdv_scratch_bf16"] is True
    assert vl_captured["backward_rtn_grad_packs"] is True
    assert vl_captured["stochastic_rounding"] is True
    # VL patch has no fuse_vproj / fp4_projections params.
    assert "fuse_vproj" not in vl_captured
    assert "fuse_attn_proj" not in vl_captured


def test_apply_qwen3_vl_skips_attention_when_disabled(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.monkeypatch.attention import nvfp4_flash_attn_vl

    called = {"n": 0}

    def fake_vl(_model, **_kwargs):
        called["n"] += 1
        return 0

    monkeypatch.setattr(
        nvfp4_flash_attn_vl, "patch_qwen3_vl_nvfp4_attention", fake_vl
    )
    pm = _patch_manager(
        {
            "model_config_type": "qwen3_vl",
            "nvfp4_training": {"enabled": True, "attention": {"enabled": False}},
        }
    )
    pm._apply_qwen3_5_native_nvfp4_patches(object())
    assert called["n"] == 0


def test_qwen3_5_packing_patch_forwards_fla_compile_boundary(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.loaders import patch_manager
    from axolotl.monkeypatch.models.qwen3_5 import modeling

    captured = []

    def fake_patch(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(patch_manager.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(modeling, "patch_qwen3_5_modeling_packing", fake_patch)

    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "sample_packing": True,
            "bf16": True,
        }
    )

    pm._apply_model_specific_patches()

    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "sample_packing": True,
            "nvfp4_training": {
                "enabled": True,
                "fla_causal_conv_compile_boundary": True,
            },
        }
    )

    pm._apply_model_specific_patches()

    pm = _patch_manager(
        {
            "model_config_type": "qwen3_5",
            "sample_packing": True,
            "torch_compile": True,
            "bf16": True,
        }
    )

    pm._apply_model_specific_patches()

    assert captured[0]["fla_causal_conv_compile_boundary"] is False
    assert captured[1]["fla_causal_conv_compile_boundary"] is True
    # torch_compile must be forwarded: it switches FusedRMSNormGated to its
    # custom-op wrapper so the decoder LOOP can compile break-free.
    assert captured[0]["torch_compile"] is False
    assert captured[1]["torch_compile"] is False
    assert captured[2]["torch_compile"] is True


def test_apply_selects_lora_compute_mode(monkeypatch):
    """adapter=lora default -> base_layer becomes an FP4 compute base (the
    pre-quantized FP4-compute base, the recommended default)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FrozenBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager({"adapter": "lora", "nvfp4_training": {"enabled": True}})
    pm._apply_nvfp4_training(model)

    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert bases and all(
        isinstance(b, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear))
        for b in bases
    )
    assert not any(
        isinstance(b, (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear)) for b in bases
    )


def test_apply_sets_shared_lora_base_fprop_flag(monkeypatch):
    _supported(monkeypatch, True)
    from axolotl.kernels import lora as lora_mod
    from axolotl.loaders.patch_manager import PatchManager
    from axolotl.utils import nvfp4_training as nvfp4_training_mod

    monkeypatch.setattr(
        nvfp4_training_mod,
        "convert_lora_base_to_nvfp4",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        PatchManager, "_nvfp4_apply_tied_or_lm_head", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        PatchManager, "_nvfp4_load_packed_sidecar", lambda *args, **kwargs: None
    )

    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {
                "enabled": True,
                "shared_lora_base_fprop": True,
                "fuse_rmsnorm": False,
            },
        }
    )
    pm._apply_nvfp4_training(object())
    assert lora_mod._NVFP4_SHARED_BASE_FPROP is True

    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {
                "enabled": True,
                "shared_lora_base_fprop": False,
                "fuse_rmsnorm": False,
            },
        }
    )
    pm._apply_nvfp4_training(object())
    assert lora_mod._NVFP4_SHARED_BASE_FPROP is False


def test_apply_selects_hp_mode_when_requested(monkeypatch):
    """base_mode: hp -> NVFP4Linear (HP frozen base, re-quantized each step)."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = _tiny_lora_model()
    pm = _patch_manager(
        {"adapter": "lora", "nvfp4_training": {"enabled": True, "base_mode": "hp"}}
    )
    pm._apply_nvfp4_training(model)
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert bases and all(isinstance(b, NVFP4Linear) for b in bases)
    assert all(not b.weight.requires_grad for b in bases)


def test_apply_selects_fft_mode_when_no_adapter(monkeypatch):
    """No adapter -> raw nn.Linear swapped to NVFP4Linear (full fine-tune)."""
    _supported(monkeypatch, True)
    from torch import nn

    from axolotl.utils.nvfp4_training import NVFP4Linear

    model = nn.Sequential(nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False))
    pm = _patch_manager({"nvfp4_training": {"enabled": True}})
    pm._apply_nvfp4_training(model)
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())


def test_apply_honors_block_exclusions(monkeypatch):
    """skip_last_n_blocks keeps the tail block's base_layer un-swapped."""
    _supported(monkeypatch, True)
    from peft.tuners.lora import Linear as LoraLinear

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
    )

    model = _tiny_lora_model()
    pm = _patch_manager(
        {
            "adapter": "lora",
            "nvfp4_training": {"enabled": True, "skip_last_n_blocks": 1},
        }
    )
    pm._apply_nvfp4_training(model)

    swapped = {
        name: isinstance(
            m.base_layer, (NVFP4ComputeBaseLinear, NVFP4FastComputeBaseLinear)
        )
        for name, m in model.named_modules()
        if isinstance(m, LoraLinear)
    }
    # layer 0 swapped, layer 1 (last block) excluded
    assert any("layers.0." in n and v for n, v in swapped.items())
    assert all(not v for n, v in swapped.items() if "layers.1." in n)


_ok, _ = nvfp4_mod.nvfp4_supported()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
def test_e2e_swap_and_train_step(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4Linear,
        NVFP4Recipe,
        convert_to_nvfp4_training,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    count = convert_to_nvfp4_training(model, NVFP4Recipe())
    assert count > 0
    assert any(isinstance(m, NVFP4Linear) for m in model.modules())

    ids = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()


@pytest.mark.skipif(not _ok, reason="requires Blackwell FP4 tensor cores")
@pytest.mark.parametrize("quantize_base", [False, True])
def test_e2e_lora_swap_and_train_step(quantize_base):
    """Both adapter modes: right NVFP4 class appears under PEFT and a step is finite."""
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import Linear as LoraLinear
    from transformers import AutoModelForCausalLM

    from axolotl.utils.nvfp4_training import (
        NVFP4FastComputeBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FrozenBaseLinear,
        NVFP4Linear,
        NVFP4Recipe,
        convert_lora_base_to_nvfp4,
    )

    base = AutoModelForCausalLM.from_pretrained(
        "axolotl-ai-co/tiny-qwen2-129m", torch_dtype=torch.bfloat16
    ).cuda()
    model = get_peft_model(
        base, LoraConfig(r=8, target_modules=["q_proj", "v_proj"], lora_alpha=16)
    )
    count = convert_lora_base_to_nvfp4(
        model, NVFP4Recipe(), quantized_storage=quantize_base
    )
    assert count > 0

    expected = (
        (NVFP4FrozenBaseLinear, NVFP4FastFrozenBaseLinear)
        if quantize_base
        else (NVFP4Linear, NVFP4FastComputeBaseLinear)
    )
    bases = [m.base_layer for m in model.modules() if isinstance(m, LoraLinear)]
    assert any(isinstance(b, expected) for b in bases)

    ids = torch.randint(0, base.config.vocab_size, (1, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss).item()


def test_fsdp_nvfp4_class_is_picklable():
    """FSDP2's FULL_STATE_DICT save pickles the frozen NVFP4 params, so the
    lazily-built FSDP-hooked subclass must live at module scope (not be a
    ``<locals>`` class) or the checkpoint save crashes with
    ``Can't get local object '_fsdp_nvfp4_class.<locals>.FSDPNVFP4Tensor'``.
    Regression guard for that FSDP checkpoint-save bug."""
    import pickle

    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    from axolotl.utils.nvfp4_training import _fsdp_nvfp4_class

    cls = _fsdp_nvfp4_class()
    assert cls.__qualname__ == "FSDPNVFP4Tensor"
    assert cls.__module__ == "axolotl.utils.nvfp4_training"
    # the exact operation that failed during the FSDP checkpoint save:
    assert pickle.loads(pickle.dumps(cls)) is cls
