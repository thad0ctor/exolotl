"""GPU wiring tests for packed-sequence (multipack) VARLEN NVFP4 attention.

Under axolotl sample packing the batch reaches the attention patch FLATTENED to
one row (Z=1) with ``attention_mask=None`` and the sample boundaries encoded in
``position_ids`` (each sample's positions restart at 0). These tests cover:

  * packed-batch detection from position_ids (incl. the cached cu_seqlens) and
    routing into ``nvfp4_flash_attn_func(..., cu_seqlens=...)`` — block-diagonal
    causal varlen attention instead of DENSE attention over the whole pack;
  * forward/grad parity vs a per-sample SDPA reference (the model's original
    attention run sample by sample), eager AND through the torch.compile opaque
    custom op;
  * dense batches staying on the unchanged dense path;
  * unsupported packed layouts (multi-row batches) falling back to the original
    forward — never silently dense-attended;
  * ``save_packs`` being overridden (one-time INFO) for packed batches (the
    forward saved-pack set is not varlen-aware), while the configured
    ``grad_dots`` mode passes through — every grad-GEMM mode composes with
    varlen, and the default (None) defers to the kernel AUTO, which resolves
    on the pack's MEAN SAMPLE LENGTH ("bf16" below 4096, else "fp4_rownorm");
  * the packed perf GATE (``packed_min_sample_len``): default 0 (gate off —
    the pre-packed training forward beats FA2-varlen at every measured packed
    mean) routes all packs to FP4 varlen; a positive threshold (legacy-fork
    escape hatch) keeps short-mean packs on the model's original (FA2-varlen)
    forward; the decision is computed once per step (cached with cu_seqlens on
    the position_ids tensor) and applies to grad, no-grad and compiled
    sub-paths alike;
  * the NO-GRAD packed path feeding the varlen-capable packed-operand forward
    from the fused producers (RoPE+quant one pass, no roped Q/K round-trip);
  * the GRAD packed path feeding PRE-PACKED q/k (fused RoPE+quant with the
    roped hp stored for the HP backward) into the training entry.

NOTE: parity/routing tests pass ``packed_min_sample_len=0`` explicitly so
they stay valid regardless of the default.
"""

import logging

import pytest
import torch
import torch.nn.functional as F

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (10, 0)
        ),
        reason="native NVFP4 attention requires sm>=100 (Blackwell)",
    ),
]

pytest.importorskip("sageattention.nvfp4")
pytest.importorskip("transformers.models.qwen3_5")

import axolotl.monkeypatch.attention.nvfp4_flash_attn as patch_mod  # noqa: E402
from axolotl.kernels.attn_nvfp4_flash import _varlen_seq_arrays  # noqa: E402
from axolotl.monkeypatch.attention.nvfp4_flash_attn import (  # noqa: E402
    _compute_packed_info,
    patch_qwen3_5_nvfp4_attention,
)

if _varlen_seq_arrays is None:  # pragma: no cover - legacy fork
    pytest.skip(
        "installed sageattention fork lacks varlen (cu_seqlens) support",
        allow_module_level=True,
    )

HEADS, KV, D = 4, 2, 128
S = 256
LENS = [96, 64, 96]
FWD_COS = 0.98
GRAD_COS = 0.95


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _build_attn(seed=0, max_pos=1024):
    """Tiny Qwen3.5 full-attention layer (no download) + its rotary embedding."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5TextRotaryEmbedding,
    )

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        hidden_size=HEADS * D,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=D,
        max_position_embeddings=max_pos,
        attention_dropout=0.0,
        attention_bias=False,
    )
    cfg._attn_implementation = "sdpa"
    attn = Qwen3_5Attention(cfg, layer_idx=0).cuda().to(torch.bfloat16)
    rot = Qwen3_5TextRotaryEmbedding(cfg).cuda()
    return attn, rot, cfg


def _cos_sin(rot, hidden_states, position_ids):
    """Qwen3.5 MRoPE cos/sin: the rotary embedding takes [axes, B, T]."""
    return rot(hidden_states, position_ids[None].expand(3, *position_ids.shape))


def _packed_inputs(hidden, seed=0, lens=LENS):
    torch.manual_seed(seed)
    s = sum(lens)
    hs = torch.randn(1, s, hidden, device="cuda", dtype=torch.bfloat16)
    pos = torch.cat([torch.arange(n) for n in lens]).unsqueeze(0).cuda()
    w = torch.randn(1, s, hidden, device="cuda", dtype=torch.bfloat16)
    return hs, pos, w


def _per_sample_reference(attn, orig_forward, rot, hs0, w, lens=LENS):
    """Per-sample SDPA reference: the ORIGINAL attention forward run on each
    sample independently (positions restarting at 0), outputs concatenated."""
    hs = hs0.clone().requires_grad_(True)
    outs, off = [], 0
    for n in lens:
        sl = hs[:, off : off + n]
        p = torch.arange(n, device="cuda").unsqueeze(0)
        o, _ = orig_forward(
            attn,
            hidden_states=sl,
            position_embeddings=_cos_sin(rot, sl, p),
            attention_mask=None,
        )
        outs.append(o)
        off += n
    out = torch.cat(outs, dim=1)
    (out.float() * w.float()).sum().backward()
    return out.detach(), hs.grad.detach()


def _run_patched(attn, rot, hs0, pos, w):
    hs = hs0.clone().requires_grad_(True)
    out, _ = attn(
        hidden_states=hs,
        position_embeddings=_cos_sin(rot, hs, pos),
        attention_mask=None,
        position_ids=pos,
    )
    (out.float() * w.float()).sum().backward()
    return out.detach(), hs.grad.detach()


def test_compute_packed_info_classification():
    """position_ids classification: packed / dense / unsupported layouts."""
    pos_packed = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    kind, cu, mean_len = _compute_packed_info(pos_packed, 9)
    assert kind == "packed"
    assert cu.tolist() == [0, 3, 5, 9]
    assert mean_len == pytest.approx(3.0)  # 9 tokens / 3 samples
    # MRoPE [axes, B, T] layout classifies identically.
    kind3, cu3, mean3 = _compute_packed_info(pos_packed[None].expand(3, 1, 9), 9)
    assert kind3 == "packed" and cu3.tolist() == [0, 3, 5, 9]
    assert mean3 == pytest.approx(3.0)
    # Single contiguous sample -> not packed.
    assert _compute_packed_info(torch.arange(9).unsqueeze(0), 9) == (
        None,
        None,
        None,
    )
    # Plain B>1 batch (one sample per row) -> not packed.
    pos_b2 = torch.arange(4).unsqueeze(0).repeat(2, 1)
    assert _compute_packed_info(pos_b2, 4) == (None, None, None)
    # Multi-row pack: packed but unsupported by the Z=1 varlen kernel.
    pos_multi = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 3, 4]])
    assert _compute_packed_info(pos_multi, 5) == ("fallback", None, None)
    # seq-length mismatch (e.g. cached decode) -> not classified.
    assert _compute_packed_info(pos_packed, 5) == (None, None, None)


def test_packed_parity_eager():
    """Packed batch through the patched attention (eager kernel) matches a
    per-sample SDPA reference: cos > 0.98 forward, > 0.95 grads."""
    attn, rot, cfg = _build_attn(seed=1)
    orig_forward = type(attn).forward
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=1)
    ref_out, ref_grad = _per_sample_reference(attn, orig_forward, rot, hs0, w)

    assert (
        patch_qwen3_5_nvfp4_attention(
            attn, train_backward=True, packed_min_sample_len=0
        )
        == 1
    )
    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    cf, cg = _cos(out, ref_out), _cos(grad, ref_grad)
    assert cf > FWD_COS, f"packed eager fwd cos={cf:.4f} <= {FWD_COS}"
    assert cg > GRAD_COS, f"packed eager grad cos={cg:.4f} <= {GRAD_COS}"

    # No-grad/eval prefill takes the varlen kernel too (fused-producer packed
    # forward when the fork supports it, else the forward-only entry).
    with torch.no_grad():
        out_ng, _ = attn(
            hidden_states=hs0,
            position_embeddings=_cos_sin(rot, hs0, pos),
            attention_mask=None,
            position_ids=pos,
        )
    cng = _cos(out_ng, ref_out)
    assert cng > FWD_COS, f"packed no-grad fwd cos={cng:.4f} <= {FWD_COS}"


def test_packed_parity_compiled_custom_op():
    """Same parity through torch.compile with the opaque differentiable custom
    op (cu_seqlens threaded through fwd custom op, ctx, and bwd custom op)."""
    attn, rot, cfg = _build_attn(seed=2)
    orig_forward = type(attn).forward
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=2)
    ref_out, ref_grad = _per_sample_reference(attn, orig_forward, rot, hs0, w)

    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, compile_custom_op=True, packed_min_sample_len=0
    )
    torch._dynamo.reset()

    def fn(hs):
        out, _ = attn(
            hidden_states=hs,
            position_embeddings=_cos_sin(rot, hs, pos),
            attention_mask=None,
            position_ids=pos,
        )
        return out

    compiled = torch.compile(fn, dynamic=False)
    hs = hs0.clone().requires_grad_(True)
    out = compiled(hs)
    (out.float() * w.float()).sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(hs.grad).all()
    cf, cg = _cos(out.detach(), ref_out), _cos(hs.grad, ref_grad)
    assert cf > FWD_COS, f"packed compiled fwd cos={cf:.4f} <= {FWD_COS}"
    assert cg > GRAD_COS, f"packed compiled grad cos={cg:.4f} <= {GRAD_COS}"


def test_packed_routing_cu_seqlens_reaches_kernel(monkeypatch):
    """Packed batch -> kernel gets cu_seqlens; dense batch -> dense path
    unchanged (no cu_seqlens); multi-row pack -> original forward (no kernel)."""
    attn, rot, cfg = _build_attn(seed=3)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=3)

    calls = []
    real = patch_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        calls.append(kwargs.get("cu_seqlens", None))
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "nvfp4_flash_attn_func", spy)

    # Packed: cu_seqlens must reach the kernel with the position_ids boundaries.
    _run_patched(attn, rot, hs0, pos, w)
    assert len(calls) == 1
    assert calls[0] is not None
    assert calls[0].tolist() == [0, 96, 160, 256]

    # Dense (single contiguous sample): the dense path, no cu_seqlens.
    calls.clear()
    pos_dense = torch.arange(S, device="cuda").unsqueeze(0)
    _run_patched(attn, rot, hs0, pos_dense, w)
    assert len(calls) == 1 and calls[0] is None

    # Multi-row pack: unsupported layout MUST bypass the NVFP4 kernel entirely
    # (original forward; never dense-attended by the FP4 path).
    calls.clear()
    hs2 = torch.randn(
        2, 128, cfg.hidden_size, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    pos_multi = torch.cat(
        [torch.arange(64), torch.arange(64)]
    ).unsqueeze(0).repeat(2, 1).cuda()
    out2, _ = attn(
        hidden_states=hs2,
        position_embeddings=_cos_sin(rot, hs2, pos_multi),
        attention_mask=None,
        position_ids=pos_multi,
    )
    assert len(calls) == 0
    assert torch.isfinite(out2).all()


def test_packed_save_packs_config_ignored_one_time_log(caplog):
    """save_packs config + packed batch: runs (no crash) with the saved-pack
    set forced off and logs a one-time INFO; the configured grad_dots mode
    (here the legacy all-FP4 backward, which now composes with varlen) still
    applies."""
    attn, rot, cfg = _build_attn(seed=4)
    patch_qwen3_5_nvfp4_attention(
        attn,
        train_backward=True,
        save_backward_packs=True,
        grad_dots="fp4_legacy",  # composes with varlen; only save_packs is forced
        packed_min_sample_len=0,
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=4)

    patch_mod._PACKED_SAVE_PACKS_LOGGED = False
    msg = "ignoring attention.backward.save_packs"
    with caplog.at_level(logging.INFO, logger=patch_mod.LOG.name):
        out, grad = _run_patched(attn, rot, hs0, pos, w)
        assert torch.isfinite(out).all() and torch.isfinite(grad).all()
        first = sum(msg in r.getMessage() for r in caplog.records)
        assert first == 1, f"expected exactly one save_packs INFO, got {first}"
        # Second packed step: no duplicate log.
        _run_patched(attn, rot, hs0, pos, w)
        total = sum(msg in r.getMessage() for r in caplog.records)
        assert total == 1, f"save_packs INFO logged {total} times, expected 1"


def test_packed_default_auto_short_mean_uses_hp_backward(monkeypatch):
    """The packed default (grad_dots=None) no longer forces a mode: the patch
    passes None through and the KERNEL AUTO decides from the pack's mean
    sample length — ~85 here, far below the 4096 crossover, so the backward
    must route through the hp ('bf16') ``_run_bwd_hp`` path."""
    if not patch_mod._TRAIN_GRAD_DOTS:
        pytest.skip("installed sageattention fork lacks backward_grad_dots")
    import sageattention.nvfp4.flash as sage_flash

    attn, rot, cfg = _build_attn(seed=11)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=11)

    hp_calls = []
    real_hp = sage_flash._run_bwd_hp

    def spy_hp(*args, **kwargs):
        hp_calls.append(1)
        return real_hp(*args, **kwargs)

    monkeypatch.setattr(sage_flash, "_run_bwd_hp", spy_hp)

    seen_modes = []
    real = patch_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        seen_modes.append(kwargs.get("backward_grad_dots", "MISSING"))
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "nvfp4_flash_attn_func", spy)

    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert seen_modes == [None], (
        "packed branch must pass the CONFIGURED grad_dots (None = kernel "
        f"AUTO) instead of forcing a mode, got {seen_modes}"
    )
    assert hp_calls, "kernel AUTO at mean ~85 must resolve to the hp backward"


def test_packed_explicit_grad_dots_mode_passes_through(monkeypatch):
    """An explicit grad_dots config applies to packed batches too: the mode
    reaches the kernel entry and the matching grad-GEMM backward runs."""
    if not patch_mod._TRAIN_GRAD_DOTS:
        pytest.skip("installed sageattention fork lacks backward_grad_dots")
    import sageattention.nvfp4.flash as sage_flash

    attn, rot, cfg = _build_attn(seed=12)
    patch_qwen3_5_nvfp4_attention(
        attn,
        train_backward=True,
        grad_dots="fp4_rownorm",
        packed_min_sample_len=0,
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=12)

    bwd_modes = []
    real_bwd = sage_flash._run_bwd

    def spy_bwd(*args, **kwargs):
        bwd_modes.append(kwargs.get("grad_dots"))
        return real_bwd(*args, **kwargs)

    monkeypatch.setattr(sage_flash, "_run_bwd", spy_bwd)

    seen_modes = []
    real = patch_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        seen_modes.append(kwargs.get("backward_grad_dots", "MISSING"))
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "nvfp4_flash_attn_func", spy)

    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert seen_modes == ["fp4_rownorm"]
    assert bwd_modes == ["fp4_rownorm"], (
        f"configured fp4_rownorm must reach _run_bwd, got {bwd_modes}"
    )


def test_packed_32k_mean_auto_routes_fp4_rownorm_parity(monkeypatch):
    """A 32k-mean pack (2 x 32768) with the default grad_dots (None): the
    kernel AUTO resolves to 'fp4_rownorm' (mean >= the 4096 crossover) and the
    grads still match the per-sample SDPA reference (cos > 0.95)."""
    if not patch_mod._TRAIN_GRAD_DOTS:
        pytest.skip("installed sageattention fork lacks backward_grad_dots")
    import sageattention.nvfp4.flash as sage_flash

    lens = [32768, 32768]
    attn, rot, cfg = _build_attn(seed=13, max_pos=65536)
    orig_forward = type(attn).forward
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=13, lens=lens)
    ref_out, ref_grad = _per_sample_reference(
        attn, orig_forward, rot, hs0, w, lens=lens
    )

    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    bwd_modes = []
    real_bwd = sage_flash._run_bwd

    def spy_bwd(*args, **kwargs):
        bwd_modes.append(kwargs.get("grad_dots"))
        return real_bwd(*args, **kwargs)

    monkeypatch.setattr(sage_flash, "_run_bwd", spy_bwd)

    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert bwd_modes == ["fp4_rownorm"], (
        "kernel AUTO at mean 32768 must resolve to fp4_rownorm, got "
        f"{bwd_modes}"
    )
    cf, cg = _cos(out, ref_out), _cos(grad, ref_grad)
    assert cf > FWD_COS, f"32k-mean packed fwd cos={cf:.4f} <= {FWD_COS}"
    assert cg > GRAD_COS, f"32k-mean packed grad cos={cg:.4f} <= {GRAD_COS}"


def test_qwen3_vl_packed_falls_back_to_original_forward(monkeypatch):
    """Qwen3-VL patch with the DEFAULT gate (packed_min_sample_len=1024): this
    short-mean pack (~85) must bypass the NVFP4 kernel (original FA2-varlen
    forward) — never silently dense-attended. The VL packed FP4-varlen path
    itself (gate off / long packs) is covered in
    test_qwen3_vl_nvfp4_packed_varlen.py."""
    pytest.importorskip("transformers.models.qwen3_vl")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    import axolotl.monkeypatch.attention.nvfp4_flash_attn_vl as vl_mod
    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    torch.manual_seed(5)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=HEADS * D,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=D,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    model = Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()
    assert patch_qwen3_vl_nvfp4_attention(model, train_backward=True) > 0

    calls = []
    real = vl_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(vl_mod, "nvfp4_flash_attn_func", spy)

    attn = model.layers[0].self_attn
    hs = torch.randn(1, S, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    pos_packed = torch.cat([torch.arange(n) for n in LENS]).unsqueeze(0).cuda()
    cos, sin = model.rotary_emb(hs, pos_packed)
    out, _ = attn(
        hidden_states=hs,
        position_embeddings=(cos, sin),
        attention_mask=None,
        position_ids=pos_packed,
    )
    assert len(calls) == 0, "packed VL batch must not hit the NVFP4 kernel"
    assert torch.isfinite(out).all()

    # Dense VL batch keeps using the NVFP4 path.
    pos_dense = torch.arange(S, device="cuda").unsqueeze(0)
    cos, sin = model.rotary_emb(hs, pos_dense)
    out2, _ = attn(
        hidden_states=hs,
        position_embeddings=(cos, sin),
        attention_mask=None,
        position_ids=pos_dense,
    )
    assert len(calls) == 1
    assert torch.isfinite(out2).all()


# ---------------------------------------------------------------------------
# Packed perf gate (packed_min_sample_len).
# ---------------------------------------------------------------------------
def _spy_fp4_entrypoints(monkeypatch):
    """Spy on every FP4 attention entry the packed branch can take; returns the
    call list (any entry hit appends its name)."""
    calls = []
    for name in (
        "nvfp4_flash_attn_func",
        "nvfp4_flash_attention",
        "nvfp4_flash_attention_packed",
    ):
        real = getattr(patch_mod, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(patch_mod, name, spy)
    return calls


def test_packed_gate_default_is_off_fp4_everywhere(monkeypatch):
    """The DEFAULT gate is now 0 (off): with the pre-packed training forward
    FP4 varlen wins at every measured packed mean, so even short-mean packs
    (the test pack's mean is ~85) go to FP4 under the default patch config."""
    attn, rot, cfg = _build_attn(seed=6)
    patch_qwen3_5_nvfp4_attention(attn, train_backward=True)  # default gate (0)
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=6)
    calls = _spy_fp4_entrypoints(monkeypatch)
    patch_mod._PACKED_GATE_LOGGED = False
    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert calls, "default gate (0) must route packed batches to FP4 varlen"


def test_packed_gate_positive_routes_short_packs_to_original(monkeypatch, caplog):
    """An explicit positive gate (1024, the legacy-fork escape hatch): the test
    pack's mean sample length (~85) is short, so grad AND no-grad steps must
    use the ORIGINAL forward (no FP4 entrypoint) with a one-time INFO stating
    the resolution and the measured mean."""
    attn, rot, cfg = _build_attn(seed=6)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=1024
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=6)
    calls = _spy_fp4_entrypoints(monkeypatch)

    patch_mod._PACKED_GATE_LOGGED = False
    msg = "packed-batch gate -> original attention"
    with caplog.at_level(logging.INFO, logger=patch_mod.LOG.name):
        out, grad = _run_patched(attn, rot, hs0, pos, w)  # grad path
        with torch.no_grad():  # no-grad path
            out_ng, _ = attn(
                hidden_states=hs0,
                position_embeddings=_cos_sin(rot, hs0, pos),
                attention_mask=None,
                position_ids=pos,
            )
        n_logs = sum(msg in r.getMessage() for r in caplog.records)
    assert calls == [], f"gated-out pack hit FP4 entrypoints: {calls}"
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert torch.isfinite(out_ng).all()
    assert n_logs == 1, f"expected exactly one gate INFO, got {n_logs}"
    assert any(
        msg in r.getMessage() and "85.3" in r.getMessage() for r in caplog.records
    ), "gate INFO must state the measured mean sample length"


@pytest.mark.parametrize(
    "min_len,expect_fp4",
    [
        (0, True),  # 0 disables the gate: always FP4 varlen
        (64, True),  # mean ~85 >= 64: FP4 varlen
        (86, False),  # mean ~85 < 86: original forward
        (10**9, False),  # huge: never FP4 for packed batches
    ],
)
def test_packed_gate_threshold_routing(monkeypatch, min_len, expect_fp4):
    attn, rot, cfg = _build_attn(seed=7)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=min_len
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=7)
    calls = _spy_fp4_entrypoints(monkeypatch)
    patch_mod._PACKED_GATE_LOGGED = False

    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    if expect_fp4:
        assert calls, f"min_len={min_len}: expected the FP4 varlen path"
    else:
        assert calls == [], f"min_len={min_len}: expected the original forward"


def test_packed_gate_decision_cached_once_per_step(monkeypatch):
    """The packed classification (and so the gate decision input) is computed
    ONCE per distinct position_ids tensor — repeated layer calls within a step
    reuse the cache; a new step's tensor recomputes."""
    attn, rot, cfg = _build_attn(seed=8)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=8)

    n = {"compute": 0}
    real = patch_mod._compute_packed_info

    def counting(*args, **kwargs):
        n["compute"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "_compute_packed_info", counting)

    with torch.no_grad():
        for _ in range(3):  # same step: same position_ids tensor (3 "layers")
            attn(
                hidden_states=hs0,
                position_embeddings=_cos_sin(rot, hs0, pos),
                attention_mask=None,
                position_ids=pos,
            )
    assert n["compute"] == 1, f"expected one classification, got {n['compute']}"

    pos2 = pos.clone()  # next step: new tensor -> one new classification
    with torch.no_grad():
        attn(
            hidden_states=hs0,
            position_embeddings=_cos_sin(rot, hs0, pos2),
            attention_mask=None,
            position_ids=pos2,
        )
    assert n["compute"] == 2


# ---------------------------------------------------------------------------
# No-grad packed path through the fused producers + packed-operand varlen fwd.
# ---------------------------------------------------------------------------
def test_packed_nograd_uses_fused_packed_forward(monkeypatch):
    """No-grad packed prefill feeds the varlen-capable packed-operand forward
    (cu_seqlens) from the fused producers; grad path keeps the autograd entry."""
    if not patch_mod._PACKED_FWD_VARLEN:
        pytest.skip("installed sageattention fork: packed fwd lacks cu_seqlens")
    attn, rot, cfg = _build_attn(seed=9)
    orig_forward = type(attn).forward
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=9)
    ref_out, _ = _per_sample_reference(attn, orig_forward, rot, hs0, w)

    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    packed_calls = []
    real_packed = patch_mod.nvfp4_flash_attention_packed

    def spy_packed(*args, **kwargs):
        packed_calls.append(kwargs.get("cu_seqlens"))
        return real_packed(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "nvfp4_flash_attention_packed", spy_packed)

    with torch.no_grad():
        out_ng, _ = attn(
            hidden_states=hs0,
            position_embeddings=_cos_sin(rot, hs0, pos),
            attention_mask=None,
            position_ids=pos,
        )
    assert len(packed_calls) == 1, "no-grad packed must use the packed fwd"
    assert packed_calls[0] is not None
    assert packed_calls[0].tolist() == [0, 96, 160, 256]
    cng = _cos(out_ng, ref_out)
    assert cng > FWD_COS, f"fused packed no-grad fwd cos={cng:.4f} <= {FWD_COS}"

    # grad path stays on the autograd entry (HP backward needs hp q/k/v).
    packed_calls.clear()
    _run_patched(attn, rot, hs0, pos, w)
    assert packed_calls == []


def test_packed_grad_uses_prepacked_qk(monkeypatch):
    """The packed-GRAD branch feeds PRE-PACKED q/k from the fused RoPE+quant
    producers: the training call carries q_packs/k_packs, and the sage forward
    quantizes ONLY V (one _quant_nvfp4 call — q/k arrive packed; the backward
    repacks from the saved hp with its own kernels, not _quant_nvfp4)."""
    if not patch_mod._TRAIN_PREPACKED:
        pytest.skip("installed sageattention fork: training entry lacks q_packs")
    import sageattention.nvfp4.flash as sage_flash

    attn, rot, cfg = _build_attn(seed=10)
    patch_qwen3_5_nvfp4_attention(
        attn, train_backward=True, packed_min_sample_len=0
    )
    hs0, pos, w = _packed_inputs(cfg.hidden_size, seed=10)

    seen = []
    real = patch_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        seen.append((kwargs.get("q_packs"), kwargs.get("k_packs")))
        return real(*args, **kwargs)

    monkeypatch.setattr(patch_mod, "nvfp4_flash_attn_func", spy)

    quant_calls = []
    real_quant = sage_flash._quant_nvfp4

    def quant_spy(x, *args, **kwargs):
        quant_calls.append(tuple(x.shape))
        return real_quant(x, *args, **kwargs)

    monkeypatch.setattr(sage_flash, "_quant_nvfp4", quant_spy)

    out, grad = _run_patched(attn, rot, hs0, pos, w)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert len(seen) == 1
    qp, kp = seen[0]
    assert qp is not None and kp is not None, "expected pre-packed q/k"
    assert qp[0].dtype == torch.uint8 and qp[0].shape == (HEADS, S, D // 2)
    assert kp[0].shape == (KV, S, D // 2)
    assert len(quant_calls) == 1, (
        f"forward should quantize ONLY V (q/k pre-packed), got _quant_nvfp4 "
        f"calls for shapes {quant_calls}"
    )
    assert quant_calls[0] == (KV, S, D)  # V [Z*Hk, Skv, D] (key-axis quant)


def test_custom_op_wrapper_rejects_invalid_combos():
    """The custom-op wrapper mirrors the kernel's knob rejections. Varlen now
    composes with EVERY grad_dots mode (the old bf16-only restriction is
    gone); only save_backward_packs and non-causal remain rejected, plus the
    grad_dots-level conflicts (fp8_rownorm + saved packs, both knobs set,
    unknown mode)."""
    from axolotl.kernels.attn_nvfp4_custom_op import nvfp4_flash_attn_train_custom_op

    q = torch.randn(1, HEADS, 64, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, KV, 64, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, KV, 64, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)
    common = dict(causal=True, num_key_value_groups=HEADS // KV, cu_seqlens=cu)
    with pytest.raises(ValueError, match="save_backward_packs"):
        nvfp4_flash_attn_train_custom_op(
            q, k, v, D**-0.5, **common, save_backward_packs=True
        )
    with pytest.raises(ValueError, match="causal"):
        nvfp4_flash_attn_train_custom_op(
            q, k, v, D**-0.5, causal=False,
            num_key_value_groups=HEADS // KV, cu_seqlens=cu,
        )
    dense = dict(causal=True, num_key_value_groups=HEADS // KV)
    with pytest.raises(ValueError, match="fp8_rownorm"):
        nvfp4_flash_attn_train_custom_op(
            q, k, v, D**-0.5, **dense,
            backward_grad_dots="fp8_rownorm", save_backward_packs=True,
        )
    with pytest.raises(ValueError, match="not both"):
        nvfp4_flash_attn_train_custom_op(
            q, k, v, D**-0.5, **dense,
            backward_grad_dots="bf16", backward_bf16_grad_dots=True,
        )
    with pytest.raises(ValueError, match="must be one of"):
        nvfp4_flash_attn_train_custom_op(
            q, k, v, D**-0.5, **dense, backward_grad_dots="fp16"
        )
    # Varlen + the legacy all-FP4 mode now RUNS (block-diagonal grads finite).
    ql = q.clone().requires_grad_(True)
    out = nvfp4_flash_attn_train_custom_op(
        ql, k, v, D**-0.5, **common, backward_grad_dots="fp4_legacy"
    )
    out.float().sum().backward()
    assert torch.isfinite(ql.grad).all()
