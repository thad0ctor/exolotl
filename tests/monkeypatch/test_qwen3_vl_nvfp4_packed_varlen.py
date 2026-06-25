"""GPU wiring tests for packed-sequence (multipack) VARLEN NVFP4 attention on
Qwen3-VL.

What the Qwen3-VL attention layer actually receives as ``position_ids``
(transformers 5.8.x, Qwen3VLTextModel.forward): the TEXT axis only —
2D ``[B, T]`` inputs (axolotl text-pack collators) are expanded to ``[4, B, T]``
with identical axes and the decoder layers get ``position_ids[0]`` back; 3-axis
mRoPE ``[3, B, T]`` tensors (``get_rope_index`` output for image batches) are
stripped to ``None`` before the decoder layers and feed ONLY the rotary
embedding. So for every packed layout the HF model can produce, position-0
resets in the attention kwarg ARE the sample boundaries.

These tests cover the VL patch's packed path:

  * boundary derivation (``_compute_packed_info_vl``) finds exactly the true
    boundaries on text-axis packs (2D and identical-axes 3D forms);
  * AMBIGUOUS signals — multi-axis mRoPE tensors, vision-style repeated /
    gridded / locally-reordered positions (incl. image-leading samples whose
    temporal axis fakes phantom position-0 resets), multi-row packs — return
    "fallback": the original forward, NEVER a wrong block-diagonal mask;
  * routing: packed batches reach ``nvfp4_flash_attn_func`` with the correct
    ``cu_seqlens`` (vision-run mRoPE cos/sin in the position embeddings);
    dense batches stay dense; adversarial position_ids bypass the kernel;
  * forward/grad parity vs a per-sample reference (the model's original
    attention run sample by sample, each with its own mRoPE cos/sin slice);
  * the packed perf gate (``packed_min_sample_len``), same semantics as the
    Qwen3.5 patch; and the save_packs override for packed batches (one-time
    INFO; the configured grad_dots mode passes through — every mode composes
    with varlen);
  * once-per-step caching of the classification.

NOTE: parity/routing tests pass ``packed_min_sample_len=0`` — the test pack's
mean sample length (256/3 ~ 85) is far below the 1024 default, which would
(correctly) gate them onto the original forward instead of FP4 varlen.
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
pytest.importorskip("transformers.models.qwen3_vl")

import axolotl.monkeypatch.attention.nvfp4_flash_attn as text_patch_mod  # noqa: E402
import axolotl.monkeypatch.attention.nvfp4_flash_attn_vl as vl_mod  # noqa: E402
from axolotl.kernels.attn_nvfp4_flash import _varlen_seq_arrays  # noqa: E402
from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (  # noqa: E402
    _compute_packed_info_vl,
    patch_qwen3_vl_nvfp4_attention,
)

if _varlen_seq_arrays is None:  # pragma: no cover - legacy fork
    pytest.skip(
        "installed sageattention fork lacks varlen (cu_seqlens) support",
        allow_module_level=True,
    )

HEADS, KV, D = 4, 2, 128
# Three samples, each: text prefix + image run (gridded mRoPE) + text suffix.
# (pre, grid_h, grid_w, post) -> lens 96 / 64 / 96, S = 256.
SAMPLES = [(24, 4, 4, 56), (8, 6, 4, 32), (40, 4, 6, 32)]
LENS = [pre + gh * gw + post for pre, gh, gw, post in SAMPLES]
S = sum(LENS)
CU = [0, LENS[0], LENS[0] + LENS[1], S]
FWD_COS = 0.98
GRAD_COS = 0.95


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _build_model(seed=0, n_layers=1):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=HEADS * D,
        intermediate_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=D,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16), cfg


def _sample_mrope(pre, gh, gw, post):
    """[3, L] mRoPE positions for one sample: text prefix (arange on all axes),
    an image run (temporal CONSTANT, gridded h/w — mirrors get_rope_index /
    get_vision_position_ids), then a text suffix continuing from max(gh, gw)."""
    t_pre = torch.arange(pre).expand(3, -1)
    cur = pre
    img_t = torch.full((gh * gw,), cur)
    img_h = (torch.arange(gh) + cur).repeat_interleave(gw)
    img_w = (torch.arange(gw) + cur).repeat(gh)
    img = torch.stack([img_t, img_h, img_w])
    cur += max(gh, gw)
    t_post = (torch.arange(post) + cur).expand(3, -1)
    return torch.cat([t_pre, img, t_post], dim=1)


def _packed_positions():
    """(text_axis [1, S], mrope_axes [3, 1, S]) for the SAMPLES pack — exactly
    what a packing-aware Qwen3-VL step yields at the attention layer: the
    flattened per-sample-arange TEXT axis as the position_ids kwarg, the 3-axis
    vision mRoPE only inside the rotary embedding's cos/sin."""
    text = torch.cat([torch.arange(n) for n in LENS]).unsqueeze(0)
    mrope = torch.cat([_sample_mrope(*s) for s in SAMPLES], dim=1).unsqueeze(1)
    return text.cuda(), mrope.cuda()


def _packed_inputs(hidden, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(1, S, hidden, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, S, hidden, device="cuda", dtype=torch.bfloat16)
    return hs, w


def _per_sample_reference(model, orig_forward, hs0, w):
    """Per-sample reference: the ORIGINAL attention forward run on each sample
    independently with that sample's own mRoPE cos/sin, outputs concatenated."""
    attn = model.layers[0].self_attn
    hs = hs0.clone().requires_grad_(True)
    outs, off = [], 0
    for spec, n in zip(SAMPLES, LENS):
        sl = hs[:, off : off + n]
        mrope = _sample_mrope(*spec).unsqueeze(1).cuda()
        o, _ = orig_forward(
            attn,
            hidden_states=sl,
            position_embeddings=model.rotary_emb(sl, mrope),
            attention_mask=None,
        )
        outs.append(o)
        off += n
    out = torch.cat(outs, dim=1)
    (out.float() * w.float()).sum().backward()
    return out.detach(), hs.grad.detach()


def _run_attn(model, hs0, w, position_ids, mrope, grad=True):
    attn = model.layers[0].self_attn
    cos_sin = model.rotary_emb(hs0, mrope)
    if not grad:
        with torch.no_grad():
            out, _ = attn(
                hidden_states=hs0,
                position_embeddings=cos_sin,
                attention_mask=None,
                position_ids=position_ids,
            )
        return out, None
    hs = hs0.clone().requires_grad_(True)
    out, _ = attn(
        hidden_states=hs,
        position_embeddings=cos_sin,
        attention_mask=None,
        position_ids=position_ids,
    )
    (out.float() * w.float()).sum().backward()
    return out.detach(), hs.grad.detach()


def _spy_kernel(monkeypatch):
    calls = []
    real = vl_mod.nvfp4_flash_attn_func

    def spy(*args, **kwargs):
        calls.append(kwargs.get("cu_seqlens", None))
        return real(*args, **kwargs)

    monkeypatch.setattr(vl_mod, "nvfp4_flash_attn_func", spy)
    return calls


# ---------------------------------------------------------------------------
# Boundary derivation.
# ---------------------------------------------------------------------------
def test_vl_packed_info_finds_true_boundaries():
    """Text-axis packs (the only packed form the HF text model forwards to
    attention) classify as packed with EXACTLY the true boundaries."""
    text, _ = _packed_positions()
    text = text.cpu()
    kind, cu, mean_len = _compute_packed_info_vl(text, S)
    assert kind == "packed"
    assert cu.tolist() == CU
    assert mean_len == pytest.approx(S / len(LENS))
    # [axes, B, T] with ALL axes identical (4-axis text-pack expansion of a 2D
    # input degenerates to this) classifies identically.
    kind3, cu3, _ = _compute_packed_info_vl(text[None].expand(3, 1, S), S)
    assert kind3 == "packed" and cu3.tolist() == CU
    # Single contiguous sample / plain batches -> not packed (dense path).
    assert _compute_packed_info_vl(torch.arange(S).unsqueeze(0), S) == (
        None,
        None,
        None,
    )
    pos_b2 = torch.arange(64).unsqueeze(0).repeat(2, 1)
    assert _compute_packed_info_vl(pos_b2, 64) == (None, None, None)
    # seq-length mismatch (e.g. cached decode) -> not classified.
    assert _compute_packed_info_vl(text, S - 1) == (None, None, None)


def test_vl_packed_info_rejects_ambiguous_signals():
    """Anything the validity check can't certify -> ("fallback", None, None):
    the original forward, never a possibly-wrong block-diagonal mask."""
    text, mrope = _packed_positions()
    text, mrope = text.cpu(), mrope.cpu()

    # (a) A real 3-axis mRoPE pack (axes differ inside vision runs): ambiguous.
    assert _compute_packed_info_vl(mrope, S) == ("fallback", None, None)

    # (b) The TEMPORAL axis alone: image runs repeat positions (p[i+1] == p[i]),
    # violating strict arange segments even though its 0-resets happen to be at
    # the true boundaries here -> conservative fallback.
    temporal = mrope[0]
    assert int((temporal == 0).sum()) >= len(LENS)  # looks packed at first sight
    assert _compute_packed_info_vl(temporal, S) == ("fallback", None, None)

    # (c) Image-LEADING sample: temporal axis [0, 0, 0, 0, 1, ...] fakes
    # PHANTOM position-0 resets (the catastrophic case) -> fallback.
    img_first = torch.cat(
        [_sample_mrope(0, 2, 2, 28), _sample_mrope(8, 2, 2, 20)], dim=1
    ).unsqueeze(1)
    phantom = img_first[0]  # temporal axis, [1, 64]
    assert int((phantom == 0).sum()) > 2  # phantom resets present
    assert _compute_packed_info_vl(phantom, 64) == ("fallback", None, None)

    # (d) Locally reordered positions inside a segment -> fallback.
    bad = torch.tensor([[0, 1, 2, 1, 2, 3, 0, 1, 2]])
    assert _compute_packed_info_vl(bad, 9) == ("fallback", None, None)

    # (e) Multi-row packs: unsupported by the Z=1 varlen kernel -> fallback.
    pos_multi = torch.tensor([[0, 1, 2, 0, 1], [0, 1, 2, 3, 4]])
    assert _compute_packed_info_vl(pos_multi, 5) == ("fallback", None, None)


# ---------------------------------------------------------------------------
# Routing.
# ---------------------------------------------------------------------------
def test_vl_packed_routing_cu_seqlens_reaches_kernel(monkeypatch):
    """Packed -> kernel gets the true cu_seqlens; dense -> dense path (no
    cu_seqlens); ambiguous position_ids -> original forward (no kernel)."""
    model, cfg = _build_model(seed=3)
    assert (
        patch_qwen3_vl_nvfp4_attention(
            model, train_backward=True, packed_min_sample_len=0
        )
        == 1
    )
    hs0, w = _packed_inputs(cfg.hidden_size, seed=3)
    text, mrope = _packed_positions()
    calls = _spy_kernel(monkeypatch)

    # Packed: cu_seqlens must reach the kernel with the true boundaries.
    _run_attn(model, hs0, w, text, mrope)
    assert len(calls) == 1
    assert calls[0] is not None and calls[0].tolist() == CU

    # Dense (single contiguous sample): dense path, no cu_seqlens.
    calls.clear()
    pos_dense = torch.arange(S, device="cuda").unsqueeze(0)
    _run_attn(model, hs0, w, pos_dense, pos_dense)
    assert len(calls) == 1 and calls[0] is None

    # Ambiguous (temporal mRoPE axis with vision repeats): MUST bypass the
    # NVFP4 kernel entirely — original forward, never a wrong mask.
    calls.clear()
    out, _ = _run_attn(model, hs0, w, mrope[0], mrope, grad=False)
    assert len(calls) == 0
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Parity.
# ---------------------------------------------------------------------------
def test_vl_packed_parity_eager():
    """Packed VL batch (vision-run mRoPE cos/sin) through the patched attention
    matches the per-sample original-forward reference: cos > 0.98 forward,
    > 0.95 grads; no-grad path matches too."""
    model, cfg = _build_model(seed=1)
    attn = model.layers[0].self_attn
    orig_forward = type(attn).forward
    hs0, w = _packed_inputs(cfg.hidden_size, seed=1)
    text, mrope = _packed_positions()
    ref_out, ref_grad = _per_sample_reference(model, orig_forward, hs0, w)

    assert (
        patch_qwen3_vl_nvfp4_attention(
            model, train_backward=True, packed_min_sample_len=0
        )
        == 1
    )
    out, grad = _run_attn(model, hs0, w, text, mrope)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    cf, cg = _cos(out, ref_out), _cos(grad, ref_grad)
    assert cf > FWD_COS, f"vl packed fwd cos={cf:.4f} <= {FWD_COS}"
    assert cg > GRAD_COS, f"vl packed grad cos={cg:.4f} <= {GRAD_COS}"

    out_ng, _ = _run_attn(model, hs0, w, text, mrope, grad=False)
    cng = _cos(out_ng, ref_out)
    assert cng > FWD_COS, f"vl packed no-grad fwd cos={cng:.4f} <= {FWD_COS}"


# ---------------------------------------------------------------------------
# Perf gate + forced-HP backward.
# ---------------------------------------------------------------------------
def test_vl_packed_gate_default_routes_short_packs_to_original(monkeypatch):
    """Default gate (1024): the test pack's mean sample length (~85) is short,
    so grad AND no-grad steps must use the ORIGINAL forward (no FP4 kernel)."""
    model, cfg = _build_model(seed=6)
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True)  # default gate
    hs0, w = _packed_inputs(cfg.hidden_size, seed=6)
    text, mrope = _packed_positions()
    calls = _spy_kernel(monkeypatch)

    out, grad = _run_attn(model, hs0, w, text, mrope)
    out_ng, _ = _run_attn(model, hs0, w, text, mrope, grad=False)
    assert calls == [], "gated-out pack hit the FP4 kernel"
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert torch.isfinite(out_ng).all()


@pytest.mark.parametrize(
    "min_len,expect_fp4",
    [
        (0, True),  # 0 disables the gate: always FP4 varlen
        (64, True),  # mean ~85 >= 64: FP4 varlen
        (86, False),  # mean ~85 < 86: original forward
        (10**9, False),  # huge: never FP4 for packed batches
    ],
)
def test_vl_packed_gate_threshold_routing(monkeypatch, min_len, expect_fp4):
    model, cfg = _build_model(seed=7)
    patch_qwen3_vl_nvfp4_attention(
        model, train_backward=True, packed_min_sample_len=min_len
    )
    hs0, w = _packed_inputs(cfg.hidden_size, seed=7)
    text, mrope = _packed_positions()
    calls = _spy_kernel(monkeypatch)

    out, grad = _run_attn(model, hs0, w, text, mrope)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    if expect_fp4:
        assert calls, f"min_len={min_len}: expected the FP4 varlen path"
    else:
        assert calls == [], f"min_len={min_len}: expected the original forward"


def test_vl_packed_save_packs_config_ignored_one_time_log(caplog):
    """save_packs config + packed VL batch: runs with the saved-pack set
    forced off (same override as the Qwen3.5 patch) and a one-time INFO; the
    configured grad_dots mode (the legacy all-FP4 backward here — it composes
    with varlen now) still applies."""
    model, cfg = _build_model(seed=4)
    patch_qwen3_vl_nvfp4_attention(
        model,
        train_backward=True,
        save_backward_packs=True,
        grad_dots="fp4_legacy",  # composes with varlen; only save_packs forced
        packed_min_sample_len=0,
    )
    hs0, w = _packed_inputs(cfg.hidden_size, seed=4)
    text, mrope = _packed_positions()

    text_patch_mod._PACKED_SAVE_PACKS_LOGGED = False
    msg = "ignoring attention.backward.save_packs"
    with caplog.at_level(logging.INFO, logger=text_patch_mod.LOG.name):
        out, grad = _run_attn(model, hs0, w, text, mrope)
        assert torch.isfinite(out).all() and torch.isfinite(grad).all()
        first = sum(msg in r.getMessage() for r in caplog.records)
        assert first == 1, f"expected exactly one save_packs INFO, got {first}"
        _run_attn(model, hs0, w, text, mrope)
        total = sum(msg in r.getMessage() for r in caplog.records)
        assert total == 1, f"save_packs INFO logged {total} times, expected 1"


def test_vl_packed_classification_cached_once_per_step(monkeypatch):
    """The VL packed classification is computed ONCE per distinct position_ids
    tensor (cached on the tensor): repeated layer calls within a step reuse it."""
    model, cfg = _build_model(seed=8)
    patch_qwen3_vl_nvfp4_attention(
        model, train_backward=True, packed_min_sample_len=0
    )
    hs0, w = _packed_inputs(cfg.hidden_size, seed=8)
    text, mrope = _packed_positions()

    n = {"compute": 0}
    real = vl_mod._compute_packed_info_vl

    def counting(*args, **kwargs):
        n["compute"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(vl_mod, "_compute_packed_info_vl", counting)

    for _ in range(3):  # same step: same position_ids tensor (3 "layers")
        _run_attn(model, hs0, w, text, mrope, grad=False)
    assert n["compute"] == 1, f"expected one classification, got {n['compute']}"

    text2 = text.clone()  # next step: new tensor -> one new classification
    _run_attn(model, hs0, w, text2, mrope, grad=False)
    assert n["compute"] == 2
