"""GPU wiring tests for the NVFP4 attention backward grad-GEMM modes.

Covers the ``grad_dots`` knob ("bf16" / "fp4_rownorm" / "fp8_rownorm" /
"fp4_legacy" / None = kernel AUTO) end to end: the eager
``nvfp4_flash_attn_func`` path, the torch.compile opaque custom op
(``nvfp4_flash_attn_train_custom_op``), and the monkeypatch plumbing
(``patch_qwen3_vl_nvfp4_attention``), plus the DEPRECATED boolean
``bf16_grad_dots`` alias (True -> "bf16", False -> "fp4_legacy"; both set ->
error). At this file's S=256 the kernel AUTO resolves to "bf16" (below the
4096 crossover), and the backward must route through ``_run_bwd_hp`` with
grad cosine > 0.98 vs an SDPA reference (the legacy all-FP4 backward sits at
~0.94 RTN / ~0.82 SR — well below this bar, so the threshold itself proves
the HP path is active). The rownorm modes get their own (slightly lower)
parity bar.
"""

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

from axolotl.kernels.attn_nvfp4_flash import (  # noqa: E402
    _run_bwd_hp,
    nvfp4_flash_attn_func,
)

if _run_bwd_hp is None:  # pragma: no cover - legacy fork
    pytest.skip(
        "installed sageattention fork lacks the HP-grad-dots backward "
        "(_run_bwd_hp); point PYTHONPATH at the train-perf fork",
        allow_module_level=True,
    )

Z, H, HK, S = 2, 4, 2, 256
NG = H // HK
GRAD_COS = 0.98
# The rownorm fp4/fp8 grad-GEMM modes quantize the grad operands (measured grad
# cos 0.97-0.99 at training scale); slightly lower wiring-parity bar.
GRAD_COS_ROWNORM = 0.94


def _qkv(d, seed):
    torch.manual_seed(seed)
    q = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(Z, HK, S, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(Z, HK, S, d, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _sdpa_ref(q, k, v, scaling, do):
    """bf16 SDPA reference grads on fresh leaves carrying the same values."""
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)
    out = F.scaled_dot_product_attention(
        qr,
        kr.repeat_interleave(NG, dim=1),
        vr.repeat_interleave(NG, dim=1),
        is_causal=True,
        scale=scaling,
    )
    out.backward(do)
    return out, qr.grad, kr.grad, vr.grad


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _assert_grads(tag, grads, ref_grads, bar=GRAD_COS):
    for name, g, r in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert g is not None and torch.isfinite(g).all(), f"{tag} {name} non-finite"
        assert g.abs().sum() > 0, f"{tag} {name} is all-zero"
        c = _cos(g, r)
        assert c > bar, f"{tag} {name} cos={c:.4f} <= {bar}"


@pytest.mark.parametrize("d", [128, 256])
def test_eager_hp_grad_dots_auto(d):
    """Eager nvfp4_flash_attn_func with grad_dots auto (None) beats the legacy
    bar — at S=256 the kernel AUTO resolves to 'bf16'."""
    q, k, v = _qkv(d, seed=10 + d)
    scaling = d**-0.5
    torch.manual_seed(99)
    do = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    _, rdq, rdk, rdv = _sdpa_ref(q, k, v, scaling, do)

    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = nvfp4_flash_attn_func(
        ql,
        kl,
        vl,
        scaling,
        causal=True,
        num_key_value_groups=NG,
        backward_grad_dots=None,  # kernel AUTO: "bf16" (S=256 < 4096 crossover)
    )
    assert torch.isfinite(out).all()
    out.backward(do)
    _assert_grads("eager", (ql.grad, kl.grad, vl.grad), (rdq, rdk, rdv))


@pytest.mark.parametrize("mode", ["fp4_rownorm", "fp8_rownorm", "bf16"])
@pytest.mark.parametrize("d", [128, 256])
def test_eager_explicit_grad_dots_modes(d, mode):
    """Each explicit grad_dots mode produces finite, SDPA-aligned grads eagerly."""
    q, k, v = _qkv(d, seed=40 + d)
    scaling = d**-0.5
    torch.manual_seed(97)
    do = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    _, rdq, rdk, rdv = _sdpa_ref(q, k, v, scaling, do)

    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = nvfp4_flash_attn_func(
        ql,
        kl,
        vl,
        scaling,
        causal=True,
        num_key_value_groups=NG,
        backward_grad_dots=mode,
    )
    assert torch.isfinite(out).all()
    out.backward(do)
    bar = GRAD_COS if mode == "bf16" else GRAD_COS_ROWNORM
    _assert_grads(f"eager[{mode}]", (ql.grad, kl.grad, vl.grad), (rdq, rdk, rdv), bar)


def test_eager_alias_and_both_set_rejection():
    """The deprecated boolean alias still works; passing both knobs raises."""
    d = 128
    scaling = d**-0.5
    q, k, v = _qkv(d, seed=50)
    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = nvfp4_flash_attn_func(
        ql,
        kl,
        vl,
        scaling,
        causal=True,
        num_key_value_groups=NG,
        backward_bf16_grad_dots=True,  # deprecated alias -> "bf16"
    )
    out.float().sum().backward()
    assert torch.isfinite(ql.grad).all()
    with pytest.raises(AssertionError, match="backward_bf16_grad_dots"):
        nvfp4_flash_attn_func(
            q.clone().requires_grad_(True),
            k,
            v,
            scaling,
            causal=True,
            num_key_value_groups=NG,
            backward_grad_dots="bf16",
            backward_bf16_grad_dots=True,
        )


@pytest.mark.parametrize("d", [128, 256])
def test_custom_op_compiled_hp_grad_dots_auto(d):
    """The opaque custom op under torch.compile routes backward through HP grads."""
    from axolotl.kernels.attn_nvfp4_custom_op import nvfp4_flash_attn_train_custom_op

    torch._dynamo.reset()
    q, k, v = _qkv(d, seed=20 + d)
    scaling = d**-0.5
    torch.manual_seed(98)
    do = torch.randn(Z, H, S, d, device="cuda", dtype=torch.bfloat16)
    _, rdq, rdk, rdv = _sdpa_ref(q, k, v, scaling, do)

    def fn(q_, k_, v_):
        return nvfp4_flash_attn_train_custom_op(
            q_,
            k_,
            v_,
            scaling,
            causal=True,
            num_key_value_groups=NG,
            backward_grad_dots=None,  # kernel AUTO: "bf16" at S=256
        )

    compiled = torch.compile(fn, dynamic=False)
    ql = q.clone().requires_grad_(True)
    kl = k.clone().requires_grad_(True)
    vl = v.clone().requires_grad_(True)
    out = compiled(ql, kl, vl)
    assert torch.isfinite(out).all()
    out.backward(do)
    _assert_grads("compiled", (ql.grad, kl.grad, vl.grad), (rdq, rdk, rdv))


def test_custom_op_backward_routing(monkeypatch):
    """The op resolves the grad-dots string with the kernel's policy: auto ->
    'bf16' at S=256; rownorm/legacy modes -> _run_bwd with that grad_dots;
    saved packs clamp 'bf16' away; the deprecated alias maps through."""
    import axolotl.kernels.attn_nvfp4_custom_op as op_mod

    d = 128
    scaling = d**-0.5
    calls = []
    bwd_modes = []
    real_hp = op_mod._run_bwd_hp
    real_bwd = op_mod._run_bwd

    def spy(*args, **kwargs):
        calls.append(1)
        return real_hp(*args, **kwargs)

    def spy_bwd(*args, **kwargs):
        bwd_modes.append(kwargs.get("grad_dots", "fp4_legacy"))
        return real_bwd(*args, **kwargs)

    monkeypatch.setattr(op_mod, "_run_bwd_hp", spy)
    monkeypatch.setattr(op_mod, "_run_bwd", spy_bwd)

    def run(**op_kwargs):
        q, k, v = _qkv(d, seed=33)
        ql = q.clone().requires_grad_(True)
        kl = k.clone().requires_grad_(True)
        vl = v.clone().requires_grad_(True)
        out = op_mod.nvfp4_flash_attn_train_custom_op(
            ql, kl, vl, scaling, causal=True, num_key_value_groups=NG, **op_kwargs
        )
        out.float().sum().backward()
        for g in (ql.grad, kl.grad, vl.grad):
            assert g is not None and torch.isfinite(g).all()

    run(backward_grad_dots=None)  # auto at S=256 -> "bf16" -> HP path
    assert len(calls) == 1 and bwd_modes == []
    run(backward_grad_dots="fp4_legacy")  # forced legacy
    assert len(calls) == 1 and bwd_modes == ["fp4_legacy"]
    run(backward_grad_dots="fp4_rownorm")
    assert len(calls) == 1 and bwd_modes[-1] == "fp4_rownorm"
    run(backward_grad_dots="fp8_rownorm")
    assert len(calls) == 1 and bwd_modes[-1] == "fp8_rownorm"
    # auto + saved packs: AUTO's "bf16" is clamped to "fp4_rownorm" (kernel rule)
    run(backward_grad_dots=None, save_backward_packs=True)
    assert len(calls) == 1 and bwd_modes[-1] == "fp4_rownorm"
    # forced "bf16" + saved packs: same clamp
    run(backward_grad_dots="bf16", save_backward_packs=True)
    assert len(calls) == 1 and bwd_modes[-1] == "fp4_rownorm"
    run(backward_grad_dots="bf16")  # forced HP
    assert len(calls) == 2
    # deprecated alias still routes: True -> "bf16", False -> "fp4_legacy"
    run(backward_bf16_grad_dots=True)
    assert len(calls) == 3
    run(backward_bf16_grad_dots=False)
    assert len(calls) == 3 and bwd_modes[-1] == "fp4_legacy"
    # both knobs set -> rejected; unknown mode -> rejected
    q, k, v = _qkv(d, seed=33)
    with pytest.raises(ValueError, match="not both"):
        op_mod.nvfp4_flash_attn_train_custom_op(
            q, k, v, scaling, causal=True, num_key_value_groups=NG,
            backward_grad_dots="bf16", backward_bf16_grad_dots=True,
        )
    with pytest.raises(ValueError, match="backward_grad_dots must be one of"):
        op_mod.nvfp4_flash_attn_train_custom_op(
            q, k, v, scaling, causal=True, num_key_value_groups=NG,
            backward_grad_dots="fp16",
        )


def _build_vl_model(seed=0, n_layers=2, head_dim=128, n_heads=4, n_kv=2):
    """Tiny Qwen3-VL text model (no download) — mirrors test_qwen3_vl_nvfp4_attn."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(seed)
    cfg = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=n_heads * head_dim,
        intermediate_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    cfg._attn_implementation = "sdpa"
    return Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()


def _vl_inputs(hidden, seq=256, seed=0):
    torch.manual_seed(seed)
    hs = torch.randn(1, seq, hidden, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(seq, device="cuda").unsqueeze(0)
    return hs, pos


def _run_vl_attention(model, layer_idx, hidden_states, position_ids):
    attn = model.layers[layer_idx].self_attn
    cos, sin = model.rotary_emb(hidden_states, position_ids)
    out, _ = attn(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )
    return out


def test_vl_monkeypatch_plumbs_grad_dots(monkeypatch):
    """patch_qwen3_vl_nvfp4_attention threads the knob into the eager backward."""
    pytest.importorskip("transformers.models.qwen3_vl")
    import sageattention.nvfp4.flash as flash_mod

    from axolotl.monkeypatch.attention.nvfp4_flash_attn_vl import (
        patch_qwen3_vl_nvfp4_attention,
    )

    calls = []
    real_hp = flash_mod._run_bwd_hp

    def spy(*args, **kwargs):
        calls.append(1)
        return real_hp(*args, **kwargs)

    monkeypatch.setattr(flash_mod, "_run_bwd_hp", spy)

    model = _build_vl_model(seed=6)
    hidden = model.config.hidden_size
    patched = patch_qwen3_vl_nvfp4_attention(
        model, train_backward=True, grad_dots=None
    )
    assert patched > 0
    assert model.layers[0].self_attn._nvfp4_grad_dots is None

    hs, pos = _vl_inputs(hidden, seed=6)
    hs = hs.clone().requires_grad_(True)
    out = _run_vl_attention(model, 0, hs, pos)
    out.float().sum().backward()
    assert len(calls) > 0, (
        "grad_dots auto (kernel AUTO at S=256) did not route through _run_bwd_hp"
    )
    assert hs.grad is not None and torch.isfinite(hs.grad).all()
    assert hs.grad.abs().sum() > 0

    # Forcing the legacy mode must fall back to the all-FP4 backward.
    calls.clear()
    patch_qwen3_vl_nvfp4_attention(
        model, train_backward=True, grad_dots="fp4_legacy"
    )
    assert model.layers[0].self_attn._nvfp4_grad_dots == "fp4_legacy"
    hs2, pos2 = _vl_inputs(hidden, seed=7)
    hs2 = hs2.clone().requires_grad_(True)
    out2 = _run_vl_attention(model, 0, hs2, pos2)
    out2.float().sum().backward()
    assert len(calls) == 0
    assert hs2.grad is not None and torch.isfinite(hs2.grad).all()

    # The deprecated boolean alias maps onto grad_dots at patch time; passing
    # both raises.
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True, bf16_grad_dots=False)
    assert model.layers[0].self_attn._nvfp4_grad_dots == "fp4_legacy"
    patch_qwen3_vl_nvfp4_attention(model, train_backward=True, bf16_grad_dots=True)
    assert model.layers[0].self_attn._nvfp4_grad_dots == "bf16"
    with pytest.raises(ValueError, match="not both"):
        patch_qwen3_vl_nvfp4_attention(
            model, train_backward=True, grad_dots="bf16", bf16_grad_dots=True
        )
