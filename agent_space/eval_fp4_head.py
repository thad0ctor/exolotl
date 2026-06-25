#!/usr/bin/env python
"""Fast A/B harness: how much FP4 lm_head accuracy does a candidate method reclaim?

PTQ-style, NO training. We take a real (or synthetic) lm_head, quantize it with the
branch's NVFP4 code path, and compare the FP4 head against the bf16/fp32 gold head on
a fixed set of real (or random) hidden states. Each candidate accuracy-recovery method
plugs in as a single callable `(W_bf16, hidden, **ctx) -> head_module_or_weight` and is
scored on the exact same metrics, so methods can be ranked in seconds.

This is a DESIGN STUB. It is wired to nothing in training; it only imports the existing
quant primitives (`NVFP4FrozenBaseLinear`) read-only. It does not touch training code,
configs, or defaults.

Usage
-----
    PY=/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_torch211_experimental/bin/python
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY agent_space/eval_fp4_head.py
    # explicit model:
    ... $PY agent_space/eval_fp4_head.py --model Qwen/Qwen3-0.6B --n-tokens 512
    # force the synthetic large-vocab path (no model needed; proves the harness):
    ... $PY agent_space/eval_fp4_head.py --synthetic

Method registry
---------------
A method is registered with @register_method("name"). It receives the gold weight, the
captured calibration hidden states, and a `ctx` dict (bias, recipe, device, dtype) and
returns either an nn.Module with a `.weight` property and `__call__(x)->logits`, or a
plain corrected weight tensor (the harness wraps it in a bf16 Linear for scoring). The
"fp4_baseline" method (the branch's current behaviour) is always registered. Add yours
(per-row scale, GPTQ/AWQ calib, outlier-keep/MixFP4, low-rank residual, logit-distill)
next to it and it gets A/B'd automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

# Make the in-repo src importable without an editable install.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


# --------------------------------------------------------------------------------------
# Quant primitive: reuse the EXACT branch code path the training swap uses.
# `swap_frozen_lm_head_tileable` builds `NVFP4FrozenBaseLinear.from_linear(src, recipe)`,
# whose `.weight` property dequantizes the FP4 store back to bf16. So the bf16 view of
# this module IS the effective FP4 head the model trains/infers against.
# --------------------------------------------------------------------------------------
def make_fp4_head(linear: nn.Linear):
    """Quantize a bf16 nn.Linear lm_head with the branch's NVFP4 code path.

    Returns the NVFP4FrozenBaseLinear (its `.weight` is the dequantized FP4 head, and
    `forward` runs the real FP4 GEMM when on a Blackwell card).
    """
    from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe

    # recipe flags (SR/RHT) are gradient-path knobs; the frozen weight quant is plain
    # RTN regardless, so the recipe choice does not change the stored FP4 weight here.
    recipe = NVFP4Recipe(stochastic_rounding=False, hadamard=False)
    return NVFP4FrozenBaseLinear.from_linear(linear, recipe, fsdp=False)


def fp4_dequant_weight(linear: nn.Linear) -> torch.Tensor:
    """The bf16 weight you'd get back out of the FP4 store: Q(W) dequantized."""
    return make_fp4_head(linear).weight.detach()


# --------------------------------------------------------------------------------------
# Method registry. A method maps (gold W, calib hidden, ctx) -> scorable head.
# --------------------------------------------------------------------------------------
MethodFn = Callable[[torch.Tensor, torch.Tensor, dict], object]
_METHODS: dict[str, MethodFn] = {}


def register_method(name: str):
    def deco(fn: MethodFn):
        _METHODS[name] = fn
        return fn

    return deco


def _linear_from_weight(W: torch.Tensor, bias, device, dtype) -> nn.Linear:
    lin = nn.Linear(W.shape[1], W.shape[0], bias=bias is not None)
    with torch.no_grad():
        lin.weight.copy_(W)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin.to(device=device, dtype=dtype)


@register_method("fp4_baseline")
def _m_fp4_baseline(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """The branch as-is: plain RTN NVFP4 of the head, no correction. The thing to beat."""
    lin = _linear_from_weight(W, ctx["bias"], ctx["device"], ctx["dtype"])
    return make_fp4_head(lin)


# ---- Stubs for the candidate methods the user wants to test. ----
# Each returns either an nn.Module (.weight + __call__) or a corrected weight tensor.
# They are registered but raise NotImplementedError until filled in, so the harness
# lists them and you implement one at a time.


@register_method("per_row_scale")
def _m_per_row_scale(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """STUB: fold a learned/closed-form per-output-row scalar s_v into the FP4 head so
    Q(W*s)/s minimizes per-row relerr. lm_head rows have very different norms (rare vs
    frequent tokens); NVFP4's per-tensor scale under-serves the small-norm rows. Cheap
    closed form: s_v = argmin ||W_v - Q(s_v W_v)/s_v||. Return corrected weight."""
    raise NotImplementedError("per_row_scale: implement and return corrected head")


@register_method("gptq_awq_calib")
def _m_gptq_awq_calib(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """STUB: activation-aware calibration. Weight per-column importance by the calib
    hidden-state second moment (AWQ) or do GPTQ Hessian-OBQ error-feedback so the
    quantization error is pushed onto unimportant directions. `hidden` is exactly the
    calibration set this needs."""
    raise NotImplementedError("gptq_awq_calib: implement using ctx hidden stats")


@register_method("outlier_keep_mixfp4")
def _m_outlier_keep(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """STUB: MixFP4 — keep the top-k outlier rows/cols (or whole rare-token rows) in
    bf16 and FP4 the rest. The qmd flags the flat lm_head weight distribution as FP4's
    weakest case; a tiny bf16 residual on outliers may recover most of the loss."""
    raise NotImplementedError("outlier_keep_mixfp4: implement mixed-precision head")


def _residual_method(calibration: str, rank: int):
    """Build a registerable method: FP4(W) + low-rank bf16 residual A@B.

    ``calibration='activation'`` is the L2QER construction (weight the quant error
    by the calib activation 2nd-moment before the SVD); ``'svd'`` is plain. Reuses
    the SHIPPED residual builder so the harness scores exactly the training code.
    """

    def method(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
        from axolotl.kernels.nvfp4_residual import _build_residual

        lin = _linear_from_weight(W, ctx["bias"], ctx["device"], ctx["dtype"])
        Wq = fp4_dequant_weight(lin).float()
        E = W.float() - Wq
        g = None
        if calibration == "activation":
            h = hidden.reshape(-1, hidden.shape[-1]).float()
            g = h.pow(2).mean(0).sqrt().clamp_min(1e-8)
        A, B = _build_residual(E, rank, g)
        dtype = ctx["dtype"]

        class _ResHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.fp4 = make_fp4_head(lin)
                self.A = A.to(dtype)
                self.B = B.to(dtype)

            @property
            def weight(self):
                return (self.fp4.weight.float() + self.A.float() @ self.B.float()).to(dtype)

            def __call__(self, x):
                base = self.fp4(x)
                return base + (x @ self.B.t()) @ self.A.t()

        return _ResHead()

    return method


@register_method("low_rank_residual")
def _m_low_rank_residual(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """FP4(W) + activation-weighted (L2QER) low-rank bf16 residual, rank 32.

    The plain-SVD residual barely helps (the e2m1 quant error is near full-rank);
    the activation weighting concentrates the logit-relevant error into a low-rank
    subspace. See residual_probe.py for the full plain-vs-L2QER rank sweep."""
    return _residual_method("activation", 32)(W, hidden, ctx)


@register_method("logit_distill")
def _m_logit_distill(W: torch.Tensor, hidden: torch.Tensor, ctx: dict):
    """STUB: short PTQ optimization of the correction (per-row scale / low-rank UV) to
    directly minimize KL(softmax(bf16 logits) || softmax(fp4 logits)) on the calib
    hidden states. This is the QAT-lite micro-run: a few hundred steps of SGD on the
    *correction only*, gold head frozen, no model body update."""
    raise NotImplementedError("logit_distill: implement KL-matching of corrected head")


# --------------------------------------------------------------------------------------
# Metrics.
# --------------------------------------------------------------------------------------
@dataclass
class MetricResult:
    name: str
    weight: dict = field(default_factory=dict)
    logit: dict = field(default_factory=dict)
    loss: dict = field(default_factory=dict)


def _scorable(method_out, ctx) -> nn.Module:
    """Normalize a method's return to a forward-able module with a `.weight`."""
    if isinstance(method_out, nn.Module):
        return method_out
    # plain corrected weight tensor -> wrap in a bf16 Linear
    return _linear_from_weight(method_out, ctx["bias"], ctx["device"], ctx["dtype"])


@torch.no_grad()
def weight_metrics(W_gold: torch.Tensor, W_q: torch.Tensor) -> dict:
    """Per-row relerr / SNR of Q(W) vs W. Per-row because lm_head rows = per-token
    logit directions; a single bad row is a token the model can never rank correctly."""
    Wg = W_gold.float()
    Wq = W_q.float()
    err = Wq - Wg
    row_relerr = err.norm(dim=1) / Wg.norm(dim=1).clamp_min(1e-12)
    sig = Wg.pow(2).sum(dim=1)
    noi = err.pow(2).sum(dim=1).clamp_min(1e-30)
    row_snr_db = 10.0 * torch.log10(sig / noi)
    return {
        "rel_fro": (err.norm() / Wg.norm().clamp_min(1e-12)).item(),
        "row_relerr_mean": row_relerr.mean().item(),
        "row_relerr_p99": torch.quantile(row_relerr, 0.99).item(),
        "row_relerr_max": row_relerr.max().item(),
        "row_snr_db_mean": row_snr_db.mean().item(),
        "row_snr_db_min": row_snr_db.min().item(),
    }


@torch.no_grad()
def logit_metrics(gold_logits: torch.Tensor, cand_logits: torch.Tensor, k: int = 5) -> dict:
    """The thing that matters: agreement of the candidate logits with the gold logits.

    All on the SAME hidden states. KL(softmax(gold)||softmax(cand)) is the training-loss
    surrogate; top-1/top-k agreement and rank correlation are the decode-quality surrogate.
    """
    g = gold_logits.float()
    c = cand_logits.float()
    mse = F.mse_loss(c, g).item()
    cos = F.cosine_similarity(c, g, dim=-1).mean().item()
    g_lsm = F.log_softmax(g, dim=-1)
    c_lsm = F.log_softmax(c, dim=-1)
    # KL(P_gold || Q_cand) = sum p (log p - log q)
    kl = (g_lsm.exp() * (g_lsm - c_lsm)).sum(-1).mean().item()
    top1 = (g.argmax(-1) == c.argmax(-1)).float().mean().item()
    gk = g.topk(k, dim=-1).indices
    ck = c.topk(k, dim=-1).indices
    # fraction of gold top-k present in cand top-k (set overlap), per row then mean
    topk_overlap = (
        (gk.unsqueeze(-1) == ck.unsqueeze(-2)).any(-1).float().mean(-1).mean().item()
    )
    # Spearman-style rank correlation over the union of gold top-k tokens: do the
    # candidate logits order the salient tokens the same way?
    rank_corr = _topk_rank_corr(g, c, gk)
    return {
        "mse": mse,
        "cosine": cos,
        "kl_gold_cand": kl,
        "top1_agree": top1,
        f"top{k}_overlap": topk_overlap,
        f"top{k}_rank_corr": rank_corr,
    }


@torch.no_grad()
def _topk_rank_corr(g: torch.Tensor, c: torch.Tensor, gk: torch.Tensor) -> float:
    """Mean Spearman rho between gold and cand orderings restricted to gold's top-k."""
    g_sel = torch.gather(g, 1, gk)
    c_sel = torch.gather(c, 1, gk)

    def _ranks(x):
        order = x.argsort(dim=1)
        r = torch.empty_like(order, dtype=torch.float)
        ar = torch.arange(x.shape[1], device=x.device, dtype=torch.float)
        r.scatter_(1, order, ar.expand_as(order))
        return r

    rg, rc = _ranks(g_sel), _ranks(c_sel)
    rg = rg - rg.mean(1, keepdim=True)
    rc = rc - rc.mean(1, keepdim=True)
    num = (rg * rc).sum(1)
    den = (rg.norm(dim=1) * rc.norm(dim=1)).clamp_min(1e-12)
    return (num / den).mean().item()


@torch.no_grad()
def loss_metrics(
    gold_logits: torch.Tensor, cand_logits: torch.Tensor, targets: torch.Tensor
) -> dict:
    """CE / PPL delta on held-out next-token targets (the held-out text set).

    `targets[i]` is the gold-head argmax for token i (self-supervised next-token proxy
    when we don't have real labels) OR real shifted input ids when a model is loaded.
    """
    ce_gold = F.cross_entropy(gold_logits.float(), targets)
    ce_cand = F.cross_entropy(cand_logits.float(), targets)
    return {
        "ce_gold": ce_gold.item(),
        "ce_cand": ce_cand.item(),
        "ce_delta": (ce_cand - ce_gold).item(),
        "ppl_gold": torch.exp(ce_gold).item(),
        "ppl_cand": torch.exp(ce_cand).item(),
        "ppl_ratio": torch.exp(ce_cand - ce_gold).item(),
    }


# --------------------------------------------------------------------------------------
# Calibration data: real hidden states from a cached model body, or synthetic.
# --------------------------------------------------------------------------------------
def _pick_cached_model() -> str | None:
    """Smallest realistic large-vocab model that's already cached.

    Both Qwen3-0.6B (vocab 151936, hidden 1024) and Llama-3.2-1B (vocab 128256,
    hidden 2048, tied) are present in ~/.cache/huggingface and exercise the flat
    large-vocab lm_head that FP4 is weakest on. Prefer Qwen3-0.6B: untied head (so the
    lm_head is a real standalone nn.Linear to quantize) and the smallest body.
    """
    from huggingface_hub import try_to_load_from_cache

    for repo in ("Qwen/Qwen3-0.6B", "NousResearch/Llama-3.2-1B", "Qwen/Qwen2.5-0.5B"):
        hit = try_to_load_from_cache(repo, "config.json")
        if isinstance(hit, str) and os.path.exists(hit):
            return repo
    return None


def load_real(model_id: str, n_tokens: int, seq_len: int, device, dtype):
    """Run the model BODY over real tokens, capture final-layer hidden states + the
    lm_head. Returns (lm_head_linear, hidden[N,H], gold_targets[N])."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    text = _calib_text()
    ids = tok(text, return_tensors="pt").input_ids[:, : n_tokens + 1].to(device)
    if ids.shape[1] < 2:
        raise RuntimeError("calibration text tokenized too short")

    # find the lm_head and the hidden state feeding it
    lm_head = model.get_output_embeddings()
    if not isinstance(lm_head, nn.Linear):
        # tied / non-Linear head: rebuild a Linear from its weight so we can quantize it
        W = lm_head.weight.detach()
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_(W)
        lm_head = lin

    captured = {}
    base = model.model if hasattr(model, "model") else model

    def hook(_m, _inp, out):
        # base model returns a BaseModelOutputWithPast, a tuple, or a tensor
        h = getattr(out, "last_hidden_state", None)
        if h is None:
            h = out[0] if isinstance(out, (tuple, list)) else out
        captured["h"] = h.detach()

    handle = base.register_forward_hook(hook)
    with torch.no_grad():
        model(ids[:, :-1])
    handle.remove()

    h = captured["h"].reshape(-1, captured["h"].shape[-1])  # [N, hidden]
    # gold next-token targets = the real shifted input ids
    targets = ids[0, 1 : 1 + h.shape[0]].to(h.device)
    return lm_head.to(device=device, dtype=dtype), h, targets


def _calib_text() -> str:
    """A few hundred tokens of generic English (stands in for wikitext). Kept inline so
    the harness needs no dataset download; swap for a real wikitext slice if desired."""
    para = (
        "The quick brown fox jumps over the lazy dog. In the field of machine learning, "
        "quantization reduces the numerical precision of model weights to save memory and "
        "increase throughput, at some cost to accuracy. The language model head projects "
        "hidden states onto a large vocabulary, producing one logit per token. Because the "
        "vocabulary is large and the weight distribution is comparatively flat, low-bit "
        "formats such as FP4 tend to lose accuracy here more than in the transformer body. "
    )
    return para * 12


def load_synthetic(vocab: int, hidden: int, n_tokens: int, device, dtype):
    """Random large-vocab head + random hidden states. Proves the harness end-to-end
    with no model download. Uses a heavy-tailed row scale so per-row metrics are
    non-degenerate (mimics rare/frequent-token norm spread)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    W = torch.randn(vocab, hidden, generator=g)
    row_scale = torch.empty(vocab).log_normal_(mean=0.0, std=1.0, generator=g)
    W = (W * row_scale.unsqueeze(1)).to(device=device, dtype=dtype)
    lin = nn.Linear(hidden, vocab, bias=False).to(device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(W)
    h = torch.randn(n_tokens, hidden, generator=g).to(device=device, dtype=dtype)
    gold = (h.float() @ W.float().t())
    targets = gold.argmax(-1)
    return lin, h, targets


# --------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _logits(head: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    out = head(hidden)
    return out[0] if isinstance(out, tuple) else out


def run(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_id = None
    if not args.synthetic:
        model_id = args.model or _pick_cached_model()

    if model_id:
        print(f"[harness] real path: {model_id}", flush=True)
        lm_head, hidden, targets = load_real(
            model_id, args.n_tokens, args.seq_len, device, dtype
        )
    else:
        if not args.synthetic:
            print("[harness] no cached model found -> synthetic large-vocab path", flush=True)
        lm_head, hidden, targets = load_synthetic(
            args.vocab, args.hidden, args.n_tokens, device, dtype
        )

    W_gold = lm_head.weight.detach()
    vocab, hidden_dim = W_gold.shape
    print(
        f"[harness] head [vocab={vocab}, hidden={hidden_dim}] "
        f"calib tokens={hidden.shape[0]} dtype={dtype}",
        flush=True,
    )

    # gold logits (bf16/fp32 head) — the reference everything is scored against
    gold_logits = _logits(lm_head, hidden)
    ctx = {
        "bias": lm_head.bias.detach() if getattr(lm_head, "bias", None) is not None else None,
        "recipe": None,
        "device": device,
        "dtype": dtype,
    }

    methods = args.methods or list(_METHODS)
    results: list[MetricResult] = []
    for name in methods:
        fn = _METHODS.get(name)
        if fn is None:
            print(f"[harness] unknown method '{name}', skipping", flush=True)
            continue
        try:
            out = fn(W_gold, hidden, ctx)
        except NotImplementedError as e:
            print(f"[harness] {name}: STUB ({e})", flush=True)
            continue
        head = _scorable(out, ctx)
        W_cand = head.weight.detach()
        cand_logits = _logits(head, hidden)
        mr = MetricResult(name)
        mr.weight = weight_metrics(W_gold, W_cand)
        mr.logit = logit_metrics(gold_logits, cand_logits, k=args.topk)
        mr.loss = loss_metrics(gold_logits, cand_logits, targets)
        results.append(mr)
        _print_result(mr)

    summary = {
        "model": model_id or f"synthetic[vocab={vocab},hidden={hidden_dim}]",
        "calib_tokens": int(hidden.shape[0]),
        "results": {
            r.name: {"weight": r.weight, "logit": r.logit, "loss": r.loss}
            for r in results
        },
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"[harness] wrote {args.json_out}", flush=True)
    return summary


def _print_result(mr: MetricResult) -> None:
    w, lg, ls = mr.weight, mr.logit, mr.loss
    print(f"\n=== {mr.name} ===", flush=True)
    print(
        f"  weight: rel_fro={w['rel_fro']:.4f} "
        f"row_relerr(mean/p99/max)={w['row_relerr_mean']:.4f}/"
        f"{w['row_relerr_p99']:.4f}/{w['row_relerr_max']:.4f} "
        f"row_snr_dB(mean/min)={w['row_snr_db_mean']:.1f}/{w['row_snr_db_min']:.1f}",
        flush=True,
    )
    kkey = [k for k in lg if k.startswith("top") and "overlap" in k][0]
    rkey = [k for k in lg if k.endswith("rank_corr")][0]
    print(
        f"  logit : mse={lg['mse']:.4g} cos={lg['cosine']:.5f} "
        f"KL(gold||cand)={lg['kl_gold_cand']:.4g} top1={lg['top1_agree']:.4f} "
        f"{kkey}={lg[kkey]:.4f} {rkey}={lg[rkey]:.4f}",
        flush=True,
    )
    print(
        f"  loss  : CE gold={ls['ce_gold']:.4f} cand={ls['ce_cand']:.4f} "
        f"delta={ls['ce_delta']:+.4f} ppl_ratio={ls['ppl_ratio']:.4f}",
        flush=True,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None, help="HF repo id; default = auto-pick cached")
    p.add_argument("--synthetic", action="store_true", help="force synthetic head path")
    p.add_argument("--n-tokens", type=int, default=512, help="calibration tokens")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab", type=int, default=151936, help="synthetic vocab size")
    p.add_argument("--hidden", type=int, default=1024, help="synthetic hidden dim")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--methods", nargs="*", default=None, help="subset of registered methods")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
