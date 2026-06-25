#!/usr/bin/env python
"""Phase-2 probe: does a low-rank bf16 residual on the frozen FP4 lm_head help?

PTQ-style, no training. Builds on the eval_fp4_head harness primitives (same FP4
quant code path the training swap uses) and answers the brief's CRITICAL question:

  The FP4 weight error E = W_bf16 - dequant(Q(W)) is ~9.5% UNIFORM e2m1 mantissa
  noise. If that error is mostly full-rank rounding noise, a rank-k SVD residual
  recovers little. We test TWO constructions over a rank sweep and report which,
  if either, actually shrinks the LOGIT / CE gap vs the bf16 head:

    1. plain  : E = U S Vt ; A = U[:,:k]*S[:k] , B = Vt[:k]      -> W ~= Q(W)+A B
    2. l2qer  : weight E by the calib activation 2nd-moment before the SVD, so the
                rank-k captures the directions that matter for the LOGITS, not the
                raw weight. SVD(E @ G) with G = diag(sqrt(mean(h^2))); then
                A = Uw[:,:k]*Sw[:k] , B = (Vwt[:k]) @ G^{-1}  (so A B ~= E but the
                truncation error is minimized in the activation-weighted norm).

We also measure the singular-value spectrum of E and of E@G to show directly how
(non-)low-rank the error is, and we report the residual+distillation composition
is unnecessary here (distillation is a body-side loss; the residual is a head-side
weight correction — they compose trivially by construction, see report).

Usage:
  PY=.../axolotl_torch211_experimental/bin/python
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src \
      $PY agent_space/residual_probe.py --model Qwen/Qwen3-0.6B --n-tokens 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "agent_space"))

from eval_fp4_head import (  # noqa: E402
    fp4_dequant_weight,
    load_real,
    load_synthetic,
    _pick_cached_model,
    logit_metrics,
    loss_metrics,
    weight_metrics,
)


# ----------------------------------------------------------------------------------
# Residual constructions.
# ----------------------------------------------------------------------------------
@torch.no_grad()
def quant_error(W_gold: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
    """E = W_bf16 - dequant(Q(W)), [V, H], in fp32."""
    Wq = fp4_dequant_weight(lin).float()
    return W_gold.float() - Wq


@torch.no_grad()
def svd_residual(E: torch.Tensor, k: int):
    """Plain truncated SVD of the quant error. Returns (A[V,k], B[k,H])."""
    # E is [V, H], V >> H, so full_matrices=False gives U[V,r], S[r], Vt[r,H],
    # r = min(V,H) = H. Cheap (H ~ 1-4k).
    U, S, Vt = torch.linalg.svd(E, full_matrices=False)
    A = U[:, :k] * S[:k].unsqueeze(0)  # [V, k]
    B = Vt[:k]  # [k, H]
    return A, B, S


@torch.no_grad()
def l2qer_residual(E: torch.Tensor, g: torch.Tensor, k: int):
    """Activation-weighted (L2QER-style) low-rank residual.

    Minimize ||(E - A B) diag(g)||_F over rank-k A,B. Substituting Ew = E diag(g),
    the optimum is the rank-k SVD of Ew: Ew = Uw Sw Vwt; then A = Uw[:,:k] Sw[:k],
    and B = Vwt[:k] diag(1/g) so that A B approximates E (un-weighted), with the
    truncation error minimized in the g-weighted (logit) norm. g = sqrt of the
    per-hidden-dim activation second moment on the calibration set.
    """
    Ew = E * g.unsqueeze(0)  # [V, H]
    Uw, Sw, Vwt = torch.linalg.svd(Ew, full_matrices=False)
    A = Uw[:, :k] * Sw[:k].unsqueeze(0)  # [V, k]
    B = Vwt[:k] / g.unsqueeze(0)  # [k, H]  (undo the column weighting)
    return A, B, Sw


# ----------------------------------------------------------------------------------
# Scorable head: FP4 store + low-rank residual added at logit time.
# ----------------------------------------------------------------------------------
class ResidualHead(nn.Module):
    """FP4 head whose logits get a low-rank bf16 correction (hidden@Bt)@At.

    weight property returns dequant(Q(W)) + A @ B (the effective corrected weight)
    so the harness's weight_metrics see the corrected head.
    """

    def __init__(self, lin: nn.Linear, A: torch.Tensor, B: torch.Tensor, dtype):
        super().__init__()
        from eval_fp4_head import make_fp4_head

        self.fp4 = make_fp4_head(lin)
        self.register_buffer("A", A.to(dtype))  # [V, k]
        self.register_buffer("B", B.to(dtype))  # [k, H]
        self.bias = lin.bias

    @property
    def weight(self) -> torch.Tensor:
        return (self.fp4.weight.float() + (self.A.float() @ self.B.float())).to(self.A.dtype)

    def forward(self, x):
        base = self.fp4(x)  # [M, V] FP4 GEMM logits
        corr = (x @ self.B.t()) @ self.A.t()  # [M,k] then [M,V]
        return base + corr


# ----------------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------------
@torch.no_grad()
def _logits(head, hidden):
    out = head(hidden)
    return out[0] if isinstance(out, tuple) else out


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_id = None if args.synthetic else (args.model or _pick_cached_model())
    if model_id:
        print(f"[probe] real path: {model_id}", flush=True)
        lin, hidden, targets = load_real(model_id, args.n_tokens, args.seq_len, device, dtype)
    else:
        print("[probe] synthetic large-vocab path", flush=True)
        lin, hidden, targets = load_synthetic(args.vocab, args.hidden, args.n_tokens, device, dtype)

    W_gold = lin.weight.detach()
    V, H = W_gold.shape
    print(f"[probe] head [vocab={V}, hidden={H}] calib tokens={hidden.shape[0]} dtype={dtype}", flush=True)

    gold_logits = _logits(lin, hidden)

    # quant error + spectra
    E = quant_error(W_gold, lin)  # [V,H] fp32
    g = hidden.float().pow(2).mean(0).sqrt().clamp_min(1e-8)  # [H] activation rms per dim

    A_plain_full, B_plain_full, S_plain = svd_residual(E, max(args.ranks))
    _, _, S_w = l2qer_residual(E, g, max(args.ranks))

    # spectrum diagnostics: how much energy lives in the leading k directions?
    def energy_frac(S, k):
        e = S.pow(2)
        return (e[:k].sum() / e.sum().clamp_min(1e-30)).item()

    fro_E = E.norm().item()
    print(f"\n[spectrum] ||E||_fro={fro_E:.3f}  rank(full)={len(S_plain)}", flush=True)
    print(f"[spectrum] plain  E   : sv[0]={S_plain[0]:.3f} sv[-1]={S_plain[-1]:.4f} "
          f"energy in top-{{8,16,32,64}}={[round(energy_frac(S_plain,k),4) for k in (8,16,32,64)]}", flush=True)
    print(f"[spectrum] l2qer  E@G : sv[0]={S_w[0]:.3f} sv[-1]={S_w[-1]:.4f} "
          f"energy in top-{{8,16,32,64}}={[round(energy_frac(S_w,k),4) for k in (8,16,32,64)]}", flush=True)

    # baseline (no residual) row
    fp4_head = ResidualHead(lin, torch.zeros(V, 1, device=device), torch.zeros(1, H, device=device), dtype)
    base_logits = _logits(fp4_head, hidden)
    results = {}

    def score(name, head):
        W_cand = head.weight.detach()
        cl = _logits(head, hidden)
        w = weight_metrics(W_gold, W_cand)
        lg = logit_metrics(gold_logits, cl, k=5)
        ls = loss_metrics(gold_logits, cl, targets)
        results[name] = {"weight": w, "logit": lg, "loss": ls}
        print(f"  {name:22s} rel_fro={w['rel_fro']:.4f} KL={lg['kl_gold_cand']:.5f} "
              f"top1={lg['top1_agree']:.4f} mse={lg['mse']:.4g} ce_delta={ls['ce_delta']:+.5f}", flush=True)

    print("\n=== rank sweep (vs bf16 gold) ===", flush=True)
    score("fp4_baseline", fp4_head)

    for k in args.ranks:
        Ap, Bp, _ = svd_residual(E, k)
        score(f"plain_k{k}", ResidualHead(lin, Ap, Bp, dtype))
    for k in args.ranks:
        Al, Bl, _ = l2qer_residual(E, g, k)
        score(f"l2qer_k{k}", ResidualHead(lin, Al, Bl, dtype))

    summary = {
        "model": model_id or f"synthetic[V={V},H={H}]",
        "calib_tokens": int(hidden.shape[0]),
        "fro_E": fro_E,
        "spectrum": {
            "plain_energy": {k: energy_frac(S_plain, k) for k in (8, 16, 32, 64)},
            "l2qer_energy": {k: energy_frac(S_w, k) for k in (8, 16, 32, 64)},
        },
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"\n[probe] wrote {args.json_out}", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--n-tokens", type=int, default=400)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--vocab", type=int, default=151936)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--ranks", type=int, nargs="*", default=[8, 16, 32, 64])
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
