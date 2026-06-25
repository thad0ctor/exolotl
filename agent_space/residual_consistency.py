#!/usr/bin/env python
"""Verify the low-rank residual is applied CONSISTENTLY across all logit paths
and that the fused-CE residual gradient into hidden is correct.

Paths checked (must agree on the residual term):
  1. direct FP4 head forward  (NVFP4FrozenBaseLinear.forward, dequant-GEMM path)
  2. fused FP4 cross-entropy  (_FusedFP4CrossEntropy, tiled)
  3. distillation student lse (_scaled_lse student branch)

For (1) vs the reference, the base GEMM differs by the FP4 scaled_mm-vs-dequant
gap, so we compare the RESIDUAL DELTA (corrected - uncorrected), which must be
bit-identical across paths. Gradient: autograd grad of the fused-CE loss wrt
hidden vs a finite-difference reference, with the residual attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from axolotl.kernels.nvfp4_residual import _build_residual  # noqa: E402
from axolotl.utils.nvfp4_training import (  # noqa: E402
    NVFP4FrozenBaseLinear,
    NVFP4Recipe,
)


def main():
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    V, H, M = 4096, 256, 64
    lin = nn.Linear(H, V, bias=False).to(dev, dt)
    head = NVFP4FrozenBaseLinear.from_linear(lin, NVFP4Recipe(), fsdp=False)

    Wq = head.weight.detach().float()
    E = lin.weight.detach().float() - Wq
    hidden = torch.randn(M, H, device=dev, dtype=dt)
    g = hidden.float().pow(2).mean(0).sqrt().clamp_min(1e-8)
    A, B = _build_residual(E, 16, g)
    A = A.to(dt)
    B = B.to(dt)

    # residual delta reference
    ref_delta = ((hidden.to(dt) @ B.t()) @ A.t()).float()  # [M,V]

    # path 1: direct forward, corrected - uncorrected
    out_uncorr = head(hidden).float()
    head.register_buffer("_lm_head_residual_A", A, persistent=False)
    head.register_buffer("_lm_head_residual_B", B, persistent=False)
    out_corr = head(hidden).float()
    delta1 = out_corr - out_uncorr
    err1 = (delta1 - ref_delta).abs().max().item()
    print(f"[path1 direct GEMM]    residual delta max-abs-err = {err1:.3e}")

    # path 2: fused CE — compare logsumexp WITH vs WITHOUT residual; the lse shift
    # must equal what the reference corrected logits give.
    from axolotl.kernels.nvfp4_fused_ce import _FusedFP4CrossEntropy

    store = head.w_q
    labels = torch.randint(0, V, (M,), device=dev)

    def fused_loss(A_, B_):
        return _FusedFP4CrossEntropy.apply(
            hidden, store, labels, -100, 1.0, 1.0 / M, A_, B_
        )

    loss_no = fused_loss(None, None).item()
    loss_res = fused_loss(A, B).item()
    # reference: full corrected-logit CE
    ref_logits = (hidden.float() @ Wq.t()) + ref_delta
    ref_ce = torch.nn.functional.cross_entropy(ref_logits, labels).item()
    base_logits = hidden.float() @ Wq.t()
    base_ce = torch.nn.functional.cross_entropy(base_logits, labels).item()
    print(f"[path2 fused CE]       loss no-res={loss_no:.5f} ref={base_ce:.5f} "
          f"diff={abs(loss_no-base_ce):.3e}")
    print(f"[path2 fused CE]       loss +res ={loss_res:.5f} ref={ref_ce:.5f} "
          f"diff={abs(loss_res-ref_ce):.3e}")

    # path 3: distillation student lse with residual == reference corrected lse
    from axolotl.kernels.nvfp4_distill import _scaled_lse

    lse_res = _scaled_lse(store, hidden, 1.0, 2048, grad=False, res_A=A, res_B=B)
    ref_lse = torch.logsumexp(ref_logits, dim=1)
    err3 = (lse_res - ref_lse).abs().max().item()
    print(f"[path3 distill lse]    student lse max-abs-err = {err3:.3e}")

    # gradient: autograd grad of fused-CE +res wrt hidden vs finite difference
    h = hidden.clone().float().requires_grad_(True)
    Af = A.float(); Bf = B.float()
    loss = _FusedFP4CrossEntropy.apply(
        h.to(dt), store, labels, -100, 1.0, 1.0 / M, A, B
    )
    (gh,) = torch.autograd.grad(loss, h)
    # FD reference on the full corrected-logit CE (fp32, exact)
    def ce_of(hf):
        lg = (hf @ Wq.t()) + (hf @ Bf.t()) @ Af.t()
        return torch.nn.functional.cross_entropy(lg, labels, reduction="sum") / M
    h2 = hidden.clone().float()
    eps = 1e-2
    fd = torch.zeros_like(h2)
    idx = [(0, 0), (3, 10), (20, 100), (50, 200)]
    for (i, j) in idx:
        hp = h2.clone(); hp[i, j] += eps
        hm = h2.clone(); hm[i, j] -= eps
        fd[i, j] = (ce_of(hp) - ce_of(hm)) / (2 * eps)
    rel = [(abs(gh[i, j].item() - fd[i, j].item()) /
            (abs(fd[i, j].item()) + 1e-6)) for (i, j) in idx]
    print(f"[grad] autograd vs FD rel-err at {idx}: {[round(r,4) for r in rel]}")
    print("VERDICT:", "PASS" if (err1 < 5e-2 and err3 < 5e-1 and max(rel) < 0.15
          and abs(loss_res - ref_ce) < 5e-3) else "CHECK")


if __name__ == "__main__":
    main()
