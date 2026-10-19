"""Low-rank bf16 residual correction for a frozen FP4 lm_head (Phase 2).

``quantize_lm_head`` freezes the output projection on its NVFP4 grid, adding
~9.5% UNIFORM e2m1 mantissa error. The quant error ``E = W_bf16 - dequant(Q(W))``
is therefore mostly FULL-RANK rounding noise: a plain rank-k SVD of ``E`` captures
almost none of it (measured: ~2-10% of E's energy in the top 64 singular
directions on real Qwen3 heads), so a plain SVD residual barely helps.

What DOES help is the L2QER (activation-weighted) construction. Weighting the
error columns by the calibration activation second moment before the SVD
concentrates the *logit-relevant* error into a low-rank subspace (measured:
50-69% of energy in the top 8, 63-82% in the top 64 — because the activations are
anisotropic, only a few hidden-dim directions actually move the logits). The
rank-k correction then captures the error that matters:

    PTQ A/B harness, Qwen3-8B head (hidden 4096), 800 calib tokens, vs bf16 head
      plain  rank-64 : KL 0.00696 -> 0.00667  (-4%,  noise)
      l2qer  rank-16 : KL 0.00696 -> 0.00298  (-57%), CE gap closed
      l2qer  rank-64 : KL 0.00696 -> 0.00268  (-61%)

Construction (run ONCE at load, after the FP4 quant):

  * plain : E = U S Vt ;  A = U[:,:k]*S[:k] [V,k] , B = Vt[:k] [k,H]
  * l2qer : g = sqrt(mean_t hidden[t]^2) per hidden dim (calib);  Ew = E*g;
            Ew = Uw Sw Vwt ;  A = Uw[:,:k]*Sw[:k] , B = (Vwt[:k]) / g
            (minimizes ||(E - A B) diag(g)||_F = the truncation error in the
            activation-weighted / logit norm, while A B still approximates E).

so the corrected head is  W ~= dequant(Q(W)) + A @ B, applied at logit time as a
two-matmul correction  (hidden @ B.t()) @ A.t()  threaded into EVERY path that
produces head logits (direct GEMM, fused CE tiles, distillation student logits),
so the paths agree bit-for-bit on the residual term.

SPEED: the residual is a PERMANENT cost (train + inference). Its own two matmuls
are ~9% of the base head GEMM at vocab 152k; materializing+adding the [M,V]
correction as a bolt-on is ~36%; fused into the CE tile loop (add per tile before
logsumexp, no extra [M,V] materialization) it lands ~10-15%. The cost is nearly
rank-independent (the [M,V] projection dominates, not the k FLOPs), so prefer the
SMALLEST k that captures the gain — rank 16 on these heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

LOG = logging.getLogger(__name__)

# Buffer names attached to the FP4 lm_head module.
_RES_A = "_lm_head_residual_A"  # [V, k] bf16
_RES_B = "_lm_head_residual_B"  # [k, H] bf16

# Buffer names attached to a frozen FP4 BASE linear (the generalization of the
# lm_head residual to every frozen base linear under LoRA/QLoRA). Distinct names
# so the lm_head paths (fused CE tiles, distillation) never double-apply.
_BASE_RES_A = "_base_residual_A"  # [out, k] bf16
_BASE_RES_B = "_base_residual_B"  # [k, in] bf16


@dataclass
class ResidualConfig:
    """Per-model residual knobs (validated copy of the schema block)."""

    enabled: bool = False
    rank: int = 32
    calibration: str = "activation"  # "activation" (L2QER) or "svd" (plain)
    calib_tokens: int = 512


def residual_buffers(module: nn.Module):
    """Return ``(A, B)`` residual buffers on ``module``, or ``(None, None)``."""
    return getattr(module, _RES_A, None), getattr(module, _RES_B, None)


def has_residual(module: nn.Module) -> bool:
    a, b = residual_buffers(module)
    return a is not None and b is not None


@torch.no_grad()
def _build_residual(
    E: torch.Tensor, rank: int, g: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank factors ``(A[V,k], B[k,H])`` of the quant error ``E[V,H]``.

    ``g`` (per-hidden-dim activation rms, [H]) selects the L2QER construction;
    ``None`` selects the plain SVD. All math in fp32; factors returned in fp32
    (the caller casts to the head dtype).
    """
    # Randomized svd_lowrank (q = 4k oversampling, niter=4): only the top-k
    # (k<=64) subspace is consumed, and a FULL svd of a 9B head's error matrix
    # ([248320, 4096] fp32) transiently allocates ~15 GiB (Ef + U + Vt +
    # cusolver workspace; measured as a 41-vs-28 GiB load-time spike that
    # inflated benchmark peaks and can OOM tight long-context loads), while
    # the randomized projection's transients are ~tens of MiB beyond Ef.
    Ef = E.float()
    k = min(rank, min(Ef.shape))
    q = min(max(4 * k, 64), min(Ef.shape))
    if g is None:
        U, S, V = torch.svd_lowrank(Ef, q=q, niter=4)
        A = U[:, :k] * S[:k].unsqueeze(0)
        B = V[:, :k].t().contiguous()
    else:
        gf = g.float().clamp_min(1e-8)
        # in-place: Ef is this function's private fp32 copy and is not needed
        # unweighted in this branch — saves a second [V, H] fp32 plane
        Ew = Ef.mul_(gf.unsqueeze(0))
        Uw, Sw, Vw = torch.svd_lowrank(Ew, q=q, niter=4)
        A = Uw[:, :k] * Sw[:k].unsqueeze(0)
        B = Vw[:, :k].t() / gf.unsqueeze(0)
    return A.contiguous(), B.contiguous()


@torch.no_grad()
def attach_residual(
    module: nn.Module,
    teacher_w: torch.Tensor,
    cfg: ResidualConfig,
    calib_hidden: torch.Tensor | None,
) -> bool:
    """Compute and attach the residual factors to an FP4 lm_head ``module``.

    ``teacher_w`` is the ORIGINAL bf16 head weight ``[V, H]`` (the same buffer the
    distillation teacher uses); ``module.weight`` is the dequantized FP4 head, so
    ``E = teacher_w - module.weight``. ``calib_hidden`` ([N, H]) drives the L2QER
    column weighting (activation rms); only its second moment is used, so a few
    hundred real tokens suffice. Plain SVD ignores it. Returns True on success.

    Registered as non-persistent buffers (reconstructable from the head + a short
    calibration forward; would otherwise bloat the checkpoint by ~2*k*(V+H) bf16).
    """
    if not cfg.enabled or cfg.rank <= 0:
        return False
    W_q = getattr(module, "weight", None)
    if W_q is None:
        return False
    dev = W_q.device
    teacher_w = teacher_w.to(device=dev, dtype=torch.float32)
    E = teacher_w - W_q.detach().float()

    g = None
    if cfg.calibration == "activation":
        if calib_hidden is None or calib_hidden.numel() == 0:
            LOG.warning(
                "lm_head_residual: calibration='activation' but no calibration "
                "hidden states captured; falling back to plain SVD."
            )
        else:
            h = calib_hidden.reshape(-1, calib_hidden.shape[-1]).to(dev).float()
            g = h.pow(2).mean(0).sqrt()  # [H] activation rms per hidden dim

    A, B = _build_residual(E, cfg.rank, g)
    dtype = W_q.dtype
    module.register_buffer(_RES_A, A.to(dtype), persistent=False)
    module.register_buffer(_RES_B, B.to(dtype), persistent=False)
    LOG.info(
        "NVFP4 lm_head_residual: attached %s residual rank=%d (%s) A%s B%s",
        cfg.calibration,
        A.shape[1],
        "L2QER" if g is not None else "plain-SVD",
        tuple(A.shape),
        tuple(B.shape),
    )
    return True


@torch.no_grad()
def attach_base_residual(
    module: nn.Module,
    w_hp: torch.Tensor,
    rank: int,
    mean_x2: torch.Tensor | None,
) -> bool:
    """Compute + attach the L2QER residual to a frozen FP4 BASE linear ``module``.

    Called AT SWAP TIME, while ``w_hp`` (the original bf16 master ``[out, in]``)
    is still in memory (the storage swap discards it right after). ``module`` is
    the freshly built NVFP4 base module, whose ``.weight`` property dequantizes
    the packed store, so ``E = w_hp - module.weight`` is the quant error.
    ``mean_x2`` is the per-INPUT-channel running mean of x^2 (fp32 ``[in]``) from
    the pre-swap calibration forward; ``g = sqrt(mean_x2)`` selects the L2QER
    construction in :func:`_build_residual` (``None`` -> plain SVD, ablation
    only — measured useless on the near-full-rank e2m1 error). Returns True on
    success.

    Buffers are non-persistent (like the lm_head residual): they reconstruct
    deterministically at load from the original weights + a short calibration
    forward, so they stay out of the adapter/model checkpoints.
    """
    if rank <= 0:
        return False
    w_dq = getattr(module, "weight", None)  # dequantized FP4 [out, in]
    if w_dq is None:
        return False
    E = (
        w_hp.detach().to(device=w_dq.device, dtype=torch.float32)
        - w_dq.detach().float()
    )
    g = None
    if mean_x2 is not None:
        g = mean_x2.to(device=E.device, dtype=torch.float32).clamp_min(0).sqrt()
    A, B = _build_residual(E, rank, g)
    dtype = w_hp.dtype
    module.register_buffer(_BASE_RES_A, A.to(dtype), persistent=False)
    module.register_buffer(_BASE_RES_B, B.to(dtype), persistent=False)
    return True


def base_residual_buffers(module: nn.Module):
    """Return ``(A, B)`` base-residual buffers on ``module``, or ``(None, None)``."""
    return getattr(module, _BASE_RES_A, None), getattr(module, _BASE_RES_B, None)


def has_base_residual(module: nn.Module) -> bool:
    a, b = base_residual_buffers(module)
    return a is not None and b is not None


def apply_residual(module: nn.Module, hidden: torch.Tensor, logits: torch.Tensor):
    """Add the low-rank correction ``(hidden @ B.t()) @ A.t()`` to full ``logits``.

    No-op (returns ``logits`` unchanged) if no residual is attached. ``hidden`` is
    ``[M, H]`` (already 2D / matching ``logits``' leading dim). Used by the direct
    GEMM head forward; the fused-CE path uses :func:`residual_tile` per vocab tile.
    """
    A, B = residual_buffers(module)
    if A is None:
        return logits
    h = hidden.reshape(-1, hidden.shape[-1])
    corr = (h.to(B.dtype) @ B.t()) @ A.t()  # [M, V]
    return logits + corr.reshape(logits.shape).to(logits.dtype)


def residual_hidden_proj(A, B, hidden: torch.Tensor) -> torch.Tensor | None:
    """Precompute ``hidden @ B.t()`` -> ``[M, k]`` once for tiled application.

    The CE/distill tile loops call this ONCE outside the loop, then
    :func:`residual_tile` per vocab tile, so ``hidden @ B.t()`` (the only
    H-contracting matmul) is not repeated per tile.
    """
    if A is None or B is None:
        return None
    return hidden.reshape(-1, hidden.shape[-1]).to(B.dtype) @ B.t()  # [M, k]


def residual_tile(A, hidden_b: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Residual logit contribution for vocab rows ``[lo, hi)``: ``hidden_b @ A[lo:hi].t()``.

    ``hidden_b`` is the ``[M, k]`` output of :func:`residual_hidden_proj`. Returns
    ``[M, hi-lo]`` to add into the tile's logits before logsumexp.
    """
    return hidden_b @ A[lo:hi].t()
