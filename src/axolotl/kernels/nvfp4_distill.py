"""KL-distillation aux-loss for a frozen FP4 lm_head (accuracy recovery).

``quantize_lm_head`` freezes the output projection on its NVFP4 grid. The frozen
weight adds ~9.5% uniform e2m1 mantissa noise (~21% argmax flips, all near-ties;
+~0.04 nats CE on an 8B head) and CANNOT be moved off the grid. The trainable
*body*, however, can adapt to it: we add a KL term that pulls the FP4 head's
logits toward the ORIGINAL bf16 head's logits, so the body learns to emit hidden
states whose FP4 projection matches the bf16 projection.

    total = CE_student + lambda * T^2 * KL(softmax(z_T / T) || softmax(z_S / T))

  * z_S : the FP4 (student) head logits  = dequant(W_fp4) @ hidden  (reuses the
          same FP4 store the model trains against).
  * z_T : the bf16 (teacher) head logits = W_bf16 @ hidden          (the original
          head weight, retained frozen at swap time — NOT the lossy dequant).

Phase-1 prototype. The FP4 head exists for SPEED, so every knob bounds the
per-step teacher cost:

  * the teacher/student logits are always computed in VOCAB TILES — peak memory
    is one ``[tokens, block]`` tile, never the full ``[tokens, vocab]``;
  * ``top_k`` restricts the KL to the teacher's top-k tokens + one aggregated
    tail bucket, bounding the softmax/KL math (the teacher matmul over the full
    vocab is still the dominant cost — that is what ``cadence`` amortizes);
  * ``cadence`` applies the term only every N steps;
  * ``teacher="precomputed"`` (stub) would read offline-cached top-k teacher
    logits per token for ~zero per-step teacher cost — the speed-optimal path.

The KL gradient flows ONLY into ``hidden`` (and thence the trainable body); both
heads are frozen, so neither weight receives a gradient from this term.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

LOG = logging.getLogger(__name__)


@dataclass
class DistillState:
    """Per-model distillation knobs, attached to the ForCausalLM at swap time.

    ``step_counter`` is a 1-element CPU tensor bumped each applied forward so the
    cadence gate is stateful without touching the Trainer's global step (which is
    not visible inside the patched model forward).
    """

    enabled: bool = False
    lambda_: float = 1.0
    temperature: float = 1.0
    top_k: int | None = None
    cadence: int = 1
    teacher: str = "live"
    vocab_block: int = 8192
    # Bumped once per distillation-eligible forward; the cadence gate fires when
    # (counter % cadence == 0). A plain Python int (forward runs eager, single
    # process per model replica; DDP replicas stay in lockstep on step count).
    step_counter: int = 0
    # diagnostics, updated in-place each applied step (read by the bench harness)
    last_kl: float = 0.0
    last_applied: bool = False
    _warned_precomputed: bool = False


def _teacher_weight(lm_head: nn.Module) -> torch.Tensor | None:
    """The retained ORIGINAL bf16 head weight ``[V, H]``, or None.

    Stored as a buffer ``_distill_teacher_w`` on the FP4 head module at swap time
    (before the bf16 source is freed). None means distillation can't run (the
    teacher was not retained) and the caller skips the term.
    """
    w = getattr(lm_head, "_distill_teacher_w", None)
    if w is None:
        return None
    return w


def _fp4_student_store(lm_head: nn.Module):
    """Row-sliceable ``[V, H]`` NVFP4 store for the FP4 student head, or None."""
    from axolotl.kernels.nvfp4_fused_ce import _nvfp4_lm_head_store

    return _nvfp4_lm_head_store(lm_head)


def _student_logits_tile(
    store, lo: int, hi: int, hidden: torch.Tensor, res_A=None, res_B=None
) -> torch.Tensor:
    """FP4 student logits for vocab rows ``[lo, hi)``: dequant tile then matmul.

    Mirrors the fused-CE tiling (``_dequant_vocab_tile``): one ``[Vb, H]`` weight
    tile is dequantized FP4->hidden.dtype on read, never the whole table. The
    matmul carries hidden's grad (the body), the weight does not (frozen).

    When a low-rank head residual is attached (``res_A`` [V,k] / ``res_B`` [k,H]),
    its contribution ``(hidden @ res_B.t()) @ res_A[lo:hi].t()`` is added so the
    STUDENT logits the KL sees are the SAME corrected logits the FP4-head forward
    and fused CE produce (the residual reduces the student error before the KL —
    they compose, no double-count).
    """
    from axolotl.kernels.nvfp4_fused_ce import _dequant_vocab_tile

    w_tile = _dequant_vocab_tile(store, lo, hi, hidden.dtype)  # [Vb, H]
    logits = hidden @ w_tile.t()  # [M, Vb]
    if res_A is not None:
        logits = logits + (hidden.to(res_B.dtype) @ res_B.t()) @ res_A[lo:hi].t()
    return logits


def _teacher_logits_tile(
    teacher_w: torch.Tensor, lo: int, hi: int, hidden: torch.Tensor
) -> torch.Tensor:
    """bf16 teacher logits for vocab rows ``[lo, hi)`` (no grad: target)."""
    with torch.no_grad():
        return hidden @ teacher_w[lo:hi].t()  # [M, Vb]


@torch.no_grad()
def _teacher_pass(
    teacher_w: torch.Tensor,
    hidden: torch.Tensor,
    vocab_block: int,
    top_k: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """One tiled teacher forward -> (lse, max, topk_vals, topk_idx).

    Returns the numerically-stable fp32 logsumexp and running max over the FULL
    vocab (so softmax probabilities are exact), plus — when ``top_k`` is set — the
    global top-k teacher logits and their vocab indices (online top-k across
    tiles, never materializing the full ``[M, V]`` logits).
    """
    M = hidden.shape[0]
    V = teacher_w.shape[0]
    device = hidden.device

    running_max = torch.full((M,), float("-inf"), device=device, dtype=torch.float32)
    running_sum = torch.zeros(M, device=device, dtype=torch.float32)

    tk_vals = tk_idx = None
    if top_k is not None:
        k = min(top_k, V)
        tk_vals = torch.full((M, k), float("-inf"), device=device, dtype=torch.float32)
        tk_idx = torch.zeros(M, k, device=device, dtype=torch.long)

    for lo in range(0, V, vocab_block):
        hi = min(lo + vocab_block, V)
        z = _teacher_logits_tile(teacher_w, lo, hi, hidden).float()  # [M, Vb]

        tile_max = z.max(dim=1).values
        new_max = torch.maximum(running_max, tile_max)
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
            z - new_max.unsqueeze(1)
        ).sum(dim=1)
        running_max = new_max

        if top_k is not None:
            # merge this tile's top-k into the running top-k (online selection)
            kt = min(top_k, hi - lo)
            t_vals, t_cols = z.topk(kt, dim=1)
            t_idx = t_cols + lo
            cat_vals = torch.cat([tk_vals, t_vals], dim=1)
            cat_idx = torch.cat([tk_idx, t_idx], dim=1)
            sel = cat_vals.topk(tk_vals.shape[1], dim=1)
            tk_vals = sel.values
            tk_idx = cat_idx.gather(1, sel.indices)

    lse = running_max + torch.log(running_sum)
    return lse, running_max, tk_vals, tk_idx


def _kl_topk(
    student_store,
    teacher_w: torch.Tensor,
    hidden: torch.Tensor,
    teacher_lse: torch.Tensor,
    tk_idx: torch.Tensor,
    temperature: float,
    vocab_block: int,
    res_A=None,
    res_B=None,
) -> torch.Tensor:
    """KL over the teacher's top-k tokens + one aggregated tail bucket.

    The teacher distribution is collapsed to (k explicit outcomes, 1 tail mass);
    the student is evaluated on the SAME partition. This bounds the per-token KL
    to k+1 terms while keeping it a proper KL between two valid distributions over
    that partition.

    Student logits at the k teacher-selected vocab rows are gathered by dequantizing
    only those rows (a tiny ``[M*k]`` row-slice), and the student tail mass is
    derived from the student's full-vocab logsumexp (one extra tiled student pass).
    Both passes are tiled; the teacher matmul is the dominant cost and is already
    paid in ``_teacher_pass``.
    """
    T = temperature
    M, k = tk_idx.shape

    # --- teacher side (no grad: target) ---
    with torch.no_grad():
        # teacher full-vocab scaled logsumexp L_t. teacher_lse is the UNscaled
        # (T=1) full lse from the tiled pass; reuse it when T==1, else retile.
        L_t = (
            teacher_lse
            if T == 1.0
            else _scaled_lse(teacher_w, hidden, T, vocab_block, grad=False)
        )
        t_rows = teacher_w[tk_idx.reshape(-1)].reshape(M, k, -1)  # [M, k, H]
        z_t_k = torch.einsum("mh,mkh->mk", hidden.float(), t_rows.float())  # [M, k]
        p_t_k = torch.exp(z_t_k / T - L_t.unsqueeze(1))  # [M, k]
        # tail = remaining probability mass as one outcome (proper partition).
        p_t_tail = (1.0 - p_t_k.sum(dim=1)).clamp_min(0.0)  # [M]

    # --- student side (carries grad into hidden) ---
    s_rows = _dequant_rows(student_store, tk_idx.reshape(-1), hidden.dtype)
    s_rows = s_rows.reshape(M, k, -1)  # [M, k, H]
    z_s_k = torch.einsum("mh,mkh->mk", hidden, s_rows)  # [M, k]
    if res_A is not None:
        # residual logit at the k teacher-selected vocab rows: gather A's rows.
        a_rows = res_A[tk_idx.reshape(-1)].reshape(M, k, -1).to(res_B.dtype)  # [M,k,r]
        hb = hidden.to(res_B.dtype) @ res_B.t()  # [M, r]
        z_s_k = z_s_k + torch.einsum("mr,mkr->mk", hb, a_rows).to(z_s_k.dtype)
    L_s = _scaled_lse(
        student_store, hidden, T, vocab_block, grad=True, res_A=res_A, res_B=res_B
    )  # [M]

    log_p_s_k = z_s_k / T - L_s.unsqueeze(1)  # [M, k]
    # student tail log-mass: log(1 - sum_k p_s_k), clamped off 0.
    p_s_tail = (1.0 - torch.exp(log_p_s_k).sum(dim=1)).clamp_min(1e-20)
    log_p_s_tail = torch.log(p_s_tail)

    # KL(P_T || P_S) over the (k + tail) partition, per token, then mean.
    kl_k = (p_t_k * (torch.log(p_t_k.clamp_min(1e-20)) - log_p_s_k)).sum(dim=1)
    kl_tail = p_t_tail * (torch.log(p_t_tail.clamp_min(1e-20)) - log_p_s_tail)
    return (kl_k + kl_tail).mean()


def _scaled_lse(
    store_or_w,
    hidden: torch.Tensor,
    T: float,
    vocab_block: int,
    grad: bool,
    res_A=None,
    res_B=None,
) -> torch.Tensor:
    """fp32 logsumexp of logits/T over the full vocab, tiled.

    ``store_or_w`` is either an NVFP4 store (student, dequant per tile) or a dense
    ``[V, H]`` teacher weight. ``grad`` controls whether the student pass tracks
    autograd (teacher is always no-grad). ``res_A``/``res_B`` (student only) add
    the low-rank head residual per tile so the student lse matches the corrected
    student logits.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.kernels.nvfp4_fused_ce import _dequant_vocab_tile

    # NVFP4Tensor IS a torch.Tensor subclass, so test the FP4 type explicitly:
    # a plain dense weight goes through the bf16 matmul, an NVFP4 store is
    # dequantized per tile (a row-slice through `store[lo:hi]` would otherwise hit
    # the NVFP4 scaled-mm path, which is the wrong arithmetic here).
    is_store = isinstance(store_or_w, NVFP4Tensor)
    V = store_or_w.shape[0]
    M = hidden.shape[0]
    device = hidden.device
    running_max = torch.full((M,), float("-inf"), device=device, dtype=torch.float32)
    running_sum = torch.zeros(M, device=device, dtype=torch.float32)
    hb = None if (res_A is None or not is_store) else (hidden.to(res_B.dtype) @ res_B.t())

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for lo in range(0, V, vocab_block):
            hi = min(lo + vocab_block, V)
            if is_store:
                w_tile = _dequant_vocab_tile(store_or_w, lo, hi, hidden.dtype)
                zt = hidden @ w_tile.t()
                if hb is not None:
                    zt = zt + hb @ res_A[lo:hi].t()
                z = zt.float() / T
            else:
                z = (hidden @ store_or_w[lo:hi].t()).float() / T
            tile_max = z.max(dim=1).values.detach()
            new_max = torch.maximum(running_max, tile_max)
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                z - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max
    return running_max + torch.log(running_sum)


def _dequant_rows(store, idx: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a scattered set of vocab rows ``idx`` from an NVFP4 store.

    NVFP4 blocks lie along the hidden dim, so each vocab row is a self-contained
    slice of qdata/scale — gathering the rows is bit-identical to dequantizing the
    full table and gathering. No grad through the frozen weight.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    _, fctx = store.__tensor_flatten__()
    sub = NVFP4Tensor.__tensor_unflatten__(
        {
            "qdata": store.qdata[idx],
            "scale": store.scale[idx],
            "per_tensor_scale": store.per_tensor_scale,
        },
        fctx,
        None,
        None,
    )
    return sub.dequantize(dtype)


def _kl_full(
    student_store,
    teacher_w: torch.Tensor,
    hidden: torch.Tensor,
    teacher_lse: torch.Tensor,
    temperature: float,
    vocab_block: int,
    res_A=None,
    res_B=None,
) -> torch.Tensor:
    """Exact full-vocab KL(P_T || P_S), tiled (no top_k bucketing).

    Two tiled passes share each vocab tile: teacher probs (no grad) and student
    log-probs (grad into hidden). Per-tile contribution to
    sum_v p_T(v) (log p_T(v) - log p_S(v)) is accumulated; the student log-norm
    uses the student's own full-vocab scaled lse.
    """
    from axolotl.kernels.nvfp4_fused_ce import _dequant_vocab_tile

    T = temperature
    V = teacher_w.shape[0]
    M = hidden.shape[0]

    t_lse = (
        teacher_lse
        if T == 1.0
        else _scaled_lse(teacher_w, hidden, T, vocab_block, grad=False)
    )
    s_lse = _scaled_lse(
        student_store, hidden, T, vocab_block, grad=True, res_A=res_A, res_B=res_B
    )
    hb = None if res_A is None else (hidden.to(res_B.dtype) @ res_B.t())  # [M,r]

    kl = hidden.new_zeros((M,), dtype=torch.float32)
    for lo in range(0, V, vocab_block):
        hi = min(lo + vocab_block, V)
        with torch.no_grad():
            z_t = (hidden @ teacher_w[lo:hi].t()).float() / T
            log_p_t = z_t - t_lse.unsqueeze(1)
            p_t = torch.exp(log_p_t)
        w_tile = _dequant_vocab_tile(student_store, lo, hi, hidden.dtype)
        z_s_lin = hidden @ w_tile.t()
        if hb is not None:
            z_s_lin = z_s_lin + hb @ res_A[lo:hi].t()
        z_s = z_s_lin.float() / T
        log_p_s = z_s - s_lse.unsqueeze(1)
        kl = kl + (p_t * (log_p_t - log_p_s)).sum(dim=1)
    return kl.mean()


def lm_head_distillation_loss(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    state: DistillState,
) -> torch.Tensor | None:
    """KL aux-loss ``lambda * T^2 * KL(teacher || student)``, or None if skipped.

    ``hidden`` is the ``[*, H]`` activation feeding the lm_head (the SAME tensor
    the student CE consumes — reused, not recomputed). Returns None when the term
    is gated off (disabled, off-cadence step, teacher not retained, or store not
    tile-able), so the caller adds nothing.
    """
    state.last_applied = False
    if not state.enabled or state.lambda_ == 0.0:
        return None

    # cadence gate: apply only every N forwards (amortize the teacher matmul)
    state.step_counter += 1
    if state.cadence > 1 and (state.step_counter % state.cadence) != 0:
        return None

    teacher_w = _teacher_weight(lm_head)
    store = _fp4_student_store(lm_head)
    if teacher_w is None or store is None:
        return None

    if state.teacher == "precomputed" and not state._warned_precomputed:
        # Phase-1 stub: no offline cache builder yet. The interface is here; until
        # a cache is attached (`lm_head._distill_teacher_cache`), fall through to
        # the live teacher so training still runs. Warn once per run.
        state._warned_precomputed = True
        if getattr(lm_head, "_distill_teacher_cache", None) is None:
            LOG.warning(
                "lm_head_distillation: teacher='precomputed' but no cache attached; "
                "using the live teacher this run (Phase-1 stub)."
            )
        # TODO(phase-2): read top-k teacher logits/idx from the attached cache keyed
        # by token id (and position), skipping the teacher matmul entirely. That is
        # the speed-optimal path (near-zero per-step teacher cost).

    h2d = hidden.reshape(-1, hidden.shape[-1])
    teacher_w = teacher_w.to(device=h2d.device, dtype=h2d.dtype)

    # Low-rank head residual (Phase 2), if attached: thread it into the STUDENT
    # logits so the KL sees the corrected student (the residual reduces the student
    # error before the KL — composes with distillation, no double-count).
    res_A = getattr(lm_head, "_lm_head_residual_A", None)
    res_B = getattr(lm_head, "_lm_head_residual_B", None)

    lse, _, tk_vals, tk_idx = _teacher_pass(
        teacher_w, h2d, state.vocab_block, state.top_k
    )

    if state.top_k is not None:
        kl = _kl_topk(
            store, teacher_w, h2d, lse, tk_idx, state.temperature, state.vocab_block,
            res_A=res_A, res_B=res_B,
        )
    else:
        kl = _kl_full(
            store, teacher_w, h2d, lse, state.temperature, state.vocab_block,
            res_A=res_A, res_B=res_B,
        )

    state.last_kl = float(kl.detach())
    state.last_applied = True
    # Hinton T^2 keeps the gradient magnitude comparable across temperatures.
    return state.lambda_ * (state.temperature**2) * kl


# --- model forward wiring -----------------------------------------------------
#
# The KL term needs the SAME hidden states that feed the student CE. We own the
# head computation in one wrapped forward: run the base model once to get hidden,
# compute the student CE (reusing the fused FP4 CE path when present, else the
# materialized head + HF loss), then add the KL term computed from the same
# hidden. This avoids a second body forward and keeps the student CE path
# bit-identical to the non-distill run.

import functools  # noqa: E402

_PATCHED_DISTILL: set = set()


def _student_ce_and_hidden(self, args, kwargs, labels, num_items_in_batch):
    """Run the base model, return (ce_loss, hidden, outputs).

    Mirrors the fused-CE forward prologue. Uses the fused FP4 CE when the head is
    a tile-able FP4 store; otherwise falls back to materialized logits + the HF
    loss function. ``hidden`` is the [*, H] activation feeding the head.
    """
    from axolotl.kernels.nvfp4_fused_ce import (
        _nvfp4_lm_head_store,
        fused_fp4_cross_entropy,
    )

    lm_head = self.get_output_embeddings()
    base = getattr(self, "model", None)
    if base is None:
        return None, None, None

    outputs = base(*args, **kwargs)
    hidden = outputs.last_hidden_state

    if _nvfp4_lm_head_store(lm_head) is not None:
        ce = fused_fp4_cross_entropy(
            hidden,
            lm_head,
            labels,
            num_items_in_batch=num_items_in_batch,
            shift=True,
        )
        if ce is not None:
            return ce, hidden, outputs

    # Fallback: materialized logits + HF loss (still gives us hidden for KL).
    logits = lm_head(hidden)
    loss_fn = getattr(self, "loss_function", None)
    vocab = logits.shape[-1]
    if loss_fn is not None:
        ce = loss_fn(
            logits, labels, vocab, num_items_in_batch=num_items_in_batch
        )
    else:
        from transformers.loss.loss_utils import ForCausalLMLoss

        ce = ForCausalLMLoss(
            logits, labels, vocab, num_items_in_batch=num_items_in_batch
        )
    return ce, hidden, outputs


def _make_distill_forward(orig_forward, state: DistillState):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        # Prefer a per-model state (set at patch time) so a re-patched class picks
        # up the current run's knobs; fall back to the closure's state.
        cur_state = getattr(self, "_nvfp4_distill_state", state)
        labels = kwargs.get("labels")
        lm_head = self.get_output_embeddings()
        # Only the training path with a retained teacher + tile-able FP4 student.
        if (
            not cur_state.enabled
            or labels is None
            or kwargs.get("logits_to_keep")
            or _teacher_weight(lm_head) is None
            or _fp4_student_store(lm_head) is None
        ):
            return orig_forward(self, *args, **kwargs)

        labels_v = kwargs.pop("labels")
        num_items = kwargs.pop("num_items_in_batch", None)
        ce, hidden, outputs = _student_ce_and_hidden(
            self, args, kwargs, labels_v, num_items
        )
        if ce is None or hidden is None:
            # couldn't take the owned path -> restore and defer to original
            kwargs["labels"] = labels_v
            if num_items is not None:
                kwargs["num_items_in_batch"] = num_items
            return orig_forward(self, *args, **kwargs)

        kl = lm_head_distillation_loss(hidden, lm_head, cur_state)
        loss = ce if kl is None else ce + kl

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward


def patch_model_distillation(model: nn.Module, state: DistillState) -> bool:
    """Patch the ForCausalLM forward to add the FP4-head KL-distillation term.

    Idempotent per ForCausalLM class. The PEFT wrapper delegates forward to the
    base model, so patching the underlying class covers LoRA too. The ``state`` is
    stashed on the model (``_nvfp4_distill_state``) for the bench harness to read.
    """
    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:
            causal = model

    # Set the live state on both the outer (harness) and the causal (forward self)
    # objects so the wrapper reads this run's knobs even when re-patched.
    model._nvfp4_distill_state = state
    causal._nvfp4_distill_state = state
    cls = causal.__class__
    if cls in _PATCHED_DISTILL:
        return True
    cls.forward = _make_distill_forward(cls.forward, state)
    _PATCHED_DISTILL.add(cls)
    LOG.info("NVFP4 lm_head_distillation: patched %s.forward", cls.__name__)
    return True
