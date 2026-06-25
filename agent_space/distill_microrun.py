#!/usr/bin/env python
"""Phase-1 FP4 lm_head KL-distillation: accuracy gap-shrink + SPEED accounting.

Uses the REAL branch code paths (the NVFP4 frozen FP4 head, the retained bf16
teacher, the tiled KL distillation loss + forward patch) on a small large-vocab
model body (Qwen3-0.6B). Two deliverables:

  ACCURACY: a short body-adaptation micro-run (the FP4 head is frozen; the body
  trains). We measure the teacher(bf16)-vs-student(FP4) gap — KL, top-1 agreement,
  CE delta — on held-out hidden states BEFORE and AFTER training, WITH vs WITHOUT
  the KL aux-loss. Distillation should shrink the gap more.

  SPEED: per-step head-forward+loss wall time for (a) plain FP4 CE, (b) + live
  full-vocab distill, (c) + top_k, (d) + cadence. Reports the % overhead of each
  lever and how much of the plain-FP4-head step time is retained.

    PY=/home/rgilbreth/Documents/GitHub/LLM-Tools/Build-Venv/_venvs/axolotl_torch211_experimental/bin/python
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 $PY agent_space/distill_microrun.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def build_fp4_head(linear: nn.Linear, retain_teacher: bool):
    """Real branch FP4 head (NVFP4FrozenBaseLinear), optionally + bf16 teacher."""
    from axolotl.utils.nvfp4_training import NVFP4FrozenBaseLinear, NVFP4Recipe

    head = NVFP4FrozenBaseLinear.from_linear(
        linear, NVFP4Recipe(stochastic_rounding=False, hadamard=False), fsdp=False
    )
    if retain_teacher:
        t = linear.weight.detach().clone().requires_grad_(False)
        head.register_buffer("_distill_teacher_w", t, persistent=False)
    return head


@torch.no_grad()
def gap_metrics(hidden, head, teacher_w):
    """teacher(bf16) vs student(FP4) gap on `hidden`: KL, top1, CE delta."""
    from axolotl.kernels.nvfp4_fused_ce import _dequant_vocab_tile, _nvfp4_lm_head_store

    store = _nvfp4_lm_head_store(head)
    V = store.shape[0]
    zt = hidden.float() @ teacher_w.float().t()
    # tiled student dequant logits
    tiles = [_dequant_vocab_tile(store, lo, min(lo + 8192, V), hidden.dtype)
             for lo in range(0, V, 8192)]
    Wq = torch.cat(tiles, 0)
    zs = hidden.float() @ Wq.float().t()
    pt = torch.softmax(zt, 1)
    kl = (pt * (torch.log_softmax(zt, 1) - torch.log_softmax(zs, 1))).sum(1).mean()
    top1 = (zt.argmax(1) == zs.argmax(1)).float().mean()
    tgt = zt.argmax(1)  # teacher argmax as the self-supervised target
    ce_t = F.cross_entropy(zt, tgt)
    ce_s = F.cross_entropy(zs, tgt)
    return {
        "kl": kl.item(),
        "top1": top1.item(),
        "ce_delta": (ce_s - ce_t).item(),
    }


def load_body(model_id, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return tok, model


_TRAIN_TEXT = (
    "In the field of machine learning, quantization reduces the numerical "
    "precision of model weights to save memory and increase throughput, at some "
    "cost to accuracy. The language model head projects hidden states onto a large "
    "vocabulary, producing one logit per token. Low-bit formats such as FP4 lose "
    "accuracy on the flat lm_head weight distribution. Distillation lets the body "
    "adapt to the frozen quantized head by matching the original head's logits. "
)
# Distinct held-out text (different topic) so the gap eval measures generalization,
# not memorization of the training tokens.
_EVAL_TEXT = (
    "The history of cartography spans many centuries, from clay tablets to "
    "satellite imagery. Early sailors navigated by the stars, charting coastlines "
    "and currents across unknown oceans. Trade routes connected distant cities, "
    "carrying spices, silk, and stories between continents. Maps shaped empires and "
    "guided explorers toward territories no European had yet described in writing. "
)


def calib_ids(tok, device, n_tokens, text=_TRAIN_TEXT):
    ids = tok(text * 40, return_tensors="pt").input_ids[:, : n_tokens + 1].to(device)
    return ids


def run_microrun(model_id, device, dtype, steps, lr, lam=1.0):
    """Body-adaptation A/B: train the last 2 decoder blocks for `steps`, with and
    without the KL aux-loss, frozen FP4 head. Report gap before/after for each."""
    from axolotl.kernels.nvfp4_distill import DistillState, lm_head_distillation_loss

    results = {}
    for use_distill in (False, True):
        torch.manual_seed(0)
        tok, model = load_body(model_id, device, dtype)
        model.train()
        ids = calib_ids(tok, device, n_tokens=512, text=_TRAIN_TEXT)
        eval_ids = calib_ids(tok, device, n_tokens=512, text=_EVAL_TEXT)

        # real FP4 head + retained teacher; swap it onto the model
        orig_head = model.get_output_embeddings()
        W = orig_head.weight.detach()
        lin = nn.Linear(W.shape[1], W.shape[0], bias=False).to(device, dtype)
        lin.weight.data.copy_(W)
        head = build_fp4_head(lin, retain_teacher=True).to(device)
        teacher_w = head._distill_teacher_w

        base = model.model  # Qwen3Model body

        # freeze everything, then unfreeze just the last 2 decoder blocks (the body
        # adapts; the FP4 head is frozen by construction)
        for p in model.parameters():
            p.requires_grad_(False)
        trainable = []
        for blk in base.layers[-2:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                trainable.append(p)
        opt = torch.optim.AdamW(trainable, lr=lr)

        # held-out hidden states for gap eval (captured once, fixed)
        def capture_hidden(input_ids):
            out = base(input_ids)
            return out.last_hidden_state.reshape(-1, out.last_hidden_state.shape[-1])

        with torch.no_grad():
            h_eval = capture_hidden(eval_ids[:, :-1]).detach()
        gap_before = gap_metrics(h_eval, head, teacher_w)

        state = DistillState(
            enabled=True, lambda_=lam, temperature=1.0, top_k=None,
            cadence=1, teacher="live", vocab_block=8192,
        )

        from axolotl.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy

        labels = ids[:, 1:].clone()
        inp = ids[:, :-1]
        for _step in range(steps):
            opt.zero_grad(set_to_none=True)
            hidden = capture_hidden(inp)  # [M, H], carries grad to body
            ce = fused_fp4_cross_entropy(
                hidden.reshape(1, -1, hidden.shape[-1]),
                head, labels, shift=False,
            )
            loss = ce
            if use_distill:
                kl = lm_head_distillation_loss(hidden, head, state)
                if kl is not None:
                    loss = loss + kl
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_eval2 = capture_hidden(eval_ids[:, :-1]).detach()
        gap_after = gap_metrics(h_eval2, head, teacher_w)
        results["distill" if use_distill else "baseline"] = {
            "before": gap_before, "after": gap_after,
        }
        del model, base, head
        torch.cuda.empty_cache()
    return results


def run_speed(model_id, device, dtype, iters, warmup):
    """Per-step head-forward+loss wall time for each speed configuration."""
    from axolotl.kernels.nvfp4_distill import DistillState, lm_head_distillation_loss
    from axolotl.kernels.nvfp4_fused_ce import fused_fp4_cross_entropy

    tok, model = load_body(model_id, device, dtype)
    model.train()
    ids = calib_ids(tok, device, n_tokens=1024)
    base = model.model
    orig_head = model.get_output_embeddings()
    W = orig_head.weight.detach()
    lin = nn.Linear(W.shape[1], W.shape[0], bias=False).to(device, dtype)
    lin.weight.data.copy_(W)
    head = build_fp4_head(lin, retain_teacher=True).to(device)

    # capture a fixed hidden (requires grad so backward exercises the body dgrad)
    inp = ids[:, :-1]
    labels = ids[:, 1:].clone()

    for p in model.parameters():
        p.requires_grad_(False)
    for p in base.layers[-1].parameters():
        p.requires_grad_(True)

    def one_step(state):
        hidden = base(inp).last_hidden_state
        h2 = hidden.reshape(1, -1, hidden.shape[-1])
        ce = fused_fp4_cross_entropy(h2, head, labels, shift=False)
        loss = ce
        if state is not None:
            kl = lm_head_distillation_loss(hidden.reshape(-1, hidden.shape[-1]), head, state)
            if kl is not None:
                loss = loss + kl
        loss.backward()
        return loss

    configs = {
        "a_plain_fp4_ce": None,
        "b_distill_full": DistillState(enabled=True, lambda_=1.0, top_k=None, cadence=1, vocab_block=8192),
        "c_distill_topk128": DistillState(enabled=True, lambda_=1.0, top_k=128, cadence=1, vocab_block=8192),
        "d_distill_topk128_cad4": DistillState(enabled=True, lambda_=1.0, top_k=128, cadence=4, vocab_block=8192),
    }
    timings = {}
    for name, state in configs.items():
        # reset cadence counter
        if state is not None:
            state.step_counter = 0
        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            one_step(state)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            one_step(state)
        torch.cuda.synchronize()
        timings[name] = (time.perf_counter() - t0) / iters * 1000.0  # ms/step
    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--speed-iters", type=int, default=20)
    ap.add_argument("--speed-warmup", type=int, default=5)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    print(f"[microrun] model={args.model} steps={args.steps} lr={args.lr}\n")

    print("=== SPEED (head-forward + loss + backward, ms/step) ===")
    timings = run_speed(args.model, device, dtype, args.speed_iters, args.speed_warmup)
    base_t = timings["a_plain_fp4_ce"]
    for name, t in timings.items():
        ov = (t - base_t) / base_t * 100.0
        retained = base_t / t * 100.0
        print(f"  {name:26s} {t:8.2f} ms/step   overhead {ov:+6.1f}%   "
              f"fp4-speed-retained {retained:5.1f}%")
    print()

    print(f"=== ACCURACY (gap teacher-bf16 vs student-FP4, {args.steps}-step body adapt) ===")
    res = run_microrun(args.model, device, dtype, args.steps, args.lr)
    for tag in ("baseline", "distill"):
        b, a = res[tag]["before"], res[tag]["after"]
        print(f"  [{tag:8s}] KL  {b['kl']:.5f} -> {a['kl']:.5f}   "
              f"top1 {b['top1']:.4f} -> {a['top1']:.4f}   "
              f"CEdelta {b['ce_delta']:+.5f} -> {a['ce_delta']:+.5f}")
    bd = res["baseline"]["after"]
    dd = res["distill"]["after"]
    print(f"\n  gap shrink (distill vs baseline, after {args.steps} steps):")
    print(f"    KL:       baseline {bd['kl']:.5f}  distill {dd['kl']:.5f}  "
          f"({(1-dd['kl']/bd['kl'])*100:+.1f}% vs baseline)")
    print(f"    top1:     baseline {bd['top1']:.4f}  distill {dd['top1']:.4f}")
    print(f"    CE delta: baseline {bd['ce_delta']:+.5f}  distill {dd['ce_delta']:+.5f}")


if __name__ == "__main__":
    main()
