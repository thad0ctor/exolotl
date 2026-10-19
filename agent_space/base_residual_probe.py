#!/usr/bin/env python
"""Base-residual probe: L2QER low-rank residuals on ALL frozen FP4 base linears.

PTQ-style A/B on a real model (no training), the generalization of
``residual_probe.py`` from the lm_head to the transformer BODY:

  1. capture per-input-channel activation second moments (mean x^2) on every
     to-be-swapped body linear with one forward of the bf16 model (calib slice);
  2. swap every eligible body linear (q/k/v/o, gate/up/down) to the frozen FP4
     base module the LoRA converter uses (MSLK-fast compute when available —
     the shipped default);
  3. score logit KL(bf16 || fp4) on a held-out eval slice, then attach the
     rank-k L2QER residuals (exactly the shipped `attach_base_residual`) and
     score again. Plain-SVD ablation + rank sweep included.

Also reports the load-time overhead (calibration forward + total SVD wall time)
and a step-time micro A/B (forward+backward of one swapped linear at
training-ish shapes, with/without the residual).

Usage:
  PY=.../axolotl_nvfp4_sage_fork/bin/python
  CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD/src \
      $PY agent_space/base_residual_probe.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "agent_space"))

from eval_fp4_head import _calib_text  # noqa: E402

from axolotl.kernels.nvfp4_residual import attach_base_residual  # noqa: E402
from axolotl.utils.nvfp4_training import (  # noqa: E402
    NVFP4ComputeBaseLinear,
    NVFP4FastComputeBaseLinear,
    NVFP4Recipe,
    _is_swappable,
    _mslk_available,
)


def _target_linears(model):
    """(name, linear) for every eligible frozen body linear (q/k/v/o, mlp)."""
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name or "layers." not in name:
            continue
        if not _is_swappable(mod):
            continue
        out.append((name, mod))
    return out


@torch.no_grad()
def _capture_mean_x2(model, targets, ids):
    """One bf16 forward with pre-hooks: name -> mean_t x_t^2 (fp32 [in])."""
    acc: dict = {}

    def make_hook(name):
        def hook(_m, args):
            x = args[0].detach()
            h = x.reshape(-1, x.shape[-1]).float()
            ent = acc.get(name)
            if ent is None:
                acc[name] = [h.pow(2).sum(0), h.shape[0]]
            else:
                ent[0] += h.pow(2).sum(0)
                ent[1] += h.shape[0]

        return hook

    handles = [m.register_forward_pre_hook(make_hook(n)) for n, m in targets]
    model(ids)
    for h in handles:
        h.remove()
    return {n: s / c for n, (s, c) in acc.items()}


@torch.no_grad()
def _eval_logits(model, ids):
    return model(ids).logits.float().reshape(-1, model.config.vocab_size)


@torch.no_grad()
def _kl_and_top1(ref_logits, cand_logits):
    """mean_t KL(P_ref || P_cand), top-1 agreement (fp32, token-chunked)."""
    kl_sum, agree, n = 0.0, 0, 0
    for lo in range(0, ref_logits.shape[0], 128):
        r = ref_logits[lo : lo + 128]
        c = cand_logits[lo : lo + 128]
        logp = F.log_softmax(r, dim=-1)
        logq = F.log_softmax(c, dim=-1)
        kl_sum += (logp.exp() * (logp - logq)).sum(-1).sum().item()
        agree += (r.argmax(-1) == c.argmax(-1)).sum().item()
        n += r.shape[0]
    return kl_sum / n, agree / n


def _set_submodule(model, name, new_mod):
    parent = model.get_submodule(name.rsplit(".", 1)[0])
    setattr(parent, name.rsplit(".", 1)[-1], new_mod)


def whole_step_ms(model, ids, iters=12):
    """Median fwd+bwd ms of the whole swapped model (training-shaped step: the
    input embeddings carry grad, so dgrad flows through every base linear)."""
    emb_layer = model.get_input_embeddings()
    times = []
    for i in range(iters + 3):
        emb = emb_layer(ids).detach().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(inputs_embeds=emb, labels=ids)
        out.loss.backward()
        torch.cuda.synchronize()
        if i >= 3:  # warmup
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e3


def step_time_ab(cls, recipe, in_f=1024, out_f=3072, m=4096, iters=60):
    """Median fwd+bwd ms of ONE swapped linear, with vs without the residual."""
    torch.manual_seed(0)
    lin = nn.Linear(in_f, out_f, bias=False).cuda().bfloat16()
    lin.weight.requires_grad_(False)
    mod = cls.from_linear(lin, recipe)
    mean_x2 = torch.rand(in_f, device="cuda") + 0.1

    def bench():
        x = torch.randn(m, in_f, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        times = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mod(x).sum().backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            x.grad = None
        times.sort()
        return times[len(times) // 2] * 1e3

    base_ms = bench()
    attach_base_residual(mod, lin.weight, 16, mean_x2)
    res_ms = bench()
    return base_ms, res_ms


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    dtype = torch.bfloat16
    recipe = NVFP4Recipe()
    fast = _mslk_available()
    cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
    print(f"[probe] model={args.model} base class={cls.__name__}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    all_ids = tok(_calib_text(), return_tensors="pt").input_ids.to(device)
    calib_ids = all_ids[:, : args.calib_tokens]
    eval_ids = all_ids[:, args.calib_tokens : args.calib_tokens + args.eval_tokens]
    print(
        f"[probe] calib tokens={calib_ids.shape[1]} eval tokens={eval_ids.shape[1]}",
        flush=True,
    )

    targets = _target_linears(model)
    print(f"[probe] {len(targets)} eligible body linears", flush=True)

    # 1. calibration (load-time overhead part 1)
    t0 = time.perf_counter()
    mean_x2 = _capture_mean_x2(model, targets, calib_ids)
    calib_s = time.perf_counter() - t0
    print(f"[probe] calibration forward: {calib_s:.2f}s", flush=True)

    # 2. bf16 reference logits on the eval slice
    ref_logits = _eval_logits(model, eval_ids)

    # 3. swap everything to the frozen FP4 base, keeping the bf16 masters around
    masters, swapped = {}, {}
    t0 = time.perf_counter()
    for name, lin in targets:
        masters[name] = lin.weight.detach().clone()
        mod = cls.from_linear(lin, recipe)
        _set_submodule(model, name, mod)
        swapped[name] = mod
    quant_s = time.perf_counter() - t0
    print(f"[probe] FP4 swap ({len(swapped)} linears): {quant_s:.2f}s", flush=True)

    fp4_logits = _eval_logits(model, eval_ids)
    kl0, top1_0 = _kl_and_top1(ref_logits, fp4_logits)
    results = {
        "fp4_baseline": {"kl": kl0, "top1": top1_0},
    }
    print(f"  fp4_baseline           KL={kl0:.5f} top1={top1_0:.4f}", flush=True)

    def attach_all(rank, use_g):
        t0 = time.perf_counter()
        for name, mod in swapped.items():
            # overwrite any previous buffers (re-attach for the sweep)
            for bname in ("_base_residual_A", "_base_residual_B"):
                if hasattr(mod, bname):
                    delattr(mod, bname)
            attach_base_residual(
                mod, masters[name], rank, mean_x2[name] if use_g else None
            )
        return time.perf_counter() - t0

    svd_s_rank16 = None
    for label, rank, use_g in (
        ("plain_k16", 16, False),
        ("l2qer_k8", 8, True),
        ("l2qer_k16", 16, True),
        ("l2qer_k32", 32, True),
    ):
        svd_s = attach_all(rank, use_g)
        if label == "l2qer_k16":
            svd_s_rank16 = svd_s
        cand = _eval_logits(model, eval_ids)
        kl, top1 = _kl_and_top1(ref_logits, cand)
        results[label] = {"kl": kl, "top1": top1, "svd_s": svd_s}
        print(
            f"  {label:22s} KL={kl:.5f} ({(kl - kl0) / kl0 * 100:+.1f}%) "
            f"top1={top1:.4f} svd={svd_s:.2f}s",
            flush=True,
        )

    # 4. whole-model step A/B: fwd+bwd through the swapped model (dgrad flows
    # through every corrected base linear), residuals on (rank 16) vs off.
    attach_all(16, True)
    step_ids = all_ids[:, :2048] if all_ids.shape[1] >= 2048 else all_ids
    on_ms = whole_step_ms(model, step_ids)
    for mod in swapped.values():
        for bname in ("_base_residual_A", "_base_residual_B"):
            if hasattr(mod, bname):
                delattr(mod, bname)
    off_ms = whole_step_ms(model, step_ids)
    print(
        f"[probe] whole-model step A/B (fwd+bwd, {step_ids.shape[1]} tokens, "
        f"eager): base={off_ms:.1f}ms +residuals={on_ms:.1f}ms "
        f"({(on_ms - off_ms) / off_ms * 100:+.1f}%)",
        flush=True,
    )

    compiled_on_ms = compiled_off_ms = None
    if args.compile_step:
        # The NVFP4 path is documented compile-first ("the speedup only
        # materializes under torch.compile"); eager numbers above are dominated
        # by ~4 extra tiny kernel launches per corrected layer. Under compile
        # the residual matmuls fuse/overlap, leaving close to the pure-FLOPs
        # cost (~0.4% of the body GEMM FLOPs at rank 16).
        torch._dynamo.reset()
        cmodel = torch.compile(model)
        attach_all(16, True)
        compiled_on_ms = whole_step_ms(cmodel, step_ids)
        for mod in swapped.values():
            for bname in ("_base_residual_A", "_base_residual_B"):
                if hasattr(mod, bname):
                    delattr(mod, bname)
        torch._dynamo.reset()
        cmodel = torch.compile(model)
        compiled_off_ms = whole_step_ms(cmodel, step_ids)
        print(
            f"[probe] whole-model step A/B (fwd+bwd, {step_ids.shape[1]} tokens, "
            f"torch.compile): base={compiled_off_ms:.1f}ms "
            f"+residuals={compiled_on_ms:.1f}ms "
            f"({(compiled_on_ms - compiled_off_ms) / compiled_off_ms * 100:+.1f}%)",
            flush=True,
        )

    # 6. step-time micro A/B on one training-ish linear (eager; per-op launch
    # overhead dominates at this size, so this is the pessimistic bound)
    del model, ref_logits, fp4_logits, masters
    torch.cuda.empty_cache()
    base_ms, res_ms = step_time_ab(cls, recipe)
    print(
        f"[probe] one-linear micro A/B ({cls.__name__} 1024->3072, M=4096, "
        f"fwd+bwd): base={base_ms:.3f}ms +residual={res_ms:.3f}ms "
        f"({(res_ms - base_ms) / base_ms * 100:+.1f}%)",
        flush=True,
    )

    summary = {
        "model": args.model,
        "base_class": cls.__name__,
        "n_linears": len(swapped),
        "calib_tokens": int(calib_ids.shape[1]),
        "eval_tokens": int(eval_ids.shape[1]),
        "load_overhead_s": {
            "calibration_forward": calib_s,
            "svd_total_rank16": svd_s_rank16,
        },
        "whole_model_step_ms": {"base": off_ms, "with_residuals": on_ms},
        "whole_model_step_compiled_ms": {
            "base": compiled_off_ms,
            "with_residuals": compiled_on_ms,
        },
        "one_linear_step_ms": {"base": base_ms, "with_residual": res_ms},
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2))
        print(f"[probe] wrote {args.json_out}", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--calib-tokens", type=int, default=512)
    p.add_argument("--eval-tokens", type=int, default=512)
    p.add_argument("--compile-step", action="store_true")
    p.add_argument("--json-out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
