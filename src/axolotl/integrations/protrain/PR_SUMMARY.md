<!--- Provide a general summary of your changes in the Title above -->

# Description

This PR adds an opt-in ProTrain integration for Axolotl under
`src/axolotl/integrations/protrain/`.

ProTrain provides automatic chunked parameter residency, CPU offload, Mode A/B/C
execution, block-level activation checkpoint/offload policy, and
checkpoint/resume support for Axolotl LoRA/QLoRA and bounded full-FT workloads,
meaning full fine-tune shapes validated within the documented hardware and
sequence-length limits.

It is disabled by default. Runtime behavior requires both:

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
protrain_auto_memory: true
```

Listing the plugin alone only registers schema. It does not alter model loading,
optimizer creation, embedding dtype conversion, checkpointing, or runtime hooks.
Existing Axolotl configs that do not enable this plugin should behave unchanged.

The integration deliberately rejects conflicting memory backends and unsafe
settings rather than silently composing them.

Detailed implementation notes, feature scope, and benchmark evidence are in
[`PR_PROPOSAL.md`](PR_PROPOSAL.md).

## Motivation and Context

Axolotl users can hit 24 GiB GPU limits when fine-tuning larger LoRA/QLoRA or
bounded full-FT workloads. ProTrain adds an Axolotl-native memory manager that
keeps the normal config-driven training flow while choosing between:

| Mode | Purpose |
|---|---|
| Mode A | GPU-resident execution when the selected model/training shape fits. |
| Mode B | Replicated CPU offload when GPU memory is binding and each rank has enough host RAM for the non-persistent chunks. |
| Mode C | ZeRO-3-style sharded CPU offload when replicated CPU offload would exceed per-rank host RAM or a sharded layout is explicitly requested. |

The main resume contract is intentionally conservative:

| Resume path | Status |
|---|---|
| Normal save/resume | Supported. |
| Same-world optimizer resume | Supported when `protrain_save_optimizer_state: true`. |
| Cross-world optimizer resume | Advanced opt-in via `protrain_allow_online_reshard: true`; fails closed without required sidecar metadata. |

### Fail-closed behavior

ProTrain prefers explicit failure over silent fallback. These cases raise at
config, startup, or resume time:

- ProTrain auto-memory enabled without plugin registration.
- DeepSpeed, FSDP, or Axolotl-level `gradient_checkpointing` combined with ProTrain.
- Unsupported optimizer family.
- Multiple forced execution modes enabled.
- Cross-world optimizer resume requested without sidecar metadata.
- PEFT or Transformers API surface outside validated guardrails.
- Runtime calibrated memory prediction exceeds configured device capacity.

## How has this been tested?

The committed validation runner is the reviewer-scale acceptance path:

```bash
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite maintainer
PYTHONPATH=src python -m axolotl.integrations.protrain.validation --suite full --gpu-devices 1,2,4,5,7 --keep-cache
```

Latest recorded validation:

GPU lanes are opt-in/self-hosted; standard CI remains CPU-only.

| Lane | Result | Coverage |
|---|---|---|
| Default ProTrain pytest | `650 passed, 17 skipped, 180 deselected` | CPU-safe unit and integration coverage. |
| `cpu-core` | PASS | Validators, cost/search math, calibrated memory gate, metadata, layout determinism, non-finite guard. |
| `cpu-surface` | PASS | Mode selection, force-mode safety, save/resume hooks, Path B LoRA ownership, debug/watchdog hooks. |
| `merge-surface` | PASS | `merge-lora` CLI and LoRA/QLoRA/rsLoRA/DoRA/MoE merge math. |
| `single-gpu-edge` | PASS | DoRA, multi-adapter switching, vision-LM ownership, LoRA offload runtime, A<->C resume. |
| `single-gpu` | PASS | 8B QLoRA 50-step train, checkpoint, 10-step resume, finite losses, bounded loss continuity. |
| `two-gpu` | PASS | Forced Mode C finite train, same-world optimizer save/resume, non-finite boundary checks. |

Critical regression coverage:

| Area | Covered by |
|---|---|
| Plugin inertness and schema-only registration | CPU/default tests. |
| DeepSpeed/FSDP/gradient-checkpointing validator rejection | CPU/default tests. |
| Calibrated memory gate | `cpu-core`. |
| Trainer resume hook restore/re-offload ordering | `cpu-surface`. |
| PEFT LoRA container hook shape preservation | CPU/default and `single-gpu-edge`. |
| Mode C DDP bypass | `cpu-surface` and `two-gpu`. |
| Same-world optimizer resume | `two-gpu`. |
| Cross-world fail-closed behavior without sidecars | CPU/default focused tests. |

Representative hardware validation summarized from `PR_PROPOSAL.md`:

| Shape | Result |
|---|---|
| 8B BF16 LoRA on 24 GiB | Resident memory drops from 15.83 GiB to 3.08 GiB. |
| Llama-13B 4-bit LoRA on one 3090 | Fits through seq=2048. |
| Qwen3.5-27B 4-bit LoRA on one 3090 | Fits at seq=128; Mode B reaches seq=256. |
| Qwen3.5-4B full-FT forced Mode C | Local 5x 3090-class train/save/resume coverage. |
| Qwen3.5-9B full-FT forced Mode C | High-memory 2-rank train/save/resume coverage. |

### Single-card efficiency: Mode A under the full Axolotl + Liger kernel stack

Forced Mode A is runtime-inert (no per-step ProTrain hooks), so it is a
zero-overhead substrate for Axolotl's and Liger's fused kernels plus
`torch.compile`. Benchmarked head-to-head against Unsloth on one RTX 3090 Ti
(sm_86, 24 GiB), Qwen3-14B QLoRA (r=64, dense targets, seq=1024, mb=1, ga=4,
`adamw_8bit`), FA2-vs-FA2 and compiled-vs-compiled, same card:

| Stack (seq 1024, FA2, torch.compile, adamw_8bit) | s/it | reserved | active |
|---|---:|---:|---:|
| ProTrain Mode A + `lora_*_kernel` + `fused_attn_kernel` + Liger FLCE/RMSNorm | 8.07 | 12.71 GiB | 12.42 GiB |
| Unsloth optimized path | 7.69 | 12.75 GiB | 12.50 GiB |

ProTrain Mode A reaches **memory parity** with Unsloth (marginally lower) at
**~5% lower throughput**, while keeping activations GPU-resident (Unsloth
CPU-offloads them). No new ProTrain code is required: Mode A here is identical
to this PR's branch, composed with existing Axolotl fused LoRA kernels, Liger
fused-linear-cross-entropy + RMSNorm, `torch.compile` (compile-safe 4-bit
dequant), and the upstream Qwen3 fused RMSNorm+RoPE kernel (`fused_attn_kernel`).
The `fused_attn_kernel` contribution grows with sequence length; at seq=1024 it
is roughly break-even on the q/k-norm+RoPE path. See the recommended config and
per-flag compatibility notes in [`PR_PROPOSAL.md`](PR_PROPOSAL.md) §6.6.

### Long-context single card (seq 32768) + cost-model accuracy

Beyond where Mode A fits, **Qwen3-14B QLoRA trains a clean step at `seq=32768`
on a single 24 GiB RTX 3090 Ti** via partial activation swap (`n_swap=24`,
`n_checkpoint=16`): loss 0.011, `max_active` 19.1 GiB, `device_reserved` 20.4
GiB (also confirmed on a 32 GiB RTX 5090). It is PCIe-bound (~minutes/step), so
it proves the memory envelope — 16k stays the fast single-card ceiling. Two PR
fixes enable it: a **variable-size slab allocator** for the swap pool
(`block/swap_pool.py`, replacing fixed equal-size slots that wasted RAM and
exhausted) and a **CPU-Adam construction hang fix** (py-cpuinfo's forked CPU
probe deadlocks after CUDA init; `chunk/optim.py` reads `/proc/cpuinfo`
directly).

A SwiGLU FFN-intermediate fix (`cost/memory.py`: count the 3 gated-MLP saved
intermediates, not 1) brings the calibrated memory gate to within **~1% of
measured peak** on Qwen3-14B QLoRA (0.9% mean / 1.3% max across seq 4k–32k) —
versus the paper's deliberate ~10% over-estimate margin (α=1.10, §3.3). Details
and the validation table are in [`PR_PROPOSAL.md`](PR_PROPOSAL.md) §6.7 and §7.6.

## AI Usage Disclaimer

Yes. ChatGPT/Codex and Claude were used to assist with implementation,
debugging, test orchestration, documentation drafting, and review triage. Human
review, local execution, and commit decisions were performed by the PR author.

## Screenshots (if appropriate)

Not applicable.

## Types of changes

- [x] New feature
- [x] Performance improvement
- [x] Documentation update
- [x] Tests / validation coverage
- [ ] Bug fix

## Social Handles (Optional)
