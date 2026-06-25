# ProTrain integration for Axolotl — current implementation and validation status

## 1. Executive summary

This document describes a ProTrain
(Yang et al., MLSys 2026, arXiv 2406.08334) integration to Axolotl as a
`BasePlugin` under `src/axolotl/integrations/protrain/`. The integration is a
from-scratch Python port of the paper's automatic memory manager, adapted to
Axolotl's config-driven Trainer, PEFT-LoRA/QLoRA, bitsandbytes 4-bit weights,
and checkpoint/resume flow.

The PR adds:

- A hierarchical chunk manager for parameter residency, CPU offload, and
  ZeRO-3-style sharded Mode C execution.
- A block manager for activation checkpointing, swapping, and offload-aware
  gather/release hooks.
- An auto-mode selector for Mode A (GPU-resident DDP), Mode B (replicated
  CPU-offload), and Mode C (ZeRO-3 sharded CPU-offload), with topology-aware
  choices for PCIe vs NVLink-class systems.
- Axolotl integration for LoRA/QLoRA, full fine-tune, torch.compile guardrails,
  optimizer wrapping, safetensors save, optimizer-state checkpointing, and
  resume. Cross-world optimizer-state resume is an explicit advanced path, not
  the default user flow.
- Config validation that rejects known-conflicting stacks rather than silently
  composing with DeepSpeed/FSDP, in-training adapter merges, or unsupported
  lazy-load paths. Post-training `merge-lora` is validated separately.

Reviewer-facing validation summary:

| Area | Evidence |
|---|---|
| 24 GiB consumer-GPU fit | 8B BF16 LoRA memory drops from 15.83 GiB to 3.08 GiB resident (§6.a). Llama-13B 4-bit LoRA fits on one 3090 through seq=2048 (§6.h). Qwen3.5-27B 4-bit LoRA fits on one 3090 at seq=128 and Mode B reaches seq=256 under the 24 GiB ceiling (§6.j, §6.dd). |
| Multi-GPU consumer rigs | 13B/9B 4-bit LoRA validates on 4× 3090 in Mode A and Mode C (§6.e). Mode A/B/C shape coverage includes bs=1/2, seq=256/512, mixed-SKU determinism, and non-NVLink Mode C DDP bypass (§6.zz, §6.zz.X, §6.uu). |
| Full fine-tune | Qwen3-0.6B full-FT validates Adam-state memory reduction across torch, bnb 8-bit, and paged Adam (§6.g). Qwen3.5-4B full-FT forced Mode C validates locally on 5× 3090-class cards with both `adamw_bnb_8bit` and `adamw_torch` (§6.z). Qwen3.5-9B full-FT forced Mode C validates train/save/resume on a high-memory 2-rank host (§6.nw). |
| Checkpointing and resume | LoRA save/resume/merge, ProTrain optimizer sidecars, safetensors final save, full-FT chunk restoration, same-world optimizer resume when enabled, fail-closed cross-world behavior by default, and explicitly enabled 4→2 / 2→4 optimizer-state resharding are covered (§6.o, §6.cc, §6.gg, §6.jj, §16.B). |
| LoRA sync and topology | Path B LoRA grad sync is default-on for PCIe and default-off for NVLink-class fabric. PCIe all-linear LoRA improves steady-state throughput by 15.1%; NVLink validation shows native NCCL buckets are faster, justifying the topology-aware default (§6.pb, §6.nv). |
| Compatibility | Standard attention, Qwen3.5 linear attention, tiny Mixtral-class MoE, Qwen3.6-35B-A3B multimodal MoE 4-bit QLoRA, torch.compile, Apex FusedAdam with the documented CUDA/toolkit constraint, LoRA-rank sweeps, gradient accumulation, and merge-lora command/reload paths are validated at the levels claimed in §6 and §12. |
| CI and tests | The default ProTrain suite is 632 passed, 17 skipped, 180 deselected on the rebased branch; the committed validation runner passes CPU, single-GPU, and two-GPU lanes on the latest local full run (§6.n, §11, §15). |
| Single-card efficiency vs Unsloth | Qwen3-14B QLoRA on one RTX 3090 Ti, Mode A composed with the full Axolotl + Liger fused-kernel stack + torch.compile reaches memory parity with Unsloth (12.71 vs 12.75 GiB reserved) at ~5% lower throughput (8.07 vs 7.69 s/it), FA2-vs-FA2 / compiled-vs-compiled / same card, both `adamw_8bit`. No new ProTrain code; Mode A is inert (§6.6). |

Current boundaries are explicit rather than hidden:

- ProTrain is mutually exclusive with DeepSpeed/FSDP in the same Axolotl run.
- Mode A full-FT at 8B+ can exceed 24 GiB cards because DDP reducer and frozen
  base load order still matter; Mode C is the intended full-FT path.
- bs=1 correctness and cost attribution are validated for the listed Mode A
  inert and Path B rerun shapes. Throughput is fixed-overhead-bound, so
  gradient accumulation is the recommended production path; CUDA Graphs remains
  optional future optimization work.
- 9B text full-FT train/save/resume is validated; exact multimodal visual-key
  checkpoint fidelity is validated on the local 4B/9B fidelity-specific paths
  called out in §16.B, not generalized beyond those shapes.

### Fail-closed behavior

ProTrain prefers explicit failure over silent fallback. The following cases
raise at config, startup, search, or resume time:

- `protrain_auto_memory: true` without the ProTrain plugin.
- DeepSpeed, FSDP, or Axolotl-level `gradient_checkpointing` combined with
  ProTrain.
- More than one forced mode enabled.
- Unsupported optimizer family.
- Cross-world optimizer resume without sidecar metadata or without
  `protrain_allow_online_reshard: true`.
- PEFT or Transformers API surfaces outside the validated guardrails.
- Calibrated runtime memory prediction exceeding the configured device budget.

> **Cost-model note.** Raw `estimate_peak` values are lower-bound search gates.
> Runtime-visible predictions are calibrated with actual chunk bytes before
> budget checks; users tuning near the 24 GiB ceiling should rely on the
> calibrated post-wrapper prediction and, when needed, the conservative
> `protrain_ckpt_internal_residual_factor` knob.

---

## 2. Background: ProTrain (paper) vs this integration

**The original paper.** ProTrain proposes automatic memory management for
ZeRO-style data-parallel LLM training. The core idea is to unify
ZeRO-3 sharding, CPU offload, gradient checkpointing, and activation
swapping into a small structured search space (the four tunable parameters
`n_persist`, `n_buffer`, `n_swap`, `n_checkpoint`), and to pick the optimum
analytically from a single profiling pass via two cost models (runtime and
peak memory). The model state is organized into chunks managed
hierarchically (inter- and intra-chunk); activations are managed at
transformer-block granularity with an interleaved layout that places
swap-targeted blocks early and unoptimized blocks late. The reference
implementation reports up to **2.71× throughput** vs DeepSpeed / Colossal-AI
/ FSDP on RTX 3090s and trains models up to **34B on a single 3090** / 75B
on a single A100.

**This integration.** A from-scratch Python port implemented as an Axolotl
`BasePlugin`, designed to compose with the Axolotl + HF Trainer training
loop, PEFT-LoRA adapters, and bitsandbytes weight quantization. The
plugin owns memory policy and the collectives required by that policy, but it
does not own process-group lifecycle, training-loop control flow, tensor
parallelism, pipeline parallelism, or FP8. Three notable deltas vs the paper's
reference design, expanded in §8:

1. A fifth search axis (`n_offload`) for block-level chunk-offload-without-
   recompute (Option B); the paper's design has four axes.
2. A per-dtype α fragmentation factor (`ALPHA_FRAGMENTATION_4BIT = 0.75` for
   bnb-4-bit; paper uses the constant α=1.10 across dtypes).
3. PEFT-LoRA container hooks and a DDP `init_sync=False` bypass — paper
   targeted full-tensor params on a vanilla DDP-or-internal-allreduce path.

The mode-selection table is also re-indexed into three modes (A: GPU-resident
DDP; B: replicated CPU-offload; C: ZeRO-3 sharded CPU-offload), with the
selector preferring A → B → C on non-NVLink PCIe because the per-chunk collectives
of Mode C dominate at typical 3090 batch sizes.

---

## 3. Architecture overview

### 3.1 Chunk manager (`chunk/manager.py`)

Model states (parameters, gradients, optimizer states) are partitioned into
chunks of size `S_chunk` (picked from a `{32, 64, 128, 256} MB` grid by
`chunk/sizing.py` against a fragmentation-waste model). The first
`n_persist` chunks remain GPU-resident across the iteration; the
remaining `N_chunk − n_persist` chunks are non-persistent and live on pinned
host memory (replicated or sharded) until gathered. A `BufferPool` of
`n_buffer` pre-allocated GPU slots stages the H2D and D2H movement so the
caching allocator never has to satisfy chunk-sized requests from the kernel
hot path.

The chunk layout (`chunk/layout.py`) groups parameters by the transformer
block they belong to and reorders them within a chunk by first-use order
observed during the trace. The execution-ordered intra-chunk layout
eliminates the back-and-forth access pattern that vanilla
initialization-order chunking exhibits, which is what enables prefetch to
hide PCIe latency.

`materialize_offload` is the one-shot transition that runs after the
searcher returns a config: persistent chunks get their final GPU buffer,
non-persistent chunks get their pinned-host pool view (replicated under Mode
B, sharded under Mode C), and `param.data` for every non-persistent param is
swapped to a zero-element placeholder until the first `gather`.

### 3.2 Block manager (`block/`)

Activations are managed at the transformer-block level. Each block carries
one of four modes (`block/strategy.py`):

| BlockMode | Behavior | When picked |
|---|---|---|
| `NONE` | activations kept resident, no recompute | last few blocks, where backward consumes them first |
| `CKPT` | `torch.utils.checkpoint` (non-reentrant) on the block; recompute in backward | middle/later blocks when bandwidth can't hide swap |
| `SWAP` | every autograd-saved tensor D2H'd to a pinned pool on `_swap_stream` in fwd, H2D'd back in bwd | early blocks where prefetch can overlap |
| `OFFLOAD` | re-gather the block's non-persistent chunk in backward (no recompute) | when chunk-offload is preferred over per-tensor swap (Option B) |

The placement rule (`block/layout_rules.py::assign_modes`) puts SWAP blocks
first, OFFLOAD/CKPT blocks in the middle, and `NONE` blocks last —
mirroring the paper's "swap-early, unopt-late, interleave" layout — and
`_looks_like_block` recognizes both standard-attention (`.attention`,
`.self_attn`) and Mamba-style linear-attention (`.linear_attn`) blocks so
Qwen3.5 / Falcon-Mamba / Zamba hybrids are correctly discovered.

### 3.3 Cost model and exhaustive searcher (`cost/`, `search/`)

Two cost models, both fed by a single profiler trace:

- **Runtime model** (`cost/runtime.py`) — implements paper Eqs. 2–7:
  `T_iter = T_FWD + max(T_BWD + T_GPU_OPTIM, T_CPU_OPTIM)`, with per-chunk
  `max(compute, communication)` rolloff. `cost/bandwidth.py` derates
  prefetch bandwidth when `n_swap > 0` competes for the same PCIe lane.
- **Memory model** (`cost/memory.py`) — implements paper Eqs. 8–11
  (op-walk peak + the recompute-bump indicator) plus the per-dtype α (see
  §8) and a `ckpt_chain_bytes` term that accounts for the
  linear-in-N_block residual that survives the backward window under
  non-reentrant checkpointing.

The searcher (`search/exhaustive.py`) enumerates the 5-axis tuple
`(n_persist, n_buffer, n_swap, n_checkpoint, n_offload)` in memory-ascending
order with OOM pruning, and returns the argmin of `T_iter` subject to
`M_peak < M_capacity`. On RTX 3090 hardware the swap path is almost never
selected (non-NVLink PCIe saturates from prefetch alone) so `n_swap = 0` is the
common case — matching the paper's RTX-3090 observation.

### 3.4 Plugin lifecycle (`plugin.py`)

The plugin is a `BasePlugin` subclass. The runtime path it owns:

1. `get_input_args` — returns `ProTrainArgs` to Axolotl's pydantic merger
   (adds `protrain_*` keys to the YAML schema).
2. `post_model_load(cfg, model)` — builds a `HardwareProfile`, runs the
   profiler (cached on disk by `(arch_hash, bs, seq, sku, world)`),
   invokes `protrain_model_wrapper(model, ...)`, and stashes the
   `WrappedModel` on `cfg` for the next hook to pick up.
3. `post_trainer_create(cfg, trainer)` — installs the ProTrain optimizer
   (`protrain_optimizer_wrapper(wrapped)`) onto `trainer.optimizer` and
   registers the checkpoint save/load callbacks. This is the canonical
   install point because Axolotl's `OptimizerMixin.create_optimizer` does
   not route through `PluginManager.create_optimizer`.
4. `_install_resume_hook` — monkey-patches `trainer._load_from_checkpoint`
   to interleave a `restore_to_gpu()` before HF copies loaded weights and
   a `materialize_offload()` + optimizer rebuild afterward. This is the
   bridge that enables cross-mode (A↔C) resume.

Activation requires **both** the plugin listed in `plugins:` AND
`protrain_auto_memory: true` (defaults to False). Listing the plugin alone
registers the args schema but leaves the runtime hooks dormant.

### 3.5 Runtime data path

```text
            ┌─────────────────────┐
 YAML       │ protrain_auto_memory│
   ─────►   │ + plugin in plugins │
            └──────────┬──────────┘
                       │
                       ▼
   ┌─────────────────────────────────────────────┐
   │ post_model_load (plugin.py)                 │
   │  1. _build_hardware_profile(cfg)            │
   │  2. profiler.run_trace (cached by arch_hash)│
   │  3. discover_blocks(model)  [block manager] │
   │  4. build_layout(chunks, exec_order)        │
   │  5. exhaustive.search(...) → CostConfig     │
   │  6. mode_select → A | B | C                 │
   │  7. protrain_model_wrapper(model, ...)      │
   │     → materialize_offload()                 │
   └──────────┬──────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────┐
   │ post_trainer_create                          │
   │  - protrain_optimizer_wrapper → optimizer   │
   │  - _install_resume_hook (cross-mode bridge) │
   └──────────┬──────────────────────────────────┘
              │
              ▼
   ┌─────────────────────────────────────────────┐
   │  Train loop (HF Trainer)                     │
   │                                              │
   │  pre-fwd hook  → scheduler.prefetch_chunks  │
   │       │                                      │
   │  block.forward                               │
   │       │                                      │
   │  post-fwd hook → release stale chunks       │
   │       │                                      │
   │  loss.backward()                             │
   │       │                                      │
   │  pre-bwd hook → ensure_chunks_resident      │
   │       │                                      │
   │  block.backward                              │
   │       │                                      │
   │  post-bwd hook → reduce_grads_and_offload   │
   │       │                                      │
   │  optimizer.step                              │
   │   - persistent chunks  → Apex FusedAdam      │
   │                          or torch AdamW      │
   │   - non-persistent     → DeepSpeedCPUAdam   │
   └──────────────────────────────────────────────┘
```

### 3.6 The three Modes (A / B / C)

The mode selector (`api/model_wrapper.py`) picks one of three composition
modes after the searcher returns a config:

| Mode | Composition | Picked when | Per-rank GPU peak | Per-rank pinned CPU | Throughput |
|---|---|---|---|---|---|
| **A** | All-persistent, outer DDP | `n_persist == N_chunk` fits | full model | ~0 | best (DDP allreduce) |
| **B** | Replicated CPU-offload | offload needed AND `cpu_ram_per_rank ≥ (N_chunk − n_persist) · S_chunk` | reduced | full non-persistent set, replicated | ~1.9× slower than A on non-NVLink PCIe |
| **C** | ZeRO-3 sharded CPU-offload | offload needed AND replication doesn't fit | reduced | non-persistent set / world_size | **PCIe**: ~1.04× slower than A for 4-bit + LoRA on 4× 3090 (§6.e: 24.68 vs 23.67 sps); full-FT scope is ~3.6× per the M7 internal benchmark. **NVLink** (2× A100-SXM4): **~1.43× FASTER than A** on the same shape (Mode C 252 vs Mode A 177 tok/s/rank, §6.nv) — sharded all-gather amortizes well over NV-class fabric. |

The selector prefers A → B → C; C is chosen only when CPU RAM is the
binding constraint. Mode C gracefully degrades to single-rank execution
at `world_size=1` (within 0.03 GiB / 1% of Mode A, §6.k). On consumer
non-NVLink topology, `n_persist=128 n_offload=0` is the auto-mode Mode B
landing point. Mode A bs=1/2 and Mode C bs=1/2 are hardware-validated
end-to-end on 4× 3090 (§6.zz).

---

## 4. Implementation in axolotl

### 4.1 Plugin attachment

Add to YAML:

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
protrain_auto_memory: true
```

The dual gate (plugin listed AND `protrain_auto_memory: true`) is enforced by
a pydantic `model_validator` in `args.py`. The same validator rejects
combinations with `deepspeed:` or `fsdp:` / `fsdp_config:` — the three
memory backends are mutually exclusive.

### 4.2 Config knobs that matter

| Knob | Default | Effect |
|---|---|---|
| `protrain_auto_memory` | `false` | Master enable. Required for any ProTrain behavior. |
| `protrain_auto_mode` | `true` | Run the mode selector. Set `false` to honor manual mode flags below. |
| `protrain_force_all_persistent` | `false` | Force Mode A (every chunk GPU-resident). Skips the search entirely. Requires `protrain_auto_mode: false`. |
| `protrain_zero3_shard` | auto | Force Mode C (ZeRO-3 sharded CPU offload). Requires `protrain_auto_mode: false`. |
| `protrain_save_optimizer_state` | `false` | Emit `protrain_optim/{gpu_optim.pt, metadata.json}` next to every HF checkpoint, for cross-mode and same-mode resume. |
| `protrain_optim_save_max_bytes` | `2 GiB` | Soft size gate for optimizer-state checkpoint writes. Raise explicitly for full-FT checkpoints. |
| `protrain_cache_dir` | `~/.cache/axolotl/protrain` | On-disk cache for profiler traces (keyed by `(arch_hash, bs, seq, sku, world)`). |
| `protrain_capacity_bytes` | auto | Optional GPU budget override in bytes; default is detected VRAM minus headroom. |
| `protrain_cpu_capacity_bytes` | auto | Optional per-rank pinned CPU RAM budget override in bytes. |
| `protrain_n_persist_override` | None | Manual `n_persist` for searcher validation. |
| `protrain_n_buffer_override` | None | Manual `n_buffer`. |
| `protrain_n_swap_override` | None | Manual `n_swap`. |
| `protrain_n_checkpoint_override` | None | Manual `n_checkpoint`. |
| `protrain_n_offload_override` | None | Manual `n_offload` (the Option B axis added by this integration). |
| `protrain_force_replicated_cpu_offload` | `false` | Force Mode B (replicated CPU offload, no sharding). Requires `protrain_auto_mode: false`. Sibling to `protrain_force_all_persistent` (Mode A) and `protrain_zero3_shard` (Mode C). |
| `protrain_own_lora_grad_sync` | topology-aware | Path B: ProTrain owns trainable LoRA-adapter grad sync via flattened all_reduce per dtype; DDP is told to ignore the same params. Explicit `true` / `false` overrides the auto decision. See §6.pb. |
| `protrain_eager_nccl_warmup` | `false` | Opt-in one-shot no-op of each per-chunk NCCL collective at `post_trainer_create` to amortize the first-iter NCCL init cost. Measured ~0.22-0.24 s warm-up wallclock. |
| `protrain_prefer_no_offload_on_non_nvlink` | `true` | Non-NVLink multi-rank tie-break that prefers `n_offload=0` within the prediction noise band. |
| `protrain_allow_online_reshard` | `false` | Opt-in Mode C optimizer-state load from a different current world size. |
| `protrain_phase2_quickstart` | `false` | Opt-in skip of the Phase-2 re-pick when Phase-1 measured time is already within the configured envelope. |
| `protrain_phase2_quickstart_envelope` | `0.30` | Relative-error envelope used by `protrain_phase2_quickstart`. |
| `protrain_persistent_huge_param_threshold_bytes` | `512 MiB` | Params at or above this size (typically `lm_head` / `embed_tokens` at scale) are pinned persistent regardless of `n_persist`, since paging them dominates per-step cost. |
| `protrain_ckpt_internal_residual_factor` | `1.0` | Scale applied to the per-block CKPT internal saved-tensor proxy (FFN-intermediate + attention scores + Q/K/V) in `estimate_peak`. Set `0.0` to disable the residual; lower values are more aggressive. Tune only when the calibrated peak diverges from measured at the runtime budget gate. |
| `embeddings_skip_upcast` | `false` (Axolotl-side) | Skips the load-time fp32 embedding upcast in `loaders/model.py::_convert_embedding_modules_dtype`. **Auto-enabled only when ProTrain is active** — the loader gates on `is_protrain_active(cfg)`, which requires both the plugin and `protrain_auto_memory: true`, so 27B + 4-bit + ProTrain works without the YAML knob while plugin-only schema registration remains inert. The flag remains available for non-ProTrain low-VRAM users. |

### 4.3 Compatibility constraints

These must be honored or the config will OOM or misbehave:

- **Do not set `gradient_checkpointing: true`.** ProTrain owns activation
  checkpointing per block via the `BlockMode.CKPT` strategy; setting the
  Axolotl-level flag installs a second, conflicting checkpoint wrapper.
- **Pass an explicit accelerate config file** (`accelerate launch
  --config_file ...`). The user-level
  `~/.cache/huggingface/accelerate/default_config.yaml` on multi-GPU rigs
  will auto-detect every visible device and force multi-rank launches even
  for single-GPU runs, ignoring `CUDA_VISIBLE_DEVICES`.
- **`deepspeed:` / `fsdp:` must be absent.** The pydantic validator rejects
  combinations.
- **For 27B-class + 4-bit on a 24 GiB 3090, no extra knob required.** The
  loader auto-defers the fp32 embedding upcast only when ProTrain is active
  (`is_protrain_active(cfg)` gate in
  `loaders/model.py::_configure_embedding_dtypes`), so the ~5 GiB
  transient that would otherwise push peak load-time memory above 24 GiB
  no longer fires. Listing the plugin without `protrain_auto_memory: true`
  remains inert. Non-ProTrain users on low-VRAM cards may still set
  `embeddings_skip_upcast: true` explicitly.

### 4.4 ProTrain config overlays and conflicts

These snippets are overlays for an otherwise valid Axolotl config, not complete
training recipes. Model, dataset, adapter, schedule, and logging keys remain
standard Axolotl YAML.

**Required enablement.** Both keys are required. Listing the plugin without
`protrain_auto_memory: true` is inert; setting `protrain_auto_memory: true`
without the plugin is rejected.

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
protrain_auto_memory: true
```

**Default mode selection.** Leave this unset or `true` for ordinary use. The
selector chooses Mode A/B/C from fit, CPU RAM, topology, and the measured
profile.

```yaml
protrain_auto_mode: true
```

**Force one mode for benchmarking or known-good deployments.** Forced modes
require `protrain_auto_mode: false`; only one force flag may be true.

```yaml
# Mode A: GPU-resident chunks.
protrain_auto_mode: false
protrain_force_all_persistent: true
```

```yaml
# Mode B: replicated CPU-offload chunks.
protrain_auto_mode: false
protrain_force_replicated_cpu_offload: true
protrain_n_persist_override: 128
protrain_n_offload_override: 0
```

```yaml
# Mode C: ZeRO-3-style sharded CPU-offload chunks.
protrain_auto_mode: false
protrain_zero3_shard: true
```

**Optimizer-state checkpointing.** This is opt-in. LoRA checkpoints normally
fit under the default size gate; full-FT checkpoints require an explicit cap
large enough for the optimizer state being written.

```yaml
protrain_save_optimizer_state: true
protrain_optim_save_max_bytes: 120000000000
```

Same-world resume uses normal Axolotl `resume_from_checkpoint`. Cross-world
Mode C optimizer resume requires explicit online reshard permission; otherwise
the loader fails closed instead of guessing.

```yaml
resume_from_checkpoint: ./outputs/run/checkpoint-25
protrain_save_optimizer_state: true
protrain_allow_online_reshard: true
```

**Supported optimizer families.** ProTrain currently drives AdamW-shaped
optimizers only:

```yaml
optimizer: adamw_torch          # also adamw_torch_fused, adamw_apex_fused,
                                # adamw_8bit, adamw_bnb_8bit,
                                # paged_adamw_8bit
```

For 8-bit AdamW variants, the validator fills `max_grad_norm: 1.0` when the
user omits clipping. Explicit user values are preserved.

**Invalid combinations rejected by validators.**

```yaml
# Rejected: ProTrain must be the only memory backend.
deepspeed: deepspeed_configs/zero3.json
fsdp:
  - full_shard
fsdp_config:
  fsdp_offload_params: true
```

```yaml
# Rejected: Axolotl-level checkpointing conflicts with ProTrain block modes.
gradient_checkpointing: true
```

```yaml
# Rejected: force flags are mutually exclusive.
protrain_auto_mode: false
protrain_force_all_persistent: true
protrain_force_replicated_cpu_offload: true
protrain_zero3_shard: true
```

```yaml
# Rejected until adapter support exists.
optimizer: adafactor
```

**Non-conflicting low-VRAM behavior.** For active 4-bit ProTrain configs, the
loader auto-defers the fp32 embedding upcast. Users do not need to set
`embeddings_skip_upcast: true` for ProTrain; that knob remains a non-ProTrain
low-VRAM escape hatch.

**Launch-time conflict to avoid.** Use an explicit accelerate config file for
single-GPU and multi-GPU runs. On multi-GPU hosts, the user-level default
accelerate config can select every visible device and override the intended
`CUDA_VISIBLE_DEVICES` shape; pass `accelerate launch --config_file ...` with
the intended `num_processes`.

---

## 5. Validation methodology

Validation combines unit coverage, GPU-gated tests, hardware benchmarks, and
checkpoint round-trips. The tiers below define the evidence used for the
claims in §1 and §12.

| Tier | Coverage | Evidence |
|---|---|---|
| Unit + dev-marker tests | `pytest tests/protrain/` with default markers | Cost-model arithmetic, chunk layout, validators, checkpoint metadata, optimizer sharding, and CPU-portable runtime invariants. |
| GPU-marker single-GPU regression | `pytest -m gpu tests/protrain/`, file-by-file | Hook ordering, autocast, bnb-4-bit, PEFT-LoRA-container behavior, and real CUDA residency paths. |
| Multi-GPU regression | `test_paged_adam_offload_mgpu`, `test_real_multigpu_cross_mode_resume_{a_to_c,c_to_a}` | NCCL gather/reduce-scatter alignment, DDP `init_sync` bypass, and cross-mode resume behavior. |
| Vanilla single-GPU baselines | LoRA on 0.6B / 2B / 8B / 9B-4bit / 13B-4bit | 24 GiB 3090 fit floor for comparable non-ProTrain runs. |
| ProTrain head-to-head | Identical-hyperparam vanilla ↔ ProTrain pairs at each model size | Direct memory and throughput deltas attributed to the integration. |
| Orthogonal sweeps | LoRA rank, gradient accumulation, sequence length, batch size, optimizer | Compatibility with standard Axolotl tuning knobs. |
| Save / merge / resume | 8 scenarios, both architectures, both adapters | `protrain_optim/` checkpoint format, safetensors save, merge-lora, and resume hooks. |
| Controls | Vanilla LoRA resume with no `max_steps` change | Baseline behavior for resume-time loss movement and scheduler changes. |

**Memory metric.** All peak figures come from the HF trainer's internal
`memory/max_active` line (which calls `torch.cuda.max_memory_allocated()`),
not from `nvidia-smi`. `nvidia-smi` indexing on the test rig is unreliable
because the rig mixes RTX 3090, 3090 Ti, and Blackwell-class GPUs, so CUDA's
default `FASTEST_FIRST` device order doesn't match `nvidia-smi`'s ordering
unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set (this caused one early
multi-GPU launch to land on the wrong devices; reproducibility env in §11).

**Architecture coverage.** Both standard-attention (`.self_attn` — Llama-2,
Llama-3, Qwen3-0.6B) and Mamba-style linear-attention (`.linear_attn` —
Qwen3.5 family) were exercised; `block/layout_rules.py:_looks_like_block`
recognizes both attention conventions.

---

## 6. Validation and benchmark evidence

All peak-memory numbers use the HF trainer's `memory/max_active`
(`torch.cuda.max_memory_allocated()`), not `nvidia-smi`. The primary consumer
validation host is a non-NVLink 3090-class PCIe rig with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; high-memory and NVLink-class runs are called
out separately. The table keeps the legacy `§6.*` labels used elsewhere in
this proposal while removing per-run log detail.

### 6.1 Fit, memory, and throughput

| Ref | Shape | Result |
|---|---|---|
| §6.a | Meta-Llama-3-8B BF16 LoRA, single 3090 | Vanilla peak 15.83 GiB; ProTrain Mode A post-offload resident peak 3.08 GiB (`n_persist=128/130`), validating the chunk-residency memory reduction. |
| §6.b | Vanilla single-3090 baselines | 0.6B BF16, 2B BF16, 8B BF16, 9B 4-bit, and 13B 4-bit fit; 13B BF16, 27B BF16, and 27B 4-bit seq=256 exceed the vanilla 24 GiB envelope. |
| §6.c | Llama-2-13B 4-bit LoRA, single 3090 | ProTrain Mode A matches vanilla's 7.91 GiB ceiling and shortens 50-step runtime from 35.13 s to 16.72 s on the validation config. |
| §6.d | Qwen3.5-9B 4-bit QLoRA, bs=4 single 3090 | ProTrain reaches 6.14 sps vs vanilla 3.95 sps (+55%) at 12.79 GiB peak. |
| §6.h | Llama-2-13B 4-bit LoRA sequence sweep | ProTrain Mode A validates seq=512, 1024, and 2048 on one 3090; seq=2048 peaks at 18.94 GiB. |
| §6.i, §6.u, §6.v | Batch-size sweeps | 13B 4-bit Mode A on 4× 3090 scales from bs=1 to bs=4 (22.6→44.8 global sps) and OOMs at bs=8/seq=256; bs=8 fits at seq=128 for the validated 9B single-GPU and 13B 4-rank shapes. |
| §6.j, §6.q, §6.dd | Qwen3.5-27B 4-bit LoRA | Single-3090 Mode A fits seq=128 at 19.98 GiB with auto-deferred embedding upcast; seq=64 head-to-head is 34% faster than vanilla; explicit Mode B validates seq=256 under the 24 GiB ceiling. |
| §6.k | Mode A/B/C single-rank behavior | Forced Mode A/B/C and auto all fit the 27B 4-bit seq=128 single-rank shape; Mode C degrades safely to single-rank behavior when `world_size=1`. |
| §6.l, §6.m | LoRA rank and gradient accumulation | Mode A holds LoRA ranks 16/32/64 and gradient accumulation 1/4/8/16 with small, monotonic memory movement. |
| §6.rr, §6.zz | Mode B and Mode A/B/C matrix | Llama-3-8B 4-bit QLoRA validates Mode B (`n_persist=128 n_offload=0`) and final Mode A/B/C coverage across bs=1/2 and seq=256/512 on 4× 3090-class GPUs. |
| §6.zz.X | Mixed-SKU plan determinism | Rank-0 broadcast of hardware inputs keeps Mode C search plans consistent across mixed 3090/3090 Ti rigs; 2-rank, 3-rank, and 4-rank validation cells complete finite. |

### 6.2 Multi-GPU, topology, and framework comparisons

| Ref | Shape | Result |
|---|---|---|
| §6.e | 13B/9B 4-bit LoRA, 4× 3090 | Llama-2-13B Mode A reaches 24.68 global sps at 9.98 GiB/rank; Mode C reaches 23.67 global sps at 9.87 GiB/rank; Qwen3.5-9B Mode A reaches 32.53 global sps. |
| §6.p | DDP scaling | 13B 4-bit Mode A shows about 3.9× effective 4-rank scaling on non-NVLink PCIe. |
| §6.s, §6.t | 27B 4-bit multi-rank ceiling | 2× 3090 Mode C fits at 20.25 GiB/rank; 4× 3090 exceeds 24 GiB cards because communication state pushes the per-rank peak past the cap. |
| §6.x, §6.ff | FSDP/DeepSpeed comparison | On Llama-2-13B 4-bit LoRA, ProTrain Mode A is the throughput leader at acceptable memory; ZeRO-3 + CPU offload is the memory leader with substantially slower steps. |
| §6.pb | Path B LoRA grad sync | PCIe all-linear LoRA r=16 improves steady-state sps/rank by 15.1%, cuts NCCL collective count by 68%, and preserves numerical equivalence across 100-step parity tests. |
| §6.nv | NVLink-class behavior | On 2× A100-SXM4 (`NV12`), Path B is slower than native NCCL buckets, so the default disables it on NVLink-class fabric; Mode C is 1.43× faster than Mode A for the validated NVLink LoRA shape. |
| §6.uu | Mode C DDP bypass | Consumer non-NVLink 4× 3090 Mode C validates that ProTrain, not DDP, owns chunk collectives after the `DistributedType.NO` bypass. |

### 6.3 Full fine-tune, optimizer state, checkpointing, and resume

| Ref | Shape | Result |
|---|---|---|
| §6.f, §6.g | Optimizer memory | At LoRA scope, bnb 8-bit Adam saves little because trainable state is small. At full-FT scope on Qwen3-0.6B, paged 8-bit Adam cuts total peak from 5.59 GiB to 2.84 GiB and composes with ProTrain. |
| §6.z | Local full-FT Mode C | Qwen3.5-4B bf16 full-FT forced Mode C validates on 5× 3090-class cards with both `adamw_bnb_8bit` and `adamw_torch`, final safetensors save, and chunk-managed param restoration. |
| §6.nw | 9B full-FT capacity validation | Qwen3.5-9B bf16 full-FT forced Mode C validates 30/30 finite train steps, final safetensors save, full optimizer-state checkpoint/resume to step 100, and second resume to step 105 on a high-memory 2-rank host. Vanilla Axolotl full-FT is faster on the same high-memory host but uses about twice the active memory. |
| §6.o, §6.gg, §6.jj | Save, merge, reload | LoRA save/resume/merge passes for standard-attention and linear-attention models; 13B/27B merge-lora command/reload paths return successfully with expected memory ceilings. No serving throughput or merged-model quality claim is made. |
| §6.cc, §6.kk-oo | Advanced optimizer resume | Full optimizer-state sidecars validate same-world resume plus explicitly enabled 4→2 and 2→4 cross-world online resharding on Qwen3.5-4B full-FT packed seq=1024/2048 shapes. Param-group identity, reorder defenses, and fail-closed sidecar checks are covered by focused tests. |
| §16.B | Multimodal checkpoint fidelity | Qwen3.5 visual-key checkpoint fidelity is validated on the local fidelity-specific paths: saved keys remain native `model.visual.*` keys with no nested `model.language_model.visual.*` or wrapper-key leakage. |

### 6.4 Compatibility and guardrails

| Ref | Area | Result |
|---|---|---|
| §6.n | Tests | Rebased branch default ProTrain suite: 632 passed, 17 skipped, 180 deselected; the committed validation runner passes CPU, single-GPU, and two-GPU lanes on 3090-class local hardware. |
| §6.w | FlashAttention | 8B BF16 LoRA + ProTrain Mode A + `flash_attention: true` validates under the extended search timeout. |
| §6.y, §6.ss | torch.compile | ProTrain hook bodies are compile-disabled where needed, NF4 dequant uses a custom op, and end-to-end torch.compile runs pass. On bs=1 QLoRA Mode A, compile gives no throughput benefit after warmup, so it is compatibility coverage rather than a speed recommendation. |
| §6.ee | Apex FusedAdam | Source-built Apex FusedAdam validates in the supported local environment; proposal claims support with the documented environment constraint. |
| §6.qq, §6.ww | Fused/custom kernels | `lora_mlp_kernel` composition avoids unsafe `n_offload>0` plans, and shape-preserving placeholders keep custom autograd backward shape checks valid. |
| §6.bb | MoE | tiny Mixtral-class MoE passes vanilla and ProTrain Mode A; Qwen3.6-35B-A3B multimodal MoE 4-bit QLoRA completes bounded-image 100-step validation on 5× 3090-class cards. This is finite-run compatibility coverage, not numerical parity coverage. |
| §6.aa | Long-horizon convergence | 1500-step Llama-13B 4-bit Mode A vs vanilla remains within expected variance, with no evidence of chunk-shuffling-induced trajectory drift. |

### 6.5 Documented limits

| Ref | Limit | Status |
|---|---|---|
| §6.t | Qwen3.5-27B 4-bit on 4× 3090 | Out of reach on 24 GiB cards in this validation; validated alternatives are single-3090 Mode A and 2× 3090 Mode C. |
| §16.B B1 | 8B+ full-FT on 24 GiB cards | Mode A can OOM because DDP reducer and base-weight load order exceed 24 GiB; Mode C is the validated full-FT path. |
| §16.B B2 | bs=1 performance | Correctness and cost attribution are validated; bs=1 remains fixed-overhead-bound, gradient accumulation is the recommended production path, and CUDA Graphs remains optional future optimization work. |
| §6.nv | NVLink LoRA workloads that already fit | Vanilla DDP QLoRA can be faster than ProTrain when memory is not the binding constraint; ProTrain is intended for capacity-bound or Mode C/offload-bound cases. |
| §7.6 | Mode B/C optimizer-state calibration | The calibrated peak gate charges *all* trainable optimizer state to the GPU peak. This is exact for Mode A (everything GPU-resident); in Mode B/C the offloaded trainables' optimizer state lives on CPU, so the GPU peak is over-predicted. The error is one-directional and safe — it only makes the gate fail closed earlier (it can never under-predict into a runtime OOM), but it may reject a large full-FT offload config that would actually fit. A precise fix would charge only the GPU-resident trainable subset; deferred as future work because the under-count failure mode it would introduce is an OOM, so it needs dedicated subset-enumeration tests before it is safe to land. |
| §6.5a | SDPA attention budgeted by dispatched backend | **Resolved.** `attn_activation_bytes` no longer assumes `attn_implementation="sdpa"` is O(seq): SDPA can fall back to the math backend (O(seq²)). A warmup-scoped, fail-safe probe records the actually-dispatched backend (`ProfilerTrace.dispatched_sdpa_backend`, via torch's eligibility API on real inputs); `_attn_is_linear_mem` budgets O(seq) only with flash/mem-efficient evidence, else conservative O(seq²) (incl. the on-demand path with no warmup, and legacy traces). `TRACE_VERSION` 26→27 re-profiles stale caches. |
| §6.5b | Mode-C 4-bit CKPT α (0.95) floored to 1.0 | **Resolved (documented as intentional + consistency fix).** The floor in `gate_consistent_alpha` is deliberate: the sub-1.0 deflation factors were phantom-compensators for the pre-SwiGLU-fix O(seq²) over-estimate; against the now-realistic sum, deflating would let the searcher accept configs the calibrated gate rejects. The constant/return site are now documented as calibration anchors (retained for per-dtype diagnostics). The runtime reverse-out (`_calibrate_peak_with_actual_chunk_bytes`) was fixed to divide by the *floored* alpha (matching how the searcher built the peak) instead of the raw sub-1.0 factor. |
| §6.5c | Enc-dec cross-attention in the empty-peaks fallback | **Resolved.** `_saved_tensor_bytes_per_block` now adds a second (cross-)attention term to decoder blocks (tree index > 0) on encoder-decoder traces, reusing the same tree signal the op-walk uses for `cross_attn_persist_bytes`. Decoder-only traces have a single tree, so it is an exact byte-for-byte no-op there; it only ever adds bytes (safe direction). |
| §6.6/§7.6 | Single-card figures pending hardware re-validation | An independent 2026-06-01 reproduction (torch 2.11.0 + current merged code, RTX 3090 Ti, same Qwen3-14B QLoRA shape) could **not reproduce the §6.6 "with compile" numbers**: `torch.compile` OOMs the 4-bit stack at ~23.5 GiB (step 0, NF4 dequant) under *both* vanilla and ProTrain, and the §6.6 "Mode A" config does not force Mode A — `protrain_force_all_persistent` is overridden by `protrain_auto_mode` (default true; set `protrain_auto_mode: false` to force it), so the searcher picks an all-resident plan with **measured ~18.4 GiB**, not the 12.71 GiB figure (which appears to be a gate *prediction*; the calibrated gate read 16.59 GiB vs 18.29 GiB actual — a pre-existing ~9% optimism, within the documented headroom, NOT introduced by the PR #37 cost-model changes). Independently, **vanilla Axolotl + its fused kernels matches Unsloth** (7.78 s/it / 12.83 GiB reserved vs Unsloth 7.69 / 12.75, same card, no ProTrain) — so single-card parity is largely Axolotl's fused-kernel stack, with ProTrain Mode A inert at this scale (consistent with the bs=1 framing in §6.6/m1). The discrepancy may be partly environmental (the §6.6 measurement stack vs this repro); §6.6/§7.6 single-card claims should be re-validated with *measured* `torch.cuda` peaks before upstreaming. Full data + configs: `Desktop/ProTrain/vanilla_baseline_and_repro_2026-06-01.md`. |

### 6.6 Single-card efficiency: Mode A under the full Axolotl + Liger kernel stack

Forced Mode A is runtime-inert: `_is_runtime_inert` short-circuits hook
installation (no per-step gather/offload; only the optimizer-step wrapper runs),
so steady-state per-step cost equals vanilla Axolotl + gradient checkpointing.
That makes Mode A a zero-overhead substrate for Axolotl's and Liger's fused
kernels plus `torch.compile`. This was benchmarked head-to-head against Unsloth
on identical hardware.

Shape: Qwen3-14B, QLoRA `r=64 alpha=64`, dense targets
(`q,k,v,o,gate,up,down`), `sequence_len=1024`, micro-batch 1, grad-accum 4,
`optimizer: adamw_8bit` (bnb 8-bit AdamW), bf16. Hardware: one RTX 3090 Ti
(sm_86, 24 GiB), `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Both frameworks ran Flash
Attention 2 and their respective `torch.compile` paths; memory is
`torch.cuda.max_memory_*` (directly comparable between the two).

| Stack | s/it (steady) | reserved | active |
|---|---:|---:|---:|
| ProTrain Mode A + `lora_mlp/qkv/o_kernel` + `fused_attn_kernel` + Liger FLCE/RMSNorm + `torch.compile` | 8.07 | 12.71 GiB | 12.42 GiB |
| Unsloth optimized path + FA2 + compile | 7.69 | 12.75 GiB | 12.50 GiB |

Result: under the full stack, ProTrain Mode A is at **memory parity** with
Unsloth (marginally lower reserved and active) and **~5% lower throughput**,
while keeping activations GPU-resident — Unsloth reaches its footprint via
CPU-offloaded activation checkpointing. Memory progression on the same card:
baseline Mode A (no fused kernels) reserves 14.83 GiB; adding the fused LoRA
kernels + cut-cross-entropy drops it to 12.79 GiB; layering Liger
FLCE/RMSNorm + `fused_attn_kernel` and enabling `torch.compile` reaches the
12.71 GiB / 8.07 s/it line above.

No new ProTrain code is required for this result: Mode A here is byte-identical
to this PR's branch. The stack composes existing Axolotl features (fused LoRA
kernels, Liger fused-linear-cross-entropy + RMSNorm, `torch.compile` with
compile-safe 4-bit dequant) with the upstream Qwen3 fused RMSNorm+RoPE kernel
(`fused_attn_kernel`, upstream Axolotl).

Recommended single-card QLoRA config (sm_86):

```yaml
plugins:
  - axolotl.integrations.protrain.ProTrainPlugin
  - axolotl.integrations.liger.LigerPlugin
protrain_auto_memory: true
protrain_force_all_persistent: true      # Mode A: all-resident, runtime inert
load_in_4bit: true
adapter: qlora
lora_mlp_kernel: true                     # Axolotl LoRA-aware fused MLP
lora_qkv_kernel: true
lora_o_kernel: true
fused_attn_kernel: true                   # upstream Qwen3 fused RMSNorm+RoPE
liger_fused_linear_cross_entropy: true    # Liger LM-head loss fusion
liger_rms_norm: true                      # Liger block-level RMSNorm
torch_compile: true                       # 4-bit dequant is compile-safe
flash_attention: true
gradient_checkpointing: false             # ProTrain owns checkpointing
trust_remote_code: false                  # required alongside lora_*_kernel
optimizer: adamw_8bit
```

Kernel-stack compatibility (what can and cannot be combined):

- `fused_attn_kernel` (q/k-norm + RoPE) is mutually exclusive with `liger_rope`
  — both fuse RoPE; enable only one.
- `lora_mlp_kernel` (LoRA-aware) is mutually exclusive with
  `liger_glu_activation`; keep `lora_mlp_kernel` for QLoRA-MLP correctness.
- `liger_fused_linear_cross_entropy` replaces `cut_cross_entropy`; use one.
- `liger_rms_norm` composes with `fused_attn_kernel`: the fused kernel owns the
  attention-internal q/k-norm, so Liger's RMSNorm swap lands on the block-level
  norms.
- `fused_attn_kernel`'s throughput win scales with sequence length; at
  `seq=1024` it is approximately break-even on the q/k-norm+RoPE path and
  contributes more at longer context.

### 6.7 Long-context single card: seq 32768 via activation swap

Where Mode A stops fitting — its all-resident + checkpoint-chain peak for
Qwen3-14B QLoRA is ~24.6 GiB at `seq=32768`, over a 24 GiB card — partial
activation swap brings it back under the ceiling. Forcing
`n_swap=24, n_checkpoint=16` (24 of 40 transformer blocks stream their
saved-for-backward activations to pinned CPU; the rest checkpoint) trains a
clean step on one RTX 3090 Ti:

| seq | config | loss | max_active | device_reserved |
|---|---|---:|---:|---:|
| 32768 | QLoRA r=64, `n_swap=24`/`n_checkpoint=16`, ~227 GiB pinned host | 0.011 | 19.1 GiB | 20.4 GiB |

Confirmed independently on a 32 GiB RTX 5090 (`n_swap=20`, loss 0.011). This is
PCIe-bound — ~minutes/step with 24 swapped blocks — so it proves the memory
envelope rather than a fast path; **16k remains the practical single-card
ceiling**. The sweet spot is counter-intuitive: too FEW swapped blocks
(`n_swap=12/16`) OOM, because each remaining checkpointed block materializes
~one full forward (~8 GiB at 32k) during its backward recompute, so MORE swap
(fewer recompute transients) LOWERS the peak.

Two fixes in this PR made the path usable:

- The activation-swap pool is now a **variable-size slab** (one pinned
  `PinnedHostMemory` region + a first-fit offset allocator with coalescing,
  `block/swap_pool.py`), sized to the real per-block saved aggregate. It replaces
  fixed equal-size slots, which both wasted pinned RAM (every slot sized to the
  max tensor) and exhausted mid-iteration (more saved tensors than slots →
  activations spilled back onto the GPU, breaking the steady-peak prediction).
- The CPU-Adam constructor no longer hangs: `DeepSpeedCPUAdam.__init__` calls
  py-cpuinfo's `get_cpu_info()`, whose forked CPU-probe subprocess deadlocks once
  the process holds a live CUDA context plus worker threads (mis-read earlier as
  the CUDA-toolkit version mismatch). `chunk/optim.py` swaps the bound
  `get_cpu_info` for a subprocess-free `/proc/cpuinfo` vendor lookup.

---

## 7. Comparison to the paper

Paper claims in the left column come from arXiv 2406.08334v2. The middle
column reports this integration's measured numbers; the right column flags
agreement or divergence.

### 7.1 Throughput vs other frameworks

| Paper claim (§5.2.2, Fig. 3, Tbl. 3) | This integration | Agreement |
|---|---|---|
| **2.71×** average throughput over DeepSpeed / Colossal-AI / FSDP on 4× RTX 3090, BF16/FP16 LLMs, full-finetune | **Not directly comparable.** This integration's "vs" target is bare Axolotl (no DeepSpeed/FSDP), with LoRA + 4-bit. Where ProTrain is the only thing that lets the run fit (13B+4-bit Mode A single 3090), it provides 2.1× the throughput of vanilla LoRA at the same memory ceiling (§6.c). | **Different baselines, but same direction.** Paper compares to other memory-mgmt frameworks; this integration compares to a no-memory-mgmt Axolotl baseline because Axolotl users typically reach for DeepSpeed/FSDP via separate `deepspeed:` / `fsdp:` flags, which are mutually exclusive with this plugin. |
| Avg throughput **2090 tokens/s** on 4× RTX 3090 across the model ladder (GPT-2 1B/10B/15B/20B, OPT-13B/30B, LLaMA-13B/34B, Mistral-7B) | Llama-13B + 4-bit + LoRA: 24.68 sps × 256 tokens = **6320 tokens/s** on 4× 3090; Qwen3.5-9B + 4-bit + LoRA: 32.53 sps × 256 = **8328 tokens/s** | **Higher in absolute terms** because (a) LoRA trains 0.48% of params not all, and (b) base weights are 4-bit not BF16. The paper's setup pre-dates bnb-4-bit + LoRA so the regimes aren't comparable. |

### 7.2 Maximum trainable model size, single RTX 3090

| Paper claim (§5.2.1, Tbl. 2) | This integration | Agreement |
|---|---|---|
| **ProTrain trains up to 34B on a single RTX 3090.** GPT-2 architecture, BF16 weights, BF16/FP32 mixed-precision, full-parameter training. | **Qwen3.5-27B + 4-bit + LoRA + ProTrain fits on a single 3090 at seq=128 with ProTrain's auto-deferred embedding upcast (peak 19.98 GiB, loss 1.075).** Llama-2-13B + 4-bit + LoRA + ProTrain Mode A fits in 7.91 GiB and extends to seq=2048 at 18.94 GiB peak. Llama-13B BF16 OOMs at `model.to()` because the 26 GB raw weights exceed 24 GB — a weight-size constraint ProTrain cannot help with on its own; the bnb-4-bit composition is required. | **Direction agrees; absolute size smaller** because the paper trains full-precision weights and full optimizer state while exhausting CPU offload, whereas this integration is validated at LoRA-rank-16 + 4-bit (which the paper did not target). 27B is the largest *class* validated. |
| DeepSpeed baseline maxes at 15B on a single 3090 | Vanilla Axolotl LoRA maxes at 8B BF16 / 13B 4-bit on a single 3090 (Llama-2-13B + 4-bit qlora vanilla peaks at 7.91 GiB) | **Different stack, broadly consistent ordering.** |

### 7.3 Memory-reduction factor

| Paper claim | This integration | Agreement |
|---|---|---|
| ProTrain trains **2.47× larger models than DeepSpeed** on RTX 3090 | Single-3090 Meta-Llama-3-8B BF16 LoRA: vanilla 15.83 GiB → ProTrain Mode A 3.08 GiB resident = **~5.1× memory reduction** post-offload on a model that already fits | **Stronger than paper on the residency metric**, but on a smaller weight class. The paper's 2.47× is a "what fits at all" metric; this integration's 5.1× is a "what stays on GPU" metric. Both validate that the chunk-residency model works as designed. |
| **74.6% Adam-state reduction** | Qwen3-0.6B full-finetune, single 3090: `adamw_torch` 5.59 GiB → `paged_adamw_8bit` 2.84 GiB = **-49% total peak**. The -49% total figure includes weights and activations; the Adam-state slice alone is m+v fp32 ~2.75 GB → ~0.7 GB paged 8-bit, which lines up with the 74.6% headline. ProTrain Mode A composes cleanly with `paged_adamw_8bit` (same 2.84 GiB peak, 5.40 sps). | **Substantiated end-to-end** at full-finetune scope on this integration (§6.g). |

### 7.4 Per-iteration overhead expectations

| Paper claim | This integration | Agreement |
|---|---|---|
| §5.2.4 (Fig. 4b): CPU parameter updates are **effectively overlapped with GPU backward**, making CPU optim time "nearly negligible" in the breakdown | Mode A keeps optim on GPU (no CPU updates). Mode C exercises the overlap; the multi-GPU Mode C run (§6.e) at 23.67 sps is within 4% of Mode A's 24.68 sps, indicating the CPU step does not dominate at this scale. | **Agree at LoRA scope.** The "negligible CPU optim time" claim is strongest at full-finetune where there's enough Adam state to keep the CPU saturated; at LoRA scope there's so little CPU work that the overlap window is unused either way. |
| Paper §5.3.4: profiling 7B Mistral with bs=4 on 3090 = **3.09 s**; 20B GPT-2 = 5.38 s; search itself = 0.06 s avg | This integration's traces are cached on disk by `(arch_hash, bs, seq, sku, world)` after the first run. Cold profile + search for Meta-Llama-3-8B at bs=1 / seq=256 / 1 GPU completed inside the 30 m budget. Warm runs of the same shape skip the profile. | **Same order of magnitude; warm-cache reuse is added by this integration.** |

### 7.5 Throughput vs ZeRO-3

| Paper claim | This integration | Agreement |
|---|---|---|
| Paper §5.3.3 (Tbl. 3): on 4× A100, ProTrain throughput vs DeepSpeed (ZeRO-3 with offload) = **1.43× on Mistral-7B**, 1.28× on GPT2-10B, 1.46× on LLaMA-13B, 1.47× on GPT2-20B | This integration on 4× 3090 non-NVLink PCIe (different hardware): Llama-13B + 4-bit + LoRA Mode A = 24.68 global sps, Mode C ("internal ZeRO-3") = 23.67 sps. Ratio ≈ **1.04×** at LoRA scope. The DESIGN.md M7 benchmark (full Llama-3B + LoRA, bs=2, seq=256) shows DDP / sharded ratio = 30.90 / 5.93 = **5.21×** on the same hardware. | **Direction matches; magnitude depends on regime.** The 1.04× LoRA-scope ratio is small because the frozen-quantized base dominates and trainable params are tiny (no per-chunk all-gather pressure). The 5.21× full-finetune ratio from the M7 internal benchmark is *larger* than the paper's A100 number, because non-NVLink PCIe hurts sharded mode harder than A100's PCIe Gen4 + NVLink. |

### 7.6 Cost-model accuracy

| Paper claim | This integration | Agreement |
|---|---|---|
| §5.3.2 + §C.2: runtime and peak-memory estimators within **4% error** on the 10B GPT-2 model; within **10% over-estimate** on peak memory across the model ladder for safety | After the SwiGLU FFN-intermediate fix (`attn_activation_bytes` now charges THREE `seq×ffn_width` intermediates per gated-MLP block — `gate_proj` out, `up_proj` out, and the `silu(gate)×up` product — not one), the calibrated gate tracks measured `max_memory_allocated` within **~1% on Qwen3-14B QLoRA** across the seq ladder: 8k 15.97/16.04, 16k 21.15/21.36, 24k 26.32/26.68, 32k-swap 18.97/19.10 GiB (mean 0.9% / max 1.3% relative; slightly *under*, covered by the 2 GiB capacity headroom). The residual drift is an O(seq) backward-gradient transient, NOT an O(seq²) attention term — a clean Mode-A seq sweep shows the per-block recompute set scales ~linearly (recompute/seq ≈ const), and an eager-vs-FA2 control confirms it (eager OOMs allocating the O(seq²) score matrix; FA2, which checkpoint-recompute re-runs, never materializes it). The per-mode α split (Mode-A 0.75 / Mode-C-CKPT 0.95) and the `_compute_ckpt_chain_bytes` per-block residual are otherwise unchanged. Final acceptance still uses the wrapper-side `_calibrate_peak_with_actual_chunk_bytes` (`api/model_wrapper.py`); the runtime "peak prediction calibrated X → Y GiB" log line is what users watch near the 24 GiB ceiling. These figures are Mode A (everything GPU-resident); in Mode B/C the gate conservatively over-charges offloaded trainable optimizer state to the GPU peak — see the §7.6 row in §6.5 (Documented limits). | **Tighter than the paper's own tolerance.** The paper's estimator carries a deliberate ~10% over-estimate (α=1.10, §3.3) as a safety margin; this integration tracks within **~1%** — and slightly *under*, leaning on the capacity headroom rather than a built-in margin. `protrain_ckpt_internal_residual_factor` (default 1.0, 0.0 disables) remains for conservative tuning. Validated post-SwiGLU-fix on Qwen3-14B QLoRA Mode-A (8k/16k/24k) and a 32k partial-swap config. |

### 7.7 CPU Adam and the overlap window

| Paper claim | This integration | Agreement |
|---|---|---|
| §A.1 + Tbl. 4: ProTrain uses DeepSpeedCPUAdam for non-persistent chunks and overlaps the CPU step with GPU backward; FusedAdam (apex) on persistent chunks | This integration uses `deepspeed.ops.adam.DeepSpeedCPUAdam` for non-persistent and `apex.optimizers.FusedAdam` when available (falling back to `torch.optim.AdamW` — non-fused — when apex isn't installed). `step_async(chunk_id)` is the path used during overlap. The HF-Trainer-side optimizer string `adamw_apex_fused` is wired through `_SUPPORTED_OPTIMIZERS`; §6.ee validates a CUDA-aligned source build plus train-time validation. | **Faithful**, with an apex-optional fallback path the paper didn't need to specify. |

### 7.8 Block-level forward prefetch / async backward reduce

| Paper claim | This integration | Agreement |
|---|---|---|
| §3.1 + Eqs. 3–7: forward issues `gather + upload` for chunks `> n_persist`; backward issues `reduce + offload` for the same, with `buffer_pool` carrying forward-resident chunks into backward to avoid double-loading | This integration implements both via `runtime/scheduler.py`'s `prefetch_chunks` / `reduce_grads_and_offload`. The `BufferPool` (`chunk/buffer_pool.py`) carries forward-resident slots into backward, exactly per paper §3.1.1. | **Faithful** to the paper. |
| §3.1.2: SWAP wrapper D2H's activations on `_swap_stream` in forward, H2D's back in backward | `block/swap.py::SwappedBlock` wraps every autograd-saved tensor (not just block output) in `saved_tensors_hooks` and routes through a pinned `ActivationSwapPool` — now a **variable-size slab** (one `PinnedHostMemory` region + a first-fit offset allocator with coalescing, `block/swap_pool.py`) sized to the per-block saved aggregate, so it charges the true sum of saved-tensor bytes instead of fixed equal-size slots. | **Faithful**, with the explicit clarification (DESIGN.md §3.1.2) that memory accounting must charge the sum of saved-tensor bytes, not just the block output — which the slab now does directly. |

### 7.9 Single-stream allocation and pinned-host allocator

| Paper claim (App B.2) | This integration | Agreement |
|---|---|---|
| Single-stream GPU allocation (routes all allocations through the default-stream heap to avoid PyTorch's per-stream free-list fragmentation) | `runtime/streams.py::SingleStreamAllocator` is wired across `BufferPool.__init__`, `chunk/manager.py` (every chunk allocation), and `block/swap.py::unpack_from_pool`. `record_stream` discipline documented in DESIGN.md and tested by `test_single_stream_allocator.py`. | **Faithful**, with the `record_stream` cross-stream-handoff contract explicitly tested. |
| Custom pinned-host allocator via `cudaHostAlloc` ctypes binding (avoids `CUDAHostAllocator`'s power-of-two rounding) | `chunk/pinned_alloc.py::PinnedHostMemory` calls `cudaHostAlloc` via ctypes for exact byte counts; falls back to `torch.empty(pin_memory=True)` if libcudart isn't loadable. Wired into `BufferPool`, `chunk/manager.py::materialize_offload`, and `block/swap_pool.py`. | **Faithful**, with the ctypes-load fallback the paper didn't need to specify. |

---

## 8. Divergences from the paper

### 8.1 Per-dtype α fragmentation factor

**Paper.** Eq. 11 of the cost model multiplies the predicted peak by a
single constant α (default 1.10) "to account for potential memory
inefficiencies due to memory fragmentation". The paper does not condition α
on parameter dtype.

**This integration.** `cost/memory.py::alpha_fragmentation_for_dtype` returns
**1.10** for fp16 / bf16 / 8-bit dtypes (`bpe ≥ 1.0`) and **0.75** for bnb
4-bit Mode-A-style callers (`bpe = 0.5` via `Params4bit` packing).
Mode-aware callers use `alpha_fragmentation_for_cfg`: 4-bit configs with
checkpointed Mode-C blocks use **0.95**, while non-4-bit dtypes keep **1.10**.
The dominant dtype is detected at `protrain_model_wrapper` construction by
aggregating logical element counts over `model.named_parameters()`.

**Why.** Measured `α_steady ≈ 0.70` across four 8B-Llama 4-bit rows
vs the paper's 1.10 default; 1.10 over-predicts bnb-4-bit Mode-A
peak by ~37% and rejects otherwise-valid 4-bit configs at the
searcher gate. 0.75 (slightly conservative vs the 0.70 empirical
floor) keeps the Mode-A search space open; 0.95 narrows the Mode-C-CKPT raw
under-prediction without replacing the wrapper-side calibrated budget gate.
Tests: `tests/protrain/test_alpha_per_dtype.py`,
`tests/protrain/test_cost_model.py`, and
`tests/protrain/test_alpha_diagnostics.py`.

### 8.2 bnb 4-bit support (paper-era ProTrain didn't have this)

**Paper.** All evaluation in §5 uses BF16 or FP16 weights with FP32
optimizer state. bnb / 4-bit quantization is mentioned only as orthogonal
related work in §D.

**This integration.** Validated end-to-end with `bitsandbytes.nn.Linear4bit`
+ `adapter: qlora` on Qwen3.5-9B, Llama-2-13B, and Qwen3.5-27B, both single
and multi-GPU. The relevant integration points:

- `Params4bit` instances are mapped to `bpe = 0.5` explicitly in the
  dominant-dtype walk (their `element_size()` is 1 but each byte packs two
  4-bit values).
- The per-dtype α (8.1) is the cost-model accommodation.
- Per-dtype-region sharding in `chunk/manager.py::_gather_sharded` (each
  chunk is decomposed into `_DtypeRegion` entries, one per
  maximal-length contiguous same-dtype run) handles the mixed-dtype
  layouts that show up when 4-bit Linear weights coexist with FP32 RMSNorm
  scales inside the same chunk.
- The Phase-2 chunked-steady cost-model measurement excludes bnb-4-bit
  companion buffers (`absmax`, `quant_map`, nested storage) when accounting
  for chunk size, so the override path's per-chunk save matches the
  analytical overlap and the buffer-shortfall surcharge accurately.

**Why.** bnb-4-bit is the single most common Axolotl-side memory technique;
shipping a memory-management plugin that ignores it would have left the
common 24 GB-3090 LoRA workflow unimproved.

### 8.3 PEFT-LoRA container hook quartet

**Paper.** Hooks are registered at transformer-block granularity, assuming
each block's parameters are full tensors that live in chunks.

**This integration.** When PEFT LoRA is wrapped on top of the base model,
LoRA factors (`lora_A` / `lora_B` / `lora_magnitude_vector` /
`lora_embedding_*`) live as trainable sub-modules *inside* base-model
blocks. The runtime block-level gather releases the underlying chunk after
the block forward, but PEFT's `LoraLayer.forward` casts the LoRA factors to
bf16 in a separate autograd op that needs them resident — producing the
canonical `ToCopyBackward0 returned an invalid gradient at index 0 — got
[N, R] but expected shape compatible with [0]` failure.

The LoRA offload path installs a **quartet** of pre-fwd, post-fwd,
pre-bwd, post-bwd hooks at every PEFT LoRA container, at both the profiler
trace surface (`profiler/on_demand.py::_find_peft_lora_containers`) and the
runtime scheduler surface (`runtime/scheduler.py::ensure_chunks_resident`).
The placeholders are `scratch.expand(slot.shape)` views (not zero-element
tensors) so `param.size()` metadata survives the release/re-gather cycle
and autograd's shape-capture doesn't trip.

**Why.** PEFT post-dates the paper; this integration needs to compose with
it cleanly because LoRA is the dominant Axolotl fine-tuning workflow.

### 8.4 Mode-C cross-rank resume bridge

**Paper.** Checkpoint/resume is not discussed in detail; the paper's
implementation assumes a contiguous training run.

**This integration.** `_install_resume_hook` (plugin.py) monkey-patches
HF's `_load_from_checkpoint` to:

1. `restore_to_gpu()` every offloaded chunk *before* HF copies loaded
   weights into full-shape `param.data` slots (otherwise HF writes into
   the zeroed non-persistent placeholders, and ProTrain's first gather
   overwrites the loaded weights with the still-zero CPU shadow).
2. Re-run `materialize_offload()` and rebuild the per-chunk optimizer
   adapter *after* HF returns.

The current resume implementation closes both same-mode and cross-mode
(A → C, C → A) resume. The cross-mode case requires reconciling
on-disk Mode A weights (full-replicated chunk views) with a Mode C runtime
layout (sharded chunks across ranks), which is non-trivial because each
rank has to find its byte range of every chunk and discard the rest.

**Why.** Long training runs need checkpoint/resume; cross-mode resume in
particular matters because operators may want to start in Mode A for early
warmup and shift to Mode C when activations grow.

### 8.5 The Option B `n_offload` axis

**Paper.** The activation block manager has three modes: `NONE`, `CKPT`,
`SWAP`. Chunk-level offload of model states is governed only by `n_persist`.

**This integration.** A fourth `BlockMode.OFFLOAD` (defined in
`block/strategy.py`) and a fifth search axis `n_offload` were added (see
`BLOCK_MODE_OFFLOAD_DESIGN.md`). An OFFLOAD block runs without
recomputation; its owning chunk is re-gathered for backward and offloaded
again after the fwd. This composes block-level activation management with
chunk-level state offload, giving the searcher a finer knob between
"checkpoint everything" (paper's high-pressure default) and "swap
activations" (PCIe-saturating).

**Why.** On non-NVLink PCIe the swap path is bandwidth-bound and rarely chosen;
the searcher otherwise had only `CKPT` to fall back on, which over-pays
in recomputation. `OFFLOAD` (re-gather without recompute) is cheaper when
the chunk happens to be small relative to the block's activation footprint.

### 8.6 NCCL late-bind for the auto-mode selector

**Paper.** Profiler is run at model-construction time; NCCL is assumed
already alive for the cost model's gather/reduce timing.

**This integration.** `protrain_model_wrapper` runs from
`plugin.post_model_load`, which fires at Axolotl's `loaders/model.py:191`
— *before* Accelerate brings up the distributed process group. So the
profiler's `measure_nccl(world_size > 1)` falls through to empty tables on
the first run, and the trace records `world=1`. `plugin.post_trainer_create`
calls `_remeasure_nccl_and_research(wrapped)` after Accelerate inits the
process group, splices the real NCCL tables into the cached trace, and
re-runs `search()` with the same layout and capacity. This affects Mode C
only (Mode A and Mode B don't consume the NCCL tables); the bootstrap
config is used for the first iteration, then the running step picks up the
post-NCCL config.

**Why.** Axolotl + HF Trainer + Accelerate ordering makes the
construction-time NCCL measurement unavailable; this integration adapts.

---

## 9. Current operating notes

The current implementation is feature-complete for LoRA / QLoRA ProTrain
training on the validated 3090-class topologies. The operating constraints are
hardware capacity for larger full-FT shapes, bounded multimodal batch sizing on
24 GiB cards, and known throughput tradeoffs.

- **8B+ full-finetune needs larger hardware than 24 GiB cards.**
  Full-finetune at 8B-class scale is hardware-bound locally even with optimizer
  state partitioning; the high-memory Qwen3.5-9B validation is recorded in
  §16.B B1.
- **bs=1 Mode A is fixed-overhead dominated.** The all-persistent inert path
  prunes ProTrain's block hooks and drain-side stream sync, and the remaining
  batch-independent cost has been profiled in §16.B B2. Use higher
  micro-batches or gradient accumulation where possible; CUDA Graphs remain
  optional future throughput work.
- **27B + 4-bit on a single 3090 is validated at seq=128 in Mode A and
  seq=256 in explicit Mode B.** Higher-throughput Mode A at longer 27B
  sequence lengths still requires multi-GPU or larger-memory hardware.
- **Cross-world Mode C full optimizer-state resume requires sidecar metadata.**
  Qwen3.5-4B full-FT packed seq=1024 and seq=2048 validate 4-rank save →
  online 2-rank resume on 3090-class hardware; packed seq=2048 also validates
  2-rank save → online 4-rank resume. Older persistent GPU optimizer
  checkpoints without `gpu_optim_rank_*.pt.meta.json` fail closed for cross-world
  resume.
- **Apex `FusedAdam` is source-build validated when `CUDA_HOME` is aligned.**
  The local system toolkit is CUDA 13.2 while torch is `cu130`; pointing the
  build at a CUDA 13.0 toolkit prefix produces a working Apex wheel, a direct
  `FusedAdam` CUDA step, and an Axolotl `adamw_apex_fused` validation run
  (§6.ee).
- **27B-class 4-bit ProTrain auto-defers the load-time fp32 embedding upcast.**
  The loader skips the upcast only when ProTrain is active; plugin-only schema
  registration remains inert. Non-ProTrain low-VRAM configs may still use
  `embeddings_skip_upcast: true` explicitly.
- **Mixed-SKU rigs should set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.** This keeps
  `CUDA_VISIBLE_DEVICES` aligned with the physical devices shown by
  `nvidia-smi`.

---

## 10. Current feature set

The current codebase exposes ProTrain as an Axolotl plugin with these
capabilities:

- **Activation and parameter residency management.** Chunked model state,
  block-level Mode A/B/C execution, pinned-host offload buffers, swap and
  prefetch streams, checkpointed blocks, and shape-preserving placeholders for
  custom-autograd compatibility.
- **Config-driven mode selection.** Users can run auto-mode or force Mode A
  (`protrain_force_all_persistent`), Mode B
  (`protrain_force_replicated_cpu_offload`), or Mode C
  (`protrain_zero3_shard`). Force-mode validators keep those choices mutually
  exclusive.
- **QLoRA and LoRA compatibility.** The integration composes with PEFT LoRA,
  bnb 4-bit QLoRA, LoRA MLP kernels, DoRA coverage, LoRA-style PEFT adapter
  surfaces whose parameter ownership matches LoRA, and `torch.compile` on the
  validated NF4 path.
- **Cost model and searcher.** The searcher accounts for dtype-specific
  fragmentation, CKPT internal saved tensors, bnb 4-bit companion buffers,
  CPU/offload overlap windows, NCCL re-measurement, non-NVLink preferences,
  and cross-rank plan determinism.
- **Runtime robustness.** Per-chunk NCCL warm-up, separate offload and prefetch
  streams, scheduler drain at optimizer boundaries, loud-inert plugin warnings,
  actionable searcher infeasibility messages, and `RuntimeError` invariants for
  hot-path residency checks are all present in the current runtime.
- **Checkpoint and resume.** ProTrain optimizer state is saved under
  `protrain_optim/`, supports same-world and cross-world resume, and includes
  stable param-name sidecars for Mode C persistent optimizer-state resharding.
- **Loader integration.** The loader auto-defers the 4-bit embedding upcast
  when ProTrain is active and installs the PEFT preparation patch through the
  same `is_protrain_active(cfg)` gate.
- **Version and CI guardrails.** Startup checks validate the PEFT and
  Transformers API surface used by the plugin; the CI version gate compares
  pyproject pins against the validated upper bounds declared in
  `src/axolotl/integrations/protrain/check.py`.
- **Telemetry.** Peak-memory reporting gathers the worst rank with a backend
  aware distributed max, so multi-rank runs report cluster-wide peak rather
  than rank-0-only peak.

---

## 11. Setup / reproducibility

Validation used Python 3.12, torch 2.11+cu13.0, `bitsandbytes` 0.49.1, and
`deepspeed` 0.18.2. The main consumer-GPU host is a non-NVLink PCIe pool of
RTX 3090 / 3090 Ti class cards; high-memory RunPod and NVLink-class A100 runs
are identified separately in §6.

Use explicit Accelerate configs instead of the user default config:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID \
TOKENIZERS_PARALLELISM=false \
accelerate launch --config_file <single-or-multigpu-accelerate.yaml> \
  -m axolotl.cli.train <protrain-config.yml>
```

The single-GPU Accelerate config sets `distributed_type: NO` and
`num_processes: 1`. Multi-GPU configs set `distributed_type: MULTI_GPU`,
`mixed_precision: bf16`, and the intended `num_processes`. Mixed-SKU rigs
should keep `CUDA_DEVICE_ORDER=PCI_BUS_ID` so `CUDA_VISIBLE_DEVICES` maps to
the physical devices shown by `nvidia-smi`.

Config constraints that affect reproducibility:

- Do not set `gradient_checkpointing: true`; ProTrain owns block-level CKPT.
- Force modes with `protrain_auto_mode: false` plus one force flag.
- 27B-class 4-bit ProTrain configs auto-defer the load-time fp32 embedding
  upcast only when `protrain_auto_memory: true` is active.

---

## 12. Validated-claims checklist

This is the claim surface the proposal relies on. Detailed evidence is in §6
and §16.

Reviewer-facing support status:

| Status | Scope |
|---|---|
| Supported and validated | LoRA/QLoRA Mode A/B/C at stated 3090-class shapes; same-world resume; final safetensors save; Path B LoRA sync with topology-aware default; active-ProTrain 4-bit embedding-upcast deferral. Advanced cross-world Mode C optimizer resume is validated for stated Qwen3.5-4B full-FT shapes and remains opt-in. |
| Supported with environment constraint | Apex FusedAdam requires a CUDA/toolkit-aligned source build; GPU and multi-GPU regression lanes require suitable local/self-hosted hardware. |
| Future optimization, not correctness gap | bs=1 CUDA Graphs / deeper launch capture work. Correctness and cost attribution are validated for the stated Mode A inert and Path B rerun shapes; production guidance is higher micro-batch or gradient accumulation. |
| Rejected by validator | DeepSpeed/FSDP composition, Axolotl-level `gradient_checkpointing`, unsafe force-mode combinations, unsupported optimizer families, and `protrain_auto_memory: true` without the ProTrain plugin. |

| Claim group | Status | Evidence |
|---|---|---|
| Consumer-card memory fit | **Validated** | 8B BF16 LoRA memory reduction, 13B 4-bit seq=2048, and 27B 4-bit single-3090 / Mode B coverage (§6.a, §6.h, §6.j, §6.dd). |
| Throughput recovery | **Validated where claimed** | 13B 4-bit single-GPU speedup, 9B bs=4 speedup, DDP scaling, and batch-size scaling through bs=4 (§6.c, §6.d, §6.e, §6.i, §6.p). |
| Mode A/B/C behavior | **Validated** | Forced and auto-mode paths, single-rank Mode C fallback, consumer non-NVLink Mode C bypass, and mixed-SKU plan determinism are covered (§6.k, §6.rr, §6.uu, §6.zz, §6.zz.X). |
| Save, merge, resume | **Validated** | Standard- and linear-attn LoRA round trips, merge-lora command/reload paths, full-FT safetensors save, same-world resume, and 4→2 / 2→4 full optimizer-state resume (§6.o, §6.cc, §6.jj, §6.nw). |
| Full fine-tune | **Validated within stated hardware boundaries** | Qwen3-0.6B Adam-state reduction, Qwen3.5-4B local Mode C full-FT, and Qwen3.5-9B high-memory train/save/resume (§6.g, §6.z, §6.nw). |
| Optimizers | **Validated** | `adamw_torch`, bnb 8-bit Adam, paged 8-bit Adam, DeepSpeed CPU Adam, and source-built Apex FusedAdam paths have targeted coverage; Apex requires the CUDA/toolkit-aligned source build documented in §6.ee. |
| LoRA / QLoRA compatibility | **Validated at claimed scope** | PEFT LoRA container hooks, DoRA/extended-target ownership, Path B LoRA grad sync, LoRA-rank sweep, gradient accumulation, and torch.compile guardrails (§6.l, §6.m, §6.pb, §6.y, §6.ss). Numerical parity evidence is scoped to the stated LoRA Path B runs. |
| Model-family coverage | **Validated at claimed scope** | Standard attention, Qwen3.5 linear attention, tiny Mixtral-class MoE, and Qwen3.6-35B-A3B multimodal MoE 4-bit QLoRA (§6.bb, §16.B). MoE/multimodal rows are finite-run and checkpoint-fidelity coverage, not broad numerical-parity claims. |
| Framework comparisons | **Measured, not overclaimed** | ProTrain, FSDP2, DeepSpeed ZeRO-2/3, ZeRO-3+CPU, vanilla DDP, PCIe, and NVLink-class behavior are separated by hardware/regime (§6.x, §6.nv). |
| CI and regression coverage | **Validated** | Default ProTrain suite result is 632 passed, 17 skipped, 180 deselected; the committed validation runner passes CPU, single-GPU, and two-GPU lanes on 3090-class local hardware (§6.n, §15). |

---

## 13. Appendix: methodology details

**Memory.** All `Peak (GiB)` figures are read from the HF trainer's
internal `memory/max_active` log line, which is `torch.cuda.max_memory_allocated()
/ 1024**3` at the time of logging. This is the authoritative torch-side
metric, distinct from `nvidia-smi`'s `memory.used` (which includes the
caching allocator's reserve and is unreliable when the rig mixes SKUs).
For multi-GPU runs `memory/max_active` is reported per-step from rank 0;
per-rank breakouts would require an instrumentation patch (a
`torch.cuda.max_memory_allocated()` collective on every rank), not done in
this validation.

**Runtime / throughput.** `train_runtime`, `train_samples_per_second`, and
`train_steps_per_second` come from the HF Trainer's
`TrainOutput.metrics` returned by `trainer.train()`. For multi-GPU runs the
HF Trainer reports these per-rank; the "global sps" column in §6.e and
§6.i multiplies by `world_size`. This is correct in the absence of gradient
accumulation across ranks (which all multi-GPU runs here disable, GA=1).

**Profile cache.** The on-disk profiler cache key is `(arch_hash, bs, seq,
sku, world)` and lives under `protrain_cache_dir` (default
`~/.cache/axolotl/protrain`). `TRACE_VERSION` is prefixed to the key, so
internal schema bumps invalidate stale entries silently. Current
`TRACE_VERSION` is `23`.

**Step count.** 50 steps for single-GPU runs and 25 steps for multi-GPU
runs were chosen to amortize trainer warmup (data prep + first-iteration
profile + materialize_offload + DDP / NCCL init typically takes 60–100 s
before the first training step) over enough measured steps to make
samples-per-second stable, without paying for a multi-hour run. They are
NOT chosen for learning quality; see §9.

---

## 14. Architecture guardrails

These design constraints are part of the current implementation rather than
follow-up work:

- **Trainer resume hook.** `plugin.py::_install_resume_hook` wraps
  `Trainer._load_from_checkpoint` so ProTrain can restore offloaded tensors to
  GPU before Hugging Face loads checkpoint state, then re-offload and rebuild
  optimizer residency afterward.
- **PEFT API surface.** The LoRA container hooks depend on PEFT's `LoraLayer`
  adapter metadata and LoRA parameter naming. Startup checks fail loudly if the
  installed PEFT version no longer exposes the validated surface.
- **Optimizer installation timing.** The ProTrain optimizer is installed in
  `post_trainer_create`, after HF Trainer has materialized merged
  `TrainingArguments` defaults for Adam betas, epsilon, weight decay, and
  learning rate.
- **Pinned-host allocation fallback.** `chunk/manager.py` prefers
  `libcudart.cudaHostAlloc` for pinned buffers and falls back to
  `torch.empty(pin_memory=True)` when libcudart is unavailable.
- **Cost-model safety.** Raw `estimate_peak` remains a search heuristic;
  wrapper-side calibration with actual chunk bytes raises the prediction before
  comparing against the runtime memory budget.

---

## 15. CI guardrails

This PR does **not** require maintainers to provide GPU-backed GitHub Actions.
Standard CI remains CPU-only. GPU coverage is supplied by the committed
validation runner for reviewers or developers who have access to qualifying
CUDA hardware; unavailable GPU lanes report `SKIP` and do not fail the runner.
The runner is environment-native by default: it executes the current checkout
with `PYTHONPATH=src`, and can also be run inside the repository's existing
Docker or `docker compose` environment when a pinned container is preferred.

The ProTrain test suite spans three tiers, each with different runner
requirements:

| Tier | Tests | CI compatibility |
|---|---|---|
| Default-marker (CPU / dev) | Default ProTrain pytest coverage spans chunk management, validators, cost/search math, layout rules, checkpointing, torch.compile compatibility, schema behavior, sentinel re-exports, alpha diagnostics, rsLoRA ownership, MoE/VLM ownership, and Path B LoRA grad sync. Manual bs=1 wall-clock microbenches are opt-in via `PROTRAIN_RUN_BS1_MICROBENCH=1`. | Run on standard Axolotl CI **without GPU**. Latest recorded result: **632 passed, 17 skipped, 180 deselected**. |
| GPU-marker (single-GPU) | ~10 tests requiring CUDA + a transformer model load (alpha measurement against a real model, profiler trace round-trip, chunk-residency end-to-end) | Opt-in local/self-hosted validation only. Marker: `@pytest.mark.gpu`. **Not blocking on default CI**. |
| Multi-GPU regression | `test_paged_adam_offload_mgpu`, `test_cross_mode_resume` plus the validation runner's two-GPU Mode C lane | Opt-in local/self-hosted validation only. Use the committed runner or manual pytest commands on a qualifying multi-GPU CUDA host. **Not blocking on default CI**. |

Maintainer acceptance recipe:

The runnable entry point is
`PYTHONPATH=src python -m axolotl.integrations.protrain.validation`
(`validation/README.md`). It emits a compact lane/hardware/status matrix plus
PASS/FAIL/SKIP/GAP details, and applies finite-loss, finite logged grad-norm
when present, no-NaN/Inf, checkpoint-artifact, and resume loss-continuity gates
where the lane trains a model. `--gpu-devices` scopes single-GPU lanes to the
first selected device and two-GPU lanes to the first two selected devices; CPU
lanes hide CUDA with `CUDA_VISIBLE_DEVICES=""`.

| Lane | Run | Acceptance bar |
|---|---|---|
| CPU/default | `--suite maintainer` | Validators, cost/search math, calibrated memory gates, checkpoint metadata, deterministic chunk layout, Mode A/B/C selection, save/resume hooks, Path B LoRA ownership, merge-lora math, and non-finite fail-closed guards pass without a GPU. |
| CPU/exhaustive | `--suite cpu-full` | The full default ProTrain pytest suite passes without requiring large-model hardware. |
| One GPU, 24 GiB | `--suite single-gpu` | 8B 4-bit QLoRA trains 50 steps from `validation/recipes/3090-8b-qlora-acceptance.yml`, writes `checkpoint-50`, resumes for additional steps, keeps logged losses and any logged grad norms finite, and preserves bounded resume loss continuity. |
| One GPU edge | `--suite single-gpu-edge` | DoRA, multi-adapter switching, vision-LM hybrid ownership, LoRA offload runtime, and A<->C cross-mode resume pass on tiny/cheap model shapes. |
| Two GPUs | `--suite two-gpu` | Forced Mode C finite-training coverage, same-world optimizer save/resume, and non-finite-boundary tests pass on the visible two-GPU set. |
| Optional four GPUs | `pytest tests/protrain/ -m gpu` on a 3090-class PCIe pool | Covers Path B / Mode C PCIe regressions and cross-mode / cross-world resume beyond the minimal maintainer lanes. |

This recipe is the committed, reviewer-scale replacement for the temporary
Desktop/local validation scripts. Historical 30B/35B/9B runs remain benchmark
evidence for the proposal, while the recipe keeps day-to-day verification small
enough for future maintainers and portable across any qualifying multi-GPU
system.

Validation artifacts are inspectable by default under `--work-dir`. `--cleanup`
removes run logs/checkpoints/configs after a suite, and `--clean --work-dir ...`
removes an existing validation work directory without running lanes. Reusable
model, dataset, asset, and prepared-dataset downloads can be retained with
`--cache-dir PATH` or `--keep-cache`; cleanup removes run artifacts but leaves
that reusable cache intact.

Latest full-run result on a local 3090-class GPU selection
(`--suite full --gpu-devices 1,2,4,5,7 --keep-cache`): all lanes passed, the
single-GPU 8B QLoRA lane completed a finite 50-step train plus 10-step resume,
the two-GPU forced Mode C lane passed optimizer save/resume and
non-finite-boundary checks, and cleanup verified that run artifacts and the
reusable validation cache can be removed after inspection.

### Version guardrails (load-bearing for monkey-patches)

The plugin monkey-patches `transformers.Trainer._load_from_checkpoint`
and depends on PEFT's `LoraLayer` internals via the container hook
quartet (`runtime/hooks.py`). Current validated bounds in
`src/axolotl/integrations/protrain/check.py`:

- `VALIDATED_TRANSFORMERS_MAX = "5.9"`; current pyproject pin:
  `transformers == 5.5.4`.
- `VALIDATED_PEFT_MAX = "0.21"`; current pyproject range:
  `peft >= 0.19.1, < 0.20.0`.

If either upper bound moves, the monkey-patch and container-hook code
paths need targeted re-validation. The patch surface is small (one
method override + four hook factories) so the integration cost of a
future version bump is bounded. The `.github/workflows/protrain-version-check.yml`
gate compares the pyproject pin against the current validated range.

**Current startup gate.** A startup assertion probes
`LoraLayer.adapter_layer_names` and the existence of
`Trainer._load_from_checkpoint` makes the failure loud at config time
rather than silent at training time if either upper bound is exceeded
without re-validation.

---

## 16. At-scale validation matrix

This table summarizes the large-run validation surface that does not fit
cleanly into the smaller single- and multi-GPU benchmark matrix.

| # | Area | Scope |
|---|---|---|
| B1 | **9B full-FT Mode C: finite high-memory train/save/resume** | Validated on 2× H100 NVL RunPod (`NODE` fabric) with Qwen3.5-9B full-FT, bf16, seq=256, bs=1, forced Mode C. The branch completes finite training, safetensors save, full sharded `protrain_optim/` save, full-state resume to step 100, and second resume to step 105. Local 24 GiB regression coverage uses Qwen3.5-4B forced Mode C because the 9B shape exceeds local card capacity. Exact Qwen3.5-9B multimodal checkpoint fidelity is validated on 2× RTX PRO 6000 Blackwell at seq=256: a seed checkpoint and checkpoint-1 resume both complete finite, the resume restores sharded `protrain_optim/`, final saves restore 9.002 GB and unwrap 32 blocks, and both final safetensors use native keys (`760` total, `333` `model.visual.*`, `0` nested visual, `0` `.block.`). Runtime resume load emits HF block-wrapper missing/unexpected-key warnings, but the saved artifacts are native-key clean. |
| B2 | **bs=1 throughput is fixed-overhead-bound; use `gradient_accumulation_steps >= 4`** | At bs=1 on Llama-3-8B 4-bit qlora Mode A, the all-persistent inert predicate fires and ProTrain installs no block hooks; the inert scheduler drain also skips side-stream synchronization and empty chunk-manager queues. The manual CPU hot-path microbench (`PROTRAIN_RUN_BS1_MICROBENCH=1`) shows bs=4 only ~1.09× bs=1 in the hook-only path, so the batch-independent cost amortizes with accumulation. Recommended config: `gradient_accumulation_steps: 4`; measured per-rank bs=1 = 0.229 sps → bs=4 via grad-accum = 0.738 sps (3.22×/sample). A Qwen3.5-9B all-linear QLoRA rerun on 4× 3090 PCIe confirms the active Path B profile: OFF completes 25 steps in **68.46 s** (`0.365` steps/s) and ON completes in **64.61 s** (`0.387` steps/s), with captured steady micro-step mean **0.648 s → 0.561 s** and peak **8.48 → 8.37 GiB**. Deeper CUDA Graphs capture remains optional future throughput work, not a correctness gap. |
| B3 | **At-scale cross-world Mode C full optimizer-state resume** | Closed for the local regression class: Qwen3.5-4B full-FT forced Mode C validates packed seq=1024 and seq=2048 4-rank save → online 2-rank full optimizer-state resume through finite steps 4-6, with final safetensors saves. The inverse packed seq=2048 2-rank save → online 4-rank path also passes under `adamw_bnb_8bit`, and seq=2048 4→2 repeats under `adamw_torch`. The path requires param-name sidecar metadata for persistent GPU optimizer state; older sidecar-less checkpoints fail closed for cross-world resume. |

---

**Document status.** Current-state proposal for the ProTrain integration branch.
Benchmarks and validation claims are the measured results in §6; current feature
surface is summarized in §10; at-scale validation is summarized in §16.
