# FP4 fused-CE re-engineer: the `fp4_matmul` tile loop is NOT eager — the +54% was a measurement artifact

**Date:** 2026-06-10 · **GPU:** RTX 5090 (sm_120), `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6` (verified)
**Worktree/branch:** `axolotl-nvfp4-lm_head` @ `a398bd895` (`nvfp4-core-lm_head`)
**Venv:** `exolotl` (torch 2.11.0+cu130, **native mslk 2026.6.6**, `_axo_stub=False` asserted) · `PYTHONPATH=$PWD/src`

## Verdict up front

**No kernel re-engineering was needed.** The brief's premise — that the `fp4_matmul=True`
tiled `_scaled_mm` + logsumexp loop runs in eager Python and never enters the compiled
graph — is **false on this stack (torch 2.11 + the b80dbd650 graph-break fix)**. Measured
cleanly, the `fp4_matmul` fused CE:

- **compiles into a single inductor graph** (`unique_graphs=1`, `graph_breaks=0`) inside the
  real `torch.compile(model)` path, *including its backward*;
- runs the **full Qwen3-1.7B training step at a net WIN vs bf16**, not a regression.

| shape (1.7B, 5090, compiled, steady-state median) | bf16 | **fp4_matmul ON** | Δ vs bf16 |
|---|---:|---:|---:|
| **1024 tok/step** (b2×s512), median of 3 clean A/B repeats | 155.7 ms | **146.1 ms** | **−6.2% (win)** |
| **2048 tok/step** (b2×s1024), single clean run | 250.8 ms | **230.4 ms** | **−8.1% (win)** |

The previously-reported **+54–82%** ("eager fp4_matmul") and the **58 ms head** are
**artifacts of the benchmark harness** (`agent_space/step_compare_compile_native.py`), not of
the kernel — see §3. Both fp4 configs above fire the native mslk activation-quant →
`torch._scaled_mm` FP4 GEMM path (proven by a `torch._scaled_mm`-call counter on an eager
probe step; bf16 = 0 calls).

---

## 1. Why the bf16-compute CE compiles — and why the fp4_matmul one compiles *too*

The brief asked to study why `_FusedFP4CrossEntropy` (bf16-compute) compiles but
`_FusedFP4ScaledMMCrossEntropy` (fp4_matmul) supposedly doesn't. The answer: **both compile,
for the same two reasons**, and the fp4 path was already built to inherit them.

1. **torch.autograd.Function is NOT dynamo-opaque on torch 2.11.** Dynamo traces
   `Function.apply` via the `autograd_function_apply` HOP and compiles *both* fwd and bwd
   into the graph. (The older `FP4_LMHEAD_*COMPILE_COMPARISON.md` notes that predate this
   assumed Functions stay eager — true on older torch, false here.) Direct proof
   (`agent_space/fp4_ce_fn_compile.py`, full V=151936 fwd+bwd):
   - bf16-compute: eager 79 ms → **compiled 10.6 ms** (7.5× — body fuses)
   - fp4_matmul:  eager 49 ms → **compiled 19.5 ms** (2.5× — body fuses)
   A Function that "stayed eager" would show ~1× here.

2. **The b80dbd650 fix removed the only real break** — the manual `__tensor_flatten__`
   round-trip in `_dequant_vocab_tile` / the `ctx.get("is_swizzled_scales")` read. The fp4
   path's `_FusedFP4ScaledMMCrossEntropy` already uses `store[lo:hi]` slicing and reads
   `tile.is_swizzled_scales` off the attribute, so it inherited the traceability for free.
   The mslk activation-quant is a registered `torch.library.custom_op`
   (`axolotl_nvfp4::quantize_single_level`, with a `register_fake`) and `torch._scaled_mm` is
   a native ATen op — both are opaque-but-traceable boxes, so the tile loop around them
   fuses without breaking.

**Decisive in-model test** (`agent_space/fp4_ce_inmodel_fuse.py`) — the fused CE called from
inside a `torch.compile`d module (mirrors the patched `_make_fused_forward`):

| path | eager | **compiled** | graphs | gb |
|---|---:|---:|---:|---:|
| fp4_matmul   | 152 ms | **18.1 ms** (8.4×) | **1** | **0** |
| bf16-compute | 216 ms | **19.4 ms** (11×)  | **1** | **0** |

Both collapse to **one graph, zero breaks**, and fp4 is slightly *faster* in-model.

---

## 2. Where the time actually goes (component dissection)

`agent_space/fp4_ce_dissect.py` (M=1024, V=151936, H=2048, block 16384, fwd-only):

| component | eager ms | compiled ms |
|---|---:|---:|
| activation quant (once/fwd) | 0.06 | — (already hoisted, ~free) |
| tiled `_scaled_mm` only (10 tiles) | 1.82 | — |
| **tiled `_scaled_mm` + tiled fp32 logsumexp/gather (fp4 fwd)** | **6.82** | **1.62** |
| SINGLE big `_scaled_mm` + 1 CE (non-tiled) | 5.56 | 1.30 |
| bf16 single cuBLAS GEMM + 1 CE | 7.31 | 3.70 |

Reads:
- **Eager, the fp32 logsumexp/gather is the cost** (6.82 − 1.82 = 5.0 ms), *not* the GEMM.
  Under compile that pointwise/reduction bookkeeping **fuses away** → fwd drops 6.82 → 1.62.
- **Compiled, the tiled fp4 fwd (1.62 ms) BEATS bf16 (3.70 ms) by 2.3×** — the real FP4
  tensor-core win shows up once the loop is compiled.
- **Tiling is NOT the gate:** the single big non-tiled `_scaled_mm` (1.30 ms) is only ~0.3 ms
  faster than the 10-tile version (1.62 ms). So "many small `_scaled_mm` launches" costs
  ~20% in fwd, far less than the once-claimed dominant factor. A single big GEMM would
  re-materialize the `[M,V]` logits (594 MiB fp32) — trading the entire memory win for ~0.3 ms.
  Not worth it; the wider-tile default (16384) already captures it.

fwd-vs-bwd split (`agent_space/fp4_ce_fwdbwd_split.py`, compiled):

| path | fwd | bwd | fwd+bwd |
|---|---:|---:|---:|
| bf16-compute | 5.00 | 5.69 | 10.69 |
| **fp4_matmul** | **1.85** | 6.70 | **8.55** |

fp4 fwd is **2.7× faster**; fp4 bwd is slightly slower (it recomputes logits via fp4
`_scaled_mm` *and* dequantizes the tile for the bf16 `dz @ w_tile` grad GEMM — the FP4
weight is packed along H, the wrong axis for the dz-contraction, so the grad GEMM can't reuse
it as FP4). Net **fp4 fwd+bwd 8.55 ms beats bf16-compute 10.69 ms by 20%**.

---

## 3. Root cause of the phantom +54% (the harness bug)

`agent_space/step_compare_compile_native.py` (the script behind every prior +54–82% number)
does, *per config in one process*:
1. compiles a **head-portion step fn** that calls `loss.backward()` / `.requires_grad_()`
   **inside the compiled region** → Dynamo cannot trace `Tensor.backward()` →
   **10 graph breaks / 9 unique subgraphs** (visible in `_vb16384.json` `fp4_on.head_graph_breaks`);
2. **then** compiles the full model in the *same* process / dynamo state.

The broken head-portion compile pollutes the dynamo cache, so the subsequent **full-step
model fragments into `unique_graphs=8`** for `fp4_on` (vs `unique_graphs=1` for bf16 in the
same file). A model split into 8 graphs pays repeated graph-entry/exit + loses cross-graph
fusion → the +54–59% full-step inflation. The "58 ms head" is the eager-fallback head-portion
itself (matches the eager dissect, 6.8 ms fwd + eager bwd).

**Fix:** measure each config in its own fresh process and **do not pre-compile a
backward-inside-graph head fn first**. `agent_space/fp4_fullstep_clean.py` (full-step only,
one config/process) yields `unique_graphs=1, gb=0` and the true numbers below. This is a
test-harness defect, not product code — `nvfp4_fused_ce.py` was already correct.

---

## 4. Clean re-measurement (the deliverable metric)

Interleaved A/B, fresh subprocess per (config, repeat), `agent_space/fp4_fullstep_ab.py`,
Qwen3-1.7B, 1024 tok/step, torch.compile inductor default + `suppress_errors=True` (axolotl
path), discard 10 warmup, median of 25 steady-state steps:

| repeat | bf16 (ms) | fp4_matmul (ms) | graphs/gb |
|---|---:|---:|---|
| 0 | 155.7 | 146.1 | 1 / 0 each |
| 1 | 166.6 | 146.9 | 1 / 0 each |
| 2 | 153.2 | 145.6 | 1 / 0 each |
| **median** | **155.7** | **146.1** | — |

**fp4_matmul vs bf16 = −6.2% (net step WIN).** Every fp4 repeat beats every bf16 repeat;
fp4 is tighter (145.6–146.9) while bf16 carries the documented S512 clock-boost noise
(153–167). vs the FP4-bf16-compute path (≈ bf16 ±2%), fp4_matmul is the faster FP4 config.

Against the brief's reference points: bf16 ≈ 151 ms (here 156), FP4-bf16-compute ≈ +1.5%,
**current "eager" fp4_matmul +54% → re-measured −6.2%.** The +54% does not reproduce under a
clean measurement; it was the harness fragmentation in §3.

## 5. Correctness (compiled fp4 == eager fp4)

`agent_space/fp4_ce_correctness.py`, M=1024/V=151936/H=2048:
- eager-vs-eager (same path) floor: `|Δloss|=0`, `grad Δ=0` (the eager fp4 path is deterministic).
- **eager fp4 vs compiled fp4:** `Δloss = 1.55e-3` (loss agrees to 4 sig figs: 12.3393 vs 12.3408),
  `grad_hidden max|Δ| = 1.91e-6`. This is **inductor fp32 reduction-reorder noise** in the
  logsumexp (matches the ACTQUANT_HOIST note's ~2.1e-6 reference), not a math change — the
  compiled path computes the same CE.
- fp4 vs bf16-compute: `Δloss = 7e-4`, grad rel `1.4e-2` — the expected FP4 quant error, unchanged.

Full Qwen3-1.7B fp4 step trains end-to-end with finite loss (`fp4_fullstep_clean.py fp4`).

## 6. Where it matters / honest framing

- The lm_head is only **~5% of a 1.7B bf16 step**, so a faster head can only move the step a
  few %. The clean **−6.2%** (1024 tok) is real but modest *because* the head is small. **The
  win grows with tokens:** at 2048 tok/step it widens to **−8.1%** (bf16 250.8 → fp4 230.4 ms),
  because the head's share of the step rises with sequence length. It scales further with
  **vocab × tokens × batch**: the fp4 head avoids the `[M,V]` fp32 logit tensor (594 MiB at
  V=151936, M=1024) AND does the projection on 4× FP4 tensor cores.
- **Memory is mixed at these shapes:** 1024 tok peak 18.77 (fp4) vs 19.08 GB (bf16) — fp4 wins;
  2048 tok peak 18.88 (fp4) vs 18.08 GB (bf16) — bf16 slightly lower here (the FP4 weight store
  + per-tile dequant/save-for-backward intermediates can exceed the avoided logit tensor at a
  given M/V/H; consistent with the isolated `FP4_ACTQUANT_HOIST.md` finding). The robust win is
  **step time**; treat the memory claim as shape-dependent, not universal.
- The one residual inefficiency is the **fp4 backward** (it can't reuse the H-packed FP4 weight
  for the dz-contraction grad GEMM, so grad_hidden stays bf16). It is already net-positive
  (fwd+bwd 8.55 < 10.69 ms), so fixing it is optional upside, not required for the win.

## 7. Recommended path
1. **Ship as-is.** The `fp4_matmul` fused CE already compiles to one graph and is a full-step
   win on the 5090. No source change to `nvfp4_fused_ce.py` is warranted by the perf data.
2. **Fix the benchmark, not the kernel:** stop pre-compiling a `backward()`-in-graph head fn
   before the full-step measurement, and measure one config per process. The corrected
   harness (`fp4_fullstep_clean.py` / `fp4_fullstep_ab.py`) supersedes the +54% numbers in
   `FP4_STEP_VOCABBLOCK_CONFIRM.md` / `FP4_MATMUL_NATIVE_MSLK.md`, whose full-step figures are
   contaminated by §3.
3. Optional future upside (not needed for parity/win): a single fused FP4-CE Triton kernel
   doing `dot_scaled` GEMM + online logsumexp would shave the ~1.6 ms fwd further and could
   give the backward an FP4 grad GEMM, but the payoff is small at 1.7B (head ≈ 5% of step) and
   only grows at very large V/seq.

## Artifacts (all in `agent_space/`, no source changes)
- `fp4_ce_dissect.py` — component timing (quant / tiled mm / CE / big-mm / bf16), eager+compiled
- `fp4_ce_fn_compile.py`, `fp4_ce_fwdbwd_split.py` — autograd.Function eager→compiled, fwd/bwd split
- `fp4_ce_inmodel_fuse.py` — decisive in-`torch.compile`-model fusion test (graphs=1, gb=0)
- `fp4_fullstep_clean.py` — clean single-config full Qwen3 step (the un-contaminated harness)
- `fp4_fullstep_ab.py` + `fp4_fullstep_ab.json` — interleaved A/B, the −6.2% result
- `fp4_ce_correctness.py` — eager-vs-compiled fp4 loss/grad equivalence
