# Base residual: L2QER low-rank correction on ALL frozen FP4 base linears — results

Generalization of the Phase-2 lm_head residual (`RESIDUAL_PHASE2_RESULTS.md`) to
every frozen FP4 base linear under LoRA/QLoRA. PTQ A/B harness
(`agent_space/base_residual_probe.py`, no training) on Qwen3-0.6B: all 196
eligible body linears (q/k/v/o, gate/up/down; lm_head excluded — it has its own
block) swapped to the shipped frozen FP4 base (`NVFP4FastComputeBaseLinear`,
the LoRA compute-mode default on this machine), scored as logit
KL(bf16 ‖ fp4-body) against the unmodified bf16 model on a held-out 512-token
eval slice (512 separate calibration tokens).

Construction is the SHIPPED code path: `attach_base_residual` →
`_build_residual` (the same whitened SVD the lm_head uses), with
`g = sqrt(mean_t x_t²)` per INPUT channel captured by forward pre-hooks on the
still-bf16 model BEFORE the quantize swap (the QLoRA storage mode discards the
bf16 master right after packing, so the error `E = W − dequant(Q(W))` must be
factored at swap time).

## Headline: logit KL vs the bf16 model (Qwen3-0.6B, full FP4 body)

| method        | KL(bf16‖cand) | vs fp4 base | top-1 agree |
|---------------|--------------:|------------:|------------:|
| fp4_baseline  | 0.07391       |      —      | 0.9512      |
| plain_k16     | 0.06978       |   −5.6%     | 0.9492      |
| l2qer_k8      | 0.05321       |  −28.0%     | 0.9551      |
| **l2qer_k16** | **0.05404**   | **−26.9%**  | **0.9570**  |
| l2qer_k32     | 0.05291       |  −28.4%     | 0.9473      |

- **Rank-16 L2QER residuals cut the FP4-body logit KL by ~27%** and raise top-1
  agreement 0.951 → 0.957. The gain saturates by rank 8–16 (rank 32 buys ~nothing,
  consistent with the lm_head finding), so the default rank is 16.
- **Plain SVD is again a dead end** (−5.6% ≈ noise): the e2m1 error is near
  full-rank; only the activation weighting concentrates the output-relevant
  error into a low-rank subspace. This is why an unavailable calibration
  DISABLES the residual (loud warning) instead of degrading to plain SVD.

## Load-time overhead (one-time, at model load)

| phase                                | wall time |
|--------------------------------------|----------:|
| calibration forward (512 tokens, hooks on 196 linears) | 0.40 s |
| 196 × SVD + attach (rank 16, GPU)    | 11.7 s    |
| (reference: the FP4 swap itself)     | 0.46 s    |
| A/B buffer memory (rank 16, whole model) | 0.62 MiB |

The SVD total is per-call-overhead dominated (196 small fp32 GPU SVDs); fine as
a load-time cost, scales linearly with layer count and ~quadratically with
hidden size (a 9B-scale body is minutes, still load-time-acceptable).

## Step-time overhead — FUSED into the LoRA path (the shipped implementation)

The original bolt-on (`out = out + (x@Bᵀ)@Aᵀ` materializing `[tokens, out]`
per layer) measured +15.7% eager / +22.4% compiled on the 0.6B whole-model
fwd+bwd and ~+50% on a lone linear — pure memory bandwidth, not FLOPs. The
shipped implementation now RIDES THE LoRA PATH instead ("path to cheap" #2
below, implemented): the residual has exactly the LoRA shape, so on the fused
LoRA kernels (`lora_qkv/o/mlp_kernel`, the production training path) the
factors are CONCATENATED into the adapter GEMMs —

  * forward: `out.addmm_(X @ cat([A_lora, res_B/s]).T → C1ᵀ, cat([B_loraᵀ,
    res_Aᵀ]) → C2, alpha=s)` — the same two skinny GEMMs LoRA already pays,
    just k columns wider (residual net scale 1: its first factor is
    pre-divided by s);
  * backward: `ext = dY @ C2ᵀ` (whose `[:, :r]` slice is exactly the `dY @ B`
    the LoRA grads consume — adapter grads UNCHANGED), `dX.addmm_(ext, C1,
    alpha=s)` carries the exact residual dgrad `(dY@A)@B`;
  * `(C1, C2)` are version-cached per module (invalidated by the optimizer's
    `_version` bump, the unsloth/DoRA cache pattern), so in steady state the
    fold adds ZERO extra kernel launches; under dynamo the cache is bypassed
    (functional build, no graph breaks — verified cold).

Measured (RTX PRO 6000 Blackwell, torch 2.12, fused-LoRA whole-model harness:
PEFT r=16 on q/k/v/o/gate/up/down + axolotl LoRA kernels + FP4 bases, 1081
tokens; arms interleaved 3x, medians):

| measurement (fwd+bwd, median)                          | base    | +residuals | overhead |
|--------------------------------------------------------|--------:|-----------:|---------:|
| whole 0.6B, fused-LoRA step, eager (wall)              | 226.0 ms| 225.1 ms   | **−0.4%** (noise) |
| whole 0.6B, fused-LoRA step, eager (GPU time)          | 38.9 ms | 39.1 ms    | **+0.4%** |
| whole 0.6B, fused-LoRA step, torch.compile (wall)      | 232.7 ms| 239.0 ms   | **+0.1% / +2.7%** (2 runs) |
| whole 0.6B, fused-LoRA step, torch.compile (GPU time)  | 45.5 ms | 45.9 ms    | **+0.9%** |
| one linear 4096→4096 + rank-16 LoRA, M=4096, eager     | 1.21 ms | 1.17–1.24 ms | **−3% to +2%** |
| one linear 1024→3072 + rank-16 LoRA, M=4096, eager     | 0.90 ms | 0.90–0.92 ms | **−0% to +2%** |
| (old bolt-on, same one-linear shapes, for reference)   |         |            | +12–18%  |

(The lone-linear bench is CPU-launch-bound — wall jitter ±3%; the fused arm
sometimes measures *faster* than no-residual because the version cache also
caches the adapter dtype casts. GPU-side kernel time is the clean signal:
+0.4% whole-model.) Note the whole-model "base" here is the fused-LoRA-kernel
step (~220 ms), not the adapterless module-forward step the original +16/+22%
was measured on.

Paths NOT on the fused kernels keep a bolt-on, but a cheaper one: plain module
forwards (no adapter) apply the correction as a single fused `torch.addmm`
(half the old memory traffic — true in-place is forbidden on a custom-Function
output view), and dropout>0 adapters accumulate `out.addmm_(X @ (res_B/s)ᵀ,
res_Aᵀ, alpha=s)` on the RAW X in place (the residual is part of the effective
frozen weight and must never see dropout). Re-measured module-forward probe
(NO LoRA — not the production training path; `base_residual_qwen3_0.6b_fused.json`):
one-linear 1024→3072 bolt-on +30.8% (was +51%); whole-model module-forward
step +20.5% eager / +21.2% compiled — essentially unchanged from the original
+16/+22% because at 1081 tokens that path is kernel-launch-bound, not
bandwidth-bound (and this box now runs a concurrent benchmark on GPU 0, so
wall numbers carry a few % of CPU-contention noise). The fix for training cost
is riding the LoRA kernels, which production NVFP4 LoRA configs use.

Numerics (fused vs bolt-on, identical inputs + fixed dY): outputs/dX match to
bf16 accumulation noise (rel ≤5.5e-3 on QKV/O; ~3e-2 through the MLP where the
down projection re-quantizes `hidden` to FP4 and amplifies 1-ulp differences);
the residual TERM itself matches the analytic `(x@Bᵀ)@Aᵀ` / `(dY@A)@B` to
<1e-2 of the full norm on every path; lora A/B grads are BITWISE identical
given the same dY; a 6-step SGD loop (optimizer-invalidated cache) tracks the
bolt-on trajectory to <1e-3.

Per the brief, the feature ships DEFAULT ON (explicit product decision), with
the measured cost documented in the schema field descriptions and
`enabled: false` as the pure-throughput opt-out.

(KL re-check on this run: fp4 baseline 0.07391, l2qer_k16 0.04986 = **−32.5%**,
top-1 0.9512 → 0.9492; rank sweep k8 −13.6%, k32 −41.7% — the rank-16 default
still holds the accuracy headline.)

## Correctness / wiring (unit-tested)

- Forward: `y = Q(W)x + (x@Bᵀ)@Aᵀ` in all four frozen base classes
  (`NVFP4FrozenBaseLinear`, `NVFP4ComputeBaseLinear`, and both MSLK-fast
  variants) AND in the fused-LoRA kernel entry points (`nvfp4_base_fprop`,
  `nvfp4_base_fprop_many`, `nvfp4_base_dgrad` — which take
  `apply_residual=False` when a fused-LoRA caller folds the residual into its
  adapter GEMMs instead), so fused-kernel runs stay consistent with module
  forwards. `tests/e2e/test_nvfp4_training.py::TestNVFP4BaseResidual::`
  `test_fused_lora_path_folds_residual` pins the folded path: exact residual
  terms in out/dX, adapter grads bitwise unchanged.
- Backward: the module-forward correction is a plain differentiable bolt-on, so
  autograd supplies the EXACT dgrad `dX += (dY@A)@B` and no wgrad (A/B are
  buffers). Verified: corrected-minus-base grad equals the analytic term to
  <1e-2 of total grad norm (bf16 accumulation rounding).
- torch.compile: 0 graph breaks with residuals attached (the None-check is a
  module attribute dynamo specializes once).
- FSDP: A/B are tiny plain bf16 buffers (replicated, not sharded); existing
  FSDP all-gather/pickling tests stay green. Buffers are non-persistent
  (reconstructed deterministically at load), so checkpoints are unchanged.
- Default-on: a bare `nvfp4_training: {enabled: true}` LoRA config calibrates
  and attaches residuals to every swapped base; `base_residual.enabled: false`
  disables; `base_mode: hp` and full-FT skip with a debug log; unavailable
  calibration disables loudly (never plain-SVD).

## Files
- `agent_space/base_residual_probe.py` — the PTQ A/B harness (KL, load
  overhead, module-forward step-time A/B; `--compile-step` for the compiled
  number).
- `agent_space/base_residual_qwen3_0.6b.json` — raw probe output (bolt-on era).
- `agent_space/base_residual_qwen3_0.6b_fused.json` — raw probe output after
  the LoRA-path fusion (KL table + module-forward bolt-on step numbers).
- shipped: `src/axolotl/kernels/nvfp4_residual.py` (`attach_base_residual`,
  reusing `_build_residual`), `src/axolotl/utils/nvfp4_training.py`
  (`_apply_base_residual[_dgrad]` fused-addmm bolt-ons with
  `apply_residual` opt-outs on `nvfp4_base_fprop[_many]`/`nvfp4_base_dgrad`,
  `BaseResidualPlan`, swap-time attach in `convert_lora_base_to_nvfp4`),
  `src/axolotl/kernels/lora.py` (the LoRA-path fold: `_build_residual_pair`,
  `_lora_residual_pair` (version-cached C1/C2), `_bwd_residual_pair`,
  `_dgrad_grad_B`/`_dx_addmm`, residual-aware `matmul_lora` /
  `_batched_lora_forward` / `_shared_nvfp4_base_fprop`),
  `src/axolotl/loaders/patch_manager.py` (`_nvfp4_prepare_base_residual`
  pre-swap calibration), schema `nvfp4_training.base_residual` (default ON),
  tests in `tests/e2e/test_nvfp4_training.py::TestNVFP4BaseResidual` (incl.
  `test_fused_lora_path_folds_residual`) and
  `tests/e2e/test_nvfp4_integration.py`.
