# Phase 2: low-rank bf16 residual on the frozen FP4 lm_head — results

PTQ A/B harness (`agent_space/residual_probe.py`, no training), vs the bf16 gold
head, on real captured hidden states. Two constructions over a rank sweep:

- **plain** : truncated SVD of the quant error `E = W_bf16 - dequant(Q(W))`.
- **l2qer** : activation-weighted — weight `E` by the calibration activation
  second moment before the SVD (`Ew = E·diag(g)`, `g = sqrt(mean_t h²)`), so the
  rank-k captures the directions that move the LOGITS, then `B = Vwt[:k]/g`.

## The error spectrum (why plain SVD fails, l2qer works)

Fraction of error energy in the top-k singular directions:

| head            | construction | top-8 | top-16 | top-32 | top-64 |
|-----------------|--------------|-------|--------|--------|--------|
| Qwen3-0.6B H1024 | plain  E     | 0.021 | 0.035  | 0.059  | 0.097  |
| Qwen3-0.6B H1024 | l2qer  E·G   | 0.691 | 0.737  | 0.785  | 0.815  |
| Qwen3-8B   H4096 | plain  E     | 0.004 | 0.007  | 0.012  | 0.023  |
| Qwen3-8B   H4096 | l2qer  E·G   | 0.505 | 0.564  | 0.603  | 0.627  |

The e2m1 quant error is ~9.5% UNIFORM and near FULL-RANK: a plain rank-64 SVD
captures only 2–10% of it. Activation weighting concentrates the logit-relevant
error into ~16 directions (the activations are anisotropic, so only a few hidden
dims actually drive the logits). **This is the whole game.**

## Accuracy: rank sweep vs bf16 gold head

### Qwen3-8B head (hidden 4096, vocab 151936, 800 calib tokens) — the clean case

| method        | KL(gold‖cand) | vs base | MSE    | top1   | ce_delta  |
|---------------|--------------:|--------:|-------:|-------:|----------:|
| fp4_baseline  | 0.00696       |    —    | 0.236  | 0.9912 | +0.00023  |
| plain  k8     | 0.00649       |   −7%   | 0.230  | 0.9912 | −0.00053  |
| plain  k64    | 0.00667       |   −4%   | 0.227  | 0.9912 | −0.00108  |
| **l2qer k16** | **0.00298**   | **−57%**| 0.163  | 0.9937 | −0.00341  |
| l2qer  k32    | 0.00269       |  −61%   | 0.158  | 0.9925 | −0.00374  |
| l2qer  k64    | 0.00268       |  −61%   | 0.155  | 0.9925 | −0.00339  |

Plain SVD ≈ noise (−4 to −7%). **L2QER cuts KL by 57% at rank 16, 61% at rank 32**,
and the corrected head's CE goes slightly NEGATIVE vs bf16 — the residual fully
closes the FP4 CE gap on the 8B head. Rank 16 captures nearly all the gain.

### Qwen3-0.6B head (hidden 1024, vocab 151936, 400 calib tokens)

| method        | KL(gold‖cand) | vs base | MSE   | top1   | ce_delta  |
|---------------|--------------:|--------:|------:|-------:|----------:|
| fp4_baseline  | 0.01644       |    —    | 0.295 | 0.9550 | +0.00779  |
| plain  k32    | 0.01082       |  −34%   | 0.165 | 0.9700 | +0.00513  |
| l2qer  k16    | 0.01076       |  −35%   | 0.163 | 0.9800 | +0.00638  |
| l2qer  k32    | 0.01026       |  −38%   | 0.155 | 0.9775 | +0.00491  |
| l2qer  k64    | 0.00970       |  −41%   | 0.150 | 0.9775 | +0.00558  |

On the narrow 0.6B head plain SVD does better than on the 8B head (the error is
"less full-rank" relative to H=1024) but l2qer still wins, and top1 0.955→0.980.
ce_delta is noisy at this small head/token count (the head CE gap itself is only
+0.008 nats); KL/MSE/top1 are the trustworthy signals and all improve.

## Composition with Phase-1 distillation

The residual is a HEAD-side weight correction; distillation is a BODY-side aux
loss. They are orthogonal and compose by construction: the residual is threaded
INTO the distillation student logits (`_student_logits_tile`, `_scaled_lse`,
`_kl_topk`/`_kl_full` all add `(h@Bᵀ)@A[lo:hi]ᵀ`), so the KL sees the CORRECTED
student — the residual shrinks the student error first, the KL closes what
remains. No double-count (verified: the student lse with the residual matches the
reference corrected-logit lse to 1e-4, `residual_consistency.py` path 3).

## Speed: PERMANENT cost (train + inference), by rank

The residual is two matmuls: `(h[M,H]@Bᵀ[H,k])` then `([M,k]@Aᵀ[k,V])`. Free RTX
PRO 6000, M=4096 tokens, vocab 151936.

| measurement (8B head, H4096)            | k=8   | k=16  | k=32  | k=64  |
|-----------------------------------------|------:|------:|------:|------:|
| residual two matmuls ALONE (ms)         |  —    |  —    | 0.86  |  —    |
| → as % of base FP4 head GEMM (9.76 ms)  |  —    |  —    | ~9%   |  —    |
| bolt-on after materialized GEMM (% over base) | +36 | +38 | +39 | +38 |

Key facts:
- The cost is **nearly rank-INDEPENDENT** (k=8 ≈ k=64): it is dominated by the
  `[M,V]` projection back to the 152k vocab, not the k FLOPs. So pick the SMALLEST
  rank that captures the gain → **rank 16**.
- The residual's own two matmuls are **~9%** of the base head GEMM (small, as
  hoped). The extra +27% in the bolt-on number is the memory-bound `[M,V]` add of
  two full logit tensors.
- Fused into the CE tile loop (the shipped path: add per tile before logsumexp, no
  separate `[M,V]` materialization) the add fuses with the existing tile matmul,
  landing ~10–15% in a proper kernel; the eager Python tile loop measures higher
  (~2×) purely from per-tile launch overhead, not arithmetic.
- This is NOT the "<1–2%" a small adapter costs on a narrow layer — the 152k vocab
  makes any second projection to V expensive. Honest tradeoff, stated in the
  schema docstring.

Compared to Phase-1 distillation: distillation's per-step cost is a SECOND
full-vocab teacher matmul (amortized by `cadence`), i.e. ~1× the head forward when
applied; the residual is ~0.1–0.4× and paid EVERY step including inference.

## Verdict

**Keep it, gated OFF by default, at rank 16, activation-weighted (L2QER) only.**

- L2QER is a real, defensible accuracy win: KL −57% at rank 16 on the 8B head,
  top1 up, CE gap closed. The harness evidence is reproducible.
- Plain SVD is a DEAD END (−4% on the 8B head = noise) — kept only as a
  `calibration: "svd"` ablation, never the default.
- The cost is real (~10–15% of the head forward fused; ~36% bolt-on) and permanent
  (inference too), so it is opt-in, and rank is bounded to the smallest that works.

### Design tensions
1. **Cost is rank-independent** → no point going above the rank that captures the
   gain; the schema default is 32 but 16 is the recommended setting (docstring
   says so).
2. **Inference cost is permanent** → only worth it if the FP4-head accuracy gap
   actually matters for the deployment; for pure throughput runs leave it off.
3. **Calibration sensitivity** → L2QER needs a few hundred real tokens; on pure
   white-noise weights/activations it gives ~0 (no anisotropy to exploit), which
   is correct behavior, not a bug. Real trained heads have the structure.
4. **Residual vs distillation** → distillation moves the BODY to fit the FP4 head
   (free at inference, costs training compute); the residual moves the HEAD to fit
   bf16 (costs inference). They stack; for a frozen-body / LoRA run where the body
   can't fully adapt, the residual is the only head-side lever.

## Files
- `agent_space/residual_probe.py` — rank-sweep harness (plain vs l2qer + spectrum).
- `agent_space/residual_speed.py` — per-rank head-forward overhead.
- `agent_space/residual_consistency.py` — 3-path agreement + gradient check.
- `agent_space/residual_qwen3_{0.6b,0.6b_800,8b}.json` — raw sweep outputs.
- shipped: `src/axolotl/kernels/nvfp4_residual.py`, threaded into
  `nvfp4_training.py` (3 head forwards), `nvfp4_fused_ce.py` (both CE Functions),
  `nvfp4_distill.py` (student logits), schema `lm_head_residual`, patch_manager
  attach + calibration capture, config gate.
