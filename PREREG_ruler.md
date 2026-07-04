# Pre-registration: the model's-own-ruler test

**Written and frozen BEFORE any measurement.** Date: 2026-07-04.

## Hypothesis

Raw decoder-cosine failed to predict function everywhere (GPT-2, Gemma, Pythia; five
proxy families). One escape hatch remains: cosine is the wrong metric — the model
reads directions through its downstream weights, so distances should be measured in
the model's warped coordinates, not flat ones.

H1: SAE feature geometry, measured by the model's OWN response to each direction
("the model's ruler"), predicts genuine function (co-firing) — beyond a rotation
null AND beyond raw cosine.

H0: even in the model's own coordinates, feature arrangement is functionally inert;
ruler-similarity does no better than its rotation null once raw cosine is controlled.

## Definitions (exact)

- Model: GPT-2-small, layer L=8 (continuity with all prior results). fp32 forward,
  float64 statistics.
- Features: K=400 alive features (log_sparsity > -5), seed 0, jbloom
  blocks.8.hook_resid_pre SAE.
- Ruler response: for context c (seq 48, last position p), base hidden states from
  one clean forward. r_i(c) = final pre-unembed residual at position p when
  h8[p] := h8[p] + 6.0 * W_dec[i], propagated through blocks 8..11 + ln_f with
  earlier positions' per-block K/V taken from the clean forward (valid by causality),
  minus the clean final residual. alpha = 6.0 (~3% of resid norm; matches
  feature_interaction.py / shared_mode.py).
- Shared-mode removal (instrument construction, applied IDENTICALLY to real and
  null): for each context, subtract the mean of r_i(c) over the K features.
- Ruler similarity: S_ruler(i,j) = cosine between concatenated deflated responses
  [r_i(c1); ...; r_i(cC)], C = 32 contexts from pile-10k docs 0..(first 32 usable).
- Function: co-firing. Activation matrix A (positions x K) via SAE encoder on
  pile-10k positions drawn from docs 500+ (DISJOINT from ruler contexts), 8000
  positions, seq 64. S_fn(i,j) = cosine between activation-pattern rows (as in
  cofire.py). Features that fire < 10 times are dropped from all statistics
  (replacement rule: report the surviving K; no resampling).
- Raw geometry: S_geo(i,j) = cosine of decoder rows.
- Statistic: Spearman over off-diagonal pairs, rho(S_ruler, S_fn).
  Incremental: partial Spearman rho(S_ruler, S_fn | S_geo) (rank-residualize both
  S_ruler and S_fn on S_geo, then Spearman of residuals).
- Null: N=8 random orthogonal rotations Q (Haar, seeds 1..8). For each, recompute
  responses with directions Q @ W_dec[i] (same contexts, same alpha, same deflation,
  same statistic). Rotation preserves all pairwise raw cosines and norms exactly;
  it destroys only feature-to-weight alignment.
  z = (real - mean(null)) / sd(null); z_partial likewise on the partial statistic.

## Gates (instrument checks; failure => INDETERMINATE-instrument, not refutation)

- G0 harness exactness: alpha=0 injection reproduces the clean final residual,
  max abs diff < 1e-3 (fp32 accumulation over 4 blocks).
- G1 split-half stability: S_ruler from contexts 1..16 vs 17..32, Spearman > 0.3
  over pairs.
- G2 sanity: Spearman(S_ruler, S_geo) > +0.2 (a smooth map preserves local
  geometry; if the ruler cannot even see raw angle, it is noise).

## Decision rule (frozen)

- CONFIRM H1: z >= +4 AND z_partial >= +4.
- REFUTE H1: gates pass AND (z <= +2 OR z_partial <= +2).
- INDETERMINATE otherwise (2 < z < 4 etc.), or any gate failure (label:
  instrument, publish as such).

## Prediction table (frozen before measurement)

| quantity | prediction | bin |
|---|---|---|
| G0 max diff | < 1e-4 | pass |
| G1 split-half | 0.5..0.85 | pass |
| G2 ruler-vs-cosine | +0.4..+0.9 | pass |
| top-1 shared energy of raw responses | > 0.95 | descriptive |
| z (ruler vs rotation null) | +1..+5 | straddles REFUTE/INDET |
| z_partial (beyond cosine) | -1..+3 | REFUTE-lean |
| overall | REFUTE 55% / INDETERMINATE 25% / CONFIRM 20% | — |

Most-likely-to-break row: **G1** — after shared-mode removal only ~1-2% of response
variance remains; split-half may collapse toward 0 (=> INDETERMINATE-instrument).

## Honest notes

- Why the rotation null is the right Occam blade: anything attributable to "similar
  directions are treated similarly by ANY smooth map" survives rotation (cosines are
  rotation-invariant); only genuine feature-weight alignment dies. H1 is exactly the
  claim that alignment carries function.
- Why co-firing: the only genuinely non-tautological function measure in the
  project (encoder+data side, disjoint text, no downstream object shared with the
  ruler).
- The Jacobian/metric family of rulers is deliberately EXCLUDED — proven tautology
  (causal_proxy autopsy, FINDINGS.md).
- Exploratory (non-confirmatory, reported separately if run): replicate at L=4;
  binary co-firing variant.

## AMENDMENT 1 (registered 2026-07-04, after run 1, before run 2)

Run 1 (full, C=32, N=8) verdict: INDETERMINATE (instrument) — G1 split-half
+0.292 vs frozen >0.3 (the pre-flagged most-likely-to-break row). Substantive
inputs landed in the indeterminate band regardless: z=+3.73, z_partial=+2.32,
real rho +0.0110 (> all 8 nulls), partial +0.0034 (tied with best null).
Run 1 is reported as instrument-failed; it carries no confirmatory weight.

Run 2 (confirmatory): identical in every respect except C: 32 -> 128 contexts
(Spearman-Brown: stability 0.29 at C=32 predicts ~0.55-0.62 at C=128) and
N: 8 -> 12 nulls (tighter null sd). Gates, statistics, decision bins unchanged.
Prediction (frozen now): G1 0.45..0.65 PASS; if run-1 signal was noise, z falls
toward 0; if real, z in +3..+7 and z_partial in +1..+4 — REFUTE still most
likely (rho ~0.01 is no map even if nonzero). CONFIRM 10% / REFUTE 60% /
INDETERMINATE 30%.
