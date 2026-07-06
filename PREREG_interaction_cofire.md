# Pre-registration: does pair-specific interaction structure predict co-firing?

**Frozen before any confirmatory measurement.** Date: 2026-07-05.

## Context and what is already known (not part of the claim)

- Interaction magnitude I(i,j) = ||mlp8(h0+di+dj) − mlp8(h0+di) − mlp8(h0+dj) + mlp8(h0)||
  was the one geometry-side signal to survive a spectrum-matched control
  (vs geometry: Spearman ≈ +0.16, FINDINGS.md).
- Instrument pilot (interaction_stability.py, seed-0 features, EXPLORATORY):
  split-half stability of I beyond the rank-1 scalar axis = +0.95. The instrument
  is stable — unlike the ruler-test responses (0.26–0.29).
- The interaction→CO-FIRING coupling has never been computed, on any data.
  This prereg is its first and confirmatory test.

## Hypothesis

H1: pair-specific interaction structure (I with its dominant scalar axis removed)
predicts co-firing similarity — beyond rotation nulls AND beyond raw decoder cosine.

H0: interactions carry no functional pair information beyond scalar + spectrum
channels; coupling is within rotation-null range or fully explained by cosine.

## Definitions (exact)

- Model/layer: GPT-2-small, block 8 MLP sublayer (mlp_sublayer of
  feature_interaction.py). fp32 forward, float64 stats.
- Features: K=200 alive (log_sparsity > −5), sampled with seed 7 from
  alive-set MINUS the pilot's seed-0 K=200 sample (disjoint from pilot).
- Base points: B=32 as in feature_interaction.base_points, contexts from
  pile-10k docs 0.., seq 48, seed 7. alpha = 6.0.
- I computed over all B base points (mean over base points of the interaction norm).
- I_res (primary): I minus its largest-|eigenvalue| rank-1 component.
- Function: co-firing S_fn on DISJOINT text (docs 500+, 8000 positions, seq 64),
  cosine of activation rows; features firing <10 dropped from all statistics.
- S_geo: cosine of decoder rows.
- Statistics over off-diagonal pairs:
  rho = Spearman(I_res, S_fn); rho_partial = partial Spearman(I_res, S_fn | S_geo)
  (rank-residualize both on S_geo ranks, Pearson of residuals).
- Nulls: N=8 Haar rotations Q (seeds 1..8) applied to the decoder sample before
  computing I (identical pipeline incl. rank-1 removal). Same S_fn.
  z = (real − mean(null)) / sd(null), likewise z_partial.

## Gates

- G0 determinism: rerunning the real condition with the same seed reproduces
  rho to 1e-9.
- G1 split-half: Spearman between I_res from base points 1..16 vs 17..32 > 0.5.
  (Pilot value on different features: 0.95. Prediction: 0.85–0.97.)

## Decision rule (frozen)

- CONFIRM H1: z ≥ +4 AND z_partial ≥ +4.
- REFUTE H1: gates pass AND (z ≤ +2 OR z_partial ≤ +2).
- INDETERMINATE otherwise; any gate failure → INDETERMINATE (instrument).

## Prediction table (frozen)

| quantity | prediction | bin |
|---|---|---|
| G1 split-half (fresh features) | 0.85–0.97 | pass |
| rho(I_res, S_fn) | +0.00..+0.06 | small |
| z | +0..+4 | straddles REFUTE/INDET |
| z_partial | −1..+3 | REFUTE-lean |
| overall | REFUTE 50% / INDETERMINATE 25% / CONFIRM 25% | — |

Most-likely-to-break row: none flagged for gates (instrument proven stable);
the substantive risk is z_partial landing in the 2–4 dead zone again.
Honest note: CONFIRM here would be the first positive functional-structure
result of the whole project; the prior against, from five negative proxies,
is heavy — hence REFUTE-lean predictions despite the healthy instrument.

## Replacement rules

- Features firing <10 on the co-firing corpus: dropped, surviving K reported.
- If any null run crashes: rerun same seed once; if it crashes again, replace
  with next seed (9, 10, ...) and report.

## RESULT (recorded 2026-07-05; confirmatory run, frozen rule)

G0 PASS (0.0e+00). G1 PASS (+0.948; predicted 0.85-0.97 — held). K=200/200 fired,
19,900 pairs. rho = +0.1130, rho_partial = +0.1076. Nulls (8 rotations):
+0.0121 ± 0.0030 / +0.0106 ± 0.0030. z = +33.5, z_partial = +32.4.

VERDICT (frozen rule): **CONFIRM H1** — the first confirm of the project.

Prediction scorecard: G1 held; rho and z came in ABOVE my frozen ranges
(predicted rho +0.00..+0.06, z +0..+4) — wrong in the pessimistic direction.

Registered next step (exploratory control, before any interpretation is
published): the OCCUPANCY channel. Base points are real residuals; rotated
decoys never occur in real residuals, so co-firing pairs could score high
interactions via joint presence in h0 rather than weight-coupling. Control:
recompute I at base points where NONE of the 200 sampled features fire. Signal
survives -> weight-coupling; vanishes -> occupancy echo (different, weaker claim).

## OCCUPANCY CONTROL RESULT (2026-07-05, exploratory as registered above)

At base points where NONE of the 200 features fires (3404/7968 positions
qualified): rho = +0.1114, partial = +0.1061 (vs +0.1130/+0.1076 at ordinary
base points) — unchanged. Nulls at the same quiet points: +0.012 avg.
z@quiet = +26.5, z_partial@quiet = +25.9.

READING: occupancy echo RULED OUT. The interaction->co-firing coupling is a
property of the MLP WEIGHTS' second-order response to the feature directions,
present even at residuals where the features are absent. The CONFIRM's strong
interpretation stands: the network's weights encode which feature pairs are
co-functional.
