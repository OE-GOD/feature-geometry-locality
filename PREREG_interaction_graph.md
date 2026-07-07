# Pre-registration: does the interaction graph's community structure align with function?

**Frozen before any confirmatory measurement on the registered sample.** Date: 2026-07-05.

## Context (known before freezing; pilot values are from OTHER feature samples)

- Confirmed (PREREG_interaction_cofire.md, seed-7 sample): pairwise interaction
  structure predicts co-firing, z=+33.5/+32.4, weight-borne (occupancy control).
- Graph pilot (interaction_graph_pilot.py, seed-0 sample, EXPLORATORY): real
  eigen-spectrum far above rotation-null envelope; split-half community ARI +0.63
  (k=6); ARI(interaction communities, co-firing communities) = +0.040 vs nulls
  +0.001 ± 0.005 (informal z +8.6). Predictions below are pilot-informed; the
  registered bet is that this GENERALIZES to a fresh sample.

## Hypotheses

H1a (replication): pairwise I_res -> co-firing coupling replicates on a third
     disjoint feature sample (z >= 4, z_partial >= 4).
H1b (new, community): spectral communities of the interaction graph align with
     spectral communities of the co-firing graph beyond rotation nulls.

## Definitions (exact; procedures identical to interaction_cofire.py /
## interaction_graph_pilot.py unless stated)

- Features: K=200, seed 13, sampled from alive MINUS seed-0 pilot sample MINUS
  seed-7 confirm sample (three-way disjoint).
- Base points: B=32, contexts docs 0.., seq 48, rng seed 13. alpha=6.0, layer 8.
- I over all B; I_res = largest-|eig| rank-1 removed. Co-firing S_fn on docs 500+
  (8000 positions, seq 64); fires<10 dropped everywhere; S_geo = decoder cosine.
- Pairwise stats (H1a): rho = Spearman(I_res, S_fn); rho_partial | S_geo.
- Community stats (H1b): spectral clustering, k=6, top-6 |eig| eigenvectors,
  row-normalized, k-means (seed 99, 8 restarts, 60 iters) — the pilot's exact
  procedure. ARI between labels(I_res[keep]) and labels(S_fn[keep, diag=0]).
- Nulls: N=8 Haar rotations (seeds 1..8) of the decoder sample; full pipeline
  including rank-1 removal and clustering. z per statistic:
  (real − mean(null)) / sd(null).

## Gates

- G0 determinism: same-seed rerun reproduces rho to 1e-9.
- G1 split-half of I_res (Spearman over pairs) > 0.5.
- G2 split-half community stability: ARI(labels(I_res half A), labels(half B))
  > 0.3 (pilot: 0.63 on a different sample).

## Decision rules (frozen, separate verdicts)

- H1a: CONFIRM z>=4 AND z_partial>=4; REFUTE either <=2 (gates G0,G1 passing);
  else INDETERMINATE.
- H1b: CONFIRM z_ARI >= 4; REFUTE z_ARI <= 2 (gates G0,G1,G2 passing);
  else INDETERMINATE.

## Prediction table (frozen)

| quantity | prediction | note |
|---|---|---|
| G1 | 0.90..0.97 | |
| G2 | 0.4..0.75 | most-likely-to-break row |
| rho (H1a) | +0.08..+0.14 | replication |
| z (H1a) | +15..+45 | CONFIRM expected (85%) |
| ARI real (H1b) | +0.015..+0.08 | absolute value SMALL |
| z_ARI (H1b) | +3..+12 | CONFIRM 55% / INDET 30% / REFUTE 15% |

Honest notes: (1) H1b's absolute effect is small even if confirmed — communities
exist and are stable, but their functional alignment (ARI ~0.04) is faint at k=6;
a confirm licenses "the organization has some community shape," NOT "features
form clean functional modules." (2) k=6 is fixed by the pilot's procedure, not
optimized on the confirmatory sample; no other k will be scored confirmatorily.
(3) G2 is the flagged break-row: community stability on a fresh sample is the
least-tested link.

## Replacement rules

As in PREREG_interaction_cofire.md (fires<10 dropped; crashed null reruns once,
then next seed with report).

## RESULT (recorded 2026-07-06; confirmatory run, frozen rules)

Third disjoint sample (seed 13). Gates: G0 exact; G1 +0.962; G2 +0.626 (the
flagged break-row PASSED — communities are stable on fresh features too).

H1a (replication): rho=+0.0785, partial=+0.0721; nulls +0.009 avg.
  z=+19.99, z_partial=+18.60 -> **CONFIRM**. The pairwise interaction->co-firing
  result now stands on TWO independent samples (z=+33.5 and +20.0).

H1b (communities): interaction-community vs co-firing-community ARI = -0.0117;
  8 rotation nulls +0.0029 ± 0.0103 -> z_ARI = -1.42 -> **REFUTE**.
  Communities are REAL and STABLE (G2 +0.63) but do NOT align with functional
  communities. Pilot's +0.04 alignment (informal z +8.6) did NOT generalize --
  it was table-specific noise; the fresh-sample prereg caught it. Frozen
  prediction (55% confirm) was WRONG; rule reads REFUTE with no wiggle.

INTERPRETATION: the pairwise functional signal (~1.3% variance) is real but too
faint to aggregate into functional modules at the mesoscale. Interaction
communities form around the dominant (non-functional) structure. "Fabric, not
modules": genuine thread-by-thread functional information that does not knot into
functional groups at k=6. This BOUNDS the SOAR I-6 hierarchy question -- naive
community detection on the interaction graph will not recover functional hierarchy.
