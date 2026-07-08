# Pre-registration: manifold-shard signature (v2, post-redteam, post-pilot)

**Frozen before any confirmatory measurement on the registered sample.** 2026-07-08.
Lineage: DESIGN_shard_test.md (v1 killed by adversarial fleet — 6 fatal; v2
derived from first principles; pilot on burned seeds fixed the statistic).

## Hypothesis

H1 (shard signature): among candidates at MATCHED |cos|-to-seed, the features
that interact most with a seed are more MUTUALLY aligned than random same-bin
draws — interaction selects mutually-consistent directions (a subspace), which
raw pairwise cosine to the seed cannot explain by construction.
H0: Delta_top ≈ 0 once seed-cos is held fixed.

## Definitions (exact; pipeline = shard_pilot.py mechanics)

- Model/SAE/layer/alpha: GPT-2-small, jbloom blocks.8.hook_resid_pre, L8, 6.0.
- Seeds: S=24 fresh random alive features, rng seed 29, excluding: burned
  200-samples (seeds 0/7/13/17), the pilot's 24 rand seeds (rng 23), and the 12
  month features. Candidates: ALL alive features (~24.5k), self excluded.
- Base points: B=16, contexts docs 0.. (pile-10k, seq 48), rng seed 29.
- Interaction: seed-vs-all I(s,j) as in shard_pilot.py; propensity-normalized
  Itilde = I / column-mean over the 24 seeds. Halves accumulated for G1.
- Clone filter: exclude |cos(seed,j)| > 0.9 from all selections.
- Bins: per seed, |cos(seed,j)| deciles by LINSPACE over observed range (same as
  pilot); bins with >= 24 members enter.
- Statistic: per seed per bin, Delta_top(bin) = pp(top-8 by Itilde) - mean over
  50 random 8-subsets of the same bin; pp = mean pairwise |cos| among the 8.
  Pool over bins (weight = bin count) and seeds -> Delta_top.
  Null: 200 within-bin permutations of Itilde; z = (real - mean)/sd of pooled
  permuted Delta_top (top-arm only; the bottom arm is DISCARDED per pilot —
  junk-clump artifact documented in DESIGN_shard_test.md).
- Sign/scale note: statistic uses |cos| (antipodal shards count as aligned).

## Gates (fail => INDETERMINATE-instrument)

- G0 determinism: same-seed rerun reproduces Delta_top to 1e-9.
- G1 split-half: mean Jaccard of top-20 partner sets (by per-half Itilde)
  across base-point halves > 0.4 (pilot: 0.63).
- G2 pipeline gate (explicitly cosine-confounded, NOT evidence for H1): running
  the 12 month features through the identical pipeline ranks their mates at
  median Itilde-rank <= 50 for >= 10/12 months (pilot: <= 14 for 11/12).

## Decision rule (frozen)

- CONFIRM H1: z >= +4 AND Delta_top > 0.
- REFUTE H1: gates pass AND z <= +2.
- INDETERMINATE otherwise.

## Prediction table (frozen; pilot-informed generalization bet)

| quantity | prediction |
|---|---|
| G1 | 0.5..0.75 pass |
| G2 | pass (11-12/12) |
| Delta_top | +0.05..+0.16 |
| z | +8..+40 |
| overall | CONFIRM 75% / INDETERMINATE 15% / REFUTE 10% |

Most-likely-to-break: none flagged for gates; the generalization risk is that
the pilot's 4.6x top-arm effect was specific to its 24 seeds.

## Interpretation bounds (pre-committed)

CONFIRM licenses: "interaction selects mutually-consistent directions beyond
seed-cosine — a subspace/shard-compatible organization." It does NOT license
"concepts are 2-4d manifolds" (no within-block coordinate tested) nor any claim
about typicality across layers/models. REFUTE bounds the BSF prior in language
at this layer/SAE. Bottom-arm phenomena are exploratory only.
