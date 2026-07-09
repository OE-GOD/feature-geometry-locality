# Pre-registration: is the within-subspace coordinate functionally load-bearing?

**Frozen before confirmatory measurement.** 2026-07-08.
Lineage: PREREG_shard.md CONFIRM (interaction-selected groups share subspaces).
Pilot (zero-compute, on shard_confirm.npz months rows — those base points are
hereby BURNED): within-months Spearman(interaction, circular distance) = -0.685
(label-perm z = -5.8); first-harmonic energy 0.305 vs 0.099 (z = +6.7).

## Hypothesis

H1: within a known cyclic concept subspace, pairwise MLP interaction is ordered
by the manifold's internal coordinate — I(i,j) decreases with circular distance,
and the interaction matrix carries circulant (first-harmonic) structure under
the true ordering. I.e., position on the manifold organizes co-processing:
the coordinate is functionally real.
H0: within-family interaction is unrelated to the known ordering (permutation-
level) once the family is fixed.

Honest scope note (pre-committed): within a circle, coordinate and pairwise
cosine are geometrically identified; H1 claims function-follows-manifold-
structure, NOT "beyond cosine". No such decomposition will be attempted.

## Design

- Families (both required, independent):
  MONTHS (12, calendar cycle) — replication on FRESH base points;
  DAYS (7, week cycle) — independent family, never measured with this statistic.
  Feature selection per family: concept_geometry.py mechanics (selectivity
  argmax over alive features), same as shard_pilot.month_features.
- Interaction: compute_seed_all mechanics, alpha=6.0, L8; candidates = the
  family's features only; B=16 FRESH base points (rng seed 31, contexts docs
  0.., pile-10k, seq 48 — different draw than rng 23/29 runs). Halves kept.
- Symmetrize: S = (I + I^T)/2 over the family submatrix (raw I; no propensity
  normalization needed within-family — noted: results with Itilde-style column
  normalization reported as secondary descriptive).
- Statistics per family (off-diagonal pairs):
  T1 = Spearman(S(i,j), circ_dist(i,j));
  T2 = mean first-harmonic energy fraction of mean-removed rows under the true
       cyclic order (as in the pilot).
  Nulls: 10,000 label permutations of the family members (relabel rows+cols
  jointly); z and permutation-p for each statistic. Permutation-p floor
  1/10001 — satisfiable for |z|>=4 equivalents; decision uses z.
- Gates: G0 determinism (same-seed rerun identical to 1e-9); G1 split-half:
  Spearman between S from base-point halves over off-diag pairs > 0.5.

## Decision rule (frozen, per family)

- Family CONFIRM: z(T1) <= -3 AND z(T2) >= +3 (both, correct signs).
- Family REFUTE: gates pass AND |z(T1)| < 2 AND |z(T2)| < 2.
- Else INDETERMINATE.
- Overall: CONFIRM = both families confirm; PARTIAL = months only (replication
  without generalization); REFUTE = both refute.

## Prediction table (frozen)

| quantity | prediction |
|---|---|
| months T1 / z | -0.4..-0.8 / -3..-8 |
| months T2 z | +3..+9 |
| days T1 / z | -0.2..-0.8 / -1.5..-5 (only 21 pairs; power-limited) |
| days T2 z | +1..+6 |
| overall | CONFIRM 55% / PARTIAL 30% / INDET 12% / REFUTE 3% |

Most-likely-to-break: days family power (7 nodes, 21 pairs) and days feature
selection quality (concept_geometry found the days circle weaker, +0.74,
carried by fewer clean features).

## Interpretation bounds (pre-committed)

CONFIRM licenses: "within known cyclic subspaces, the internal coordinate
organizes the model's pairwise co-processing — shard subspaces are manifolds
with functionally real positions, not unstructured clumps." NOT licensed:
typicality across arbitrary (non-cyclic, discovered) subspaces; any behavioral/
steering claim; "beyond cosine" (see scope note).
