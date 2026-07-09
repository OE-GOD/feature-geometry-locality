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

## RESULT (2026-07-09; confirmatory run, frozen rules)

MONTHS (fresh base points): G0/G1 pass (split-half +0.967). T1 = -0.617,
z1 = -5.93 (perm-p 1e-4 floor); T2 = +0.569, z2 = +5.85 (floor). **CONFIRM** —
the pilot replicated; the calendar coordinate organizes co-processing.
DAYS: G0/G1 pass (split-half +0.932) BUT the feature selection assigned the
SAME feature (9727) to Wednesday AND Thursday — the family had 6 distinct nodes,
not 7 (objective instrument defect, visible in the printed mapping; this was the
pre-flagged most-likely-to-break row). T1 = -0.486, z1 = -2.55 (correct
direction, p=.0044, below the frozen |z|>=3 bar); T2 z2 = +0.99 n.s.
**INDETERMINATE.** OVERALL (frozen rule): **PARTIAL**.

## AMENDMENT 1 (registered 2026-07-09, before any rerun)

Defect: family_features argmax-per-word permits duplicate assignments. Fix
(mechanical, outcome-independent): greedy distinct assignment — iteratively pick
the (word, feature) pair with the highest selectivity among unassigned words and
unused features. Months selection is unchanged by this fix (its 12 argmax
features are already distinct); DAYS is rerun ONCE with distinct selection,
same frozen statistics/thresholds/nulls, fresh rng-31 base points as before.
Pre-commitment: ONE rerun only; if days stays below threshold with 7 distinct
features, the verdict stands as the honest boundary (PARTIAL: months-specific
until another family is found). Prediction: days CONFIRM 55% / INDETERMINATE
30% / REFUTE 15%.

## AMENDED DAYS RERUN (2026-07-09; one rerun per Amendment 1 — FINAL)

Distinct selection: Wednesday now 3697 (Thursday keeps 9727); 7 distinct nodes.
Gates pass (split-half +0.975). T1 = -0.183, z1 = -1.59 (n.s.); T2 z2 = -0.18
(n.s.). Frozen rule: **REFUTE** for days. Note: the original run's stronger
T1 (-0.49) was partly an artifact of the duplicate-feature collision.

FINAL OVERALL (frozen rule): **PARTIAL** — months CONFIRM (twice: pilot +
fresh base points), days REFUTE. Interpretation within pre-committed bounds:
within-subspace coordinates CAN be functionally load-bearing (months is an
existence proof: position on the circle organizes co-processing, Spearman
-0.62, circulant z +5.9, replicated), but this is NOT automatic for every
cyclic family — the days features at L8 do not carry it (whether because the
model's days-circle is weaker here or because the SAE's day features are less
cleanly individuated — e.g. the shared "midweek" detector — is not resolved by
this design). No further reruns per pre-commitment. Reproducibility note: the
days rerun overwrote coordinate_confirm.npz with days-only arrays; months
numbers are recorded here and in the run logs, regenerable via
`python3 coordinate_confirm.py --families months`.
