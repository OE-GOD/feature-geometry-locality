# DESIGN (pre-prereg, under adversarial review): the manifold-shard signature test

Motivated by Goodfire's Block-Sparse Featurizers paper (arXiv 2606.25234, Fel/Kowal
et al., incl. Sharkey): concepts are 2-4 dim subspaces; an SAE shatters each into
several single-direction shards. Prediction for language SAEs: functionally-linked
feature GROUPS should be COPLANAR (low-rank pairwise-cosine Gram) beyond
magnitude-matched nulls. This is a higher-order pattern in pairwise geometry
(arrangement), invisible to the marginal pair-level tests that produced this
project's five negatives.

## Claim to test

H1: interaction-defined groups of SAE features are coplanar (rank-b concentrated,
b~2-3) beyond (a) random groups and (b) cosine-MAGNITUDE-matched random groups.
I.e., the network's functional coupling selects feature sets whose directions are
ARRANGED like shards of a low-dimensional subspace.

## First-principles constraints derived before design

1. GRAM DETERMINISM: group singular values = eigenvalues of the pairwise-cosine
   Gram. So coplanarity IS a function of pairwise cosines; the claim must be about
   ARRANGEMENT, and the decisive null must match the |cos| magnitude histogram of
   real groups (only arrangement free to differ). Unit-normalize decoder rows so
   norms cannot leak.
2. SHARDS ARE ALTERNATIVES, NOT COMPANIONS: shards of one manifold anti-co-fire at
   position level (one token = one month / one orientation). Co-firing selects
   companions (different concepts). Grouping signal = INTERACTION (shared
   downstream readers), possibly interaction-excess-over-co-firing.
3. POSITIVE CONTROL EXISTS: the months-circle (concept_geometry.py) is a known
   shard group. Gates: (i) months group must register coplanar vs matched nulls
   (metric validity); (ii) month features must be preferentially linked in the
   interaction graph (grouping-signal validity). Fail either -> instrument.
4. SELECTION CIRCULARITY RISK: interaction is computed THROUGH the decoder
   directions. If second-order MLP coupling is intrinsically higher for coplanar
   pairs under ANY map, interaction-groups would be coplanar tautologically.
   Control: rotation null — group rotated directions by THEIR interaction matrix,
   measure coplanarity of those groups identically. Real >> rotated => the
   coplanarity-selection is alignment-specific, not a property of any smooth map.

## Sketch of procedure (to be frozen in PREREG after adversarial review)

- Features: fresh disjoint sample (seed 17), K=200, layer 8, jbloom SAE.
- Interaction matrix I over B=32 base points (existing infrastructure);
  normalize: In(i,j) = I(i,j)/sqrt(deg_i deg_j) or I_res (choose ONE after pilot
  on seed-0/months, before freezing).
- Groups: for each feature i, S_i = {i} + its top-(m-1) partners by In, m=4
  (and m=6 secondary).
- Statistic per group: E_2(S) = (lam1+lam2)/sum(lam) of the m x m cosine Gram
  (unit rows). Primary statistic: mean E_2 over groups.
- Nulls:
  N1 random groups (same m, same pool);
  N2 magnitude-matched random groups (bin-match on (mean|cos|, max|cos|));
  N3 rotation null (rotate decoders, recompute I, regroup, remeasure) — kills the
  selection-circularity channel.
  z against each; CONFIRM requires beating N2 AND N3 (frozen thresholds TBD).
- Gates: months positive control (metric + linkage); split-half stability of
  groups (partner sets stable across base-point halves); determinism.
- Contrast (exploratory): co-firing-defined groups predicted LESS coplanar than
  interaction-defined groups (companions vs shards).

## Known open design questions for the adversarial review

- Is E_2 the right statistic vs stable rank vs effective rank (entropy)?
- m=4 fixed: does the group size interact with the b~2-4 claim? Overlapping
  groups: how much does overlap inflate effective significance?
- Is the magnitude-matched null constructible in practice (enough random groups
  in each (mean,max) bin)? Satisfiability check needed.
- Does interaction normalization choice (In vs I_res) change what "top partners"
  means in a way that smuggles in geometry?
- Months control: month features were selected by ENCODER activation
  (concept_geometry.py), interaction groups by MLP response — is the control
  probative for this grouping signal?
- Frequency/firing-rate confounds in partner selection?
- What would the Goodfire authors say this design misses about their claim?

---

# REDTEAM VERDICT (2026-07-08) + REDESIGN (derived from first principles)

Adversarial fleet (6 lenses, simulation-backed): 32 surviving findings, 6 FATAL.
The design above is DEAD as written. Fatal channels: (1) N3 rotation null is
pre-satisfied by our own z=+33 pairwise result — attacker sim shows z=+28..+63
in shard-free worlds; (2) N2 magnitude-matched null is near-deterministic where
constructible and unconstructible in the high-|cos| tail where shard groups
live (only near-duplicate clones occupy the tail); (3) within-sample partner
search is backwards — shard-mates of a feature are specific features, almost
never inside a random K=200/24576 sample (months: 0/12 in seed-17); (4) E_2
cannot distinguish a shard PLANE from a selection CONE around the seed; (5) the
months gates were circular/vacuous; (6) "arrangement beyond magnitudes" is
nearly vacuous once the signed Gram determines the configuration.

## Redesign, each element derived from the failure it fixes

FP1. What is a shard group, geometrically? Features spanning ONE b-dim subspace.
  Consequence: partners must be sought over the FULL dictionary (24,576), never
  within a sample — the mates are specific features. (fixes fatal 3)

FP2. What distinguishes a plane from a cone? In a cone around the seed, partners
  are individually seed-similar but mutually spread: pp-|cos| ~ product of
  sp-cosines. In a shared plane, partners are ALSO mutually similar: pp-|cos|
  high at the same sp-|cos|. So the discriminating observable is
  PARTNER-PARTNER structure AT MATCHED SEED-PARTNER |cos|. (fixes fatal 4)

FP3. What null cannot be pre-satisfied by the known pairwise cos-coupling?
  Only one that HOLDS the seed-cos channel fixed by construction: bin candidate
  partners by |cos(seed,j)|; WITHIN each bin, contrast top-interaction vs
  bottom-interaction partners; null = permutation of interaction ranks within
  bin. Anything driven by |cos(seed,·)| is identical across contrast arms by
  construction. (replaces N2+N3; fixes fatals 1,2,6)

CONFIRMATORY STATISTIC (to be frozen after pilot):
  Delta = mean pp-|cos| (top-I partners) − mean pp-|cos| (bottom-I partners),
  computed within seed-cos bins, pooled over bins and seeds; z vs
  within-bin permutation null (N=1000 permutations, constructible by design).
  Shards predict Delta > 0 (interaction selects mutually-consistent directions,
  a subspace); cone/propensity predicts Delta ≈ 0.

INSTRUMENT REPAIRS:
  - Clone filter: exclude |cos|>0.9 partners (attack_duplicates channel).
  - Hub/propensity normalization: with S seeds, column-normalize I(seed,j) by
    per-j mean over seeds (kills the "same hub partners for every seed" channel);
    diagnostic: cross-seed partner-list overlap.
  - Stability gate: mean Jaccard of top-20 partner sets across base-point halves
    (the well-defined statistic the fleet demanded).
  - Months control, made probative: use the 12 month features AS SEEDS over the
    full dictionary; MEASURE (not assume) whether months rank in each other's
    partner lists and whether month groups show pp-plane structure. Either
    outcome is informative about the grouping signal; metric-validity is checked
    on the known circle directly.
  - Burned samples: seed-0, seed-7, seed-13 (used), seed-17 (touched by the
    probe per the fleet). Confirmatory seeds drawn fresh (seed >= 19).

PILOT (exploratory, throwaway seeds) decides before freezing: feasibility of
seed-vs-all interaction on CPU, partner-list stability, months linkage, honest
prediction bins for Delta. Implementation delegated to Codex
(Fable 5 orchestrates/reviews). Prereg only after pilot.
