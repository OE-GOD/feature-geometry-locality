# Findings: is GPT-2-small SAE feature geometry local or global?

Status: **verified negative + a reusable methodological result.** (2026-07-01)

Motivation: Sharkey et al. 2025, *Open Problems in Mechanistic Interpretability*
§2.1.2c — SDL/SAEs decompose networks into single directions and leave feature
geometry unexplained. If only *local* geometric relationships matter, a
bag-of-features reading survives; if *global* all-to-all relationships matter,
single-direction SDL is the wrong format. We tried to measure which, on the
jbloom GPT-2-small residual SAEs.

## What we tested and what happened

Every method below asks the same thing: across feature pairs, does geometric
similarity `Sgeo = cos(decoder_i, decoder_j)` predict functional similarity
`Sfn = cos(function_i, function_j)`?

1. **Direct-logit-attribution proxy** (`run_layer.py`). Function = feature's
   direct effect on output logits. First looked local-favorable (a monotone
   "halo": geometric neighbours are functionally similar).
   **Refuted.** The proxy is *linear* in the decoder direction, so `Sfn` is just
   cosine under a fixed quadratic form `M = W_U^T W_U`. A **spectrum-matched
   random unembedding** (same singular values, random singular vectors)
   reproduces the halo bin-for-bin. Null-subtracted `delta_R2` is NEGATIVE at
   every layer except the embedding (`null_control.py`).

2. **Causal proxy** (`causal_proxy.py`). Function = mean output-logit shift from
   injecting `alpha*decoder_i` into the residual stream and running the rest of
   GPT-2. Intended to be nonlinear + downstream-aware. Beat a **label-permutation
   null** at 12-200 sigma across layers, robust to context and readout.
   **Refuted.** At on-distribution injection scale (1-6% of the ~198 residual
   norm) the model responds *linearly*: `F ~ alpha*J*d`, verified by
   `corr(cos F, cos under M=J^T J) = 1.0000`. So the same spectrum tautology
   applies. Against the correct spectrum-matched null, `delta_R2` drops from
   0.0024 (z=12) to 0.0003 (**z=0.33, n.s.**). The label-permutation null is
   strictly too weak: it is beaten by *any* fixed linear map, so beating it is
   not evidence of network-specific structure.

3. **Deflated causal signal** (`decisive_test.py`). Removing the dominant shared
   logit-shift axis (98.3% of variance) multiplied the coupling 50x vs the
   permutation null (z=483) — the one thing left untested against the right null.
   **Refuted.** Against the spectrum-matched null, the deflated real `R2` is
   *below* random (r=1: real 0.13 vs null 0.21, z=-8.4). Mechanism: once the
   dominant axis is gone the metric is near-isotropic, so `Sfn ~ Sgeo` trivially
   (geometry predicting itself); a random map is even more isotropic and scores
   higher.

## The verified conclusion

Under injection/attribution proxies that are linear in the on-distribution
regime, **GPT-2-small SAE decoder geometry shows no network-specific
local functional structure.** Every apparent "local halo" is a spectrum /
near-isotropy artifact of comparing two quadratic forms, not a fact about how
the network organizes features. The local-vs-global question is *not* answered
"local" by these methods; the apparent locality is measurement, not signal.

## The methodological result (the reusable part)

- **Never headline a geometry->function statistic without a spectrum-matched
  null.** Report `real_R2 - null_R2` where the null is a random map with the
  *same singular spectrum*. Raw `R2`, and the label-permutation null, are both
  beaten by any fixed linear map.
- A **linear function proxy cannot answer this question** — its verdict is
  forced by the proxy's spectrum. You need a genuinely nonlinear,
  on-distribution, downstream-aware proxy, scored against a matched null.
- Computational-match principle made concrete: an over-expressive / mis-nulled
  instrument manufactures structure the network does not have.

## The nonlinear on-distribution frontier — now tested, also negative

The one untested regime was a *genuinely nonlinear, on-distribution* proxy: run
the SAE encoder over a real corpus, find where each feature actually fires,
ablate its real contribution, and measure the true downstream logit shift (not a
tangent-space perturbation). Built in `real_effect.py` + `nlcp_compare.py`.

**Result: negative, robustly.** The real ablation-effect matrix is again ~99.5%
one shared axis. The decisive test uses the LINEAR direct-logit proxy as a
built-in negative control (it is a proven spectrum artifact). Only the
un-deflated comparison passes that control (the deflated levels flag the linear
artifact too, z_LIN=27–76, so they have no discriminating power). At the
trustworthy level, the real nonlinear coupling is *below* the null under two
independent nulls (spectrum-matched and rank-matched), z_REAL ≈ −6 to −7. So SAE
geometry carries no network-specific functional structure even when we measure
what features actually do when they fire.

Added methodological lessons: (a) deflation is treacherous when the unembedding
spectrum is concentrated — it can manufacture above-null coupling for a *known*
artifact; (b) a refuted linear proxy makes a decisive built-in control, sharper
than synthetic ground-truth worlds. See `specs/nonlinear-causal-proxy/`.

**Robust across depth.** Depth sweep (layers 2/4/8/11, deflate-0, both nulls):
the coupling gets monotonically MORE negative with depth (z_REAL: layer 4 ≈ −6,
layer 8 ≈ −6/−7, layer 11 ≈ −24/−37). The only non-negative point is layer 2,
and it is null-dependent (rank-matched +4.3, spectrum-map −3.9, borderline
control) — it fails the "beat both nulls" bar and is most parsimoniously the
fading tail of residual EMBEDDING geometry (layer 0 was the one trivially
positive layer). So SAE feature geometry carries no network-specific functional
structure at any mid/late layer, increasingly so with depth; the only geometric
structure is the trivial embedding geometry at the earliest layers, gone by layer 4.

**Why every proxy is ~99.5% one shared axis (characterized).** That dominant
component equals the mean ablation effect (cos=1.000, 94% of features same-sign);
it promotes GPT-2 glitch tokens (RandomRedditor, rawdownload, byte/control chars)
and demotes common punctuation/whitespace. It is a generic confidence-collapse /
residual-mass-removal direction — ablating ANY feature flattens the output
distribution toward degenerate tokens. It measures how much mass was removed, not
what the feature means, which is why it swamps the feature-specific signal.

**A non-ablation proxy agrees (`cofire.py`).** To sidestep the ablation
confound entirely, define function by CO-FIRING: two features are related if they
activate in the same contexts (SAE encoder over 8000 real Pile positions, no
ablation). Result: decoder geometry predicts co-firing essentially not at all —
coupling R²≈0.003, below the rank-matched null (z≈−16) at every deflation. And
co-firing has its OWN ~99.98% shared axis (a generic activity-level mode), so the
single-dominant-shared-mode phenomenon is pervasive across SAE feature statistics,
not specific to ablation. So whether function is a feature's output EFFECT or WHEN
it fires, decoder geometry does not predict it.

## The reframe: a narrow, decoder-side positive (adversarially bounded)

A first-principles reframe (`concept_geometry.py`) broke the deadlock — then a
three-lens adversarial pass (`attack_*.py`, `concept_geometry_generalize.py`)
narrowed it hard. Both matter.

The reframe: the five negatives all used *global, all-pairs* cosine vs a *scalar*
similarity, which washes structure out. Instead, inject external information (so
it can't be a spectrum tautology): for a family the model represents with strong
low-dimensional geometry — cyclic concepts (months/days, Engels 2024) — select
each concept's SAE feature by encoder activation (independent of geometry) and ask
whether the decoder directions recover the known order. Raw result: months recover
the calendar circle at corr +0.78–0.83, z~+6, perm_p=0.000 across layers 4/6/8/10.

**But adversarial testing bounds it on three orthogonal axes:**
1. **Inheritance, not discovery.** The MODEL residual already recovers the circle
   everywhere; the decoder only *matches* it (mean delta +0.04, paired t=1.39
   n.s., sign-flips negative for days). cos(decoder, residual)=0.54,
   decoder-vs-model matrix agreement 0.91 — the decoder geometry *is* the model
   geometry. It reflects, it doesn't add. (One non-trivial bit: the *encoder*
   recovers nothing, so the reflection is reconstruction/decoder-side, not a
   symmetric enc=dec=model triviality.)
2. **Conditional, not general.** Holds only where the model already has strong
   low-D structure (months, days, and — new — the linear family "numbers"). For
   arbitrary categories the SAE decoder geometry is *decoupled* from the model's
   (fruits r≈0 at all layers; animals fails 3/4). And the target test alone
   false-positives: for planets the model has no distance geometry, yet the SAE
   "recovers" a hand-picked linear order — so a MODEL positive-control gate is
   mandatory.
3. **Modest and fragile.** ~0.52 of the +0.83 is a free geometric-shape ceiling
   (best circular ordering of spectrum-matched random directions); honest excess
   over shape is z~+3–4 (months), z~+1.7 borderline (days). Carried entirely by
   the single top-detector feature per concept — the 2nd-most-selective gives ~0.

**Honest resolution of Sharkey 2.1.2c: NOT "local geometry is function-reflecting."**
It's: *SAE decoder geometry reflects function exactly to the extent the model's
residual stream already does* — a scoped, quantified, decoder-side corollary of
the linear-representation hypothesis, positive locally where the model has strong
structure, negative globally, decoupled where the model has no structure. The
load-bearing object throughout is the MODEL's geometry; the SAE neither reveals
function the model lacks nor globally exposes function the model has.

## The strongest result: interactions have a real-but-minor local component

First-principles reframe (Five Whys): every proxy above measured a feature's
ISOLATED effect, but the network uses features in COMBINATION — the nonlinearity
IS the interaction, and Sharkey's fork is about relational structure. So measure
the relation directly (`feature_interaction.py`): the MLP's second-order coupling
of feature pairs on real residuals, and ask whether geometrically-close features
interact more (local) or not (global).

**This is the ONLY positive that clears the spectrum-matched-random control** that
refuted every flat-geometry result. Adversarially verified (3 skeptics + judge):
(a) real & network-specific — close features interact more; random directions
through the same MLP give ~0 in all 12 layer×seed configs and every control. (b)
not a duplicate/self-curvature artifact (near-dup pairs ~0; excluding all cos>0.3
barely moves it; signal lives in distinct cos~0.1–0.5 features). (c) robust to
control choice (double-centering, Mantel, partial correlation agree). BUT narrowed:
honest magnitude is ~+0.15–0.17 (rank-robust Spearman), not the Pearson +0.30
(tail-inflated ~2×); and it is operationally MINOR — geometry adds only ~2%
incremental R² over a global per-feature "interaction-propensity" scalar, which
out-predicts geometric neighbors ~2:1 at recovering a feature's true interaction
partners.

**Final answer to Sharkey 2.1.2c:** SAE/feature geometry reflects the network's
functional structure only weakly and **mostly globally, with a small, real,
network-specific local component that appears in interactions but not flat
geometry.** The dominant organizing signal is a global per-feature propensity;
geometry is not a reliable route to which features a given feature interacts with.
Real, controlled, quantified — but global-sufficient, local-insufficient, not a
foundation for bag-of-features *local* interpretability.

## The scale test: does the negative hold at 17× the parameters? (critical-mass probe)

Everything above is GPT-2-small (117M). The sharpest doubt about it: maybe rich
functional geometry only *emerges* past a scale threshold (a "critical mass") and
117M is below it. Tested by porting the battery to **Gemma-2-2b (2B)** with
**Gemma Scope** SAEs (JumpReLU) — a ~17× jump. Matched K=400, mid-depth layer,
spectrum/rank-matched nulls throughout.

Two rungs first (GPT-2 vs Gemma), then a clean same-family ladder to remove the
family confound.

**Rung comparison (GPT-2 vs Gemma, matched K=400):**

| metric | GPT-2 (117M) | Gemma-2 (2B) | reading |
|---|---|---|---|
| decoder isotropy (top-1 SVD) | 0.025 | 0.008 | geometry *thinner*, not richer |
| co-firing shared axis (binary, mag-free) | 69% | 7% | recipe-dependent — see ladder |
| geometry→co-firing coupling z | −16 | −7 (−31 deflated) | **below null** at both |

**Same-family ladder (Pythia, one architecture, one top-k SAE recipe, scale is the
only variable — the confound-killer):**

| Pythia rung | binary shared axis | geometry→co-firing z (deflate 0) |
|---|---|---|
| 70m | 0.047 | −55 |
| 160m | 0.037 | −37 |
| 410m | 0.078 | −40 |

Two conclusions:

1. **"Geometry does not predict genuine function" is UNIVERSAL, not a small-model
   artifact.** Decoder geometry predicts co-firing *worse* than a rank-matched null
   at every point tested — GPT-2 (117M), Gemma-2 (2B), and the whole Pythia ladder
   (70m/160m/410m) — at every deflation level. Scale-invariant *and* family-invariant.
   The decoder got *more* isotropic with scale, not more structured. The
   critical-mass hypothesis (rich functional geometry emerging past a threshold) is
   **decisively unsupported**.

2. **The "~99% feature-agnostic shared mode" is an SAE-recipe artifact, not a
   model/scale property.** Its size is set by the SAE's sparsity mechanism, not by
   the network: ReLU (GPT-2 jbloom) 69%, JumpReLU (Gemma Scope) 7%, top-k (Pythia)
   ~5%. In the controlled Pythia ladder it is **flat across a 6× scale range**
   (0.047 → 0.037 → 0.078), so it is not a scale phenomenon at all. This corrects
   *two* earlier readings: (a) my first Gemma read, "the shared mode shrinks at
   scale," was confounded by recipe — the ladder refutes it; (b) post 3's mechanism,
   "the map is blank *because* the response is ~99% feature-agnostic," was reporting
   an artifact of the jbloom ReLU SAE, and is falsified independently by Gemma, where
   the co-firing response is largely feature-*specific* (7% shared) yet geometry
   *still* fails to organize it. Feature-agnosticism is neither universal nor the
   cause of the blank map.

**Net:** the family confound is resolved by the ladder. The only fact that survives
across recipe, scale, and family is the negative — decoder geometry is not a map of
function. Everything I previously offered as a *mechanism* for the blankness
(feature-agnostic response, shrink-at-scale) was measuring the SAE instrument, not
the network. Honest edge: the ladder uses top-k SAEs (a third recipe) and MLP-site
SAEs at 410m (residual only available at 70m/160m); the negative holds identically
across all of them.

## Reproduce

```
python3 gemma_scale.py                               # scale: isotropy + direct-logit shared mode + spectrum null
python3 robustness_scale.py                          # scale: layer sweep + shared-mode deflation
python3 gemma_cofire.py both                         # scale: activation co-firing shared axis + geometry z
python3 pythia_ladder.py                             # scale: clean same-family ladder (Pythia 70m/160m/410m)
python3 validate.py                                  # metric sanity on synthetic worlds
python3 run_layer.py --hook blocks.8.hook_resid_pre  # direct proxy (naive)
python3 null_control.py --hook blocks.8.hook_resid_pre  # spectrum null -> negative delta
python3 decisive_test.py                             # linear deflated signal vs spectrum null
python3 real_effect.py --k 200 --contexts 24         # nonlinear on-distribution ablation effects
python3 nlcp_validate.py                             # synthetic validation of the metric
python3 nlcp_compare.py                              # nonlinear vs null, w/ linear-proxy control
```
