# Spec: nonlinear on-distribution causal proxy

## Why

The linear proxies (direct logit attribution; small-α injection) were all
**refuted**: they're linear in the decoder direction, so `Sfn` is cosine under a
fixed quadratic form, and a spectrum-matched random map reproduces the halo
(`null_control.py`, `decisive_test.py`). The one untested regime is a function
proxy that is **genuinely nonlinear** *and* **on-distribution** — measuring what
a feature actually does when it fires, not a tangent perturbation.

Goal: decide whether GPT-2-small SAE decoder geometry predicts real downstream
function **above the spectrum null that killed the linear proxies.**

## The function proxy (on-distribution, nonlinear)

For feature `i`, its function = its **real ablation effect on outputs**:
1. Run a corpus. Encode real residual activations with the SAE encoder to find
   positions where feature `i` actually fires (`a_i > 0`).
2. At those positions, remove the feature's real contribution
   (`a_i · W_dec[i]`) from the residual stream and continue the forward pass.
3. `F_i = mean over firing positions of (clean_logits − ablated_logits)`.

This is nonlinear (passes through all downstream layers) and on-distribution
(only where the feature fires, at its real activation magnitude). Contrast with
the injection proxy, which added `α·W_dec[i]` everywhere at an arbitrary scale.

## The decisive comparison (the null)

The label-permutation null is too weak (any fixed map beats it). The honest
question after the linear refutation: does the **nonlinear on-distribution**
coupling exceed what the linear/spectrum story already produces? Compare, on the
same feature subset and geometry:

- **REAL**: geometry→function coupling of the nonlinear ablation `F`.
- **LINEAR**: same coupling for the direct-logit-attribution `F` (the refuted proxy).
- **SPECTRUM NULL**: random map with the linear proxy's singular spectrum
  (reuse `null_control.py`).

Verdict logic:
- REAL ≈ LINEAR ≈ SPECTRUM → nonlinearity adds nothing; the negative result
  holds even on-distribution. (Most likely outcome given prior evidence.)
- REAL > SPECTRUM at several sigma → genuine nonlinear geometric structure. A
  real positive finding.

## Slices

- **Slice 1 (Codex): real-effect harvester** — `real_effect.py`. Encode real
  activations, find firing positions, ablate real feature contributions, measure
  mean logit-shift. Output `F` (K×vocab) + firing counts. CPU, small scale.
  Correctness is independent of the metric design, so it's safe to build now.
- **Slice 2 (orchestrator design): comparison + null** — REAL vs LINEAR vs
  spectrum null, reusing `null_control.py`. Careful research design.
- **Slice 3 (orchestrator design + Codex impl): synthetic validation** — a
  ground-truth world with known nonlinear network-specific structure that the
  metric must detect, and a spectrum-only world it must report as null. Design
  is subtle (smooth ⇒ locally recoverable); orchestrator owns it.

Discipline: validate on synthetic ground truth before any pod-scale run
(`validate.py` pattern). CPU-first; stage the large run.

## Status

- **Slice 1: DONE + accepted (Codex-built, orchestrator-reviewed).**
  `real_effect.py` measures real ablation effects. Encoder formula, ablation
  (via `a_i·W_dec[i]`, zero at non-firing positions), and effect computation
  verified correct by reading + independent re-run (recon cos 0.864,
  deterministic). Semantic decode confirms per-feature-distinct promoted tokens.
- **LOAD-BEARING FINDING for Slice 2:** the real ablation-effect matrix `F` has
  **99.5% of its variance in a single shared axis** (top-1 SVD) — the same
  common-logit-shift component that fooled the injection proxy (98.3%). So the
  nonlinear F is dominated by a shared component too. Slice 2 MUST deflate the
  top-r shared component, then score the residual coupling against a
  spectrum-matched null.
- **Slice 2 null design (decided):** work from `F`'s own gram. `F = U Σ Vᵀ`;
  real function embedding `Y_real = U Σ` (K×K, same gram as F). Spectrum null:
  `Y_null = Q·diag(Σ)` for random orthogonal Q (K×K) — preserves F's singular
  spectrum, randomizes per-feature directions. For deflation r in {0,1,3}:
  deflate top-r SVD of Y, compute Sfn, binned R² vs Sgeo (reuse
  `null_control.binned_curve`); compare REAL vs NULL vs the linear-proxy
  baseline. This is `decisive_test.py` generalized to an arbitrary (nonlinear) F.

- **Slice 2: BUILT (`nlcp_compare.py`), preliminary result — NO trustworthy
  positive.** First null draft was too weak (function randomized independently of
  geometry = permutation-strength; both REAL and the refuted LINEAR beat it —
  caught + fixed). Corrected to the strong spectrum-matched-random-linear-map
  null. On real k=200 (198 fired features): raw (deflate 0) REAL R²=0.0016 is
  BELOW null 0.036 (z=−7.1) → negative result appears to hold on-distribution.
  The deflate-3 "z=+21 network-specific" is a CONFOUND: the refuted LINEAR scores
  even higher (0.37), because M=wteᵀwte has ~3–4 effective dims so deflation guts
  the null for a rank reason, not geometry. Same isotropy/deflation trap as before.
- **Slice 3: DONE (`nlcp_validate.py`).** Synthetic worlds (POSITIVE,
  SPECTRUM_NULL, HIGH_RANK_NULL) × two nulls (spectrum-map, rank-matched). Both
  nulls PASS all synthetic worlds at all deflations — so my synthetic worlds did
  NOT reproduce the real deflate-3 anomaly. The decisive control turned out to be
  the built-in one: the LINEAR proxy is a proven spectrum artifact, so any
  deflation level that flags it (z_LIN>3) is where the metric false-positives.

- **FINAL VERDICT: negative, robustly.** On real k=200, only deflate-0 passes the
  linear control (z_LIN=−6.2 spectrum / −9.4 rank-matched), and there z_REAL=−7.1
  / −6.1 → REAL below null under BOTH nulls. deflate-1/3 flag the refuted linear
  proxy (z_LIN=27–76) → untrustworthy, discarded. The nonlinear on-distribution
  ablation proxy does NOT beat the null: SAE geometry carries no network-specific
  functional structure detectable here, even measuring real firing effects.
  Methodological upshot: deflation is treacherous with a concentrated unembedding
  spectrum; the linear proxy is a decisive built-in negative control (caught what
  synthetic worlds missed).

- **The 99.5% shared axis, characterized (mechanistic).** It equals the mean
  ablation-effect direction (cos=1.000); 94% of features load same-sign; it
  promotes GPT-2 glitch tokens (RandomRedditor, rawdownload, byte/control chars)
  and demotes common punctuation/whitespace (…, quotes, nbsp). Signature of a
  generic confidence-collapse / residual-mass-removal effect: ablating ANY
  feature flattens the distribution toward degenerate tokens. Feature-agnostic —
  it measures how much mass was removed, not what the feature means. This is the
  same shared component that dominated every prior proxy (98-99.5%).

- **Shared-axis-free proxy: attempted, NOT achievable at the source
  (real_effect.py --ablation normmatch / randctrl).** (1) norm-matched ablation
  (remove feature direction, preserve residual norm): shared axis unchanged
  (0.995) so it is not a magnitude effect. (2) magnitude-matched random-direction
  control (subtract a random-dir ablation of equal per-position magnitude): shared
  axis 0.989 -- its character changes (confidence-collapse gone) but a new
  dominant mode replaces it. (3) Decoder rows are near-isotropic (top-1 SVD energy
  0.023, ||mean row||/mean||row|| = 0.066), so the shared OUTPUT axis is NOT from
  decoder geometry -- it is the MODEL's low-rank response to perturbing its real
  feature directions. Upshot: feature-ablation effects are intrinsically ~99%
  low-rank; a shared-axis-free ablation proxy can't be built this way. The
  geometry-to-function confound is fundamental -- explains why every proxy hit it.
  Negative verdict stands and is now mechanistically explained.
