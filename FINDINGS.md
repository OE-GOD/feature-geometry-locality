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

The genuinely-open remainder is now narrow: other SAE families/architectures;
non-cosine or hierarchical geometry measures; and function proxies for the
sub-dominant (post-shared-axis) structure that come with a validated null (the
shared axis is pervasive, so any such proxy needs the deflation controls from
`nlcp_validate.py` / the built-in linear control).

## Reproduce

```
python3 validate.py                                  # metric sanity on synthetic worlds
python3 run_layer.py --hook blocks.8.hook_resid_pre  # direct proxy (naive)
python3 null_control.py --hook blocks.8.hook_resid_pre  # spectrum null -> negative delta
python3 decisive_test.py                             # linear deflated signal vs spectrum null
python3 real_effect.py --k 200 --contexts 24         # nonlinear on-distribution ablation effects
python3 nlcp_validate.py                             # synthetic validation of the metric
python3 nlcp_compare.py                              # nonlinear vs null, w/ linear-proxy control
```
