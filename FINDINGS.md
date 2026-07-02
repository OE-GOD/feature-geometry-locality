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
   0.0024 (z=13) to 0.0003 (**z=0.33, n.s.**). The label-permutation null is
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

## What remains genuinely open (needs a pod / real activations)

The only untested regime is a *genuinely nonlinear, on-distribution* function
proxy — e.g. run the SAE encoder over a real corpus, find where each feature
actually fires, and measure its true downstream causal effect on model outputs
(not a small tangent-space perturbation). If any layer's `delta_R2` beats the
matched null there, that is a real, publishable local-structure result. Current
evidence (signal flat-then-decreasing with injection scale) predicts it will
not, but it has not been tested and is the honest frontier.

## Reproduce

```
python3 validate.py                                  # metric sanity on synthetic worlds
python3 run_layer.py --hook blocks.8.hook_resid_pre  # direct proxy (naive)
python3 null_control.py --hook blocks.8.hook_resid_pre  # spectrum null -> negative delta
python3 causal_proxy.py --layer 8 --k 200            # causal proxy vs permutation null
python3 decisive_test.py                             # deflated signal vs spectrum null
```
