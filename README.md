# Is SAE feature geometry local or global?

A small, validated probe for the open problem raised in Sharkey et al. (2025),
*Open Problems in Mechanistic Interpretability*, §2.1.2c ("SDL leaves feature
geometry unexplained"):

> If understanding the **global** geometry (all-to-all relationships) of
> features is essential, this might pose a fundamental problem for current
> approaches. However, if only **local** geometric relationships need to be
> understood, a 'bag of features' approach may be more feasible.

This repo turns that dichotomy into a measurable, training-free quantity and
validates it against ground truth before spending any GPU.

## The metric (`locality.py`)

For N features we have decoder directions `D` (geometry) and function vectors
`F` (how the model uses each feature). For every pair we compute geometric
similarity `Sgeo = cos(D_i,D_j)` and functional similarity `Sfn =
cos(F_i,F_j)`, then ask **where along the `Sgeo` axis functional similarity is
explained**:

- signal concentrated in **near** pairs (`Sgeo` high) -> **LOCAL** (a
  feature's role is set by its neighbours; bag-of-features survives)
- signal concentrated in **far** pairs (`Sgeo` very negative / antipodal) ->
  **GLOBAL** (you need an all-to-all relationship to a distant feature)
- `Sgeo` explains ~none of `Sfn` -> **NULL** (geometry doesn't track this
  function proxy)

Output: `locality in [0,1]` (1 = local, 0 = global), an `R2`, and a verdict.

## Validate first (`validate.py`)

Three synthetic worlds with known answers, checked across seeds:

| world          | construction                                   | expected |
|----------------|------------------------------------------------|----------|
| `world_local`  | blobs; each blob has its own function          | LOCAL    |
| `world_global` | antipodal pairs with opposite function         | GLOBAL   |
| `world_null`   | geometry and function independent              | NULL     |

```
python3 validate.py     # must print PASS before running on real models
```

Currently passes on all three worlds across 3 seeds.

## Run on a real SAE (`run_gpt2_sae.py`) -- pod

Needs `sae_lens` + `transformer_lens`. Function proxy = **direct logit
effect** `F_i = W_dec[i] @ W_U`. Subsamples K features (all-pairs is O(K^2)).

```
python3 run_gpt2_sae.py --release gpt2-small-res-jb \
                        --sae-id blocks.8.hook_resid_pre --k 3000
```

Sweep `--sae-id` across layers: locality may change with depth, which is
itself a result worth plotting.

## RESULT SO FAR (and a refuted first draft)

A 13-layer sweep of GPT-2-small residual SAEs (jbloom release) with the
**direct-logit-attribution** proxy first looked favorable: a thin monotone
halo of functionally-similar geometric neighbours, i.e. "local, bag-of-features
survives." **Adversarial review refuted that reading**, and the refutation is
now baked into the repo as a mandatory control.

### The tautology (see `null_control.py`)

The direct-logit proxy is *linear* in the decoder direction: `F = wte @ (ln_w *
(d - mean))`, so `Sfn` is just cosine of `d` under the fixed quadratic form
`M = wte^T wte`, while `Sgeo` is cosine under the identity. Comparing two
quadratic forms *mechanically* produces a monotone halo for ANY map with `wte`'s
singular spectrum. A **spectrum-matched random unembedding** (same eigenvalues,
random eigenvectors) reproduces the halo bin-for-bin. Measured as
`delta_R2 = real_R2 - null_R2`:

| layer | real R2 | null R2 | delta | reading |
|-------|--------:|--------:|------:|---------|
| 0 (embedding) | 0.315 | 0.039 | **+0.275** | real, but trivially = token embeddings |
| 1-10 (mid)    | 0.005-0.014 | 0.029-0.049 | **-0.02..-0.04** | real map WEAKER than random null |
| 11 resid_post | 0.268 | 0.396 | **-0.129** | weaker than null overall |

Every non-embedding layer has **negative delta** — the direct-logit halo carries
no network-specific structure. **Always subtract the spectrum-matched null.**

### The escape (see `causal_proxy.py`)

To answer the actual local-vs-global question you need a proxy that is (a)
non-linear in `d` (so the null argument dies) and (b) downstream-aware (so it
isn't blind to how mid-layer features act). The causal proxy injects `alpha*d_i`
into the residual stream at the feature's layer, runs the rest of GPT-2, and
records the mean output-logit shift over real contexts. Its null is a
**label permutation** (shuffle which causal-effect vector pairs with which
decoder direction). Early result at layer 8: `delta_R2 > 0`, several sigma above
the permutation null — the opposite sign from the direct-logit proxy. [Full
causal sweep in progress.]

## Caveats (read before trusting a verdict)

- **Never report a raw geometry->function statistic.** Report it minus the
  appropriate null (spectrum-matched for linear proxies, label-permutation for
  causal). The naive version measured the proxy, not the network.
- **Smooth = local, by construction.** Any smooth map of geometry is locally
  recoverable, so a LOCAL verdict from a linear proxy is nearly vacuous; a
  GLOBAL verdict is the informative one.
- **Computational match.** Keep the function proxy no more expressive than the
  mechanism it stands for -- and no *less*. The direct-logit proxy failed both
  ways at once: tautologically coupled to geometry AND blind to downstream
  computation.
```
