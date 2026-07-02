# A random matrix debunked my SAE-geometry result. Three times.

*How trying to answer an open problem in mechanistic interpretability turned into
a lesson about the null you're not using.*

## The question

Sparse autoencoders (SAEs) decompose a network's activations into a big
dictionary of directions, and we call each direction a "feature." One of the
open problems Sharkey et al. raise in *Open Problems in Mechanistic
Interpretability* (2025, §2.1.2c) is that this decomposition **leaves feature
geometry unexplained**. Treating features as an unordered "bag of directions" is
only fair if their *arrangement* carries no information. But if the position of
one feature relative to others encodes how the network treats them, then a list
of single directions is missing something.

They frame it as a fork:

- If only **local** geometry matters (a feature's role is set by its nearest
  neighbours), the bag-of-features picture survives and interpretability stays
  tractable.
- If **global**, all-to-all geometry matters, single-direction SAEs may be the
  wrong kind of object entirely.

I wanted to measure which. I picked GPT-2-small and the well-known jbloom
residual-stream SAEs (~24k features per layer), and I set up what looked like a
clean test.

## The metric

For every pair of features *i, j*, compute two cosine similarities:

- **Sgeo** = cosine of their decoder directions — how geometrically close they are.
- **Sfn** = cosine of their *function vectors* — how similarly the model uses them.

Then ask: **does high Sgeo predict high Sfn?** And crucially, *where* along the
Sgeo axis does the agreement live — in near pairs (local) or far/antipodal pairs
(global)?

The whole thing hinges on that word "function." My first choice was the standard
one.

## Round 1: the direct-logit proxy, and a beautiful halo

The cheapest proxy for "what a feature does" is its **direct logit effect**: push
the decoder direction through the unembedding and see which tokens it promotes.
Two features that promote the same tokens are doing the same job.

I ran it across all 13 hook points. And there it was: a clean, monotone **halo**.
Geometrically-close features were functionally similar; the mean Sfn rose
smoothly from ~0 for orthogonal pairs to ~0.9 for near-parallel ones. It looked
like good news for the bag-of-features picture — local structure, tractable.

Then I ran an adversarial check that I should have run first.

## The refutation: a spectrum-matched null

Here's the problem. The direct-logit proxy is **linear** in the decoder
direction: `F = W_U · d`. So Sfn is nothing but cosine of *d* under a fixed
quadratic form `M = W_Uᵀ W_U`, while Sgeo is cosine under the identity. (More
precisely M also folds in the fixed final-LayerNorm centering and gain, but those
are held constant in the null below, so nothing hinges on the simplification.)
Comparing two quadratic forms **mechanically** produces a monotone halo — for
*any* map with the unembedding's singular spectrum.

So I built the control: replace the real unembedding with a **spectrum-matched
random** one — same singular values, random singular vectors — and recompute the
halo. If the halo is about GPT-2, the random matrix shouldn't reproduce it.

It reproduced it bin-for-bin. Worse: measured as `delta_R2 = real_R2 − null_R2`,
the real map was *weaker* than random at every layer except layer 0:

| layer | real R² | null R² | delta |
|-------|--------:|--------:|------:|
| 0 (embedding) | 0.315 | 0.039 | **+0.275** |
| 1–10 (mid)    | 0.005–0.014 | 0.029–0.049 | **−0.02 … −0.04** |
| 11 (output)   | 0.268 | 0.396 | **−0.129** |

The only positive-delta layer is layer 0 — which is trivially the token
embeddings aligning with the unembedding. Everywhere else, my "local structure"
was an artifact of the proxy's spectrum. The halo said nothing about the network.

## Round 2: the causal proxy, and z = 200

The obvious fix: use a proxy that isn't a linear function of the decoder
direction. So I built a **causal** one. Inject `α · dᵢ` into the residual stream
at the feature's own layer, run the *rest* of GPT-2, and record the mean shift in
output logits over a set of real contexts. This passes through downstream
attention and MLPs — nonlinear, downstream-aware. Surely the spectrum argument
dies here.

It looked spectacular. Against a **label-permutation null** (shuffle which
causal-effect vector belongs to which decoder direction), the signal was positive
at every real layer, from z ≈ 12 at layer 8 up to **z ≈ 202** at the output. It
was robust across injection scale (α = 2, 6, 12), across different context sets,
across last-position vs mean readout. And there was a lovely internal sanity
check: at layer 0, injecting into the embedding does the same generic thing to
every feature (baseline Sfn = 0.96), and the metric correctly reported *no*
feature-specific structure there.

I was ready to believe it. So I sent it to the skeptics.

## The refutation: the null was too weak

Two things came back, and they were devastating in the precise way good reviews
are.

First: **at realistic injection scale, the causal proxy is essentially linear.**
A typical token's residual stream at layer 8 has norm ~105 (the *mean*, ~198, is
inflated by a few outlier positions), and the decoder directions are unit-norm,
so α = 2–12 is only a ~2–11% perturbation. In that regime the model responds
almost linearly: `F ≈ α · J · d` for the local Jacobian J. Measuring it directly,
the cosine structure of the *actual forward-pass* effects correlates **0.96** with
the linear-J prediction at α = 2 and **0.89** at α = 6 (it only drops to 0.71 once
you push to α = 12). So at realistic scale it's the same kind of object as the
direct-logit proxy — a linear map, just with J instead of the unembedding.

Second, and this is the real lesson: **the label-permutation null is strictly too
weak.** *Any* fixed linear map preserves the geometry→function structure that a
label shuffle destroys. So beating the permutation null proves nothing about the
network — a spectrum-matched random linear map beats it too.

When I scored the causal proxy against the *correct* spectrum-matched null (built
from the causal Jacobian), the signal collapsed: `delta_R2` fell from 0.0024
(z ≈ 12) to **0.0003 (z = 0.33)** — indistinguishable from zero. The celebrated
"sign flip from the direct proxy" was an artifact of comparing against a weaker
baseline. Against the right null, both proxies are dead.

## Round 3: the last open door

There was one thing left. A skeptic noticed that ~98% of the variance in the
causal-effect matrix lives in a *single* shared "logit-shift" axis — a common
component that all features push. If you **deflate** that dominant axis and look
at what's underneath, the geometry→function coupling *jumps* 50× (z ≈ 483
against the permutation null). Maybe the real local structure was hiding beneath
the shared axis all along.

This was the only claim nobody had tested against the spectrum null. So I tested
it. Take the deflated signal, and compare it to a spectrum-matched random map put
through the *same* deflation:

| deflate rank | real R² | null R² | z |
|-------------:|--------:|--------:|---:|
| 0 | 0.0027 | 0.0024 | 0.3 |
| 1 | 0.133 | 0.213 | **−8.4** |
| 3 | 0.202 | 0.292 | **−5.6** |

Deflation *does* multiply the signal — but the random map's signal grows even
more. Real is *below* null. And the mechanism is clear once you see it: removing
the one dominant axis leaves a nearly **isotropic** metric, so Sfn ≈ Sgeo
trivially — geometry "predicts" function only because it's predicting itself, and
a random map is even more isotropic. The last door was the same artifact in a new
costume.

## The verified conclusion

Under geometry→function proxies that are linear in the on-distribution regime —
which covers both the standard direct-logit proxy and small-perturbation causal
injection — **GPT-2-small SAE decoder geometry shows no network-specific local
functional structure.** Every "local halo" I found, across three rounds, was a
spectrum/isotropy artifact of comparing two quadratic forms. The local-vs-global
question is *not* answered "local" by these methods; the apparent locality is
measurement, not signal.

I did not solve Sharkey et al.'s open problem. But I think I showed something
useful: a whole family of popular measurement strategies **can't** answer it, and
exactly why.

## The reusable lesson

**Never headline a geometry→function statistic without a spectrum-matched null.**
Report `real − null`, where the null is a random map with the *same singular
spectrum* as your real one. Raw R², and the intuitive label-permutation null, are
both beaten by any fixed linear map — so a positive result against either can be
pure tautology. If your function proxy is linear in the feature direction (and
many are), its verdict is largely *forced* by its spectrum, and you're measuring
your proxy, not your network.

## What's still genuinely open

Everything I tested lives in the linear regime. The honest frontier is a proxy
that is *genuinely* nonlinear, *on-distribution*, and downstream-aware — for
example, run the SAE encoder over a real corpus, find where each feature actually
fires, and measure its true effect on model outputs, then score *that* against a
matched null. Current evidence (the signal is flat-then-decreasing as you push
the injection harder) predicts it won't reveal structure either — but it's
untested, and it's where a real positive result would have to come from.

## Coda

The thing I keep coming back to is that I found the same false positive **three
times**, dressed differently each time, and each time it felt real until a random
matrix with the right eigenvalues did the same thing. The load-bearing part of
this project wasn't any single metric. It was the null — and the discipline of
letting an adversary pick it.
