# A random matrix debunked my SAE-geometry result. Every time.

*Trying to answer an open problem in mechanistic interpretability turned into a
lesson about the null you're not using — and, five proxies later, a genuinely
robust negative with a mechanism attached.*

## The question

Sparse autoencoders (SAEs) decompose a network's activations into a big
dictionary of directions, and we call each direction a "feature." One of the open
problems Sharkey et al. raise in *Open Problems in Mechanistic Interpretability*
(2025, §2.1.2c) is that this decomposition **leaves feature geometry
unexplained**. Treating features as an unordered "bag of directions" is only fair
if their *arrangement* carries no information. But if the position of one feature
relative to others encodes how the network treats them, a list of single
directions is missing something.

They frame it as a fork: if only **local** geometry matters (a feature's role is
set by its nearest neighbours), the bag-of-features picture survives; if **global**
all-to-all geometry matters, single-direction SAEs may be the wrong object. I
wanted to measure which, on GPT-2-small and the well-known jbloom residual-stream
SAEs (~24k features per layer). It turned into five attempts to define "function,"
and a random matrix that killed each one.

## The metric

For every pair of features *i, j*, compute two cosine similarities: **Sgeo**
(cosine of their decoder directions — how geometrically close) and **Sfn** (cosine
of their *function vectors* — how similarly the model uses them). Then ask: **does
high Sgeo predict high Sfn?** Everything hinges on that word "function."

## Round 1: the direct-logit proxy, and a beautiful halo

The cheapest proxy: a feature's **direct logit effect** — push the decoder
direction through the unembedding, see which tokens it promotes. I ran it across
all 13 hook points, and there it was — a clean, monotone **halo**: Sfn rose
smoothly with Sgeo, from ~0 for orthogonal pairs to ~0.9 for near-parallel ones.
Good news for bag-of-features. Then I ran the check I should have run first.

## The refutation: a spectrum-matched null

The direct-logit proxy is **linear** in the decoder direction: `F = W_U · d`. So
Sfn is just cosine of *d* under a fixed quadratic form `M = W_Uᵀ W_U`, while Sgeo
is cosine under the identity. Comparing two quadratic forms *mechanically*
produces a monotone halo — for **any** map with the unembedding's singular
spectrum. So I replaced the real unembedding with a **spectrum-matched random**
one (same singular values, random singular vectors) and recomputed.

It reproduced the halo bin-for-bin. Measured as `delta_R2 = real − null`, the real
map was *weaker* than random at every layer except layer 0 (which is trivially the
token embeddings): mid-layers sat at delta ≈ −0.02 to −0.04. The "local structure"
was an artifact of the proxy's spectrum.

## Round 2: the causal proxy, and z = 200

Fix: use a proxy that isn't linear in the decoder direction. Inject `α · dᵢ` into
the residual stream at the feature's layer, run the *rest* of GPT-2, record the
mean output-logit shift. Against a **label-permutation null**, the signal was
positive at every layer, up to **z ≈ 202** at the output, robust across injection
scale and context. I was ready to believe it.

Two problems. First, at realistic injection scale (α = 2–12 is a ~2–11%
perturbation of the residual), the model responds **almost linearly** — the cosine
structure of the measured effects correlates 0.96–0.89 with a linear-Jacobian
prediction. Same kind of object as Round 1. Second, and this is the lesson: the
**label-permutation null is strictly too weak** — *any* fixed linear map beats it,
so beating it proves nothing. Against the correct spectrum-matched null, the
signal collapsed from z ≈ 12 to **z = 0.33**. Dead.

## Round 3: the last (linear) door

~98% of the variance in the causal-effect matrix lives in one shared "logit-shift"
axis. **Deflate** it, and the geometry→function coupling jumps 50× against the
permutation null. Maybe the real structure was hiding underneath. But scored
against a spectrum-matched null put through the *same* deflation, the real signal
lands **below** null (z = −8.4 at deflate-1). The mechanism: removing the dominant
axis leaves a near-**isotropic** metric, so Sfn ≈ Sgeo trivially — geometry
"predicts" function only because it's predicting itself, and a random map is even
more isotropic. Same artifact, new costume.

## The reusable lesson

**Never headline a geometry→function statistic without a spectrum-matched null** —
a random map with the *same singular spectrum*. Raw R² and the intuitive
label-permutation null are both beaten by any fixed linear map, so a positive
against either can be pure tautology. If your function proxy is linear in the
feature direction — and many are — its verdict is *forced* by its spectrum. You're
measuring your proxy, not your network.

## Round 4: the honest frontier — genuinely nonlinear, on-distribution

Everything so far was linear-in-the-regime. The honest test is a proxy that's
*genuinely nonlinear and on-distribution*: run the SAE encoder over real text, find
where each feature actually **fires**, ablate its real contribution at those
positions, and measure the true logit shift. What a feature *does when it's there*.

That effect matrix is, again, **99.5% one shared axis**. So the deflation trap
recurs — but this time there's a decisive built-in control: the refuted linear
proxy is a *known* spectrum artifact, so any analysis setting that flags **it** as
"real" is untrustworthy by construction. Only the un-deflated comparison passes
that control, and there the real nonlinear coupling sits **below** the null under
two independent nulls (z ≈ −6 to −7 at layer 8). Across depth it gets
monotonically *more* negative (layer 4 ≈ −6, layer 11 ≈ −24 to −37); the lone
ambiguous point is layer 2, which fails the "beat both nulls" bar and is best read
as the fading tail of embedding geometry. The negative holds — even measuring what
features actually do when they fire.

### What that 99.5% shared axis actually is

It's worth naming, because it's the whole confound. The dominant axis equals the
**mean** ablation effect (cos = 1.000; 94% of features load the same sign). It
promotes GPT-2 glitch tokens — `RandomRedditor`, `rawdownload`, byte and control
characters — and demotes common punctuation and whitespace. That's the signature
of a generic **confidence collapse**: ablate *any* feature, the model gets less
confident and its distribution flattens toward degenerate tokens. It measures how
much residual mass you deleted, not what the feature means.

Can you remove it at the source instead of via treacherous deflation? I tried two
principled controls. **Norm-matched ablation** (remove the feature's direction but
preserve the residual's norm): shared axis unchanged — so it isn't a magnitude
effect. **Magnitude-matched random-direction control** (subtract an equal-size
random-direction ablation): the confidence-collapse signature changes, but a *new*
dominant mode instantly replaces it. And the decoder directions are nearly
**isotropic** (top-1 SVD energy 2.3%), so the shared *output* axis isn't inherited
from geometry — GPT-2 itself funnels perturbations along its real feature
directions into a low-rank response. The confound is **fundamental** to
ablation-based function measurement. That's why every proxy hit it.

## Round 5: a completely different kind of "function" — co-firing

If *effect* is hopelessly confounded, measure function a different way entirely:
**co-firing**. Two features are related if they *activate in the same contexts*.
No ablation, no output perturbation — just the SAE encoder over 8,000 real Pile
positions. Does decoder geometry predict which features fire together?

No. Coupling R² ≈ 0.003 — essentially zero, and below the null (z ≈ −16) at every
deflation. And co-firing has its *own* ~99.98% shared axis (a generic
activity-level mode), so the single-dominant-mode phenomenon is pervasive across
SAE statistics, not specific to ablation. Whether function is a feature's output
**effect** or **when it fires**, decoder geometry does not predict it.

## The verified conclusion

Across five proxies spanning three genuinely independent notions of "function" —
linear direct effect, nonlinear on-distribution effect, and co-firing —
**GPT-2-small SAE decoder geometry shows no network-specific functional
structure.** The local-vs-global question is not answered "local" by any of these;
the only geometric structure is the trivial embedding geometry at the earliest
layers, gone by layer 4. I didn't solve Sharkey et al.'s open problem, but I
mapped out a wide family of measurement strategies that **can't** answer it — and
*why*, down to a mechanism (the model's low-rank response to perturbing its own
features).

Two things I'd hand to the next person. **A method:** a refuted proxy makes a
decisive built-in negative control — sharper than synthetic ground-truth worlds,
which missed the real spectral pathology. **A caution:** SAE feature statistics
are pervasively rank-one-dominated (~99% shared mode, twice, from unrelated
causes), so any sub-dominant-structure claim needs the deflation controls, not raw
cosines.

## Coda

The load-bearing part of this project was never a metric. It was the null — and
the discipline of letting an adversary pick it. I found the same false positive
five times, each dressed differently, and each time it felt real until a random
matrix with the right eigenvalues did exactly the same thing.

The genuinely-open remainder is narrow, and it's where I'm headed next: *cosine*
geometry says nothing, but does *hierarchical* geometry — containment,
parent/child structure — predict function where flat similarity doesn't? That's a
different question, and this whole negative is what motivates asking it.
