"""
Is the decoder circle FORCED by decoder<->residual alignment?

Mechanism claim: decoder recovers the circle only because each decoder dir
points ~54% toward its concept's model residual R_i (which already sits on the
model's circle). If so, ANY 'fake decoder' = R_i blended with noise to match the
observed per-item cos(dec_i,R_i) should recover the circle just as well -- the
recovery is a foregone consequence of pointing at the right concepts, not an SAE
discovery.

For each (family, layer): take real R (model residuals), build fake decoders at
a grid of alignment levels c = mean cos to R_i, and report corr-to-circle. Show
the REAL decoder's (c, corr) lands on the same curve as random-noise fakes.
"""
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from attack_foregone import (FAMILIES, concept_residual, circular_target,
                             cos_mat, corr_to)


def rownorm(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def fake_decoder(R, c, rng):
    """Unit vectors each at cos=c to the corresponding R_i, else random."""
    n, d = R.shape
    Rn = rownorm(R)
    out = np.empty_like(R)
    for i in range(n):
        g = rng.standard_normal(d)
        g = g - (g @ Rn[i]) * Rn[i]            # component orthogonal to R_i
        g = g / np.linalg.norm(g)
        out[i] = c * Rn[i] + np.sqrt(max(1 - c * c, 0)) * g
    return out


def main():
    tok, model = load_gpt2()
    rng = np.random.default_rng(0)
    print("fam    L  | real:cos(dec,R) real:corr | fake-decoder corr at matched cos "
          "(mean+/-sd, 20 draws) | fake at cos=1.0")
    for family in ["months", "days"]:
        for layer in [4, 8, 10]:
            words = FAMILIES[family]
            n = len(words)
            hook = f"blocks.{layer}.hook_resid_pre"
            W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
            R = np.stack([concept_residual(layer, w, tok, model) for w in words])
            A = np.maximum((R - b_dec) @ W_enc + b_enc, 0.0)
            alive = log_sp > DEAD_LOG_SPARSITY
            sel = []
            for m in range(n):
                score = A[m] - A[np.arange(n) != m].mean(0)
                score[~alive] = -np.inf
                sel.append(int(np.argmax(score)))
            sel = np.array(sel)
            D = W_dec[sel]
            Rn, Dn = rownorm(R), rownorm(D)
            c_real = float(np.mean(np.sum(Dn * Rn, axis=1)))
            T = circular_target(n)
            real_corr = corr_to(cos_mat(D), T)

            draws = [corr_to(cos_mat(fake_decoder(R, c_real, rng)), T)
                     for _ in range(20)]
            draws = np.array(draws)
            corr_c1 = corr_to(cos_mat(R), T)   # cos=1 fake == model itself
            print(f"{family:<6} {layer:>2} |   {c_real:+.3f}      {real_corr:+.3f} | "
                  f"           {draws.mean():+.3f} +/- {draws.std():.3f}              "
                  f"|   {corr_c1:+.3f}")


if __name__ == "__main__":
    main()
