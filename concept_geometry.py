"""
Does SAE decoder geometry recover a KNOWN structured concept family?

First-principles reframe of the geometry->function question. Flat all-pairs
cosine washes out structure; instead test a family with KNOWN geometry: cyclic
concepts (months, days) that GPT-2 represents as circles (Engels et al. 2024).

For each concept in a cyclic family:
  - elicit its residual (mean over templates ending in the concept word),
  - find the most SELECTIVE SAE feature (fires for this concept, not the others).
Then ask whether the pairwise geometry recovers the cyclic order, via correlation
with the circular target cos(2*pi*(i-j)/n):
  - MODEL positive control: cos(residual_i, residual_j) vs circular target.
  - SAE DECODER: cos(W_dec_i, W_dec_j) vs circular target.
  - NULL: permute the concept<->feature assignment (does the real order beat chance?).

Outcomes: SAE recovers the circle (geometry reflects function for structured
families) / SAE destroys it (a finding about SAEs) / neither has it.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY

FAMILIES = {
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"],
}
TEMPLATES = [
    "The month is {}", "My favorite is {}", "It happened in {}",
    "The event took place in {}", "She was born in {}", "We met in {}",
]


def concept_residual(layer, word, tok, model):
    """Mean layer-L resid_pre at the concept's final token, over templates."""
    block = model.transformer.h[layer]
    vecs = []
    for t in TEMPLATES:
        ids = tok(t.format(word), return_tensors="pt")["input_ids"]
        rec = []

        def pre(_m, args):
            rec.append(args[0].detach())
            return None

        h = block.register_forward_pre_hook(pre)
        with torch.no_grad():
            model(ids)
        h.remove()
        vecs.append(rec[0][0, -1].double().numpy())          # final-token residual
    return np.mean(vecs, axis=0)


def circular_target(n):
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts @ pts.T                                       # cos(2pi(i-j)/n)


def cos_mat(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Xn @ Xn.T


def offdiag_corr(S, T):
    iu = np.triu_indices(S.shape[0], 1)
    return float(np.corrcoef(S[iu], T[iu])[0, 1])


def perm_null(S, T, rng, n=2000):
    iu = np.triu_indices(S.shape[0], 1)
    t = T[iu]
    obs = np.corrcoef(S[iu], t)[0, 1]
    K = S.shape[0]
    ge = 0
    draws = []
    for _ in range(n):
        p = rng.permutation(K)
        Sp = S[np.ix_(p, p)]
        r = np.corrcoef(Sp[iu], t)[0, 1]
        draws.append(r)
        if r >= obs:
            ge += 1
    draws = np.array(draws)
    return obs, (ge + 1) / (n + 1), float(draws.mean()), float(draws.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--family", choices=list(FAMILIES), default="months")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    words = FAMILIES[args.family]
    n = len(words)
    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    tok, model = load_gpt2()
    rng = np.random.default_rng(args.seed)

    R = np.stack([concept_residual(args.layer, w, tok, model) for w in words])  # n x d
    # selective SAE feature per concept
    A = np.maximum((R - b_dec) @ W_enc + b_enc, 0.0)          # n x n_features
    alive = log_sp > DEAD_LOG_SPARSITY
    sel = []
    for m in range(n):
        others = A[np.arange(n) != m].mean(0)
        score = A[m] - others                                # selectivity per feature
        score[~alive] = -np.inf                              # mask the SCORE, not A
        sel.append(int(np.argmax(score)))
    sel = np.array(sel)
    n_distinct = len(set(sel.tolist()))
    selectivity = np.mean([A[m, sel[m]] - A[np.arange(n) != m, sel[m]].mean()
                           for m in range(n)])

    T = circular_target(n)
    Smodel = cos_mat(R)
    Sdec = cos_mat(W_dec[sel])

    print(f"family={args.family} (n={n})  layer={args.layer}  "
          f"distinct selective features={n_distinct}/{n}  "
          f"mean selectivity={selectivity:.2f}")
    for name, S in [("MODEL residual (positive control)", Smodel),
                    ("SAE decoder directions", Sdec)]:
        obs, p, nmean, nsd = perm_null(S, T, rng)
        z = (obs - nmean) / (nsd + 1e-9)
        tag = "CIRCULAR structure recovered" if p < 0.05 else "no circular structure"
        print(f"  {name:<34} corr_to_circle={obs:+.3f}  perm_p={p:.3f}  "
              f"z={z:+.1f}  -> {tag}")


if __name__ == "__main__":
    main()
