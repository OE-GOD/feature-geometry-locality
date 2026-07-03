"""
Are the network's feature INTERACTIONS local or global? (Sharkey 2.1.2c, done right)

Root diagnosis (Five Whys): every prior proxy measured a feature's ISOLATED
marginal effect, but the network uses features in COMBINATION -- the nonlinearity
IS the interaction. And the paper's local-vs-global fork is about relational
structure. So measure the relation directly: the MLP's second-order coupling of
feature pairs, on top of real residuals.

    interaction(i,j) = || mlp(h0 + d_i + d_j) - mlp(h0 + d_i) - mlp(h0 + d_j)
                          + mlp(h0) ||        (averaged over real base points h0)

Nonzero only where the MLP genuinely MIXES features i and j (the actual
computation), and it lives in the gelu nonlinearity, so it is not the linear-read
tautology. Then ask: do strongly-interacting pairs have close decoder geometry
(LOCAL -> bag-of-features feasible) or is interaction geometry-independent
(GLOBAL -> to understand a feature you need all the others)?

Readout: correlation of interaction with cos(d_i,d_j), vs a feature-label
permutation null; plus the binned interaction-vs-geometry curve.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts


def mlp_sublayer(block, x):
    return block.mlp(block.ln_2(x))                          # GPT-2 MLP sublayer


def base_points(layer, contexts, model, n_base, rng):
    block = model.transformer.h[layer]
    rec = []

    def pre(_m, args):
        rec.append(args[0].detach())
        return None

    h = block.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(contexts)
    h.remove()
    H = rec[0].reshape(-1, rec[0].shape[-1])
    pick = rng.choice(H.shape[0], size=min(n_base, H.shape[0]), replace=False)
    return H[pick]                                           # (n_base, d) real residuals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--scale", type=float, default=6.0)     # feature-write magnitude
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--random", action="store_true", help="spectrum-matched null: random directions")
    args = ap.parse_args()

    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    tok, model = load_gpt2()
    Wsel = W_dec[idx].astype(np.float32)
    if args.random:                                          # spectrum-matched-ish null
        g = rng.standard_normal((idx.size, Wsel.shape[1])).astype(np.float32)
        Wsel = g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(
            Wsel, axis=1, keepdims=True)

    contexts = build_pile_contexts(tok, args.docs, args.seq, args.docs * args.seq)
    H0 = base_points(args.layer, contexts, model, args.n_base, rng)  # (B, d) torch
    block = model.transformer.h[args.layer]
    D = torch.from_numpy(Wsel).float() * args.scale    # (K, d) scaled writes
    K = D.shape[0]
    iu = np.triu_indices(K, 1)
    ii, jj = iu

    inter = np.zeros(len(ii), dtype=np.float64)
    with torch.no_grad():
        pair_writes = D[ii] + D[jj]                          # (npairs, d)
        for b in range(H0.shape[0]):
            h0 = H0[b:b + 1]                                 # (1, d)
            f0 = mlp_sublayer(block, h0)                     # (1, d)
            f1 = mlp_sublayer(block, h0 + D)                 # (K, d)
            fp = mlp_sublayer(block, h0 + pair_writes)       # (npairs, d) batched
            resid = fp - f1[ii] - f1[jj] + f0                # second-order interaction
            inter += resid.norm(dim=-1).double().numpy()
    inter /= H0.shape[0]

    Dn = Wsel / np.linalg.norm(Wsel, axis=1, keepdims=True)
    Sgeo = Dn @ Dn.T
    sgeo = Sgeo[iu]

    # correlation of interaction with geometric proximity, vs feature-label permutation
    obs = np.corrcoef(inter, sgeo)[0, 1]
    ge = 0
    for _ in range(args.perm):
        p = rng.permutation(K)
        Sp = (Dn[p] @ Dn[p].T)[iu]
        if np.corrcoef(inter, Sp)[0, 1] >= obs:
            ge += 1
    p_val = (ge + 1) / (args.perm + 1)

    # CONTROL: is +corr a genuine pairwise-geometry effect, or just per-feature
    # interactivity clustering? Residualize interaction on each pair's per-feature
    # interactivity (sum + product), then correlate the residual with geometry.
    I = np.zeros((K, K))
    I[iu] = inter
    I = I + I.T
    act = I.sum(1) / (K - 1)                                 # per-feature interactivity
    fs, fp = act[ii] + act[jj], act[ii] * act[jj]
    X = np.stack([fs, fp, np.ones_like(fs)], 1)
    inter_res = inter - X @ np.linalg.lstsq(X, inter, rcond=None)[0]
    r_ctrl = np.corrcoef(inter_res, sgeo)[0, 1]
    sv = np.linalg.svd(I, compute_uv=False)
    shared = sv[0] ** 2 / (sv ** 2).sum()

    print(f"layer {args.layer}  K={K}  pairs={len(ii)}  base_points={H0.shape[0]}  "
          f"scale={args.scale}")
    print(f"corr(interaction, cos-geometry)          = {obs:+.3f}   perm_p = {p_val:.3f}")
    print(f"corr AFTER removing per-feature interactivity = {r_ctrl:+.3f}   "
          f"(genuine pairwise-geometry effect if it survives)")
    print(f"interaction matrix top-1 SVD energy = {shared:.3f}  "
          f"(is interaction ALSO shared-mode dominated?)")
    print(f"  interpretation: >0 & sig -> LOCAL (features interact with geometric "
          f"neighbors); ~0 -> GLOBAL")
    print("\n  cos-bin   n       mean interaction")
    edges = np.linspace(-1, 1, 11)
    binidx = np.clip(np.digitize(sgeo, edges) - 1, 0, 9)
    for b in range(10):
        m = binidx == b
        if m.sum() >= 20:
            c = 0.5 * (edges[b] + edges[b + 1])
            print(f"  {c:+.2f}   {int(m.sum()):>6}   {inter[m].mean():>10.2f}")


if __name__ == "__main__":
    main()
