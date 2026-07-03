"""
ATTACK: is the +0.30 controlled local effect carried by near-duplicate / feature-split
pairs (cos high, d_i+d_j ~ 2 d_i -> second-order term is just self-curvature)?

Computes interaction(i,j) and cos(i,j) once, then:
 (1) bins the CONTROLLED interaction residual by cos -> where does the signal live?
 (2) recomputes raw + controlled corr EXCLUDING pairs with cos > {0.7,0.5,0.3}
 (3) quantifies near-duplicate prevalence among sampled features (and vs full dict)
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts


def mlp_sublayer(block, x):
    return block.mlp(block.ln_2(x))


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
    return H[pick]


def controlled_corr(inter, sgeo, act, ii, jj, mask=None):
    """corr(residual-of-inter-on-per-feature-interactivity, sgeo) on optional pair subset."""
    if mask is None:
        mask = np.ones(len(inter), dtype=bool)
    fs = act[ii] + act[jj]
    fp = act[ii] * act[jj]
    X = np.stack([fs[mask], fp[mask], np.ones(mask.sum())], 1)
    y = inter[mask]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return np.corrcoef(res, sgeo[mask])[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    tok, model = load_gpt2()
    Wsel = W_dec[idx].astype(np.float32)

    contexts = build_pile_contexts(tok, args.docs, args.seq, args.docs * args.seq)
    H0 = base_points(args.layer, contexts, model, args.n_base, rng)
    block = model.transformer.h[args.layer]
    D = torch.from_numpy(Wsel).float() * args.scale
    K = D.shape[0]
    iu = np.triu_indices(K, 1)
    ii, jj = iu

    inter = np.zeros(len(ii), dtype=np.float64)
    with torch.no_grad():
        pair_writes = D[ii] + D[jj]
        for b in range(H0.shape[0]):
            h0 = H0[b:b + 1]
            f0 = mlp_sublayer(block, h0)
            f1 = mlp_sublayer(block, h0 + D)
            fp = mlp_sublayer(block, h0 + pair_writes)
            resid = fp - f1[ii] - f1[jj] + f0
            inter += resid.norm(dim=-1).double().numpy()
    inter /= H0.shape[0]

    Dn = Wsel / np.linalg.norm(Wsel, axis=1, keepdims=True)
    Sgeo = Dn @ Dn.T
    sgeo = Sgeo[iu]

    # per-feature interactivity (as in the original control)
    I = np.zeros((K, K))
    I[iu] = inter
    I = I + I.T
    act = I.sum(1) / (K - 1)

    raw = np.corrcoef(inter, sgeo)[0, 1]
    ctrl = controlled_corr(inter, sgeo, act, ii, jj)
    print(f"layer {args.layer}  K={K}  pairs={len(ii)}  base_points={H0.shape[0]}  scale={args.scale}")
    print(f"[reproduce]  raw corr = {raw:+.3f}   controlled corr = {ctrl:+.3f}")

    # cos distribution among pairs
    print("\n[cos distribution over pairs]")
    for lo, hi in [(-1,0),(0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)]:
        m = (sgeo>=lo)&(sgeo<hi)
        print(f"  cos [{lo:+.2f},{hi:+.2f}):  {int(m.sum()):>7}  ({100*m.mean():5.2f}%)")

    # (1) controlled residual binned by cos: where does the signal live?
    fs, fp = act[ii]+act[jj], act[ii]*act[jj]
    X = np.stack([fs, fp, np.ones_like(fs)], 1)
    inter_res = inter - X @ np.linalg.lstsq(X, inter, rcond=None)[0]
    print("\n[controlled residual binned by cos]  (mean residual should track cos if local is real)")
    edges = [-1,-0.1,0,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.01]
    for a,bb in zip(edges[:-1],edges[1:]):
        m=(sgeo>=a)&(sgeo<bb)
        if m.sum()>=5:
            print(f"  cos [{a:+.2f},{bb:+.2f}):  n={int(m.sum()):>7}  mean_resid={inter_res[m].mean():+.4f}  mean_inter={inter[m].mean():.3f}")

    # (2) exclude near-duplicate pairs and recompute
    print("\n[exclude high-cos pairs, recompute raw + controlled corr on distinct pairs]")
    for thr in [1.01, 0.7, 0.5, 0.3, 0.2, 0.1]:
        m = sgeo < thr
        if m.sum() < 50:
            continue
        r = np.corrcoef(inter[m], sgeo[m])[0,1]
        c = controlled_corr(inter, sgeo, act, ii, jj, mask=m)
        print(f"  keep cos<{thr:.2f}:  n={int(m.sum()):>7}  raw={r:+.3f}  controlled={c:+.3f}")

    # (2b) also POSITIVE-cos-only distinct band: does local signal exist among moderate cos?
    print("\n[within positive-cos bands only]")
    for lo,hi in [(0,0.1),(0,0.2),(0.05,0.3),(0,0.3),(0,0.5)]:
        m=(sgeo>=lo)&(sgeo<hi)
        if m.sum()<50: continue
        r=np.corrcoef(inter[m],sgeo[m])[0,1]
        c=controlled_corr(inter,sgeo,act,ii,jj,mask=m)
        print(f"  cos in [{lo:.2f},{hi:.2f}):  n={int(m.sum()):>7}  raw={r:+.3f}  controlled={c:+.3f}")

    # (3) near-duplicate prevalence
    print("\n[near-duplicate prevalence]")
    # within sampled set
    Sabs = Sgeo.copy(); np.fill_diagonal(Sabs, -np.inf)
    nn_in = Sabs.max(1)
    for t in [0.5,0.7,0.9]:
        print(f"  within-sample: frac features with a neighbor cos>{t}: {(nn_in>t).mean():.3f}")
    # vs full alive dictionary (true nearest neighbor of each sampled feature)
    Wa = W_dec[alive].astype(np.float32)
    Wan = Wa / np.linalg.norm(Wa, axis=1, keepdims=True)
    Wsn = Dn  # sampled normalized
    C = Wsn @ Wan.T  # (K, n_alive)
    # zero out self (each sampled idx corresponds to a column in alive)
    alive_pos = {a:p for p,a in enumerate(alive)}
    for r,fidx in enumerate(idx):
        C[r, alive_pos[fidx]] = -np.inf
    nn_full = C.max(1)
    for t in [0.5,0.7,0.9]:
        print(f"  vs full dict: frac sampled features with a neighbor cos>{t}: {(nn_full>t).mean():.3f}")
    print(f"  max pair cos among sampled: {sgeo.max():.3f}   median |cos|: {np.median(np.abs(sgeo)):.4f}")

    np.savez(f"/private/tmp/claude-501/-Users-oe/832c8e58-6b79-4f8b-b508-1a24d9b54677/scratchpad/inter_L{args.layer}_k{K}_s{args.seed}.npz",
             inter=inter, sgeo=sgeo, act=act, idx=idx, ii=ii, jj=jj)


if __name__ == "__main__":
    main()
