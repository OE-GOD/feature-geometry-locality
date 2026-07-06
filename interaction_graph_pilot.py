"""
EXPLORATORY pilot: does the interaction graph have mesoscale STRUCTURE
(communities), and does that structure align with function?

Runs on the PILOT feature sample (seed 0) so a later confirmatory prereg can use
fresh features. Questions:
  1. Dimensionality: how many eigenvalues of I_res sit outside the rotation-null
     envelope? (how many real structural dimensions beyond the scalar axis)
  2. Stability: split-half ARI of spectral communities (is the clustering a coin?)
  3. Function (peek, pilot-only): ARI between interaction communities and
     co-firing communities, vs the same ARI for rotation-null interactions.
"""

import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts, harvest
from feature_interaction import base_points
from ruler_test import pile_contexts, cos_rows, rand_orth
from interaction_cofire import compute_halves, LAYER, ALPHA
from interaction_stability import rank1_removed

torch.set_num_threads(8)
K = 200
B = 32
N_NULL = 4
KC = 6            # communities for clustering passes
SEED = 0


def kmeans(X, k, rng, iters=60, restarts=8):
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        C = X[rng.choice(X.shape[0], k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            lab = d.argmin(1)
            C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                          for j in range(k)])
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def spectral_labels(M, k, rng):
    w, V = np.linalg.eigh(M)
    order = np.argsort(-np.abs(w))[:k]
    X = V[:, order]
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return kmeans(X, k, rng)


def ari(a, b):
    ka, kb = a.max() + 1, b.max() + 1
    n = a.size
    ct = np.zeros((ka, kb))
    for i in range(n):
        ct[a[i], b[i]] += 1
    comb = lambda x: x * (x - 1) / 2
    sij = comb(ct).sum()
    sa, sb = comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp = sa * sb / comb(n)
    mx = (sa + sb) / 2
    return (sij - exp) / (mx - exp + 1e-12)


def main():
    rng = np.random.default_rng(SEED)
    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    idx = np.sort(rng.choice(alive, K, replace=False))          # pilot sample (seed 0)
    D_np = W_dec[idx].astype(np.float64)
    D = torch.tensor(D_np, dtype=torch.float32) * ALPHA
    iu = np.triu_indices(K, 1)

    tok, model = load_gpt2()
    block = model.transformer.h[LAYER]
    contexts = build_pile_contexts(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, B, rng)

    IA, IB = compute_halves(block, H0, D, iu)
    Ra, Rb = rank1_removed(IA), rank1_removed(IB)
    R = rank1_removed((IA + IB) / 2)

    # 1. dimensionality vs rotation nulls
    ev_real = np.sort(np.abs(np.linalg.eigvalsh(R)))[::-1]
    ev_nulls = []
    null_R = []
    for t in range(N_NULL):
        Q = rand_orth(D_np.shape[1], np.random.default_rng(t + 1))
        Dq = torch.tensor(D_np @ Q.T, dtype=torch.float32) * ALPHA
        nA, nB = compute_halves(block, H0, Dq, iu)
        Rn = rank1_removed((nA + nB) / 2)
        null_R.append(Rn)
        ev_nulls.append(np.sort(np.abs(np.linalg.eigvalsh(Rn)))[::-1])
        print(f"[null {t}] done", flush=True)
    env = np.max(np.stack(ev_nulls), axis=0)
    n_dims = int((ev_real > env).sum())
    print(f"\n[1] real |eig| above the null envelope at same rank: {n_dims} dims "
          f"(top-5 real {np.round(ev_real[:5],1)} vs null-max {np.round(env[:5],1)})",
          flush=True)

    # 2. split-half community stability
    crng = np.random.default_rng(99)
    la = spectral_labels(Ra, KC, crng)
    lb = spectral_labels(Rb, KC, crng)
    print(f"[2] split-half community ARI (k={KC}): {ari(la, lb):+.3f}", flush=True)

    # 3. functional alignment (pilot peek)
    ctx_fn = pile_contexts(tok, 8000 // 64, 64, doc_start=500)
    Aacts = harvest(LAYER, idx, ctx_fn, model, W_enc, b_enc, b_dec)
    keep = (Aacts > 0).sum(0) >= 10
    Sfn = cos_rows(Aacts.T[keep].astype(np.float64))
    np.fill_diagonal(Sfn, 0.0)
    lfn = spectral_labels(Sfn, KC, crng)
    li = spectral_labels(R[np.ix_(keep, keep)], KC, crng)
    a_real = ari(li, lfn)
    a_nulls = [ari(spectral_labels(Rn[np.ix_(keep, keep)], KC, crng), lfn)
               for Rn in null_R]
    a_nulls = np.array(a_nulls)
    print(f"[3] ARI(interaction communities, co-firing communities): real {a_real:+.3f}"
          f"  nulls {a_nulls.mean():+.3f} ± {a_nulls.std():.3f}"
          f"  informal z {(a_real - a_nulls.mean()) / (a_nulls.std() + 1e-12):+.1f}",
          flush=True)
    print("\npilot reading: dims>0 + stable communities + real ARI >> null ->"
          " prereg the confirmatory graph test on fresh features")


if __name__ == "__main__":
    main()
