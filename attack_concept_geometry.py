"""
Adversarial audit of the "SAE decoder recovers the calendar circle" positive.

Three attacks:
  (1) SPECTRUM-MATCHED null (the one the earlier work demanded):
      replace the 12 real selected decoder rows with 12 random directions
      carrying W_dec's singular spectrum (row-space covariance). Score them
        (a) in the fixed calendar order (does spectrum alone reproduce +0.83?)
        (b) under BEST circular alignment (PCA->angle-sort, the natural
            "recover a circle" procedure) -- ceiling: can ANY 12 anisotropic
            directions be arranged into a good circle?
      Compare real's fixed-order corr and real's best-align corr against these.
  (2) RANDOM-FEATURE control: 12 random ALIVE features (not concept-selective)
      dropped into the month slots. Fixed-order + best-align.
  (3) SELECTION confound: pick the k-th most selective feature per month
      (k=1,2,3,5,10) and a min-selectivity gate; also the LEAST-selective
      alive feature per month. Does the circle survive weak selection?
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from concept_geometry import (FAMILIES, TEMPLATES, concept_residual,
                              circular_target, cos_mat, perm_null)


def offdiag(M):
    iu = np.triu_indices(M.shape[0], 1)
    return M[iu]


def corr_to_circle(X, T):
    """corr of pairwise-cosine(X) offdiag with circular target T offdiag."""
    S = cos_mat(X)
    return float(np.corrcoef(offdiag(S), offdiag(T))[0, 1])


def best_align_corr(X, T, n_perm=0, rng=None):
    """Ceiling corr to the circle under reordering.

    PCA->angle-sort gives the natural 'recover the circle' ordering (the
    circular target is invariant to cyclic shift + reflection, so angle-sort
    is a principled near-optimal aligner). If n_perm>0 also take the max over
    n_perm random permutations and return the larger of the two.
    """
    Xc = X - X.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T                      # 12 x 2
    ang = np.arctan2(coords[:, 1], coords[:, 0])
    order = np.argsort(ang)
    S = cos_mat(X[order])
    best = float(np.corrcoef(offdiag(S), offdiag(T))[0, 1])
    if n_perm and rng is not None:
        base = cos_mat(X)
        t = offdiag(T)
        K = X.shape[0]
        for _ in range(n_perm):
            p = rng.permutation(K)
            r = float(np.corrcoef(offdiag(base[np.ix_(p, p)]), t)[0, 1])
            if r > best:
                best = r
    return best


def spectrum_dirs(W_dec, k, rng):
    """k random directions carrying W_dec's row-space covariance (spectrum)."""
    C = (W_dec.T @ W_dec) / W_dec.shape[0]       # 768 x 768, eigenvalues = s^2/N
    L = np.linalg.cholesky(C + 1e-9 * np.eye(C.shape[0]))
    z = rng.standard_normal((k, C.shape[0]))
    return z @ L.T                                # k x 768, matched anisotropy


def selectivity_ranked(A, n, alive, rank):
    """For each concept m, return the feature at the given selectivity `rank`
    (0 = most selective). rank may be an int or 'min' for least-selective alive."""
    sel = []
    scores = []
    for m in range(n):
        others = A[np.arange(n) != m].mean(0)
        score = A[m] - others
        score[~alive] = -np.inf
        order = np.argsort(score)[::-1]          # high->low selectivity
        if rank == "min":
            valid = order[np.isfinite(score[order])]
            f = int(valid[-1])
        else:
            f = int(order[rank])
        sel.append(f)
        scores.append(float(score[f]))
    return np.array(sel), float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--family", choices=list(FAMILIES), default="months")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ndraw", type=int, default=2000)
    args = ap.parse_args()

    words = FAMILIES[args.family]
    n = len(words)
    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    tok, model = load_gpt2()
    rng = np.random.default_rng(args.seed)

    R = np.stack([concept_residual(args.layer, w, tok, model) for w in words])
    A = np.maximum((R - b_dec) @ W_enc + b_enc, 0.0)
    alive = log_sp > DEAD_LOG_SPARSITY
    T = circular_target(n)

    # baseline: most-selective per concept (the published protocol)
    sel0, meansel0 = selectivity_ranked(A, n, alive, 0)
    real_dirs = W_dec[sel0]
    real_fixed = corr_to_circle(real_dirs, T)
    real_best = best_align_corr(real_dirs, T, n_perm=50000, rng=np.random.default_rng(1))
    model_fixed = corr_to_circle(R, T)

    print(f"=== family={args.family} layer={args.layer} n={n} ===")
    print(f"MODEL residual fixed-order corr        = {model_fixed:+.3f}")
    print(f"SAE decoder fixed-order corr (k=0 sel)  = {real_fixed:+.3f}  "
          f"(mean_selectivity={meansel0:.2f})")
    print(f"SAE decoder BEST-ALIGN corr             = {real_best:+.3f}\n")

    # ---- Attack 1: spectrum-matched null ----
    fixed_draws, best_draws = [], []
    for _ in range(args.ndraw):
        D = spectrum_dirs(W_dec, n, rng)
        fixed_draws.append(corr_to_circle(D, T))
        best_draws.append(best_align_corr(D, T))
    fixed_draws = np.array(fixed_draws)
    best_draws = np.array(best_draws)

    def summ(name, draws, obs):
        m, sd = draws.mean(), draws.std()
        z = (obs - m) / (sd + 1e-9)
        p = (np.sum(draws >= obs) + 1) / (len(draws) + 1)
        print(f"  {name:<42} null_mean={m:+.3f} sd={sd:.3f}  "
              f"obs={obs:+.3f}  z={z:+.2f}  p={p:.4f}")

    print("Attack 1  SPECTRUM-MATCHED null (12 random dirs w/ decoder spectrum):")
    summ("fixed(calendar)-order: real vs null", fixed_draws, real_fixed)
    summ("BEST-ALIGN: real vs null-best-align", best_draws, real_best)
    print(f"    [null best-align: mean={best_draws.mean():+.3f} "
          f"p95={np.percentile(best_draws,95):+.3f} "
          f"max={best_draws.max():+.3f}]\n")

    # ---- Attack 2: random alive-feature control ----
    aidx = np.where(alive)[0]
    rfixed, rbest = [], []
    for _ in range(args.ndraw):
        pick = rng.choice(aidx, size=n, replace=False)
        D = W_dec[pick]
        rfixed.append(corr_to_circle(D, T))
        rbest.append(best_align_corr(D, T))
    rfixed = np.array(rfixed); rbest = np.array(rbest)
    print("Attack 2  RANDOM-FEATURE control (12 random alive features in slots):")
    summ("fixed-order: real vs random-feature", rfixed, real_fixed)
    summ("BEST-ALIGN: real vs random-feature-best", rbest, real_best)
    print(f"    [rand-feat best-align: mean={rbest.mean():+.3f} "
          f"p95={np.percentile(rbest,95):+.3f} max={rbest.max():+.3f}]\n")

    # ---- Attack 3: selection confound ----
    print("Attack 3  SELECTION confound (k-th most selective feature per month):")
    permrng = np.random.default_rng(args.seed)
    for rank in [0, 1, 2, 4, 9, "min"]:
        sel, ms = selectivity_ranked(A, n, alive, rank)
        ndist = len(set(sel.tolist()))
        c = corr_to_circle(W_dec[sel], T)
        # permutation p on THIS selection
        obs, p, nm, nsd = perm_null(cos_mat(W_dec[sel]), T, permrng, n=2000)
        z = (obs - nm) / (nsd + 1e-9)
        rlabel = "min(least-sel)" if rank == "min" else f"rank{rank}"
        print(f"  {rlabel:<14} distinct={ndist}/{n} mean_sel={ms:7.2f}  "
              f"corr={c:+.3f}  perm_z={z:+.2f}  perm_p={p:.4f}")


if __name__ == "__main__":
    main()
