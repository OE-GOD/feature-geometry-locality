"""
EXPLORATORY instrument pilot (NOT confirmatory, no prereg claims): is the
feature-interaction matrix's structure stable enough to build on?

Lesson from the ruler test: check the coin exists before designing the experiment.
The interaction matrix I(i,j) = ||mlp(h0+di+dj) - mlp(h0+di) - mlp(h0+dj) + mlp(h0)||
was the one signal that survived the spectrum-matched control. The next project
(interaction GRAPH: clusters/hierarchy) only makes sense if I's structure BEYOND
the known per-feature interactivity scalar replicates across disjoint data.

Split base points into two disjoint halves; compute I_A, I_B; report split-half
stability (Spearman over off-diag pairs) at three levels:
  raw            — includes the scalar (expected high; the scalar is a real coin)
  rank1-removed  — subtract best rank-1 fit (the scalar axis) from each half
  degree-normed  — I(i,j) / sqrt(deg_i * deg_j)  (configuration-model residual)
If rank1-removed / degree-normed stability ~ 0 -> empty scale, stop the project.
If solid (>~0.4) -> a real coin; design the preregistered cluster/hierarchy test.
"""

import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts
from feature_interaction import mlp_sublayer, base_points

LAYER = 8
K = 200
B = 32
ALPHA = 6.0
torch.set_num_threads(8)


def half_matrix(block, H0, D, iu):
    """Accumulate interaction magnitudes over the given base points."""
    Kf = D.shape[0]
    ii, jj = iu
    pw = D[ii] + D[jj]
    I = np.zeros(ii.size)
    with torch.no_grad():
        for b in range(H0.shape[0]):
            h0 = H0[b:b + 1]
            f0 = mlp_sublayer(block, h0)
            f1 = mlp_sublayer(block, h0 + D)
            fp = mlp_sublayer(block, h0 + pw)
            I += (fp - f1[ii] - f1[jj] + f0).norm(dim=-1).double().numpy()
    M = np.zeros((Kf, Kf))
    M[iu] = I / H0.shape[0]
    return M + M.T


def rank1_removed(M):
    w, V = np.linalg.eigh(M)
    i = np.argmax(np.abs(w))
    return M - w[i] * np.outer(V[:, i], V[:, i])


def degree_normed(M):
    deg = M.sum(1)
    return M / np.sqrt(np.outer(deg, deg) + 1e-12)


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def scorr(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def main():
    rng = np.random.default_rng(0)
    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    idx = np.sort(rng.choice(alive, K, replace=False))
    D = torch.tensor(W_dec[idx], dtype=torch.float32) * ALPHA

    tok, model = load_gpt2()
    contexts = build_pile_contexts(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, B, rng)
    block = model.transformer.h[LAYER]
    iu = np.triu_indices(K, 1)

    IA = half_matrix(block, H0[: B // 2], D, iu)
    IB = half_matrix(block, H0[B // 2:], D, iu)
    print(f"K={K} features, {iu[0].size} pairs, {B//2}+{B//2} disjoint base points")
    print(f"split-half stability (Spearman over pairs):")
    print(f"  raw            : {scorr(IA[iu], IB[iu]):+.3f}   (scalar included)")
    print(f"  rank1-removed  : {scorr(rank1_removed(IA)[iu], rank1_removed(IB)[iu]):+.3f}")
    print(f"  degree-normed  : {scorr(degree_normed(IA)[iu], degree_normed(IB)[iu]):+.3f}")
    # descriptive: how dominant is the scalar axis?
    w = np.linalg.eigh((IA + IB) / 2)[0]
    print(f"  top-eig share of full matrix: {np.abs(w).max()/np.abs(w).sum():.3f}")
    print("\nreading: residual stability ~0 -> empty scale, stop. >~0.4 -> real coin;")
    print("design the preregistered cluster/hierarchy test on the interaction graph.")


if __name__ == "__main__":
    main()
