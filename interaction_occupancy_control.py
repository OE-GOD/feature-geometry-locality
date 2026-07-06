"""
EXPLORATORY control on the preregistered CONFIRM (see PREREG_interaction_cofire.md
RESULT section): is interaction->co-firing driven by WEIGHT-COUPLING or by an
OCCUPANCY echo (co-firing features being jointly present in the real base points)?

Recompute the identical statistic at base points where NONE of the 200 sampled
features fires (their SAE activations are all zero there), plus rotation nulls at
those same base points. If rho holds ~ +0.11 with z >> nulls -> weight-coupling
(occupancy ruled out). If it collapses toward the null -> the confirm was an
occupancy echo.
"""

import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts, harvest
from ruler_test import pile_contexts, cos_rows, spearman, partial_spearman, rand_orth
from interaction_cofire import compute_halves, stat_pair, LAYER, ALPHA

torch.set_num_threads(8)
N_NULL = 4


def inactive_base_points(contexts, model, idx, W_enc, b_enc, b_dec, n_base, rng):
    """Real residuals at positions where NONE of the features in idx fires."""
    block = model.transformer.h[LAYER]
    rec = []

    def pre(_m, args):
        rec.append(args[0].detach())
        return None

    h = block.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(contexts)
    h.remove()
    H = rec[0].reshape(-1, rec[0].shape[-1])
    Hn = H.double().numpy()
    acts = np.maximum((Hn - b_dec) @ W_enc[:, idx] + b_enc[idx], 0.0)
    quiet = np.where((acts > 0).sum(1) == 0)[0]
    print(f"[occ] {quiet.size}/{H.shape[0]} positions have zero activity "
          f"across all {idx.size} features", flush=True)
    pick = rng.choice(quiet, size=min(n_base, quiet.size), replace=False)
    return H[pick]


def main():
    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    pilot_idx = set(np.sort(np.random.default_rng(0).choice(alive, 200, replace=False)))
    pool = np.array([a for a in alive if a not in pilot_idx])
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(pool, size=200, replace=False))     # same sample as confirm
    D_np = W_dec[idx].astype(np.float64)
    D = torch.tensor(D_np, dtype=torch.float32) * ALPHA
    K = idx.size
    iu = np.triu_indices(K, 1)

    tok, model = load_gpt2()
    block = model.transformer.h[LAYER]
    contexts = build_pile_contexts(tok, 400, 48, 8000)
    H0 = inactive_base_points(contexts, model, idx, W_enc, b_enc, b_dec, 32, rng)

    ctx_fn = pile_contexts(tok, 8000 // 64, 64, doc_start=500)
    A = harvest(LAYER, idx, ctx_fn, model, W_enc, b_enc, b_dec)
    keep = (A > 0).sum(0) >= 10
    iu_k = np.triu_indices(int(keep.sum()), 1)
    s_fn = cos_rows(A.T[keep].astype(np.float64))[iu_k]
    s_geo = cos_rows(D_np[keep])[iu_k]

    IA, IB = compute_halves(block, H0, D, iu)
    rho, part = stat_pair((IA + IB) / 2, keep, s_fn, s_geo, iu_k)
    print(f"[real@quiet] rho={rho:+.4f} partial={part:+.4f} "
          f"(confirm run at ordinary base points: +0.1130 / +0.1076)", flush=True)

    nr, npart = [], []
    for t in range(N_NULL):
        Q = rand_orth(D_np.shape[1], np.random.default_rng(t + 1))
        Dq = torch.tensor(D_np @ Q.T, dtype=torch.float32) * ALPHA
        nA, nB = compute_halves(block, H0, Dq, iu)
        r, p = stat_pair((nA + nB) / 2, keep, s_fn, s_geo, iu_k)
        nr.append(r)
        npart.append(p)
        print(f"[null {t}@quiet] rho={r:+.4f} partial={p:+.4f}", flush=True)

    nr, npart = np.array(nr), np.array(npart)
    print(f"\nz@quiet         = {(rho - nr.mean()) / (nr.std() + 1e-12):+.2f}")
    print(f"z_partial@quiet = {(part - npart.mean()) / (npart.std() + 1e-12):+.2f}")
    print("reading: rho holds ~0.11, z >> 0 -> WEIGHT-COUPLING (occupancy ruled out)")
    print("         rho collapses toward null -> OCCUPANCY ECHO (weaker claim)")


if __name__ == "__main__":
    main()
