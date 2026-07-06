"""
CONFIRMATORY: does pair-specific interaction structure predict co-firing?
PREREGISTERED: PREREG_interaction_cofire.md (frozen at commit 2abd7a0).

I(i,j) over B=32 base points; I_res = I minus dominant rank-1; S_fn = co-firing
on disjoint text; rho = Spearman(I_res, S_fn), rho_partial controls raw cosine;
8 rotation nulls (identical pipeline). Fresh features: seed 7, disjoint from the
pilot's seed-0 sample. Gates: G0 determinism, G1 split-half > 0.5.
Checkpoints after every condition.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts, harvest
from feature_interaction import mlp_sublayer, base_points
from ruler_test import (pile_contexts, cos_rows, spearman, partial_spearman,
                        rand_orth)
from interaction_stability import half_matrix, rank1_removed

LAYER = 8
ALPHA = 6.0
torch.set_num_threads(8)


def compute_halves(block, H0, D, iu):
    """Interaction matrix accumulated separately over the two base-point halves."""
    h = H0.shape[0] // 2
    return (half_matrix(block, H0[:h], D, iu),
            half_matrix(block, H0[h:], D, iu))


def stat_pair(I, keep, s_fn, s_geo, iu_k):
    R = rank1_removed(I[np.ix_(keep, keep)])
    v = R[iu_k]
    return spearman(v, s_fn), partial_spearman(v, s_fn, s_geo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--bp", type=int, default=32)
    ap.add_argument("--nulls", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.k, args.bp, args.nulls = 30, 4, 2
    tag = "smoke" if args.smoke else "full"

    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    pilot_idx = set(np.sort(np.random.default_rng(0).choice(alive, 200, replace=False)))
    pool = np.array([a for a in alive if a not in pilot_idx])
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(pool, size=min(args.k, pool.size), replace=False))
    D_np = W_dec[idx].astype(np.float64)
    D = torch.tensor(D_np, dtype=torch.float32) * ALPHA
    K = idx.size
    iu = np.triu_indices(K, 1)

    tok, model = load_gpt2()
    block = model.transformer.h[LAYER]
    contexts = build_pile_contexts(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, args.bp, rng)

    # co-firing on disjoint text
    ctx_fn = pile_contexts(tok, 8000 // 64, 64, doc_start=500)
    A = harvest(LAYER, idx, ctx_fn, model, W_enc, b_enc, b_dec)
    keep = (A > 0).sum(0) >= 10
    Kk = int(keep.sum())
    iu_k = np.triu_indices(Kk, 1)
    s_fn = cos_rows(A.T[keep].astype(np.float64))[iu_k]
    s_geo = cos_rows(D_np[keep])[iu_k]
    print(f"[fn] {Kk}/{K} features fire>=10; pairs={iu_k[0].size}", flush=True)

    # real condition (halves give G1 for free), computed TWICE for G0
    IA, IB = compute_halves(block, H0, D, iu)
    I_real = (IA + IB) / 2
    rho1, part1 = stat_pair(I_real, keep, s_fn, s_geo, iu_k)
    IA2, IB2 = compute_halves(block, H0, D, iu)
    rho2, _ = stat_pair((IA2 + IB2) / 2, keep, s_fn, s_geo, iu_k)
    g0 = abs(rho1 - rho2)
    g1 = spearman(rank1_removed(IA[np.ix_(keep, keep)])[iu_k],
                  rank1_removed(IB[np.ix_(keep, keep)])[iu_k])
    print(f"[G0] |rho - rho_rerun| = {g0:.2e} ({'PASS' if g0 < 1e-9 else 'FAIL'})",
          flush=True)
    print(f"[G1] split-half (rank1-removed) = {g1:+.3f} "
          f"({'PASS' if g1 > 0.5 else 'FAIL'})", flush=True)
    print(f"[real] rho={rho1:+.4f}  partial|geo={part1:+.4f}", flush=True)
    np.savez(f"icf_{tag}_real.npz", rho=rho1, part=part1, g0=g0, g1=g1)

    null_rho, null_part = [], []
    for t in range(args.nulls):
        Q = rand_orth(D_np.shape[1], np.random.default_rng(t + 1))
        Dq = torch.tensor(D_np @ Q.T, dtype=torch.float32) * ALPHA
        nA, nB = compute_halves(block, H0, Dq, iu)
        r, p = stat_pair((nA + nB) / 2, keep, s_fn, s_geo, iu_k)
        null_rho.append(r)
        null_part.append(p)
        np.savez(f"icf_{tag}_null{t}.npz", rho=r, part=p)
        print(f"[null {t}] rho={r:+.4f} partial={p:+.4f}", flush=True)

    nr, np_ = np.array(null_rho), np.array(null_part)
    z = (rho1 - nr.mean()) / (nr.std() + 1e-12)
    zp = (part1 - np_.mean()) / (np_.std() + 1e-12)
    print(f"\n=== VERDICT INPUTS (PREREG_interaction_cofire.md) ===")
    print(f"z        = {z:+.2f}  (real {rho1:+.4f} vs null {nr.mean():+.4f} ± {nr.std():.4f})")
    print(f"z_partial= {zp:+.2f}  (real {part1:+.4f} vs null {np_.mean():+.4f} ± {np_.std():.4f})")
    gates = (g0 < 1e-9) and (g1 > 0.5)
    if not gates:
        v = "INDETERMINATE (instrument gate failure)"
    elif z >= 4 and zp >= 4:
        v = "CONFIRM H1"
    elif z <= 2 or zp <= 2:
        v = "REFUTE H1"
    else:
        v = "INDETERMINATE"
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
