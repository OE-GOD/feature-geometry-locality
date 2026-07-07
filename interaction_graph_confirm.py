"""
CONFIRMATORY: interaction-graph communities vs function.
PREREGISTERED: PREREG_interaction_graph.md (frozen at commit 6ddfa70).

H1a: pairwise I_res -> co-firing replicates on a third disjoint sample (seed 13).
H1b: spectral communities of I_res align with co-firing communities vs 8 rotation
nulls (k=6, pilot's exact clustering procedure).
Pre-data ambiguity resolution (committed before running): G2 and all community
labels are computed on the fires>=10 subset, the same domain as every other
confirmatory statistic.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts, harvest
from feature_interaction import base_points
from ruler_test import pile_contexts, cos_rows, spearman, rand_orth
from interaction_cofire import compute_halves, stat_pair, LAYER, ALPHA
from interaction_stability import rank1_removed
from interaction_graph_pilot import spectral_labels, ari

torch.set_num_threads(8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--bp", type=int, default=32)
    ap.add_argument("--nulls", type=int, default=8)
    ap.add_argument("--kc", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.k, args.bp, args.nulls, args.kc = 30, 4, 2, 3
    tag = "smoke" if args.smoke else "full"

    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    s0 = set(np.random.default_rng(0).choice(alive, 200, replace=False))
    pool1 = np.array([a for a in alive if a not in s0])
    s7 = set(np.random.default_rng(7).choice(pool1, 200, replace=False))
    pool = np.array([a for a in alive if a not in s0 and a not in s7])
    rng = np.random.default_rng(13)
    idx = np.sort(rng.choice(pool, size=min(args.k, pool.size), replace=False))
    D_np = W_dec[idx].astype(np.float64)
    D = torch.tensor(D_np, dtype=torch.float32) * ALPHA
    K = idx.size
    iu = np.triu_indices(K, 1)

    tok, model = load_gpt2()
    block = model.transformer.h[LAYER]
    contexts = build_pile_contexts(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, args.bp, rng)

    ctx_fn = pile_contexts(tok, 8000 // 64, 64, doc_start=500)
    A = harvest(LAYER, idx, ctx_fn, model, W_enc, b_enc, b_dec)
    keep = (A > 0).sum(0) >= 10
    Kk = int(keep.sum())
    iu_k = np.triu_indices(Kk, 1)
    s_fn = cos_rows(A.T[keep].astype(np.float64))[iu_k]
    s_geo = cos_rows(D_np[keep])[iu_k]
    Sfn_mat = cos_rows(A.T[keep].astype(np.float64))
    np.fill_diagonal(Sfn_mat, 0.0)
    print(f"[fn] {Kk}/{K} fire>=10; pairs={iu_k[0].size}", flush=True)

    crng = np.random.default_rng(99)
    lfn = spectral_labels(Sfn_mat, args.kc, crng)

    # real + gates
    IA, IB = compute_halves(block, H0, D, iu)
    I_real = (IA + IB) / 2
    rho1, part1 = stat_pair(I_real, keep, s_fn, s_geo, iu_k)
    IA2, IB2 = compute_halves(block, H0, D, iu)
    rho2, _ = stat_pair((IA2 + IB2) / 2, keep, s_fn, s_geo, iu_k)
    g0 = abs(rho1 - rho2)
    Ra = rank1_removed(IA[np.ix_(keep, keep)])
    Rb = rank1_removed(IB[np.ix_(keep, keep)])
    g1 = spearman(Ra[iu_k], Rb[iu_k])
    g2 = ari(spectral_labels(Ra, args.kc, crng), spectral_labels(Rb, args.kc, crng))
    R = rank1_removed(I_real[np.ix_(keep, keep)])
    li = spectral_labels(R, args.kc, crng)
    ari_real = ari(li, lfn)
    print(f"[G0] {g0:.2e} ({'PASS' if g0 < 1e-9 else 'FAIL'})", flush=True)
    print(f"[G1] split-half pairs = {g1:+.3f} ({'PASS' if g1 > 0.5 else 'FAIL'})",
          flush=True)
    print(f"[G2] split-half communities ARI = {g2:+.3f} "
          f"({'PASS' if g2 > 0.3 else 'FAIL'})", flush=True)
    print(f"[real] rho={rho1:+.4f} partial={part1:+.4f} ARI={ari_real:+.4f}",
          flush=True)
    np.savez(f"igc_{tag}_real.npz", rho=rho1, part=part1, ari=ari_real,
             g0=g0, g1=g1, g2=g2)

    nr, np_, na = [], [], []
    for t in range(args.nulls):
        Q = rand_orth(D_np.shape[1], np.random.default_rng(t + 1))
        Dq = torch.tensor(D_np @ Q.T, dtype=torch.float32) * ALPHA
        nA, nB = compute_halves(block, H0, Dq, iu)
        In = (nA + nB) / 2
        r, p = stat_pair(In, keep, s_fn, s_geo, iu_k)
        an = ari(spectral_labels(rank1_removed(In[np.ix_(keep, keep)]),
                                 args.kc, crng), lfn)
        nr.append(r); np_.append(p); na.append(an)
        np.savez(f"igc_{tag}_null{t}.npz", rho=r, part=p, ari=an)
        print(f"[null {t}] rho={r:+.4f} partial={p:+.4f} ARI={an:+.4f}", flush=True)

    nr, np_, na = map(np.array, (nr, np_, na))
    z = (rho1 - nr.mean()) / (nr.std() + 1e-12)
    zp = (part1 - np_.mean()) / (np_.std() + 1e-12)
    za = (ari_real - na.mean()) / (na.std() + 1e-12)
    print(f"\n=== VERDICT INPUTS (PREREG_interaction_graph.md) ===")
    print(f"H1a: z={z:+.2f} z_partial={zp:+.2f}")
    print(f"H1b: ARI real {ari_real:+.4f} vs null {na.mean():+.4f} ± {na.std():.4f}"
          f" -> z_ARI={za:+.2f}")
    gates_a = (g0 < 1e-9) and (g1 > 0.5)
    gates_b = gates_a and (g2 > 0.3)
    va = ("INDETERMINATE (instrument)" if not gates_a else
          "CONFIRM" if (z >= 4 and zp >= 4) else
          "REFUTE" if (z <= 2 or zp <= 2) else "INDETERMINATE")
    vb = ("INDETERMINATE (instrument)" if not gates_b else
          "CONFIRM" if za >= 4 else
          "REFUTE" if za <= 2 else "INDETERMINATE")
    print(f"VERDICT H1a (replication): {va}")
    print(f"VERDICT H1b (communities): {vb}")


if __name__ == "__main__":
    main()
