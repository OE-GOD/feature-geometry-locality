"""Duplicate-confound probe: is the rank-1 halo just near-duplicate features?"""
import json
import numpy as np
from run_layer import fetch, DEAD_LOG_SPARSITY

HOOK = "blocks.8.hook_resid_pre"
K = 6000
SEED = 0
DUP_CUT = 0.9

W_dec, log_sparsity, wte, ln_w = fetch(HOOK)
alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
rng = np.random.default_rng(SEED)
if K < alive.size:
    alive = np.sort(rng.choice(alive, size=K, replace=False))
D = W_dec[alive]
n = D.shape[0]

with np.errstate(all="ignore"):
    Ds = (D - D.mean(axis=1, keepdims=True)) * ln_w
    M = wte.T @ wte
    G = Ds @ M @ Ds.T
    fnorm = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    Sfn = (G / np.outer(fnorm, fnorm)).astype(np.float64)
    Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    Sgeo = (Dn @ Dn.T).astype(np.float64)

iu = np.triu_indices(n, k=1)
sg, sf = Sgeo[iu], Sfn[iu]

bands = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5)]
band_out = []
for lo, hi in bands:
    m = (sg > lo) & (sg <= hi)
    band_out.append({"band": f"({lo},{hi if hi<=1 else 1.0}]",
                     "count": int(m.sum()),
                     "mean_sfn": round(float(sf[m].mean()), 4) if m.any() else None})

rows = np.arange(n)

def rank1(Sgeo_mat):
    nb = np.argmax(Sgeo_mat, axis=1)
    return nb, float(Sfn[rows, nb].mean()), float(Sgeo_mat[rows, nb].mean())

Sg_a = Sgeo.copy()
np.fill_diagonal(Sg_a, -np.inf)
nb_a, sfn_a, sgeo_a = rank1(Sg_a)
frac_dup_rank1 = float((Sg_a[rows, nb_a] > DUP_CUT).mean())

Sg_b = Sg_a.copy()
Sg_b[Sg_b > DUP_CUT] = -np.inf   # mask near-duplicate pairs
nb_b, sfn_b, sgeo_b = rank1(Sg_b)

baseline = float(sf.mean())

res = {
    "hook": HOOK, "k": K, "seed": SEED, "n": n,
    "bands": band_out,
    "baseline_sfn": round(baseline, 4),
    "rank1_all": {"mean_sfn": round(sfn_a, 4), "mean_sgeo": round(sgeo_a, 4)},
    "rank1_excl_dup": {"mean_sfn": round(sfn_b, 4), "mean_sgeo": round(sgeo_b, 4)},
    "frac_rank1_neighbor_sgeo_gt_0.9": round(frac_dup_rank1, 4),
}
print(json.dumps(res, indent=2))
