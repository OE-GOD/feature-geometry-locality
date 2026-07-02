"""
Spectrum-matched null control -- the fix demanded by both REFUTED skeptics.

The bug it corrects
-------------------
Our function proxy is LINEAR in the decoder direction: F = wte @ (ln_w * (d-mean)).
So Sfn is just cosine of d under a fixed quadratic form M = wte^T wte, while
Sgeo is cosine under the identity. Comparing two quadratic forms *mechanically*
produces a monotone Sgeo->Sfn "halo" and a negative-cosine "antipodal tail" for
ANY map with wte's singular spectrum -- even a random one. So the raw halo says
nothing network-specific.

The control
-----------
Keep the decoder geometry D exactly. Replace the real metric M = wte^T wte with
M_rand = Q diag(eig(M)) Q^T for random orthogonal Q (same eigenvalue spectrum,
random eigenvectors). Recompute the Sgeo->Sfn curve. Only the part of the real
curve that EXCEEDS this null is network-specific functional structure.

If real - null <= 0, the "structure" is a tautology of the proxy.

Usage: python3 null_control.py --hook blocks.8.hook_resid_pre --k 3000 --rot 5
"""

import argparse
import json
import numpy as np

from run_layer import fetch, DEAD_LOG_SPARSITY

BINS = np.linspace(-1.0, 1.0, 21)
CENTERS = 0.5 * (BINS[:-1] + BINS[1:])


def binned_curve(sgeo, sfn):
    idx = np.clip(np.digitize(sgeo, BINS) - 1, 0, len(CENTERS) - 1)
    mean = np.full(len(CENTERS), np.nan)
    cnt = np.zeros(len(CENTERS))
    for b in range(len(CENTERS)):
        m = idx == b
        cnt[b] = m.sum()
        if cnt[b]:
            mean[b] = sfn[m].mean()
    overall = sfn.mean()
    pred = np.where(np.isnan(mean[idx]), overall, mean[idx])
    ss_res = np.sum((sfn - pred) ** 2)
    ss_tot = np.sum((sfn - overall) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return mean, cnt, r2


def cos_under_metric(y, M):
    with np.errstate(all="ignore"):
        G = y @ M @ y.T
        nrm = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        return G / np.outer(nrm, nrm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default="blocks.8.hook_resid_pre")
    ap.add_argument("--k", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot", type=int, default=5, help="random rotations")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W_dec, log_sparsity, wte, ln_w = fetch(args.hook)
    alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    if args.k < alive.size:
        alive = np.sort(rng.choice(alive, size=args.k, replace=False))
    D = W_dec[alive]

    y = (D - D.mean(axis=1, keepdims=True)) * ln_w          # centered+scaled
    M = wte.T @ wte                                          # real metric
    evals = np.clip(np.linalg.eigvalsh(M), 0, None)         # spectrum to match

    Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(all="ignore"):
        Sgeo = Dn @ Dn.T
    iu = np.triu_indices(D.shape[0], k=1)
    sgeo = Sgeo[iu].astype(np.float64)

    sfn_real = cos_under_metric(y, M)[iu].astype(np.float64)
    real_mean, cnt, real_r2 = binned_curve(sgeo, sfn_real)

    rand_means, rand_r2 = [], []
    d = M.shape[0]
    for r in range(args.rot):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        M_rand = (Q * evals) @ Q.T
        sfn_r = cos_under_metric(y, M_rand)[iu].astype(np.float64)
        m, _, r2 = binned_curve(sgeo, sfn_r)
        rand_means.append(m)
        rand_r2.append(r2)
    rand_mean = np.nanmean(rand_means, axis=0)
    rand_sd = np.nanstd(rand_means, axis=0)

    delta = real_mean - rand_mean
    result = {
        "hook": args.hook, "k": int(D.shape[0]), "rot": args.rot,
        "real_r2": round(real_r2, 4),
        "null_r2_mean": round(float(np.mean(rand_r2)), 4),
        "null_r2_sd": round(float(np.std(rand_r2)), 4),
        "delta_r2": round(real_r2 - float(np.mean(rand_r2)), 4),
        "bins": [],
    }
    for c, rm, nm, ns, dl, n in zip(CENTERS, real_mean, rand_mean, rand_sd,
                                    delta, cnt):
        if n > 0:
            result["bins"].append({
                "sgeo": round(float(c), 3), "n": int(n),
                "real_sfn": None if rm != rm else round(float(rm), 4),
                "null_sfn": None if nm != nm else round(float(nm), 4),
                "null_sd": None if ns != ns else round(float(ns), 4),
                "delta": None if dl != dl else round(float(dl), 4),
            })

    out_path = args.out or f"null_{args.hook}_k{args.k}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{args.hook}  real_R2={result['real_r2']}  "
          f"null_R2={result['null_r2_mean']}±{result['null_r2_sd']}  "
          f"delta_R2={result['delta_r2']}")
    print("  sgeo   n        real    null(±sd)      delta")
    for b in result["bins"]:
        if b["n"] >= 20:
            star = "  <-- excess" if (b["delta"] or 0) > 3 * (b["null_sd"] or 1) \
                else ""
            print(f"  {b['sgeo']:+.2f} {b['n']:>8}   {b['real_sfn']:+.3f}   "
                  f"{b['null_sfn']:+.3f}(±{b['null_sd']:.3f})  "
                  f"{b['delta']:+.3f}{star}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
