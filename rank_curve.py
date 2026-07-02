"""
Rank-based locality curve -- the concentrated-geometry instrument.

The binned-R2 metric (locality.py) is variance-weighted over ALL pairs, so
when 94%+ of real SAE pairs are mutually near-orthogonal, a real but thin
"halo" of related neighbours contributes ~zero variance and reads as NULL.

This instrument asks the question per-feature instead:
    For each feature, how functionally similar is it to its rank-k NEAREST
    geometric neighbour (k = 1, 2, 4, ... 256)? And to its geometric ANTIPODE
    (the most-opposite feature)? Compared to a random-pair baseline?

Reading the output:
  - halo that decays to baseline within small k  -> LOCAL structure only
  - elevated |Sfn| at the antipode               -> genuine far/global structure
  - flat at baseline everywhere                  -> geometry uninformative (NULL)

Calibration on the synthetic ground-truth worlds (--synthetic local|global|null)
shows the expected signatures before you trust the real-data read.

Usage:
  python3 rank_curve.py --hook blocks.8.hook_resid_pre --k 4000
  python3 rank_curve.py --synthetic global
"""

import argparse
import json
import numpy as np

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def full_sims_from_vectors(D, F):
    def cos(X):
        Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        with np.errstate(all="ignore"):
            return (Xn @ Xn.T).astype(np.float64)
    return cos(D), cos(F)


def rank_analysis(Sgeo, Sfn):
    n = Sgeo.shape[0]
    np.fill_diagonal(Sgeo, -np.inf)      # exclude self from neighbours
    order = np.argsort(-Sgeo, axis=1)    # descending geometric similarity

    rows = np.arange(n)
    out = {"n": n, "ranks": [], "baseline_sfn": None, "antipode": {}}

    iu = np.triu_indices(n, k=1)
    baseline = float(Sfn[iu].mean())
    out["baseline_sfn"] = round(baseline, 4)

    for k in RANKS:
        if k >= n:
            break
        nb = order[:, k - 1]
        sfn_k = Sfn[rows, nb]
        sgeo_k = Sgeo[rows, nb]
        out["ranks"].append({
            "k": k,
            "mean_sfn": round(float(sfn_k.mean()), 4),
            "sem_sfn": round(float(sfn_k.std(ddof=1) / np.sqrt(n)), 4),
            "mean_sgeo": round(float(sgeo_k.mean()), 4),
        })

    # order[:, -1] is the self row (diagonal = -inf sorts last in descending
    # order), so the true most-negative neighbour is second-to-last
    anti = order[:, -2]
    sfn_a = Sfn[rows, anti]
    sgeo_a = Sgeo[rows, anti]
    out["antipode"] = {
        "mean_sfn": round(float(sfn_a.mean()), 4),
        "sem_sfn": round(float(sfn_a.std(ddof=1) / np.sqrt(n)), 4),
        "mean_sgeo": round(float(sgeo_a.mean()), 4),
        # fraction of features whose antipode is functionally OPPOSITE
        # (beyond what random pairs show)
        "frac_sfn_below_-0.2": round(float((sfn_a < -0.2).mean()), 4),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default=None)
    ap.add_argument("--k", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", choices=["local", "global", "null"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.synthetic:
        import worlds
        gen = {"local": worlds.world_local, "global": worlds.world_global,
               "null": worlds.world_null}[args.synthetic]
        D, F = gen(seed=args.seed)
        Sgeo, Sfn = full_sims_from_vectors(D, F)
        label = f"synthetic_{args.synthetic}"
    else:
        from run_layer import fetch, DEAD_LOG_SPARSITY
        W_dec, log_sparsity, wte, ln_w = fetch(args.hook)
        alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
        rng = np.random.default_rng(args.seed)
        if args.k < alive.size:
            alive = np.sort(rng.choice(alive, size=args.k, replace=False))
        D = W_dec[alive]
        with np.errstate(all="ignore"):
            Ds = (D - D.mean(axis=1, keepdims=True)) * ln_w
            M = wte.T @ wte
            G = Ds @ M @ Ds.T
            fnorm = np.sqrt(np.clip(np.diag(G), 1e-12, None))
            Sfn = (G / np.outer(fnorm, fnorm)).astype(np.float64)
            Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True),
                             1e-12, None)
            Sgeo = (Dn @ Dn.T).astype(np.float64)
        label = args.hook

    res = {"label": label, "seed": args.seed, **rank_analysis(Sgeo, Sfn)}
    out_path = args.out or f"rank_{label}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"{label}  baseline Sfn = {res['baseline_sfn']}")
    for r in res["ranks"]:
        print(f"  rank {r['k']:>4}: Sfn {r['mean_sfn']:+.3f} ± {r['sem_sfn']:.3f}"
              f"   (Sgeo {r['mean_sgeo']:+.3f})")
    a = res["antipode"]
    print(f"  antipode : Sfn {a['mean_sfn']:+.3f} ± {a['sem_sfn']:.3f}"
          f"   (Sgeo {a['mean_sgeo']:+.3f}, frac<-0.2: {a['frac_sfn_below_-0.2']})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
