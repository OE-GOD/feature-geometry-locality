"""
Attack: is the local interaction component (a) general across layers/seeds vs L8/seed0
cherry-pick, (b) big enough to matter given the 99% global mode, (c) local-SUFFICIENT
(geometric neighbors predict interaction partners) or only weakly-local-flavored?

Computes, for real and spectrum-matched-random directions, across layers x seeds:
  - raw corr(interaction, cos), controlled corr (residualize per-feature interactivity)
  - REAL-minus-RANDOM controlled corr (the network-specific local signal)
  - EFFECT SIZE: R^2 of interaction explained by global-interactivity model,
    by geometry alone, and INCREMENTAL R^2 that geometry adds on top of global.
  - PREDICTIVE: for each feature, does its geometric-neighbor ranking recover its
    true top interaction partners better than chance, vs the GLOBAL predictor, vs
    the full-information upper bound (precision@k, Spearman).
"""
import argparse, json, numpy as np, torch
from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import build_pile_contexts


def mlp_sublayer(block, x):
    return block.mlp(block.ln_2(x))


def base_points(layer, contexts, model, n_base, rng):
    block = model.transformer.h[layer]
    rec = []
    def pre(_m, a):
        rec.append(a[0].detach()); return None
    h = block.register_forward_pre_hook(pre)
    with torch.no_grad():
        model(contexts)
    h.remove()
    H = rec[0].reshape(-1, rec[0].shape[-1])
    pick = rng.choice(H.shape[0], size=min(n_base, H.shape[0]), replace=False)
    return H[pick]


def compute_interaction(layer, k, scale, n_base, seed, contexts, model, use_random):
    hook = f"blocks.{layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(alive, size=min(k, alive.size), replace=False))
    Wsel = W_dec[idx].astype(np.float32)
    if use_random:
        g = rng.standard_normal((idx.size, Wsel.shape[1])).astype(np.float32)
        Wsel = g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(
            Wsel, axis=1, keepdims=True)
    H0 = base_points(layer, contexts, model, n_base, rng)
    block = model.transformer.h[layer]
    D = torch.from_numpy(Wsel).float() * scale
    K = D.shape[0]
    iu = np.triu_indices(K, 1)
    ii, jj = iu
    inter = np.zeros(len(ii), dtype=np.float64)
    with torch.no_grad():
        pair_writes = D[ii] + D[jj]
        for b in range(H0.shape[0]):
            h0 = H0[b:b+1]
            f0 = mlp_sublayer(block, h0)
            f1 = mlp_sublayer(block, h0 + D)
            fp = mlp_sublayer(block, h0 + pair_writes)
            resid = fp - f1[ii] - f1[jj] + f0
            inter += resid.norm(dim=-1).double().numpy()
    inter /= H0.shape[0]
    Dn = Wsel / np.linalg.norm(Wsel, axis=1, keepdims=True)
    sgeo = (Dn @ Dn.T)[iu]
    return inter, sgeo, iu, K


def r2(y, X):
    """OLS R^2 of y on design X (already includes intercept col)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var()


def analyze(inter, sgeo, iu, K):
    ii, jj = iu
    I = np.zeros((K, K)); I[iu] = inter; I = I + I.T
    act = I.sum(1) / (K - 1)
    fs, fp = act[ii] + act[jj], act[ii] * act[jj]
    one = np.ones_like(inter)

    raw = np.corrcoef(inter, sgeo)[0, 1]
    # controlled corr (residualize interactivity, correlate residual w/ geometry)
    Xg = np.stack([fs, fp, one], 1)
    inter_res = inter - Xg @ np.linalg.lstsq(Xg, inter, rcond=None)[0]
    ctrl = np.corrcoef(inter_res, sgeo)[0, 1]

    sv = np.linalg.svd(I, compute_uv=False)
    svd_energy = sv[0]**2 / (sv**2).sum()

    # ---- EFFECT SIZE ----
    r2_geom_only = r2(inter, np.stack([sgeo, one], 1))
    r2_global = r2(inter, Xg)                                   # per-feature interactivity
    r2_global_geom = r2(inter, np.stack([fs, fp, sgeo, one], 1))
    r2_incremental = r2_global_geom - r2_global                # geometry ON TOP of global
    return dict(raw=raw, ctrl=ctrl, svd_energy=svd_energy,
                r2_geom_only=r2_geom_only, r2_global=r2_global,
                r2_global_geom=r2_global_geom, r2_incremental=r2_incremental,
                act=act, I=I)


def predictive(I, sgeo, iu, K, m_frac=0.1):
    """For each feature, does its geometric-neighbor set recover its true top
    interaction partners? Compare precision@m for: geometry, global(act), chance,
    and the full-info upper bound (=1)."""
    ii, jj = iu
    Sgeo = np.zeros((K, K)); Sgeo[iu] = sgeo; Sgeo = Sgeo + Sgeo.T
    np.fill_diagonal(Sgeo, -np.inf)
    Iw = I.copy(); np.fill_diagonal(Iw, -np.inf)
    act = I.sum(1) / (K - 1)
    m = max(1, int(round((K - 1) * m_frac)))
    chance = m / (K - 1)
    prec_geom, prec_glob, spear_geom, spear_glob = [], [], [], []
    for i in range(K):
        true_top = set(np.argsort(Iw[i])[::-1][:m])
        geom_top = set(np.argsort(Sgeo[i])[::-1][:m])
        # global predictor: partners with highest per-feature interactivity
        actp = act.copy(); actp[i] = -np.inf
        glob_top = set(np.argsort(actp)[::-1][:m])
        prec_geom.append(len(true_top & geom_top) / m)
        prec_glob.append(len(true_top & glob_top) / m)
        # per-row spearman of predictor vs true interaction (exclude self)
        oth = [j for j in range(K) if j != i]
        iv = Iw[i, oth]
        def sp(x):
            a = np.argsort(np.argsort(x)); b = np.argsort(np.argsort(iv))
            return np.corrcoef(a, b)[0, 1]
        spear_geom.append(sp(Sgeo[i, oth]))
        spear_glob.append(sp(act[oth]))
    return dict(m=m, chance=chance,
                prec_geom=float(np.mean(prec_geom)),
                prec_glob=float(np.mean(prec_glob)),
                spear_geom=float(np.mean(spear_geom)),
                spear_glob=float(np.mean(spear_glob)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok, model = load_gpt2()
    contexts = build_pile_contexts(tok, args.docs, args.seq, args.docs * args.seq)

    rows = []
    for layer in args.layers:
        for seed in args.seeds:
            rec = {"layer": layer, "seed": seed}
            for tag, use_rand in [("real", False), ("rand", True)]:
                inter, sgeo, iu, K = compute_interaction(
                    layer, args.k, args.scale, args.n_base, seed, contexts, model, use_rand)
                a = analyze(inter, sgeo, iu, K)
                rec[tag] = {kk: float(vv) for kk, vv in a.items()
                            if kk not in ("act", "I")}
                if tag == "real":
                    rec[tag]["pred"] = predictive(a["I"], sgeo, iu, K)
            rec["real_minus_rand_ctrl"] = rec["real"]["ctrl"] - rec["rand"]["ctrl"]
            rec["real_minus_rand_incR2"] = rec["real"]["r2_incremental"] - rec["rand"]["r2_incremental"]
            rows.append(rec)
            p = rec["real"]["pred"]
            print(f"L{layer} s{seed}: real raw={rec['real']['raw']:+.3f} ctrl={rec['real']['ctrl']:+.3f} "
                  f"| rand ctrl={rec['rand']['ctrl']:+.3f} | R-R ctrl={rec['real_minus_rand_ctrl']:+.3f} "
                  f"| svd={rec['real']['svd_energy']:.3f} "
                  f"| R2geom={rec['real']['r2_geom_only']:.4f} R2glob={rec['real']['r2_global']:.4f} "
                  f"incR2={rec['real']['r2_incremental']:.4f} "
                  f"| pred@{p['m']}: geom={p['prec_geom']:.3f} glob={p['prec_glob']:.3f} chance={p['chance']:.3f} "
                  f"spr_geom={p['spear_geom']:+.3f} spr_glob={p['spear_glob']:+.3f}")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
