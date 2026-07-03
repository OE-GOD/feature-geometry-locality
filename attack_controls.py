"""
Attack on the +0.30 headline: is it a real un-masking of a local interaction
component, or an artifact of the specific sum+product residualization?

Alternative, cleaner controls, each run on REAL and on --random:
  (0) raw Pearson corr(inter, cos)
  (1) DOUBLE-CENTER the KxK interaction matrix (remove ALL additive per-feature
      structure), correlate with cos. Two variants:
        (1a) center inter only (full matrix incl. zero diagonal)
        (1b) off-diagonal double-centering of inter only
        (1c) double-center BOTH inter and cos (Mantel-style), off-diagonal
  (2) Spearman rank corr (raw and on double-centered inter)
  (3) proper partial correlation: residualize BOTH inter and cos on the
      per-feature interactivity controls (sum+product), then Pearson.
  Plus the original script's control (residualize inter only) for reference,
  and diagnostics: corr(cos, controls).
All correlations get a feature-label permutation p-value.
"""
import argparse
import numpy as np
import torch
from scipy.stats import rankdata

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


def spearman(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]


def double_center_full(M):
    r = M.mean(1, keepdims=True); c = M.mean(0, keepdims=True); g = M.mean()
    return M - r - c + g


def double_center_offdiag(M):
    """Row/col means computed over off-diagonal entries only (diagonal is a
    structural zero here, not a real self-interaction)."""
    K = M.shape[0]
    mask = ~np.eye(K, dtype=bool)
    row = (M * mask).sum(1) / (K - 1)
    g = M[mask].mean()
    out = M - row[:, None] - row[None, :] + g
    return out


def compute(idx, W_dec, model, layer, contexts, n_base, scale, rng, random):
    Wsel = W_dec[idx].astype(np.float32)
    if random:
        g = rng.standard_normal((idx.size, Wsel.shape[1])).astype(np.float32)
        Wsel = g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(
            Wsel, axis=1, keepdims=True)
    H0 = base_points(layer, contexts, model, n_base, rng)
    block = model.transformer.h[layer]
    D = torch.from_numpy(Wsel).float() * scale
    K = D.shape[0]
    iu = np.triu_indices(K, 1); ii, jj = iu
    inter = np.zeros(len(ii), dtype=np.float64)
    with torch.no_grad():
        pw = D[ii] + D[jj]
        for b in range(H0.shape[0]):
            h0 = H0[b:b + 1]
            f0 = mlp_sublayer(block, h0)
            f1 = mlp_sublayer(block, h0 + D)
            fp = mlp_sublayer(block, h0 + pw)
            resid = fp - f1[ii] - f1[jj] + f0
            inter += resid.norm(dim=-1).double().numpy()
    inter /= H0.shape[0]
    Dn = Wsel / np.linalg.norm(Wsel, axis=1, keepdims=True)
    Sgeo = Dn @ Dn.T
    return inter, Sgeo, iu, K


def perm_p(stat_fn, obs, K, iu, nperm, rng):
    """Two-tailed-ish: fraction of permutations with stat >= obs (right tail)."""
    ge = 0
    for _ in range(nperm):
        p = rng.permutation(K)
        if stat_fn(p) >= obs:
            ge += 1
    return (ge + 1) / (nperm + 1)


def analyze(name, inter, Sgeo, iu, K, nperm, rng):
    ii, jj = iu
    sgeo = Sgeo[iu]
    I = np.zeros((K, K)); I[iu] = inter; I = I + I.T

    # raw
    raw = np.corrcoef(inter, sgeo)[0, 1]
    raw_sp = spearman(inter, sgeo)

    # double-center inter (full, incl zero diag)
    Ic_full = double_center_full(I)[iu]
    dc_full = np.corrcoef(Ic_full, sgeo)[0, 1]
    # double-center inter (off-diagonal)
    Ic_off = double_center_offdiag(I)[iu]
    dc_off = np.corrcoef(Ic_off, sgeo)[0, 1]
    dc_off_sp = spearman(Ic_off, sgeo)
    # double-center BOTH (Mantel-style), off-diagonal
    Sc_off = double_center_offdiag(Sgeo)[iu]
    dc_both = np.corrcoef(Ic_off, Sc_off)[0, 1]

    # original control: residualize inter only on sum+product
    act = I.sum(1) / (K - 1)
    fs, fp = act[ii] + act[jj], act[ii] * act[jj]
    X = np.stack([fs, fp, np.ones_like(fs)], 1)
    inter_r = inter - X @ np.linalg.lstsq(X, inter, rcond=None)[0]
    orig_ctrl = np.corrcoef(inter_r, sgeo)[0, 1]
    # proper partial: residualize BOTH on controls
    sgeo_r = sgeo - X @ np.linalg.lstsq(X, sgeo, rcond=None)[0]
    partial = np.corrcoef(inter_r, sgeo_r)[0, 1]
    partial_sp = spearman(inter_r, sgeo_r)

    # diagnostics: does geometry correlate with the controls?
    c_geo_fs = np.corrcoef(sgeo, fs)[0, 1]
    c_geo_fp = np.corrcoef(sgeo, fp)[0, 1]

    # permutation p-values (permute feature labels of geometry)
    def stat_raw(p): return np.corrcoef(inter, (Sgeo[np.ix_(p, p)])[iu])[0, 1]
    def stat_dc_off(p): return np.corrcoef(Ic_off, (Sgeo[np.ix_(p, p)])[iu])[0, 1]
    def stat_dc_both(p):
        Sp = double_center_offdiag(Sgeo[np.ix_(p, p)])[iu]
        return np.corrcoef(Ic_off, Sp)[0, 1]
    def stat_partial(p):
        Sp = (Sgeo[np.ix_(p, p)])[iu]
        Sp_r = Sp - X @ np.linalg.lstsq(X, Sp, rcond=None)[0]
        return np.corrcoef(inter_r, Sp_r)[0, 1]

    p_raw = perm_p(stat_raw, raw, K, iu, nperm, rng)
    p_dc_off = perm_p(stat_dc_off, dc_off, K, iu, nperm, rng)
    p_dc_both = perm_p(stat_dc_both, dc_both, K, iu, nperm, rng)
    p_partial = perm_p(stat_partial, partial, K, iu, nperm, rng)

    print(f"\n===== {name} =====")
    print(f"  raw Pearson corr(inter,cos)             = {raw:+.4f}  perm_p={p_raw:.3f}")
    print(f"  raw Spearman                            = {raw_sp:+.4f}")
    print(f"  double-center inter (full, +0diag)      = {dc_full:+.4f}")
    print(f"  double-center inter (off-diag) Pearson  = {dc_off:+.4f}  perm_p={p_dc_off:.3f}")
    print(f"  double-center inter (off-diag) Spearman = {dc_off_sp:+.4f}")
    print(f"  double-center BOTH (Mantel, off-diag)   = {dc_both:+.4f}  perm_p={p_dc_both:.3f}")
    print(f"  original ctrl (resid inter only)        = {orig_ctrl:+.4f}")
    print(f"  proper partial (resid BOTH) Pearson     = {partial:+.4f}  perm_p={p_partial:.3f}")
    print(f"  proper partial Spearman                 = {partial_sp:+.4f}")
    print(f"  diag: corr(cos, sum-interactivity)      = {c_geo_fs:+.4f}")
    print(f"  diag: corr(cos, prod-interactivity)     = {c_geo_fp:+.4f}")
    return dict(raw=raw, dc_off=dc_off, dc_both=dc_both, orig=orig_ctrl,
                partial=partial)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()

    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    tok, model = load_gpt2()
    contexts = build_pile_contexts(tok, args.docs, args.seq, args.docs * args.seq)

    # REAL (fresh rng stream per run so base points reproducible-ish)
    r1 = np.random.default_rng(args.seed + 100)
    inter, Sgeo, iu, K = compute(idx, W_dec, model, args.layer, contexts,
                                 args.n_base, args.scale, r1, False)
    analyze("REAL", inter, Sgeo, iu, K, args.perm, r1)

    r2 = np.random.default_rng(args.seed + 100)
    interR, SgeoR, iuR, KR = compute(idx, W_dec, model, args.layer, contexts,
                                     args.n_base, args.scale, r2, True)
    analyze("RANDOM (spectrum-matched)", interR, SgeoR, iuR, KR, args.perm, r2)


if __name__ == "__main__":
    main()
