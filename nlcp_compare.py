"""
Slice 2: does the NONLINEAR on-distribution coupling beat the STRONG null?

The strong null (validated in decisive_test.py): a spectrum-matched random
linear map applied to the real geometry. The linear proxies were refuted because
their geometry->function coupling did NOT exceed this null. The honest test for
the nonlinear ablation-effect F is the same one: does F's coupling to the decoder
geometry exceed a random linear map carrying the unembedding metric's spectrum?

Construction (per feature set of K fired features):
  D    = decoder directions (geometry). Sgeo = cos(D).
  y    = (D - mean) * ln_w                      (LayerNorm-folded geometry)
  M    = wte^T wte,  spectrum sqrtL = sqrt(eig M)
  REAL   function matrix = F (nonlinear ablation effects, from real_effect.py)
  LINEAR function matrix = y @ wte^T            (the refuted direct-logit proxy)
  NULL   embedding       = (y @ Q) * sqrtL,  Q random orthogonal 768x768
                           -> gram y Q diag(eig M) Q^T y^T: same spectrum,
                              random eigenvectors, applied to the real geometry.

Effect matrices are ~99.5% one shared axis, so deflate top-r and score the
residual. REAL > NULL at several sigma after deflation = genuine nonlinear
geometric structure. REAL <= NULL = the negative result holds on-distribution.

NOTE: an earlier version of this script used Y_null = Q*diag(s) (function
randomized independently of geometry) — a label-permutation-strength null that
any smooth map beats. That is the wrong (too weak) null; this uses the strong one.
"""

import argparse
import numpy as np

from run_layer import fetch, DEAD_LOG_SPARSITY
from null_control import binned_curve

DEFLATE = [0, 1, 3]


def cos_rows(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(all="ignore"):
        return Xn @ Xn.T


def coupling_r2(Y, sgeo_pairs, iu, r):
    """Binned R2 of Sfn(Y) vs Sgeo, after deflating top-r SVD components of Y."""
    if r > 0:
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        s = s.copy()
        s[:r] = 0.0
        Y = (U * s) @ Vt
    Sfn = cos_rows(Y)
    return binned_curve(sgeo_pairs, Sfn[iu])[2]


def rank_matched_null(y, F, sgeo, iu, r, rot, rng):
    """Null with F's OWN singular spectrum on geometry-derived random eigenvectors
    (so deflation cannot advantage rank). Validated in nlcp_validate.py."""
    K_, d = y.shape
    sqrt_lam = np.sqrt(np.clip(np.linalg.eigvalsh(F @ F.T)[::-1], 0, None))
    out = []
    for _ in range(rot):
        B = y @ rng.standard_normal((d, K_))
        _, U = np.linalg.eigh(B @ B.T)
        out.append(coupling_r2(U[:, ::-1] * sqrt_lam, sgeo, iu, r))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="real_effect_L8_k200_s0.npz")
    ap.add_argument("--hook", default="blocks.8.hook_resid_pre")
    ap.add_argument("--rot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz)
    F_all, idx_all = d["F"].astype(np.float64), d["idx"]
    keep = np.linalg.norm(F_all, axis=1) > 0
    idx, F_real = idx_all[keep], F_all[keep]
    K = idx.size

    W_dec, log_sparsity, wte, ln_w = fetch(args.hook)
    D = W_dec[idx].astype(np.float64)
    y = (D - D.mean(axis=1, keepdims=True)) * ln_w          # LN-folded geometry
    with np.errstate(all="ignore"):
        M = wte.T @ wte
    sqrtL = np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None))
    F_lin = y @ wte.T                                        # refuted linear proxy

    iu = np.triu_indices(K, 1)
    sgeo = cos_rows(D)[iu]
    rng = np.random.default_rng(args.seed)
    dim = y.shape[1]

    def spectrum_map_null(rr):
        out = [coupling_r2((y @ np.linalg.qr(rng.standard_normal((dim, dim)))[0]) * sqrtL,
                           sgeo, iu, rr) for _ in range(args.rot)]
        return float(np.mean(out)), float(np.std(out))

    print(f"npz={args.npz}  K={K} fired features  rot={args.rot}")
    print(f"top-1 SVD energy of F_real: "
          f"{(np.linalg.svd(F_real, compute_uv=False)[0]**2/(F_real**2).sum()):.4f}")
    print("\nLINEAR proxy = built-in NEGATIVE control (a refuted spectrum artifact).")
    print("Trust a row ONLY where the control is not flagged (z_LIN <= 3); then read z_REAL.\n")
    print(f"{'null':<13} {'defl':>4} {'z_REAL':>7} {'z_LIN(ctrl)':>11}  verdict")
    for null_name in ("spectrum-map", "rank-matched"):
        for r in DEFLATE:
            real = coupling_r2(F_real, sgeo, iu, r)
            lin = coupling_r2(F_lin, sgeo, iu, r)
            if null_name == "spectrum-map":
                nm, ns = spectrum_map_null(r)          # same null for REAL and LIN
                zr, zl = (real - nm) / (ns + 1e-9), (lin - nm) / (ns + 1e-9)
            else:
                nmr, nsr = rank_matched_null(y, F_real, sgeo, iu, r, args.rot, rng)
                nml, nsl = rank_matched_null(y, F_lin, sgeo, iu, r, args.rot, rng)
                zr, zl = (real - nmr) / (nsr + 1e-9), (lin - nml) / (nsl + 1e-9)
            if zl > 3:
                verdict = "UNTRUSTWORTHY (control flagged)"
            elif zr > 3:
                verdict = "REAL network-specific"
            else:
                verdict = "negative (REAL not > null)"
            print(f"{null_name:<13} {r:>4} {zr:>7.1f} {zl:>11.1f}  {verdict}")


if __name__ == "__main__":
    main()
