"""
DECISIVE TEST (judge's priority #1).

Skeptic 1 killed the raw causal signal: at 1-6% injection the causal proxy is a
linear map F = alpha*J*d, so Sfn = cos(d) under M = J^T J, and a spectrum-matched
random map reproduces it (delta_R2 vs spectrum null = 0.0003, z=0.33).

Skeptic 2 found that DEFLATING the dominant shared logit-shift axis (98.3% of F's
energy) multiplies the coupling 54x -- but only tested it vs the weak permutation
null. The open question: does the deflated coupling survive the SPECTRUM null?

This script builds the linear-regime function embedding
    Y_real = (D @ evecs(M)) * sqrt(eig(M))      (Gram = D M D^T, == measured F)
and the spectrum-matched null
    Y_rand = (D @ Q) * sqrt(eig(M))             (same eigenvalue spectrum, random
                                                 eigenvectors Q)
then, for each deflation rank r, removes the top-r SVD components of Y and asks
whether real R2(Sgeo->Sfn) exceeds the random-map R2.

Interpretation:
  real >> random after deflation  -> the sub-dominant coupling depends on J's
      real eigenVECTORS (network-specific), not just its spectrum -> REAL finding.
  real ~= random after deflation  -> even the deflated signal is a spectrum
      artifact -> the whole line of inquiry is dead.
"""

import numpy as np
from run_layer import fetch, DEAD_LOG_SPARSITY
from null_control import binned_curve

HOOK = "blocks.8.hook_resid_pre"
K, SEED, N_ROT = 200, 0, 10
DEFLATE_RANKS = [0, 1, 3, 10]


def sfn_r2(Y, sgeo, iu, r):
    if r > 0:
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        s = s.copy()
        s[:r] = 0.0
        Y = (U * s) @ Vt
    Yn = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(all="ignore"):
        Sfn = Yn @ Yn.T
    _, _, r2 = binned_curve(sgeo, Sfn[iu])
    return r2


def main():
    W_dec, log_sp, _, _ = fetch(HOOK)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(alive, size=K, replace=False))
    D = W_dec[idx].astype(np.float64)                      # K x 768

    Jt = np.load("Jt_L8_c8.npy")                           # 768 x V  (= J^T)
    M = Jt @ Jt.T                                          # 768 x 768
    evals, evecs = np.linalg.eigh(M)
    evals = np.clip(evals, 0, None)
    sqrtL = np.sqrt(evals)

    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        Sgeo = Dn @ Dn.T
    iu = np.triu_indices(K, 1)
    sgeo = Sgeo[iu]

    Y_real = (D @ evecs) * sqrtL                           # Gram = D M D^T

    print(f"layer 8, k={K}, spectrum-matched null with {N_ROT} rotations")
    print(f"M top-5 eig fraction: "
          f"{(np.sort(evals)[::-1][:5] / evals.sum()).round(4)}")
    print(f"\n{'deflate_r':>9} {'real_R2':>9} {'null_R2':>16} "
          f"{'delta':>9} {'z':>7}   verdict")
    for r in DEFLATE_RANKS:
        real = sfn_r2(Y_real, sgeo, iu, r)
        rand = []
        for _ in range(N_ROT):
            Q, _ = np.linalg.qr(rng.standard_normal((D.shape[1], D.shape[1])))
            rand.append(sfn_r2((D @ Q) * sqrtL, sgeo, iu, r))
        rmean, rsd = float(np.mean(rand)), float(np.std(rand))
        delta = real - rmean
        z = delta / (rsd + 1e-9)
        tag = "network-specific" if z > 3 else ("n.s." if abs(z) < 3 else "below-null")
        print(f"{r:>9} {real:>9.4f} {rmean:>9.4f}±{rsd:>5.4f} "
              f"{delta:>+9.4f} {z:>7.1f}   {tag}")


if __name__ == "__main__":
    main()
