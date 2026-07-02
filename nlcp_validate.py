"""
Slice 3: synthetic ground-truth validation for the nonlinear-coupling metric.

Question: which (null, deflation) configuration correctly separates genuine
geometry-specific structure from the rank/deflation artifact that made deflate-3
look positive on real data? We must trust a config on KNOWN worlds before we
trust any z-score on the real ablation effects.

Three worlds (function matrix F, K x v, over shared geometry D):
  POSITIVE       F = y @ W_aligned^T : the map's big singular directions align
                 with the geometry's high-variance directions -> genuine extra
                 geometry coupling. MUST be flagged (REAL > NULL).
  SPECTRUM_NULL  F = y @ W_rand^T    : a generic spectrum-matched linear map of
                 geometry. MUST clear (REAL ~ NULL).
  HIGH_RANK_NULL F = big shared axis + small generic-map + geometry-independent
                 high-rank noise. MUST clear -- but a null that is low-rank
                 (concentrated unembedding spectrum) gets gutted by deflation
                 while F's noise persists, so a bad config FALSE-POSITIVES here.
                 This reproduces the real-data deflate-3 confound.

Two candidate nulls:
  spectrum-map : Y = (y @ Q) * sqrtL(M=W^T W)         [current, decisive_test-style]
  rank-matched : Y = U_B * sqrt(eig(F F^T)), U_B geometry-derived random
                 -> SAME spectrum as the real F, so deflation cannot advantage rank.

A config PASSES iff on all three worlds it reports POSITIVE as network-specific
and both NULL worlds as not-significant.
"""

import numpy as np

from nlcp_compare import cos_rows, coupling_r2

D_MODEL, VOCAB, K = 64, 128, 150
DEFLATE = [0, 1, 3]
ROT = 12
Z_HIT = 3.0


def concentrated_spectrum(n, decay=0.6, scale=20.0):
    return scale * decay ** np.arange(n)          # few large, long small tail


def orthocols(rng, rows, cols):
    Q, _ = np.linalg.qr(rng.standard_normal((rows, cols)))
    return Q[:, :cols]


def build_worlds(rng):
    d, v = D_MODEL, VOCAB
    sv = concentrated_spectrum(min(d, v))         # "unembedding" spectrum
    D = rng.standard_normal((K, d))
    y = D - D.mean(axis=1, keepdims=True)

    # canonical unembedding W (v x d) with spectrum sv -> M = W^T W, sqrtL
    A = orthocols(rng, v, min(d, v))
    B = orthocols(rng, d, min(d, v))
    W = (A * sv) @ B.T                             # v x d
    M = W.T @ W
    sqrtL = np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None))

    # geometry's high-variance directions
    _, _, Vt_y = np.linalg.svd(y, full_matrices=False)   # Vt_y: (min(K,d), d)
    Vy = Vt_y.T                                    # d x r

    # POSITIVE: map's big singular dirs = geometry's big dirs
    r = Vy.shape[1]
    W_pos = (orthocols(rng, v, r) * sv[:r]) @ Vy.T
    F_pos = y @ W_pos.T

    # SPECTRUM_NULL: generic spectrum-matched random map
    A2, B2 = orthocols(rng, v, min(d, v)), orthocols(rng, d, min(d, v))
    W_rand = (A2 * sv) @ B2.T
    F_spec = y @ W_rand.T

    # HIGH_RANK_NULL: dominant shared axis + small generic map + hi-rank noise
    s0 = rng.standard_normal(v); s0 /= np.linalg.norm(s0)
    shared = np.outer(rng.standard_normal(K) * 40.0, s0)
    noise = rng.standard_normal((K, v)) * 1.5
    F_hrn = shared + 0.3 * F_spec + noise

    worlds = {"POSITIVE": F_pos, "SPECTRUM_NULL": F_spec, "HIGH_RANK_NULL": F_hrn}
    return D, y, sqrtL, worlds


def spectrum_map_null(y, sqrtL, sgeo, iu, r, rng):
    dim = y.shape[1]
    out = [coupling_r2((y @ orthocols(rng, dim, dim)) * sqrtL, sgeo, iu, r)
           for _ in range(ROT)]
    return float(np.mean(out)), float(np.std(out))


def rank_matched_null(y, F, sgeo, iu, r, rng):
    K_, d = y.shape
    lam = np.clip(np.linalg.eigvalsh(F @ F.T)[::-1], 0, None)   # F's spectrum, desc
    sqrt_lam = np.sqrt(lam)
    out = []
    for _ in range(ROT):
        B = y @ rng.standard_normal((d, K_))       # geometry-derived random K x K
        w, U = np.linalg.eigh(B @ B.T)
        U = U[:, ::-1]                              # eigenvectors, desc
        out.append(coupling_r2(U * sqrt_lam, sgeo, iu, r))
    return float(np.mean(out)), float(np.std(out))


def main():
    rng = np.random.default_rng(0)
    D, y, sqrtL, worlds = build_worlds(rng)
    iu = np.triu_indices(K, 1)
    sgeo = cos_rows(D)[iu]

    for null_name, null_fn in [("spectrum-map", "spec"), ("rank-matched", "rank")]:
        print(f"\n===== NULL = {null_name} =====")
        print(f"{'world':<15} {'defl':>4} {'REAL':>7} {'NULL':>15} {'z':>7}  verdict")
        verdicts = {}
        for wname, F in worlds.items():
            for r in DEFLATE:
                real = coupling_r2(F, sgeo, iu, r)
                if null_fn == "spec":
                    nmean, nsd = spectrum_map_null(y, sqrtL, sgeo, iu, r, rng)
                else:
                    nmean, nsd = rank_matched_null(y, F, sgeo, iu, r, rng)
                z = (real - nmean) / (nsd + 1e-9)
                hit = z > Z_HIT
                verdicts.setdefault(wname, {})[r] = hit
                tag = "network-specific" if hit else ("below" if z < -Z_HIT else "n.s.")
                print(f"{wname:<15} {r:>4} {real:>7.3f} {nmean:>8.3f}±{nsd:>5.3f} "
                      f"{z:>7.1f}  {tag}")
        # a deflation level PASSES if POSITIVE hits and both NULLs don't
        print("  pass by deflation level (POS flagged, both NULLs cleared):")
        for r in DEFLATE:
            ok = (verdicts["POSITIVE"][r] and not verdicts["SPECTRUM_NULL"][r]
                  and not verdicts["HIGH_RANK_NULL"][r])
            print(f"    deflate {r}: {'PASS' if ok else 'fail'}")


if __name__ == "__main__":
    main()
