"""
Core metric: is the geometry->function relationship LOCAL or GLOBAL?

Motivation (Sharkey et al. 2025, "Open Problems in Mechanistic
Interpretability", Sec 2.1.2c "SDL leaves feature geometry unexplained"):

    If understanding the GLOBAL geometry (all-to-all relationships) of
    features is essential, this poses a fundamental problem for SDL/SAEs.
    But if only LOCAL geometric relationships need to be understood, a
    'bag of features' approach may still be feasible.

Operationalization
------------------
We have, for N features:
    D : (N, d_model)  decoder directions  -> the GEOMETRY (positions)
    F : (N, v)        function vectors     -> how the model USES each feature
                                              (e.g. direct logit effect)

For every feature pair (i, j) we compute:
    Sgeo(i,j) = cos(D_i, D_j)   how geometrically close they are
    Sfn (i,j) = cos(F_i, F_j)   how functionally similar they are

Then we ask: WHERE along the Sgeo axis does functional similarity get
explained?
    - If Sfn is predicted by Sgeo mostly through NEAR pairs (Sgeo high),
      then a feature's role is set by its geometric neighbours -> LOCAL,
      bag-of-features survives.
    - If the signal lives in FAR pairs (Sgeo very negative / antipodal),
      then you must know an all-to-all relationship to a distant feature
      -> GLOBAL, trouble for SDL.
    - If Sgeo explains ~none of Sfn -> geometry doesn't explain function
      here (NULL); the question doesn't even apply.

This is deliberately a descriptive, training-free metric: no fitted model,
no cross-validation leakage, cheap enough to run on real SAEs.
"""

import numpy as np

R2_NULL_THRESHOLD = 0.05     # below this, geometry doesn't explain function
LOCALITY_LOCAL = 0.60        # >= this -> LOCAL
LOCALITY_GLOBAL = 0.40       # <= this -> GLOBAL   (between the two -> MIXED)

NEAR_CUT = 0.5               # Sgeo >= NEAR_CUT counts as a near pair
FAR_CUT = -0.5               # Sgeo <= FAR_CUT counts as a far pair
MID_CUT = 0.25               # |Sgeo| < MID_CUT is the neutral baseline


def _unit_rows(X):
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def pair_similarities(D, F):
    """Return upper-triangle pairwise (Sgeo, Sfn) arrays."""
    Dn, Fn = _unit_rows(D), _unit_rows(F)
    with np.errstate(all="ignore"):  # Accelerate BLAS emits spurious warnings
        Sgeo_full = Dn @ Dn.T
        Sfn_full = Fn @ Fn.T
    iu = np.triu_indices(D.shape[0], k=1)
    return Sgeo_full[iu], Sfn_full[iu]


def _region_mean(sfn, mask):
    return float(np.mean(sfn[mask])) if np.any(mask) else np.nan


def locality_metric(D, F, nbins=20):
    """
    Compute the locality diagnostic from raw geometry/function vectors.

    Returns a dict:
        r2            : variance of Sfn explained by Sgeo (piecewise-constant)
        near_strength : how much functional similarity NEAR pairs carry
                        relative to neutral pairs
        far_strength  : same, for FAR pairs
        locality      : |near| / (|near| + |far|)  in [0,1]; 1=local, 0=global
        verdict       : 'LOCAL' | 'GLOBAL' | 'MIXED' | 'NULL'
        curve         : (bin_centers, bin_mean_sfn, bin_count) for plotting
    """
    sgeo, sfn = pair_similarities(D, F)
    res = locality_from_pairs(sgeo, sfn, nbins=nbins)
    res["n_features"] = D.shape[0]
    return res


def locality_from_pairs(sgeo, sfn, nbins=20):
    """Same diagnostic, from precomputed pairwise similarities (upper triangle)."""

    # --- R2 of a piecewise-constant fit Sfn ~ bin(Sgeo) ---
    edges = np.linspace(-1.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(sgeo, edges) - 1, 0, nbins - 1)
    bin_mean = np.full(nbins, np.nan)
    bin_count = np.zeros(nbins)
    for b in range(nbins):
        m = idx == b
        bin_count[b] = m.sum()
        if bin_count[b] > 0:
            bin_mean[b] = sfn[m].mean()
    pred = bin_mean[idx]
    overall = sfn.mean()
    ss_res = np.sum((sfn - pred) ** 2)
    ss_tot = np.sum((sfn - overall) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # --- where does the signal live: near vs far end of the Sgeo axis ---
    baseline = _region_mean(sfn, np.abs(sgeo) < MID_CUT)
    near = _region_mean(sfn, sgeo >= NEAR_CUT)
    far = _region_mean(sfn, sgeo <= FAR_CUT)
    if np.isnan(baseline):
        baseline = overall
    near_strength = 0.0 if np.isnan(near) else abs(near - baseline)
    far_strength = 0.0 if np.isnan(far) else abs(far - baseline)

    denom = near_strength + far_strength
    locality = float(near_strength / denom) if denom > 1e-9 else np.nan

    # --- verdict ---
    if r2 < R2_NULL_THRESHOLD or np.isnan(locality):
        verdict = "NULL"
    elif locality >= LOCALITY_LOCAL:
        verdict = "LOCAL"
    elif locality <= LOCALITY_GLOBAL:
        verdict = "GLOBAL"
    else:
        verdict = "MIXED"

    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "r2": r2,
        "near_strength": near_strength,
        "far_strength": far_strength,
        "locality": locality,
        "verdict": verdict,
        "curve": (centers, bin_mean, bin_count),
        "n_features": None,
        "n_pairs": sgeo.size,
    }


def format_result(name, res):
    loc = res["locality"]
    loc_s = "  nan" if loc != loc else f"{loc:5.2f}"
    return (
        f"{name:<10} verdict={res['verdict']:<7} "
        f"R2={res['r2']:5.2f}  locality={loc_s}  "
        f"near={res['near_strength']:.2f} far={res['far_strength']:.2f}  "
        f"(N={res['n_features']}, pairs={res['n_pairs']})"
    )
