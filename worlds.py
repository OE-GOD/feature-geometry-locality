"""
Synthetic ground-truth worlds with KNOWN local/global structure.

These exist to validate that locality_metric() actually reports what we
think it does BEFORE we spend GPU time on a real SAE. Each generator returns
(D, F): geometry and function vectors.

We use a seeded numpy Generator throughout so the validation is
deterministic and reproducible.
"""

import numpy as np


def _unit(rng, shape):
    x = rng.standard_normal(shape)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def world_local(n=500, d=64, v=64, n_blobs=25, noise=0.15, seed=0):
    """
    LOCAL world: features cluster into blobs; each blob has its own function.

    Geometrically-close features (same blob) are functionally similar;
    far features are unrelated. A feature's role is fully determined by its
    local neighbourhood -> the metric should say LOCAL.
    """
    rng = np.random.default_rng(seed)
    centers = _unit(rng, (n_blobs, d))
    blob_fn = _unit(rng, (n_blobs, v))
    assign = rng.integers(0, n_blobs, size=n)
    D = centers[assign] + noise * rng.standard_normal((n, d))
    F = blob_fn[assign] + noise * rng.standard_normal((n, v))
    return D, F


def world_global(n=500, d=64, v=64, noise=0.03, seed=0):
    """
    GLOBAL world: features come in antipodal pairs (D and -D).

    A pair shares an axis with a random function g; the two ends have
    OPPOSITE function (+g and -g). Crucially, two features that are
    geometrically CLOSE (different axes that happen to point similarly)
    have INDEPENDENT functions -> the local neighbourhood is uninformative.
    The only functional structure is the antipodal (farthest-possible)
    relationship -> the metric should say GLOBAL.
    """
    rng = np.random.default_rng(seed)
    n_axes = n // 2
    axes = _unit(rng, (n_axes, d))
    axis_fn = _unit(rng, (n_axes, v))
    D = np.empty((2 * n_axes, d))
    F = np.empty((2 * n_axes, v))
    D[0::2] = axes + noise * rng.standard_normal((n_axes, d))
    D[1::2] = -axes + noise * rng.standard_normal((n_axes, d))
    F[0::2] = axis_fn + noise * rng.standard_normal((n_axes, v))
    F[1::2] = -axis_fn + noise * rng.standard_normal((n_axes, v))
    return D, F


def world_null(n=500, d=64, v=64, seed=0):
    """
    NULL world: geometry and function are independent random directions.
    Geometry explains nothing about function -> the metric should say NULL.
    """
    rng = np.random.default_rng(seed)
    return _unit(rng, (n, d)), _unit(rng, (n, v))
