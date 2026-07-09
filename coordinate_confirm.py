"""
Confirmatory cyclic-coordinate interaction test.

Frozen by PREREG_coordinate.md.  This file intentionally keeps the confirmatory
measurement self-contained while reusing the pilot's feature selection context
builder and interaction kernel.
"""

import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from concept_geometry import FAMILIES, concept_residual
from feature_interaction import base_points
from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from shard_pilot import ALPHA, CHUNK, HOOK, LAYER
from shard_pilot import build_contexts_safe, compute_seed_all, month_features


torch.set_num_threads(8)


def rank_average(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx = rank_average(x[ok])
    ry = rank_average(y[ok])
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    return float(np.sum(rx * ry) / den) if den > 0 else np.nan


def family_features(name, layer, W_dec, W_enc, b_enc, b_dec, log_sp, tok, model):
    if name == "months":
        return month_features(layer, W_dec, W_enc, b_enc, b_dec, log_sp, tok, model)
    words = FAMILIES[name]
    R = np.stack([concept_residual(layer, w, tok, model) for w in words])
    with np.errstate(all="ignore"):
        A = np.maximum(
            (R.astype(np.float64) - b_dec.astype(np.float64))
            @ W_enc.astype(np.float64)
            + b_enc.astype(np.float64),
            0.0,
        )
    alive = log_sp > DEAD_LOG_SPARSITY
    sel = []
    for m in range(len(words)):
        others = A[np.arange(len(words)) != m].mean(0)
        score = A[m] - others
        score[~alive] = -np.inf
        sel.append(int(np.argmax(score)))
    return words, np.array(sel, dtype=np.int64)


def circular_distances(n):
    pos = np.arange(n)
    d = np.abs(pos[:, None] - pos[None, :])
    return np.minimum(d, n - d).astype(np.float64)


def sym_raw(I):
    return (I.astype(np.float64) + I.astype(np.float64).T) / 2.0


def t1_spearman(S):
    iu = np.triu_indices(S.shape[0], 1)
    return spearman(S[iu], circular_distances(S.shape[0])[iu])


def row_first_harmonic_energy(row):
    row = np.asarray(row, dtype=np.float64)
    row = row[np.isfinite(row)]
    if row.size < 3:
        return np.nan
    row = row - row.mean()
    fft = np.fft.fft(row)
    power = np.abs(fft) ** 2
    den = power[1:].sum()
    if den <= 0:
        return np.nan
    return float((power[1] + power[-1]) / den)


def t2_first_harmonic(S):
    n = S.shape[0]
    vals = []
    for i in range(n):
        cols = (i + np.arange(1, n)) % n
        vals.append(row_first_harmonic_energy(S[i, cols]))
    return float(np.nanmean(vals))


def stats_from_I(I):
    S = sym_raw(I)
    return np.array([t1_spearman(S), t2_first_harmonic(S)], dtype=np.float64)


def permutation_stats(S, n_perm, rng):
    obs_t1 = t1_spearman(S)
    obs_t2 = t2_first_harmonic(S)
    null = np.zeros((n_perm, 2), dtype=np.float64)
    for r in range(n_perm):
        p = rng.permutation(S.shape[0])
        Sp = S[np.ix_(p, p)]
        null[r, 0] = t1_spearman(Sp)
        null[r, 1] = t2_first_harmonic(Sp)
    means = np.nanmean(null, axis=0)
    sds = np.nanstd(null, axis=0)
    z = (np.array([obs_t1, obs_t2], dtype=np.float64) - means) / (sds + 1e-12)
    p_t1 = (np.sum(null[:, 0] <= obs_t1) + 1.0) / (n_perm + 1.0)
    p_t2 = (np.sum(null[:, 1] >= obs_t2) + 1.0) / (n_perm + 1.0)
    return {
        "T1": float(obs_t1),
        "T2": float(obs_t2),
        "z1": float(z[0]),
        "z2": float(z[1]),
        "p1": float(p_t1),
        "p2": float(p_t2),
        "null": null,
        "null_mean": means,
        "null_sd": sds,
    }


def column_normalized(I):
    X = I.astype(np.float64, copy=True)
    col = np.nanmean(X, axis=0)
    with np.errstate(all="ignore", divide="ignore"):
        return X / np.clip(col[None, :], 1e-12, None)


def determinism_gate(I):
    a = stats_from_I(I)
    b = stats_from_I(I)
    return bool(np.all(np.isfinite(a)) and np.max(np.abs(a - b)) <= 1e-9)


def split_half_gate(IA, IB):
    SA = sym_raw(IA)
    SB = sym_raw(IB)
    iu = np.triu_indices(SA.shape[0], 1)
    r = spearman(SA[iu], SB[iu])
    return bool(np.isfinite(r) and r > 0.5), float(r)


def family_verdict(stats, g0, g1):
    if not (g0 and g1):
        return "INDETERMINATE (instrument)"
    if stats["z1"] <= -3.0 and stats["z2"] >= 3.0:
        return "CONFIRM"
    if abs(stats["z1"]) < 2.0 and abs(stats["z2"]) < 2.0:
        return "REFUTE"
    return "INDETERMINATE"


def overall_verdict(verdicts):
    months = verdicts.get("months")
    days = verdicts.get("days")
    if months == "CONFIRM" and days == "CONFIRM":
        return "CONFIRM"
    if months == "CONFIRM" and days != "CONFIRM":
        return "PARTIAL"
    if months == "REFUTE" and days == "REFUTE":
        return "REFUTE"
    return "INDETERMINATE"


def print_stats(prefix, stats):
    print(
        f"{prefix} T1={stats['T1']:+.6f} z1={stats['z1']:+.3f} p1={stats['p1']:.6f} "
        f"T2={stats['T2']:+.6f} z2={stats['z2']:+.3f} p2={stats['p2']:.6f}",
        flush=True,
    )


def run_family(name, words, idx, args, W_dec, tok, model):
    cand_idx = idx.astype(np.int64)
    seed_idx = idx.astype(np.int64)
    cand_pos = {int(v): i for i, v in enumerate(cand_idx)}
    rng31 = np.random.default_rng(31)
    contexts = build_contexts_safe(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, args.bp, rng31)
    block = model.transformer.h[LAYER]

    print(
        f"family={name} words={len(words)} base_points={H0.shape[0]} permutations={args.perm}",
        flush=True,
    )
    print(f"{name}_features {dict((w, int(i)) for w, i in zip(words, idx))}", flush=True)
    IA, IB, bp_times, _D_cand, _D_seed = compute_seed_all(
        block, H0, W_dec, cand_idx, seed_idx, cand_pos, ALPHA, CHUNK
    )
    I = (IA + IB) / 2.0
    S = sym_raw(I)
    g0 = determinism_gate(I)
    g1, split_r = split_half_gate(IA, IB)
    stats = permutation_stats(S, args.perm, np.random.default_rng(61))
    desc = permutation_stats(sym_raw(column_normalized(I)), args.perm, np.random.default_rng(61))
    verdict = family_verdict(stats, g0, g1)

    print(
        f"GATES family={name} G0={g0} G1={g1} split_half_spearman={split_r:+.6f}",
        flush=True,
    )
    print_stats(f"CONFIRMATORY family={name}", stats)
    print_stats(f"DESCRIPTIVE_It family={name}", desc)
    print(f"VERDICT family={name} {verdict}", flush=True)
    return {
        "words": np.array(words, dtype=object),
        "idx": idx.astype(np.int64),
        "I": I,
        "IA": IA,
        "IB": IB,
        "bp_times": bp_times,
        "stats": stats,
        "desc": desc,
        "g0": g0,
        "g1": g1,
        "split_r": split_r,
        "verdict": verdict,
    }


def save_results(path, results, overall):
    payload = {"overall_verdict": np.array(overall, dtype=object)}
    for name, res in results.items():
        prefix = f"{name}_"
        payload[prefix + "words"] = res["words"]
        payload[prefix + "idx"] = res["idx"]
        payload[prefix + "I"] = res["I"]
        payload[prefix + "IA"] = res["IA"]
        payload[prefix + "IB"] = res["IB"]
        payload[prefix + "bp_times"] = res["bp_times"]
        payload[prefix + "stats"] = np.array(
            [res["stats"]["T1"], res["stats"]["T2"], res["stats"]["z1"],
             res["stats"]["z2"], res["stats"]["p1"], res["stats"]["p2"]],
            dtype=np.float64,
        )
        payload[prefix + "null"] = res["stats"]["null"]
        payload[prefix + "null_mean"] = res["stats"]["null_mean"]
        payload[prefix + "null_sd"] = res["stats"]["null_sd"]
        payload[prefix + "desc_stats"] = np.array(
            [res["desc"]["T1"], res["desc"]["T2"], res["desc"]["z1"],
             res["desc"]["z2"], res["desc"]["p1"], res["desc"]["p2"]],
            dtype=np.float64,
        )
        payload[prefix + "gates"] = np.array(
            [float(res["g0"]), float(res["g1"]), res["split_r"]],
            dtype=np.float64,
        )
        payload[prefix + "verdict"] = np.array(res["verdict"], dtype=object)
    np.savez(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bp", type=int, default=16)
    ap.add_argument("--perm", type=int, default=10000)
    args = ap.parse_args()
    if args.smoke:
        args.bp = 2
        args.perm = 200

    t0 = time.monotonic()
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(HOOK)
    tok, model = load_gpt2()
    family_names = ["months"] if args.smoke else ["months", "days"]
    results = {}
    print(f"setup layer={LAYER} alpha={ALPHA} hook={HOOK}", flush=True)
    for name in family_names:
        words, idx = family_features(name, LAYER, W_dec, W_enc, b_enc, b_dec,
                                     log_sp, tok, model)
        results[name] = run_family(name, words, idx, args, W_dec, tok, model)
    overall = overall_verdict({name: res["verdict"] for name, res in results.items()})
    print(f"OVERALL {overall}", flush=True)
    save_results("coordinate_confirm.npz", results, overall)
    print(f"saved coordinate_confirm.npz elapsed_seconds={time.monotonic() - t0:.2f}", flush=True)


if __name__ == "__main__":
    main()
