"""
Confirmatory manifold-shard signature test.

Frozen spec: PREREG_shard.md.  This script intentionally reuses the pilot's
seed-vs-all interaction machinery; only the preregistered sample, statistic,
gates, checkpoint, and verdict logic live here.
"""

import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from concept_geometry import FAMILIES
from feature_interaction import base_points
from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from shard_pilot import (
    ALPHA,
    CHUNK,
    HOOK,
    LAYER,
    build_contexts_safe,
    burned_samples,
    compute_seed_all,
    cos_rows,
    jaccard,
    month_features,
    pair_abs_cos_mean,
    rank_positions,
    top_set,
)


torch.set_num_threads(8)


def pilot_random_seeds(alive, burned, n_random=24):
    rng = np.random.default_rng(23)
    pool = np.array([a for a in alive if int(a) not in burned], dtype=np.int64)
    return np.sort(rng.choice(pool, size=min(n_random, pool.size), replace=False))


def confirm_random_seeds(alive, burned, pilot_idx, month_idx, n_random=24):
    excluded = set(int(x) for x in burned)
    excluded.update(int(x) for x in pilot_idx)
    excluded.update(int(x) for x in month_idx)
    pool = np.array([a for a in alive if int(a) not in excluded], dtype=np.int64)
    rng = np.random.default_rng(29)
    return np.sort(rng.choice(pool, size=min(n_random, pool.size), replace=False))


def bin_masks_for_seed(abs_seed_cos, clone_row):
    valid = (~clone_row) & np.isfinite(abs_seed_cos)
    vals = abs_seed_cos[valid]
    if vals.size == 0:
        return []
    edges = np.linspace(float(vals.min()), float(vals.max()), 11)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    bin_id = np.full(abs_seed_cos.shape[0], -1, dtype=np.int64)
    bin_id[valid] = np.clip(np.digitize(abs_seed_cos[valid], edges) - 1, 0, 9)
    return [valid & (bin_id == b) for b in range(10)]


def eligible_bins(abs_seed_cos, clone, seed_rows):
    out = []
    for s in seed_rows:
        seed_bins = []
        for m in bin_masks_for_seed(abs_seed_cos[s], clone[s]):
            if int(m.sum()) >= 24:
                seed_bins.append(np.where(m)[0].astype(np.int64))
        out.append(seed_bins)
    return out


def delta_components(score, Dn_cand, bins_by_seed, seed_rows, n_subsets, rng):
    components = []
    per_seed = []
    for out_s, s in enumerate(seed_rows):
        seed_num = 0.0
        seed_den = 0.0
        for b, cols in enumerate(bins_by_seed[out_s]):
            vals = score[s, cols]
            ok = np.isfinite(vals)
            cols_ok = cols[ok]
            vals_ok = vals[ok]
            if cols_ok.size < 24:
                continue
            order = np.argsort(vals_ok)[::-1]
            top = cols_ok[order[:8]]
            pp_top = pair_abs_cos_mean(Dn_cand, top)
            refs = np.empty(n_subsets, dtype=np.float64)
            for r in range(n_subsets):
                refs[r] = pair_abs_cos_mean(
                    Dn_cand, rng.choice(cols_ok, size=8, replace=False)
                )
            pp_rand = float(refs.mean())
            delta = float(pp_top - pp_rand)
            w = float(cols_ok.size)
            components.append((int(s), int(b), int(cols_ok.size), pp_top, pp_rand, delta))
            seed_num += w * delta
            seed_den += w
        per_seed.append(seed_num / seed_den if seed_den > 0 else np.nan)
    pooled = float(np.nanmean(np.array(per_seed, dtype=np.float64)))
    return pooled, np.array(components, dtype=np.float64), np.array(per_seed, dtype=np.float64)


def permuted_score(score, bins_by_seed, seed_rows, rng):
    ps = score.copy()
    for out_s, s in enumerate(seed_rows):
        for cols in bins_by_seed[out_s]:
            vals = ps[s, cols].copy()
            finite = np.isfinite(vals)
            shuf = vals[finite].copy()
            rng.shuffle(shuf)
            vals[finite] = shuf
            ps[s, cols] = vals
    return ps


def stat_from_saved(score, Dn_cand, abs_seed_cos, clone, seed_rows, n_subsets, rng_seed):
    bins = eligible_bins(abs_seed_cos, clone, seed_rows)
    rng = np.random.default_rng(rng_seed)
    return delta_components(score, Dn_cand, bins, seed_rows, n_subsets, rng)


def split_half_jaccard(ItA, ItB, clone, seed_rows):
    vals = []
    for s in seed_rows:
        valid = ~clone[s]
        vals.append(jaccard(top_set(ItA[s], valid, 20), top_set(ItB[s], valid, 20)))
    return np.array(vals, dtype=np.float64)


def month_gate(It, clone, month_words, month_idx, seed_idx, cand_pos):
    seed_pos = {int(v): i for i, v in enumerate(seed_idx)}
    month_cand_pos = [cand_pos.get(int(i), -1) for i in month_idx]
    rows = []
    medians = []
    for m, feat in enumerate(month_idx):
        if int(feat) not in seed_pos:
            continue
        s = seed_pos[int(feat)]
        targets = [month_cand_pos[k] for k in range(len(month_idx))
                   if k != m and month_cand_pos[k] >= 0]
        ranks = rank_positions(It[s], targets, ~clone[s])
        med = float(np.median(ranks)) if ranks else np.nan
        medians.append(med)
        rows.append((month_words[m], int(feat), med, ranks))
    return rows, np.array(medians, dtype=np.float64)


def verdict(gates_pass, delta, z):
    if not gates_pass:
        return "INDETERMINATE (instrument gate failure)"
    if z >= 4.0 and delta > 0.0:
        return "CONFIRM H1"
    if z <= 2.0:
        return "REFUTE H1"
    return "INDETERMINATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--bp", type=int, default=16)
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--random-subsets", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 3
        args.bp = 2
        args.perms = 20
        args.random_subsets = 10

    t_start = time.time()
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(HOOK)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    cand_idx = alive[:3000] if args.smoke else alive
    cand_pos = {int(v): i for i, v in enumerate(cand_idx)}
    burned = burned_samples(alive)

    tok, model = load_gpt2()
    month_words, all_month_idx = month_features(
        LAYER, W_dec, W_enc, b_enc, b_dec, log_sp, tok, model
    )
    pilot_idx = pilot_random_seeds(alive, burned, 24)
    confirm_idx_full = confirm_random_seeds(alive, burned, pilot_idx, all_month_idx, 24)
    confirm_idx = confirm_idx_full[:args.seeds]
    month_idx = all_month_idx[:2] if args.smoke else all_month_idx
    seed_idx = np.concatenate([confirm_idx, month_idx]).astype(np.int64)
    seed_names = [f"confirm_{int(i)}" for i in confirm_idx]
    seed_names += [f"month_{w}_{int(i)}" for w, i in zip(month_words, month_idx)]
    n_confirm = len(confirm_idx)
    confirm_rows = np.arange(n_confirm, dtype=np.int64)
    month_rows = np.arange(n_confirm, len(seed_idx), dtype=np.int64)

    print(f"setup layer={LAYER} alpha={ALPHA} hook={HOOK}", flush=True)
    print(f"candidate_pool={len(cand_idx)} alive_total={len(alive)} "
          f"confirm_seeds={n_confirm} month_seeds={len(month_idx)} bp={args.bp}",
          flush=True)
    print(f"confirm_seed_rng=29 pilot_seed_rng=23 base_point_rng=29", flush=True)
    print(f"month_features {dict((w, int(i)) for w, i in zip(month_words, all_month_idx))}",
          flush=True)

    rng29 = np.random.default_rng(29)
    contexts = build_contexts_safe(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, args.bp, rng29)
    block = model.transformer.h[LAYER]
    IA, IB, bp_times, D_cand, D_seed = compute_seed_all(
        block, H0, W_dec, cand_idx, seed_idx, cand_pos, ALPHA, CHUNK
    )

    I = (IA + IB) / 2.0
    denom = np.nanmean(I[:n_confirm], axis=0)
    # Month rows use the same confirmatory-row denominator; G2 tests pipeline
    # recovery without letting month seeds change propensity normalization.
    It = I / np.clip(denom[None, :], 1e-12, None)
    denom_a = np.nanmean(IA[:n_confirm], axis=0)
    denom_b = np.nanmean(IB[:n_confirm], axis=0)
    ItA = IA / np.clip(denom_a[None, :], 1e-12, None)
    ItB = IB / np.clip(denom_b[None, :], 1e-12, None)

    Dn_cand = cos_rows(D_cand).astype(np.float64, copy=False)
    Dn_seed = cos_rows(D_seed).astype(np.float64, copy=False)
    with np.errstate(all="ignore"):
        abs_seed_cos = np.abs(Dn_seed @ Dn_cand.T).astype(np.float64)
    clone = abs_seed_cos > 0.9
    for s, idx in enumerate(seed_idx):
        if int(idx) in cand_pos:
            clone[s, cand_pos[int(idx)]] = True

    real_delta, real_components, per_seed_delta = stat_from_saved(
        It, Dn_cand, abs_seed_cos, clone, confirm_rows, args.random_subsets, 47
    )
    np.savez(
        "shard_confirm.npz",
        candidate_idx=cand_idx.astype(np.int64),
        seed_idx=seed_idx.astype(np.int64),
        seed_names=np.array(seed_names),
        confirm_idx=confirm_idx.astype(np.int64),
        month_idx=month_idx.astype(np.int64),
        I=I.astype(np.float64),
        IA=IA.astype(np.float64),
        IB=IB.astype(np.float64),
        Itilde=It.astype(np.float64),
        ItildeA=ItA.astype(np.float64),
        ItildeB=ItB.astype(np.float64),
        abs_seed_cos=abs_seed_cos.astype(np.float64),
        clone=clone,
        Dn_cand=Dn_cand.astype(np.float64),
        delta_components=real_components.astype(np.float64),
        per_seed_delta=per_seed_delta.astype(np.float64),
        bp_times=bp_times.astype(np.float64),
        n_confirm=np.array(n_confirm, dtype=np.int64),
    )
    print(f"checkpoint=shard_confirm.npz real_delta={real_delta:+.6f}", flush=True)

    ck = np.load("shard_confirm.npz", allow_pickle=False)
    It = ck["Itilde"].astype(np.float64)
    ItA = ck["ItildeA"].astype(np.float64)
    ItB = ck["ItildeB"].astype(np.float64)
    Dn_cand = ck["Dn_cand"].astype(np.float64)
    abs_seed_cos = ck["abs_seed_cos"].astype(np.float64)
    clone = ck["clone"].astype(bool)
    n_confirm = int(ck["n_confirm"])
    confirm_rows = np.arange(n_confirm, dtype=np.int64)
    seed_idx_saved = ck["seed_idx"].astype(np.int64)
    month_idx_saved = ck["month_idx"].astype(np.int64)

    print("\n=== GATES ===", flush=True)
    d1, _, _ = stat_from_saved(
        It, Dn_cand, abs_seed_cos, clone, confirm_rows, args.random_subsets, 47
    )
    d2, _, _ = stat_from_saved(
        It, Dn_cand, abs_seed_cos, clone, confirm_rows, args.random_subsets, 47
    )
    g0_diff = abs(d1 - d2)
    g0 = g0_diff <= 1e-9
    print(f"G0 determinism diff={g0_diff:.3e} pass={g0}", flush=True)

    jac = split_half_jaccard(ItA, ItB, clone, confirm_rows)
    g1_mean = float(np.nanmean(jac))
    g1 = g1_mean > 0.4
    print(f"G1 split_half_top20_jaccard mean={g1_mean:.4f} "
          f"min={float(np.nanmin(jac)):.4f} pass={g1}", flush=True)

    month_rows_out, month_medians = month_gate(
        It, clone, month_words, month_idx_saved, seed_idx_saved, cand_pos
    )
    g2_count = int(np.sum(month_medians <= 50.0))
    g2_needed = min(10, len(month_idx_saved))
    g2 = g2_count >= g2_needed
    for word, feat, med, ranks in month_rows_out:
        shown = ", ".join(str(int(r)) for r in ranks)
        print(f"G2 {word:<9} seed={feat} median_rank={med:.1f} ranks=[{shown}]",
              flush=True)
    print(f"G2 months_median_rank_le50 count={g2_count}/{len(month_idx_saved)} "
          f"needed={g2_needed} pass={g2}", flush=True)

    print("\n=== STATISTIC ===", flush=True)
    print(f"Delta_top={d1:+.6f}", flush=True)
    bins = eligible_bins(abs_seed_cos, clone, confirm_rows)
    null = np.empty(args.perms, dtype=np.float64)
    for p in range(args.perms):
        prng = np.random.default_rng(53 + p)
        ps = permuted_score(It, bins, confirm_rows, prng)
        rrng = np.random.default_rng(53 + p)
        null[p], _, _ = delta_components(
            ps, Dn_cand, bins, confirm_rows, args.random_subsets, rrng
        )
        if (p + 1) % max(1, args.perms // 10) == 0 or p == args.perms - 1:
            print(f"[perm] {p + 1}/{args.perms}", flush=True)
    null_mean = float(np.nanmean(null))
    null_sd = float(np.nanstd(null))
    z = float((d1 - null_mean) / (null_sd + 1e-12))

    print("\n=== VERDICT INPUTS ===", flush=True)
    print(f"n_confirm={n_confirm} n_month={len(month_idx_saved)} candidates={Dn_cand.shape[0]} "
          f"bp={args.bp} perms={args.perms} random_subsets={args.random_subsets}",
          flush=True)
    print(f"G0={g0} G1={g1} G2={g2} gates_pass={bool(g0 and g1 and g2)}", flush=True)
    print(f"Delta_top={d1:+.6f} null_mean={null_mean:+.6f} "
          f"null_sd={null_sd:.6f} z={z:+.2f}", flush=True)
    print(f"VERDICT: {verdict(bool(g0 and g1 and g2), d1, z)}", flush=True)

    full_seconds = (
        float(np.nanmean(ck["bp_times"])) * 16.0 / max(1, args.bp)
        * (len(alive) / len(cand_idx)) * (36.0 / len(seed_idx_saved))
    )
    print("\n=== FEASIBILITY ===", flush=True)
    print(f"seconds_per_base_point={float(np.nanmean(ck['bp_times'])):.2f} "
          f"projected_full_run_seconds={full_seconds:.1f} "
          f"projected_full_run_hours={full_seconds / 3600.0:.2f} "
          f"total_seconds={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
