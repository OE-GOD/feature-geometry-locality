"""
Exploratory pilot for the redesigned manifold-shard test.

For each seed feature, measure its MLP second-order interaction against the
full alive SAE dictionary, normalize away global partner propensity, and report
the stability / hub / shard-geometry diagnostics needed before preregistration.
"""

import argparse
import time

import numpy as np
import torch

from cofire import build_pile_contexts
from concept_geometry import FAMILIES, concept_residual
from feature_interaction import base_points, mlp_sublayer
from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY


LAYER = 8
ALPHA = 6.0
HOOK = "blocks.8.hook_resid_pre"
CHUNK = 2048

torch.set_num_threads(8)


def cos_rows(X):
    X = X.astype(np.float64, copy=False)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def burned_samples(alive):
    s0 = set(np.random.default_rng(0).choice(alive, 200, replace=False))
    pool1 = np.array([a for a in alive if a not in s0])
    s7 = set(np.random.default_rng(7).choice(pool1, 200, replace=False))
    pool2 = np.array([a for a in alive if a not in s0 and a not in s7])
    s13 = set(np.random.default_rng(13).choice(pool2, 200, replace=False))
    pool3 = np.array([a for a in alive if a not in s0 and a not in s7 and a not in s13])
    s17 = set(np.random.default_rng(17).choice(pool3, 200, replace=False))
    return s0 | s7 | s13 | s17


def month_features(layer, W_dec, W_enc, b_enc, b_dec, log_sp, tok, model):
    words = FAMILIES["months"]
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


def build_contexts_safe(tok, n_docs, seq, max_pos):
    try:
        return build_pile_contexts(tok, n_docs, seq, max_pos)
    except PermissionError:
        from datasets import Dataset

        path = (
            "/Users/oe/.cache/huggingface/datasets/NeelNanda___pile-10k/default/"
            "0.0.0/127bfedcd5047750df5ccf3a12979a47bfa0bafa/pile-10k-train.arrow"
        )
        ds = Dataset.from_file(path)
        toks = []
        for i in range(min(n_docs, len(ds))):
            toks.extend(tok(ds[i]["text"])["input_ids"])
            if len(toks) >= max_pos:
                break
        toks = toks[:max_pos]
        toks = toks[: (len(toks) // seq) * seq]
        return torch.tensor(toks).reshape(-1, seq)


def choose_seeds(alive, burned, month_idx, n_random, smoke):
    rng = np.random.default_rng(23)
    pool = np.array([a for a in alive if a not in burned])
    random_idx = np.sort(rng.choice(pool, size=min(n_random, pool.size), replace=False))
    if smoke:
        random_idx = random_idx[:3]
        month_idx = month_idx[:2]
    names = [f"rand_{int(i)}" for i in random_idx]
    names += [f"month_{w}_{int(i)}" for w, i in zip(FAMILIES["months"], month_idx)]
    seed_idx = np.concatenate([random_idx, month_idx]).astype(np.int64)
    return names, seed_idx


def compute_seed_all(block, H0, W_dec, cand_idx, seed_idx, cand_pos, alpha, chunk):
    D_cand_np = W_dec[cand_idx].astype(np.float32, copy=False)
    D_seed_np = W_dec[seed_idx].astype(np.float32, copy=False)
    D_cand = torch.from_numpy(D_cand_np)
    D_seed = torch.from_numpy(D_seed_np)
    S, K = len(seed_idx), len(cand_idx)
    d_model = D_cand.shape[1]
    h = H0.shape[0] // 2
    IA = np.zeros((S, K), dtype=np.float64)
    IB = np.zeros((S, K), dtype=np.float64)
    bp_times = []

    with torch.no_grad():
        for b in range(H0.shape[0]):
            t0 = time.time()
            h0 = H0[b:b + 1].float()
            f0 = mlp_sublayer(block, h0)
            F1 = torch.empty((K, d_model), dtype=torch.float32)
            for lo in range(0, K, chunk):
                hi = min(lo + chunk, K)
                F1[lo:hi] = mlp_sublayer(block, h0 + alpha * D_cand[lo:hi])
            acc = IA if b < h else IB
            for s in range(S):
                f1s = mlp_sublayer(block, h0 + alpha * D_seed[s:s + 1])
                for lo in range(0, K, chunk):
                    hi = min(lo + chunk, K)
                    fp = mlp_sublayer(
                        block, h0 + alpha * D_seed[s:s + 1] + alpha * D_cand[lo:hi]
                    )
                    resid = fp - F1[lo:hi] - f1s + f0
                    acc[s, lo:hi] += resid.norm(dim=-1).double().cpu().numpy()
            bp_times.append(time.time() - t0)
            print(f"[progress] base_point {b + 1}/{H0.shape[0]} seconds={bp_times[-1]:.2f}",
                  flush=True)

    if h > 0:
        IA /= h
    if H0.shape[0] - h > 0:
        IB /= H0.shape[0] - h
    for s, idx in enumerate(seed_idx):
        if int(idx) in cand_pos:
            IA[s, cand_pos[int(idx)]] = np.nan
            IB[s, cand_pos[int(idx)]] = np.nan
    return IA, IB, np.array(bp_times, dtype=np.float64), D_cand_np, D_seed_np


def top_set(row, valid, n):
    ok = valid & np.isfinite(row)
    if ok.sum() == 0:
        return set()
    cols = np.where(ok)[0]
    order = np.argsort(row[cols])[::-1]
    return set(cols[order[:min(n, cols.size)]].tolist())


def jaccard(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / max(1, len(a | b))


def rank_positions(row, targets, valid):
    score = row.copy()
    score[~valid] = -np.inf
    score[~np.isfinite(score)] = -np.inf
    order = np.argsort(score)[::-1]
    ranks = np.empty(score.size, dtype=np.int64)
    ranks[order] = np.arange(1, score.size + 1)
    return [int(ranks[t]) for t in targets if t >= 0]


def rank2_energy(G):
    sv = np.linalg.svd(G.astype(np.float64), compute_uv=False)
    return float((sv[:2] ** 2).sum() / np.clip((sv ** 2).sum(), 1e-12, None))


def group_geometry(seed_idx, score, Dn_cand, Dn_seed, clone, seed_group):
    vals = []
    for s in range(len(seed_idx)):
        partners = list(top_set(score[s], ~clone[s], 6))
        if len(partners) < 2:
            vals.append((np.nan, np.nan, np.nan))
            continue
        P = Dn_cand[partners]
        sp = np.abs(P @ Dn_seed[s]).mean()
        G = np.abs(P @ P.T)
        pp = G[np.triu_indices(len(partners), 1)].mean()
        e2 = rank2_energy(G)
        vals.append((float(sp), float(pp), e2))
        print(f"  {seed_group[s]:<24} seed_partner_abs_cos={sp:.4f} "
              f"partner_partner_abs_cos={pp:.4f} rank2_energy={e2:.4f}", flush=True)
    return np.array(vals, dtype=np.float64)


def pair_abs_cos_mean(Dn, cols):
    if len(cols) < 2:
        return np.nan
    with np.errstate(all="ignore"):
        G = np.abs(Dn[cols] @ Dn[cols].T)
    return float(G[np.triu_indices(len(cols), 1)].mean())


def conditional_contrast(score, abs_seed_cos, Dn_cand, clone, seed_mask, n_perm, rng):
    edges = np.linspace(float(np.nanmin(abs_seed_cos)), float(np.nanmax(abs_seed_cos)), 11)
    if edges[0] == edges[-1]:
        edges[-1] = edges[0] + 1e-9
    bin_id = np.clip(np.digitize(abs_seed_cos, edges) - 1, 0, 9)
    occ = np.array([(bin_id == b).sum() for b in range(10)], dtype=np.int64)

    def one(scores):
        num = 0.0
        den = 0.0
        for s in np.where(seed_mask)[0]:
            for b in range(10):
                m = ((bin_id[s] == b) & (~clone[s]) & np.isfinite(scores[s]))
                cols = np.where(m)[0]
                if cols.size < 24:
                    continue
                order = np.argsort(scores[s, cols])
                bot = cols[order[:8]]
                top = cols[order[-8:]]
                delta = pair_abs_cos_mean(Dn_cand, top) - pair_abs_cos_mean(Dn_cand, bot)
                w = float(cols.size)
                num += w * delta
                den += w
        return num / den if den > 0 else np.nan

    real = one(score)
    null = []
    for _ in range(n_perm):
        ps = score.copy()
        for s in np.where(seed_mask)[0]:
            for b in range(10):
                m = ((bin_id[s] == b) & (~clone[s]) & np.isfinite(ps[s]))
                vals = ps[s, m].copy()
                rng.shuffle(vals)
                ps[s, m] = vals
        null.append(one(ps))
    null = np.array(null, dtype=np.float64)
    z = (real - np.nanmean(null)) / (np.nanstd(null) + 1e-12)
    return occ, real, float(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--bp", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_perm = 20 if args.smoke else 200
    if args.smoke:
        args.bp = 2

    t_start = time.time()
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(HOOK)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    cand_idx = alive[:3000] if args.smoke else alive
    cand_pos = {int(v): i for i, v in enumerate(cand_idx)}
    burned = burned_samples(alive)

    tok, model = load_gpt2()
    month_words, month_idx = month_features(LAYER, W_dec, W_enc, b_enc, b_dec,
                                            log_sp, tok, model)
    seed_names, seed_idx = choose_seeds(alive, burned, month_idx, args.seeds, args.smoke)
    seed_is_month = np.array([name.startswith("month_") for name in seed_names])
    seed_is_random = ~seed_is_month

    print(f"setup layer={LAYER} alpha={ALPHA} hook={HOOK}", flush=True)
    print(f"candidate_pool={len(cand_idx)} alive_total={len(alive)} seeds={len(seed_idx)} "
          f"random={int(seed_is_random.sum())} month={int(seed_is_month.sum())}",
          flush=True)
    month_map = {w: int(i) for w, i in zip(month_words, month_idx)}
    print(f"month_features {month_map}", flush=True)

    rng23 = np.random.default_rng(23)
    contexts = build_contexts_safe(tok, 400, 48, 8000)
    H0 = base_points(LAYER, contexts, model, args.bp, rng23)
    block = model.transformer.h[LAYER]
    IA, IB, bp_times, D_cand, D_seed = compute_seed_all(
        block, H0, W_dec, cand_idx, seed_idx, cand_pos, ALPHA, CHUNK
    )
    I = (IA + IB) / 2
    denom = np.nanmean(I, axis=0)
    It = I / np.clip(denom[None, :], 1e-12, None)
    ItA = IA / np.clip(np.nanmean(IA, axis=0)[None, :], 1e-12, None)
    ItB = IB / np.clip(np.nanmean(IB, axis=0)[None, :], 1e-12, None)

    Dn_cand = cos_rows(D_cand)
    Dn_seed = cos_rows(D_seed)
    with np.errstate(all="ignore"):
        abs_seed_cos = np.abs(Dn_seed @ Dn_cand.T)
    clone = abs_seed_cos > 0.9
    for s, idx in enumerate(seed_idx):
        if int(idx) in cand_pos:
            clone[s, cand_pos[int(idx)]] = True

    full_projected = bp_times.mean() * 16 * (len(alive) / len(cand_idx)) * (36 / len(seed_idx))
    print("\n=== FEASIBILITY ===", flush=True)
    print(f"seconds_per_base_point={bp_times.mean():.2f} "
          f"projected_full_run_seconds={full_projected:.1f} "
          f"projected_full_run_hours={full_projected / 3600:.2f}", flush=True)

    print("\n=== STABILITY ===", flush=True)
    jac = np.array([jaccard(top_set(ItA[s], ~clone[s], 20),
                            top_set(ItB[s], ~clone[s], 20))
                    for s in range(len(seed_idx))], dtype=np.float64)
    print(f"top20_split_half_jaccard mean={np.nanmean(jac):.4f} "
          f"min={np.nanmin(jac):.4f}", flush=True)

    print("\n=== HUB DIAGNOSTIC ===", flush=True)
    raw_sets = [top_set(I[s], ~clone[s], 20) for s in range(len(seed_idx))]
    norm_sets = [top_set(It[s], ~clone[s], 20) for s in range(len(seed_idx))]
    pairs = [(a, b) for a in range(len(seed_idx)) for b in range(a + 1, len(seed_idx))]
    raw_j = np.array([jaccard(raw_sets[a], raw_sets[b]) for a, b in pairs])
    norm_j = np.array([jaccard(norm_sets[a], norm_sets[b]) for a, b in pairs])
    print(f"cross_seed_top20_jaccard raw_I_mean={np.nanmean(raw_j):.4f} "
          f"Itilde_mean={np.nanmean(norm_j):.4f}", flush=True)

    print("\n=== MONTHS LINKAGE ===", flush=True)
    month_pos = [cand_pos.get(int(i), -1) for i in month_idx]
    for m, feat in enumerate(month_idx):
        rows = np.where(seed_idx == feat)[0]
        if len(rows) == 0:
            continue
        s = int(rows[0])
        targets = [month_pos[k] for k in range(12) if k != m and month_pos[k] >= 0]
        ranks = rank_positions(It[s], targets, ~clone[s])
        ranks_all = []
        rp = 0
        for k in range(12):
            if k == m:
                continue
            if month_pos[k] >= 0:
                ranks_all.append(str(ranks[rp]))
                rp += 1
            else:
                ranks_all.append("NA")
        med = float(np.median(ranks)) if ranks else np.nan
        top50 = int(np.sum(np.array(ranks) <= 50)) if ranks else 0
        print(f"{month_words[m]:<9} seed={int(feat)} median_rank={med:.1f} "
              f"top50_count={top50}/11 ranks=[{', '.join(ranks_all)}]", flush=True)

    print("\n=== GROUP GEOMETRY ===", flush=True)
    geom = group_geometry(seed_idx, It, Dn_cand, Dn_seed, clone, seed_names)
    for label, mask in [("random", seed_is_random), ("month", seed_is_month)]:
        g = geom[mask]
        print(f"{label}_mean seed_partner_abs_cos={np.nanmean(g[:, 0]):.4f} "
              f"partner_partner_abs_cos={np.nanmean(g[:, 1]):.4f} "
              f"rank2_energy={np.nanmean(g[:, 2]):.4f}", flush=True)

    print("\n=== CONDITIONAL CONTRAST ===", flush=True)
    prng = np.random.default_rng(123)
    occ, dr, zr = conditional_contrast(It, abs_seed_cos, Dn_cand, clone,
                                       seed_is_random, n_perm, prng)
    print(f"bin_occupancies={occ.tolist()}", flush=True)
    print(f"random Delta_pooled={dr:+.6f} permutation_z={zr:+.2f}", flush=True)
    _, dm, zm = conditional_contrast(It, abs_seed_cos, Dn_cand, clone,
                                     seed_is_month, n_perm, prng)
    print(f"month Delta_pooled={dm:+.6f} permutation_z={zm:+.2f}", flush=True)

    np.savez(
        "shard_pilot.npz",
        candidate_idx=cand_idx.astype(np.int64),
        seed_idx=seed_idx.astype(np.int64),
        seed_names=np.array(seed_names),
        month_idx=month_idx.astype(np.int64),
        I=I.astype(np.float32),
        IA=IA.astype(np.float32),
        IB=IB.astype(np.float32),
        Itilde=It.astype(np.float32),
        abs_seed_cos=abs_seed_cos.astype(np.float32),
        split_jaccard=jac.astype(np.float64),
        bp_times=bp_times.astype(np.float64),
    )
    print(f"\nwrote shard_pilot.npz total_seconds={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
