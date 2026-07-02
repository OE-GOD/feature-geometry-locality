"""
Lightweight real-SAE locality run -- no sae_lens needed (works on py3.9).

Downloads directly from HuggingFace:
  - SAE decoder W_dec (+ feature sparsity) from jbloom/GPT2-Small-SAEs-Reformatted
  - GPT-2 tied embedding wte + final-LN gain from the `gpt2` repo

Function proxy = direct logit attribution through the final-LN linearization:
    F_i = ((d_i - mean(d_i)) * ln_f.weight) @ wte.T
(--proxy raw drops the centering/gain for sensitivity checks.)

Functional cosine is computed WITHOUT materializing the (n, d_vocab) logit
matrix: F F^T = Ds (wte^T wte) Ds^T, and wte^T wte is only (d_model, d_model).
Keeps memory ~constant in vocab size so many layers can run concurrently.

Rigor points:
  - dead-feature filter via log10 sparsity (dead directions are noise);
    runs BOTH filtered and unfiltered so the filter can't drive conclusions
  - reports pair counts in near/far Sgeo regions, so an empty far bin cannot
    silently masquerade as a LOCAL verdict

Usage: python3 run_layer.py --hook blocks.8.hook_resid_pre --k 4000
"""

import argparse
import json
import numpy as np

from locality import locality_from_pairs, NEAR_CUT, FAR_CUT

SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
GPT2_REPO = "gpt2"
DEAD_LOG_SPARSITY = -9.0  # log10 firing frequency below this = dead


def load_np(path, *keys):
    from safetensors import safe_open
    out = []
    with safe_open(path, framework="np") as f:
        names = set(f.keys())
        for k in keys:
            if k not in names:
                raise KeyError(f"{k} not in {path}; has {sorted(names)[:10]}...")
            out.append(f.get_tensor(k).astype(np.float32))
    return out


def fetch(hook):
    from huggingface_hub import hf_hub_download
    w_path = hf_hub_download(SAE_REPO, f"{hook}/sae_weights.safetensors")
    s_path = hf_hub_download(SAE_REPO, f"{hook}/sparsity.safetensors")
    g_path = hf_hub_download(GPT2_REPO, "model.safetensors")
    (W_dec,) = load_np(w_path, "W_dec")                     # (n_feat, d_model)
    (log_sparsity,) = load_np(s_path, "sparsity")           # log10 firing freq
    wte, ln_w = load_np(g_path, "wte.weight", "ln_f.weight")  # (V,d), (d,)
    return W_dec, log_sparsity, wte, ln_w


def pairwise_sims(D, wte, ln_w, proxy):
    """Return upper-triangle (sgeo, sfn) using the vocab-free Gram trick."""
    with np.errstate(all="ignore"):
        if proxy == "ln":
            Ds = (D - D.mean(axis=1, keepdims=True)) * ln_w
        else:
            Ds = D
        M = wte.T @ wte                                   # (d, d)
        G = Ds @ M @ Ds.T                                 # = F F^T
        fnorm = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        Sfn = G / np.outer(fnorm, fnorm)

        Dn = D / np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
        Sgeo = Dn @ Dn.T

    iu = np.triu_indices(D.shape[0], k=1)
    return Sgeo[iu].astype(np.float64), Sfn[iu].astype(np.float64)


def run_subset(W_dec, wte, ln_w, idx, k, seed, proxy):
    rng = np.random.default_rng(seed)
    if k < idx.size:
        idx = np.sort(rng.choice(idx, size=k, replace=False))
    sgeo, sfn = pairwise_sims(W_dec[idx], wte, ln_w, proxy)
    res = locality_from_pairs(sgeo, sfn)
    centers, bin_mean, bin_count = res["curve"]
    return {
        "n_used": int(idx.size),
        "verdict": res["verdict"],
        "r2": round(res["r2"], 4),
        "locality": None if res["locality"] != res["locality"]
        else round(res["locality"], 4),
        "near_strength": round(res["near_strength"], 4),
        "far_strength": round(res["far_strength"], 4),
        "near_pairs": int((sgeo >= NEAR_CUT).sum()),
        "far_pairs": int((sgeo <= FAR_CUT).sum()),
        "total_pairs": int(sgeo.size),
        "sgeo_min": round(float(sgeo.min()), 4),
        "sgeo_max": round(float(sgeo.max()), 4),
        "curve_sgeo": [round(float(c), 3) for c in centers],
        "curve_sfn": [None if m != m else round(float(m), 4) for m in bin_mean],
        "curve_n": [int(c) for c in bin_count],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default="blocks.8.hook_resid_pre")
    ap.add_argument("--k", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proxy", choices=["ln", "raw"], default="ln")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W_dec, log_sparsity, wte, ln_w = fetch(args.hook)
    alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
    everything = np.arange(W_dec.shape[0])

    result = {
        "hook": args.hook,
        "proxy": args.proxy,
        "n_features_total": int(W_dec.shape[0]),
        "n_alive": int(alive.size),
        "k": args.k,
        "seed": args.seed,
        "alive_only": run_subset(W_dec, wte, ln_w, alive, args.k, args.seed,
                                 args.proxy),
        "unfiltered": run_subset(W_dec, wte, ln_w, everything, args.k,
                                 args.seed, args.proxy),
    }
    out_path = args.out or f"result_{args.hook}_{args.proxy}_k{args.k}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    a = result["alive_only"]
    print(f"ALIVE   verdict={a['verdict']} R2={a['r2']} locality={a['locality']} "
          f"near_pairs={a['near_pairs']} far_pairs={a['far_pairs']}")
    u = result["unfiltered"]
    print(f"UNFILT  verdict={u['verdict']} R2={u['r2']} locality={u['locality']} "
          f"near_pairs={u['near_pairs']} far_pairs={u['far_pairs']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
