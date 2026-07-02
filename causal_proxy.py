"""
Causal function proxy -- escapes the linear-tautology refutation.

Both REFUTED skeptics showed the direct-logit proxy F = M @ d is a fixed linear
map of the decoder direction, so its Sgeo->Sfn "halo" is reproduced by a
spectrum-matched random map: it measures the proxy, not the network.

This proxy instead measures a feature's DOWNSTREAM CAUSAL effect:
  inject alpha * d_i into the residual stream at the feature's own layer, run
  the REST of GPT-2 forward, and record the mean change in output logits over a
  set of real contexts. F_i = E_context[ logits(inject d_i) - logits(base) ].

Why this answers the objection:
  - Non-linear in d_i (passes through downstream attention+MLP), so Sfn is NOT
    cosine under a fixed quadratic form -> the spectrum-matched-null argument
    does not apply.
  - Downstream-aware -> not blind to how mid-layer features actually act.

Null for THIS proxy: permute the mapping between decoder directions and their
causal-effect vectors (shuffle feature labels). If geometry predicts causal
function beyond that permutation null, the structure is real and network-specific.

Small by design -- a CPU proof-of-concept + validation gate before a pod run at
full K. Usage:
  python3 causal_proxy.py --layer 8 --k 150 --contexts 16 --alpha 6.0
"""

import argparse
import json
import numpy as np
import torch

from run_layer import fetch, DEAD_LOG_SPARSITY

SAMPLE_TEXT = """The mitochondria is the powerhouse of the cell, converting
nutrients into energy. In 1969 the Apollo program landed the first humans on
the Moon. Interest rates set by the central bank influence inflation and
unemployment across the economy. She practiced the violin every morning before
school, hoping to join the orchestra. The river flooded the valley after three
days of relentless monsoon rain. Quantum entanglement links two particles so
that measuring one instantly determines the other. The chef reduced the sauce
slowly, whisking in butter until it turned glossy. Ancient Roman aqueducts
carried fresh water across vast distances using only gravity. The startup
raised a Series B round to expand its logistics network overseas. Photosynthesis
converts carbon dioxide and sunlight into glucose and oxygen."""


def build_contexts(tokenizer, n_ctx, seq_len):
    ids = tokenizer(SAMPLE_TEXT.replace("\n", " "), return_tensors="pt")["input_ids"][0]
    chunks = []
    step = max(1, (ids.shape[0] - seq_len) // n_ctx)
    for i in range(0, ids.shape[0] - seq_len, step):
        chunks.append(ids[i:i + seq_len])
        if len(chunks) >= n_ctx:
            break
    return torch.stack(chunks)  # (n_ctx, seq_len)


def causal_effects(layer, idx, W_dec, alpha, contexts, model):
    """Return (len(idx), vocab) mean logit-shift vectors."""
    block = model.transformer.h[layer]
    state = {"dir": None}

    def pre_hook(module, args):
        if state["dir"] is None:
            return None
        hidden = args[0]
        return (hidden + alpha * state["dir"],) + args[1:]

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            # reduce to the (V,) mean-logit vector BEFORE any float64 cast --
            # never materialize a doubled (C, S, V) tensor (that was ~300MB of
            # alloc churn per forward and dominated runtime)
            base_mean = model(contexts).logits.mean(dim=(0, 1))    # (V,) float32
            V = base_mean.shape[-1]
            out = np.empty((len(idx), V), dtype=np.float64)
            for row, fi in enumerate(idx):
                state["dir"] = torch.tensor(W_dec[fi], dtype=torch.float32)
                inj_mean = model(contexts).logits.mean(dim=(0, 1))  # (V,) float32
                out[row] = (inj_mean - base_mean).double().numpy()
                state["dir"] = None
    finally:
        handle.remove()
    return out


def cos_pairs(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(all="ignore"):
        S = Xn @ Xn.T
    return S


def binned(sgeo, sfn, nb=20):
    edges = np.linspace(-1, 1, nb + 1)
    idx = np.clip(np.digitize(sgeo, edges) - 1, 0, nb - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for b in range(nb):
        m = idx == b
        if m.sum():
            rows.append((round(float(centers[b]), 3), int(m.sum()),
                         round(float(sfn[m].mean()), 4)))
    overall = sfn.mean()
    pred = np.array([sfn[idx == i].mean() if (idx == i).any() else overall
                     for i in idx])
    r2 = float(1 - np.sum((sfn - pred) ** 2) / np.sum((sfn - overall) ** 2))
    return rows, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--contexts", type=int, default=16)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=20, help="label-permutation null draws")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, log_sparsity, _, _ = fetch(hook)
    alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    contexts = build_contexts(tok, args.contexts, args.seq)

    F = causal_effects(args.layer, idx, W_dec, args.alpha, contexts, model)
    D = W_dec[idx]

    Sgeo = cos_pairs(D)
    Sfn = cos_pairs(F)
    iu = np.triu_indices(len(idx), k=1)
    sgeo, sfn = Sgeo[iu], Sfn[iu]

    rows, real_r2 = binned(sgeo, sfn)

    # permutation null: shuffle which causal vector belongs to which direction
    perm_r2 = []
    for p in range(args.perm):
        pp = rng.permutation(len(idx))
        Sfn_p = cos_pairs(F[pp])
        _, r2p = binned(sgeo, Sfn_p[iu])
        perm_r2.append(r2p)
    perm_mean, perm_sd = float(np.mean(perm_r2)), float(np.std(perm_r2))

    # rank-1 geometric-neighbor causal similarity vs its permutation null
    np.fill_diagonal(Sgeo, -np.inf)
    nn = np.argsort(-Sgeo, axis=1)[:, 0]
    rank1 = float(Sfn[np.arange(len(idx)), nn].mean())
    baseline = float(sfn.mean())

    result = {
        "layer": args.layer, "k": int(len(idx)), "contexts": int(args.contexts),
        "alpha": args.alpha, "seq": args.seq,
        "real_r2": round(real_r2, 4),
        "perm_null_r2": round(perm_mean, 4), "perm_null_sd": round(perm_sd, 4),
        "delta_r2": round(real_r2 - perm_mean, 4),
        "z_vs_null": round((real_r2 - perm_mean) / (perm_sd + 1e-9), 2),
        "rank1_neighbor_sfn": round(rank1, 4),
        "baseline_sfn": round(baseline, 4),
        "curve": rows,
    }
    out_path = args.out or f"causal_L{args.layer}_k{args.k}_a{args.alpha}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"layer {args.layer}  k={len(idx)}  alpha={args.alpha}  contexts={args.contexts}")
    print(f"  real_R2={result['real_r2']}  perm_null_R2="
          f"{result['perm_null_r2']}±{result['perm_null_sd']}  "
          f"delta={result['delta_r2']}  z={result['z_vs_null']}")
    print(f"  rank1_neighbor_sfn={result['rank1_neighbor_sfn']}  "
          f"baseline_sfn={result['baseline_sfn']}")
    print("  Sgeo   n   mean_causal_Sfn")
    for c, n, m in rows:
        if n >= 5:
            print(f"  {c:+.2f} {n:>5}   {m:+.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
