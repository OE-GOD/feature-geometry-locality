"""
Non-ablation function proxy: feature CO-FIRING.

Every prior proxy measured a feature's EFFECT (ablation/injection -> output
change) and was confounded by GPT-2's low-rank ablation response (~99% one shared
axis, a property of the model, not removable at the source). Co-firing sidesteps
that entirely: function = the contexts in which a feature ACTIVATES, read straight
off the SAE encoder on real text (NeelNanda/pile-10k). No ablation, no output
perturbation.

Question: does decoder geometry Sgeo=cos(W_dec_i,W_dec_j) predict co-firing
Sfn=cos(a_i, a_j) (a_i = feature i's activation pattern across corpus positions)?

Reports the structure (is co-firing also dominated by a shared "general activity"
axis?), the raw geometry->co-firing coupling, and the rank-matched null at
deflate 0/1/3.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from nlcp_compare import cos_rows, coupling_r2, rank_matched_null


def build_pile_contexts(tok, n_docs, seq, max_pos):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    toks = []
    for i in range(min(n_docs, len(ds))):
        toks.extend(tok(ds[i]["text"])["input_ids"])
        if len(toks) >= max_pos:
            break
    toks = toks[:max_pos]
    toks = toks[: (len(toks) // seq) * seq]
    return torch.tensor(toks).reshape(-1, seq)


def harvest(layer, idx, contexts, model, W_enc, b_enc, b_dec, batch=16):
    block = model.transformer.h[layer]
    Wsub, be = W_enc[:, idx], b_enc[idx]
    chunks = []
    for i in range(0, contexts.shape[0], batch):
        rec = []

        def pre(_m, args):
            rec.append(args[0].detach())
            return None

        h = block.register_forward_pre_hook(pre)
        with torch.no_grad():
            model(contexts[i:i + batch])
        h.remove()
        H = rec[0].reshape(-1, rec[0].shape[-1]).double().numpy()
        chunks.append(np.maximum((H - b_dec) @ Wsub + be, 0.0))
    return np.concatenate(chunks, 0)                              # positions x K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--max-pos", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot", type=int, default=10)
    args = ap.parse_args()

    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))

    tok, model = load_gpt2()
    contexts = build_pile_contexts(tok, args.docs, args.seq, args.max_pos)

    A = harvest(args.layer, idx, contexts, model, W_enc, b_enc, b_dec)
    fires = (A > 0).sum(0)
    keep = fires >= 10
    idx, A = idx[keep], A[:, keep]
    K = idx.size
    F = A.T.astype(np.float64)                                   # K x positions

    D = W_dec[idx].astype(np.float64)
    y = D - D.mean(axis=1, keepdims=True)
    iu = np.triu_indices(K, 1)
    sgeo = cos_rows(D)[iu]

    sv = np.linalg.svd(F, compute_uv=False)
    print(f"layer {args.layer}  K={K} features (fired>=10)  positions={A.shape[0]}  "
          f"mean fires/feat={fires[keep].mean():.0f}")
    print(f"co-firing matrix top-1 SVD energy = {sv[0]**2/(sv**2).sum():.4f}  "
          f"(cf. ablation effects ~0.99)")
    print(f"\n{'defl':>4} {'coupling_R2':>11} {'rank-null R2':>16} {'z':>7}  verdict")
    for r in (0, 1, 3):
        real = coupling_r2(F, sgeo, iu, r)
        nm, ns = rank_matched_null(y, F, sgeo, iu, r, args.rot, rng)
        z = (real - nm) / (ns + 1e-9)
        tag = ("geometry predicts co-firing" if z > 3 else
               "below null" if z < -3 else "n.s.")
        print(f"{r:>4} {real:>11.4f} {nm:>11.4f}±{ns:>5.4f} {z:>7.1f}  {tag}")


if __name__ == "__main__":
    main()
