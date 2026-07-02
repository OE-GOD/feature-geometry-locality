"""
Real-model run: is a trained SAE's feature geometry LOCAL or GLOBAL?

STAGED FOR THE POD -- not run during local validation. Needs `sae_lens`,
`transformer_lens`, a GPU is nice-to-have but not required for the analysis
(it's a few matmuls on the decoder + unembedding).

Pipeline
--------
1. Load a pretrained SAE (decoder directions W_dec = the GEOMETRY).
2. Load the base model's unembedding W_U.
3. Function vector for feature i = its DIRECT LOGIT EFFECT:
       F_i = W_dec[i] @ W_U        (shape: d_vocab)
   i.e. "which tokens does writing this feature into the residual push?"
   This is the cheapest defensible proxy for "how the model uses feature i".
4. Subsample K features (all-pairs is O(K^2); K~3000 -> ~4.5M pairs, fine).
5. Run locality_metric on (W_dec_subset, F_subset).

Interpreting the verdict
------------------------
LOCAL  -> functionally-similar features are geometric NEIGHBOURS. A
          bag-of-features / local reading of the SAE is defensible.
GLOBAL -> functional similarity is carried by FAR (all-to-all) relationships.
          Single-direction SDL is missing structure; Sharkey et al.'s
          "fundamental problem" bites for this layer.
NULL   -> decoder geometry doesn't track direct logit effect at all; try a
          richer function proxy (downstream/causal effect) before concluding.

Config below targets gpt2-small residual SAEs (Joseph Bloom's release).
Swap RELEASE / SAE_ID / HOOK to sweep layers -- locality may differ by depth,
which is itself a result worth plotting.

Usage:  python3 run_gpt2_sae.py --release gpt2-small-res-jb \
                                --sae-id blocks.8.hook_resid_pre --k 3000
"""

import argparse
import json
import numpy as np

from locality import locality_metric, format_result


def load_sae_and_unembed(release, sae_id, device):
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    if isinstance(sae, tuple):        # some versions return (sae, cfg, sparsity)
        sae = sae[0]
    W_dec = sae.W_dec.detach().float().cpu().numpy()          # (n_features, d_model)

    model = HookedTransformer.from_pretrained("gpt2", device=device)
    W_U = model.W_U.detach().float().cpu().numpy()            # (d_model, d_vocab)
    return W_dec, W_U


def subsample(n, k, seed=0):
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=k, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="gpt2-small-res-jb")
    ap.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    ap.add_argument("--k", type=int, default=3000, help="features to subsample")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="result_gpt2.json")
    args = ap.parse_args()

    print(f"loading {args.release} / {args.sae_id} ...")
    W_dec, W_U = load_sae_and_unembed(args.release, args.sae_id, args.device)
    n_features = W_dec.shape[0]
    print(f"  {n_features} features, d_model={W_dec.shape[1]}, d_vocab={W_U.shape[1]}")

    idx = subsample(n_features, args.k, args.seed)
    D = W_dec[idx]                    # geometry
    F = D @ W_U                       # direct logit effect = function
    print(f"  analyzing {len(idx)} features")

    res = locality_metric(D, F)
    print(format_result(args.sae_id, res))

    centers, bin_mean, bin_count = res["curve"]
    out = {
        "release": args.release,
        "sae_id": args.sae_id,
        "k": int(len(idx)),
        "verdict": res["verdict"],
        "r2": res["r2"],
        "locality": None if res["locality"] != res["locality"] else res["locality"],
        "near_strength": res["near_strength"],
        "far_strength": res["far_strength"],
        "curve": {
            "sgeo_bin_center": centers.tolist(),
            "mean_sfn": [None if x != x else float(x) for x in bin_mean],
            "count": bin_count.tolist(),
        },
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
