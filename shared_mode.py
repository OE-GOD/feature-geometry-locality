"""
Is the pervasive 99% shared mode the ANSWER, not the confound?

First-principles reframe: every proxy in this project (ablation effect, co-firing,
MLP interaction) came back ~99% dominated by one shared per-feature mode, which we
kept DEFLATING as a nuisance. Maybe it IS the functional structure — a single
per-feature "importance" scalar — and the pairwise geometry we hunted was always
the noise. That would explain why pairwise geometry never worked and why a global
per-feature scalar out-predicted geometric neighbors 2:1.

Test on a common feature set: extract each proxy's per-feature shared-mode loading,
plus the TRIVIAL candidates (decoder norm, firing frequency). Are the shared modes
ONE scalar? Is that scalar just norm/frequency (trivial), or something more? (Kill
test built in: if it's ~decoder-norm or ~frequency, the reframe is trivial.)
"""

import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from cofire import harvest as cofire_harvest, build_pile_contexts
from feature_interaction import mlp_sublayer, base_points

LAYER = 8


def top_loading(M):                       # per-row loading on the top shared axis
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    u1 = U[:, 0]
    return -u1 if u1.mean() < 0 else u1


def ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def scorr(a, b):                          # Spearman
    return np.corrcoef(ranks(a), ranks(b))[0, 1]


def main():
    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    d = np.load(f"real_effect_L{LAYER}_k200_s0.npz")
    idx, F, fc = d["idx"], d["F"].astype(np.float64), d["firing_counts"]
    keep = np.linalg.norm(F, axis=1) > 0
    idx, F, fc = idx[keep], F[keep], fc[keep]
    K = idx.size

    decoder_norm = np.linalg.norm(W_dec[idx], axis=1)
    freq = 10.0 ** log_sp[idx]
    abl_shared = top_loading(F)                       # confidence-collapse loading
    eff_mag = np.linalg.norm(F, axis=1)               # total ablation effect size

    tok, model = load_gpt2()
    contexts = build_pile_contexts(tok, 200, 48, 8000)
    A = cofire_harvest(LAYER, idx, contexts, model, W_enc, b_enc, b_dec)  # pos x K
    activity = A.mean(0)
    cofire_shared = top_loading(A.T)

    # interaction propensity: per-feature mean 2nd-order MLP coupling
    block = model.transformer.h[LAYER]
    H0 = base_points(LAYER, contexts, model, 8, np.random.default_rng(0))
    D = torch.from_numpy(W_dec[idx]).float() * 6.0
    I = np.zeros((K, K))
    iu = np.triu_indices(K, 1)
    ii, jj = iu
    pw = D[ii] + D[jj]
    with torch.no_grad():
        for b in range(H0.shape[0]):
            h0 = H0[b:b + 1]
            f0 = mlp_sublayer(block, h0)
            f1 = mlp_sublayer(block, h0 + D)
            fp = mlp_sublayer(block, h0 + pw)
            I[iu] += (fp - f1[ii] - f1[jj] + f0).norm(dim=-1).double().numpy()
    I = I + I.T
    inter_prop = I.sum(1) / (K - 1)

    names = ["dec_norm", "freq", "firing_ct", "abl_shared", "cofire_sh",
             "activity", "inter_prop", "eff_mag"]
    vecs = [decoder_norm, freq, fc, abl_shared, cofire_shared, activity,
            inter_prop, eff_mag]
    print(f"Spearman correlations among per-feature scalars (K={K}, layer {LAYER}):\n")
    print("           " + " ".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<10} " + " ".join(f"{scorr(vecs[i], vecs[j]):>+9.2f}"
                                     for j in range(len(names))))
    print("\nKEY: do abl_shared / cofire_sh / inter_prop agree (one scalar)?  "
          "and is that scalar ~= dec_norm or freq (trivial) or not?")


if __name__ == "__main__":
    main()
