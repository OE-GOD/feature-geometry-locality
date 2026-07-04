"""
Stress-test the direct-logit scale flip found in gemma_scale.py.

Is z<0 at 117M -> z>0 at 2B a robust property, or a fragile artifact of
(a) mismatched depth, or (b) the shared-mode-fraction difference?

Checks, both weights-only:
  - sweep layers (match relative depth: GPT-2 L/12 vs Gemma L/26)
  - recompute the geometry->logit coupling z AFTER deflating the top shared
    mode from F at BOTH real and null (fair) -> does the flip survive?
"""

import glob
import numpy as np
from safetensors import safe_open
from gemma_scale import gained, rand_orth, offdiag_spearman, cos_gram

HOME = glob.os.path.expanduser("~")
K, N_NULL, SEED = 1200, 12, 0


def gpt2_unembed():
    gp = glob.glob(f"{HOME}/.cache/huggingface/hub/models--gpt2/snapshots/*/model.safetensors")[0]
    with safe_open(gp, framework="np") as f:
        wte = f.get_tensor("wte.weight").astype(np.float64)
        ln_w = f.get_tensor("ln_f.weight").astype(np.float64)
    return wte, ("ln", ln_w)


def gpt2_dec(layer):
    p = glob.glob(f"{HOME}/.cache/huggingface/hub/models--jbloom--GPT2-Small-SAEs-Reformatted/"
                  f"snapshots/*/blocks.{layer}.hook_resid_pre/sae_weights.safetensors")
    if not p:
        return None
    with safe_open(p[0], framework="np") as f:
        return f.get_tensor("W_dec").astype(np.float64)


def gemma_unembed():
    snap = glob.glob(f"{HOME}/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/")[0]
    with safe_open(snap + "model-00001-of-00003.safetensors", framework="np") as f:
        emb = f.get_tensor("model.embed_tokens.weight").astype(np.float64)
    with safe_open(snap + "model-00003-of-00003.safetensors", framework="np") as f:
        ng = f.get_tensor("model.norm.weight").astype(np.float64)
    return emb, ("rms", 1.0 + ng)


def gemma_dec(layer):
    p = glob.glob(f"{HOME}/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/"
                  f"snapshots/*/layer_{layer}/width_16k/*/params.npz")
    if not p:
        return None
    return np.load(p[0])["W_dec"].astype(np.float64)


def coupling(GramF, S_geom, deflate):
    G = GramF.copy()
    if deflate:                       # remove top shared eigenmode (rank-1)
        ev, U = np.linalg.eigh(G)
        G = G - ev[-1] * np.outer(U[:, -1], U[:, -1])
    rn = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return offdiag_spearman(S_geom, G / np.outer(rn, rn))


def analyze(D, M, norm, rng):
    Fg = gained(D, norm)
    GramF = Fg @ M @ Fg.T
    S_geom = cos_gram(D)
    ev = np.linalg.eigvalsh(GramF)
    shared = ev[-1] / ev.sum()
    w, V = np.linalg.eigh(M); w = np.clip(w, 0, None)
    d = M.shape[0]
    out = {}
    for defl in (False, True):
        real = coupling(GramF, S_geom, defl)
        nulls = np.empty(N_NULL)
        for t in range(N_NULL):
            Q = rand_orth(d, rng)
            nulls[t] = coupling(Fg @ ((Q * w) @ Q.T) @ Fg.T, S_geom, defl)
        out["z_defl" if defl else "z"] = (real - nulls.mean()) / (nulls.std() + 1e-12)
    out["shared"] = shared
    return out


def run(name, unembed_fn, dec_fn, n_layers, layers):
    W_U, norm = unembed_fn()
    M = W_U.T @ W_U
    print(f"\n{name} ({n_layers} layers)  |  M is {M.shape[0]}x{M.shape[0]}")
    print(f"{'layer':>6}{'reldepth':>9}{'shared':>9}{'z(raw)':>9}{'z(deflated)':>13}")
    for L in layers:
        D = dec_fn(L)
        if D is None:
            continue
        rng = np.random.default_rng(SEED)
        idx = rng.choice(D.shape[0], min(K, D.shape[0]), replace=False)
        r = analyze(D[idx], M, norm, rng)
        print(f"{L:>6}{L / n_layers:>9.2f}{r['shared']:>9.3f}{r['z']:>+9.2f}{r['z_defl']:>+13.2f}")


def main():
    run("GPT-2-117M", gpt2_unembed, gpt2_dec, 12, [1, 4, 7, 8, 10, 11])
    run("Gemma-2-2b", gemma_unembed, gemma_dec, 26, [9, 12, 15, 18, 20, 22, 24])
    print("\nz(raw): geometry->logit coupling vs spectrum null. z(deflated): same after")
    print("removing the shared mode. Flip survives deflation => not a shared-mode artifact.")


if __name__ == "__main__":
    main()
