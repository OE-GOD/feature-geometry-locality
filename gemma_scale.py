"""
Critical-mass test: do the root findings hold at 17x scale?

Rung A: GPT-2-small (117M), jbloom SAE, LayerNorm.
Rung B: Gemma-2-2b (2B), Gemma Scope SAE (JumpReLU), RMSNorm.

Three matched, weights-only diagnostics per scale (no forward passes):
  (1) decoder isotropy         top-1 SVD energy of W_dec (structure in geometry?)
  (2) direct-logit shared mode top-1 eigenvalue frac of F F^T  (feature-agnostic response?)
  (3) geometry->function coupling vs a spectrum-matched null on the unembed metric
      (does the SPECIFIC unembed structure couple geometry to logits beyond its spectrum?)

Direct-logit F_i = (final-norm-gained decoder direction i) @ W_U^T.
Gram trick: F F^T = Fg @ (W_U^T W_U) @ Fg^T, so we never form the K x vocab matrix.
Spectrum null: replace M=W_U^T W_U with Q diag(eig(M)) Q^T, random orthogonal Q
(identical eigenvalue spectrum, structure-free eigenvectors). Real >> null => genuine
structure; real ~ null => the coupling is just the metric's spectrum (an artifact).
"""

import glob
import numpy as np
from safetensors import safe_open

K = 1200
N_NULL = 12
SEED = 0


def load_gpt2_arrays():
    p = glob.glob(f"{glob.os.path.expanduser('~')}/.cache/huggingface/hub/"
                  "models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/"
                  "blocks.8.hook_resid_pre/sae_weights.safetensors")[0]
    with safe_open(p, framework="np") as f:
        W_dec = f.get_tensor("W_dec").astype(np.float64)          # (K_all, 768)
    # log firing sparsity to drop dead features
    sp = glob.glob(f"{glob.os.path.expanduser('~')}/.cache/huggingface/hub/"
                   "models--jbloom--GPT2-Small-SAEs-Reformatted/snapshots/*/"
                   "blocks.8.hook_resid_pre/sparsity.safetensors")[0]
    with safe_open(sp, framework="np") as f:
        log_sp = f.get_tensor("sparsity").astype(np.float64)
    # gpt2 unembed (tied wte) + final layernorm
    gp = glob.glob(f"{glob.os.path.expanduser('~')}/.cache/huggingface/hub/"
                   "models--gpt2/snapshots/*/model.safetensors")[0]
    with safe_open(gp, framework="np") as f:
        wte = f.get_tensor("wte.weight").astype(np.float64)        # (50257,768)
        ln_w = f.get_tensor("ln_f.weight").astype(np.float64)
    alive = np.where(log_sp > -5.0)[0]
    return W_dec, wte, ("ln", ln_w), alive


def load_gemma_arrays():
    p = glob.glob(f"{glob.os.path.expanduser('~')}/.cache/huggingface/hub/"
                  "models--google--gemma-scope-2b-pt-res/snapshots/*/"
                  "layer_12/width_16k/*/params.npz")[0]
    W_dec = np.load(p)["W_dec"].astype(np.float64)                 # (16384, 2304)
    snap = glob.glob(f"{glob.os.path.expanduser('~')}/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    with safe_open(snap + "model-00001-of-00003.safetensors", framework="np") as f:
        emb = f.get_tensor("model.embed_tokens.weight").astype(np.float64)  # (256000,2304)
    with safe_open(snap + "model-00003-of-00003.safetensors", framework="np") as f:
        ng = f.get_tensor("model.norm.weight").astype(np.float64)
    alive = np.arange(W_dec.shape[0])          # weights-only proxy: firing-agnostic
    return W_dec, emb, ("rms", 1.0 + ng), alive                   # Gemma: gain = 1+w


def gained(W_dec_sub, norm):
    kind, g = norm
    if kind == "rms":
        return W_dec_sub * g                                       # no centering
    # LayerNorm: center then scale by gain (std folded as fixed)
    return (W_dec_sub - W_dec_sub.mean(1, keepdims=True)) * g


def rand_orth(d, rng):
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))                                 # Haar-ish


def offdiag_spearman(S_geom, S_F):
    iu = np.triu_indices(S_geom.shape[0], 1)
    a, b = S_geom[iu], S_F[iu]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return np.corrcoef(ra, rb)[0, 1]


def cos_gram(X):
    n = np.linalg.norm(X, axis=1)
    Xn = X / n[:, None]
    return Xn @ Xn.T


def run_scale(name, loader):
    rng = np.random.default_rng(SEED)
    W_dec, W_U, norm, alive = loader()
    idx = rng.choice(alive, min(K, alive.size), replace=False)
    D = W_dec[idx]                                                 # (K,d) real directions
    d = D.shape[1]

    # (1) decoder isotropy
    s = np.linalg.svd(D, compute_uv=False)
    iso = s[0] ** 2 / (s ** 2).sum()

    # metric M = W_U^T W_U  (d x d), and gained directions
    M = W_U.T @ W_U
    Fg = gained(D, norm)                                           # (K,d)

    # (2) direct-logit shared mode: top eig frac of F F^T = Fg M Fg^T
    GramF = Fg @ M @ Fg.T
    ev = np.linalg.eigvalsh(GramF)
    shared = ev[-1] / ev.sum()

    # F cosine similarity (from Gram) and decoder-geometry cosine
    rn = np.sqrt(np.clip(np.diag(GramF), 1e-30, None))
    S_F = GramF / np.outer(rn, rn)
    S_geom = cos_gram(D)
    coup_real = offdiag_spearman(S_geom, S_F)

    # (3) spectrum-matched null on M
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0, None)
    coup_null = np.empty(N_NULL)
    for t in range(N_NULL):
        Q = rand_orth(d, rng)
        Mn = (Q * w) @ Q.T
        Gn = Fg @ Mn @ Fg.T
        rnn = np.sqrt(np.clip(np.diag(Gn), 1e-30, None))
        coup_null[t] = offdiag_spearman(S_geom, Gn / np.outer(rnn, rnn))
    z = (coup_real - coup_null.mean()) / (coup_null.std() + 1e-12)
    return dict(name=name, K=idx.size, d=d, iso=iso, shared=shared,
                coup_real=coup_real, coup_null=coup_null.mean(),
                coup_null_sd=coup_null.std(), z=z)


def main():
    rows = [run_scale("GPT2-117M L8", load_gpt2_arrays),
            run_scale("Gemma-2B L12", load_gemma_arrays)]
    print(f"\n{'scale':<14}{'K':>6}{'d':>6}{'dec_iso':>9}{'dl_shared':>11}"
          f"{'coup_real':>11}{'coup_null':>11}{'z':>9}")
    for r in rows:
        print(f"{r['name']:<14}{r['K']:>6}{r['d']:>6}{r['iso']:>9.4f}"
              f"{r['shared']:>11.4f}{r['coup_real']:>+11.4f}"
              f"{r['coup_null']:>+11.4f}{r['z']:>+9.2f}")
    print("\ndec_iso   : top-1 SVD energy of decoder (isotropy: lower = more structureless)")
    print("dl_shared : top eigenvalue frac of direct-logit F F^T (feature-agnostic response)")
    print("z         : geometry->logit coupling, real vs spectrum-matched null on unembed")
    print("            z<=0 => coupling is a spectrum artifact (NEGATIVE, no rich geometry)")
    print("            z>>0 => real structural coupling emerges at scale (CRITICAL MASS)")


if __name__ == "__main__":
    main()
