"""
The model's-own-ruler test. PREREGISTERED: see PREREG_ruler.md (frozen before data).

Ruler: inject alpha*W_dec[i] at the LAST position of a real context at layer 8,
propagate through blocks 8..11 + ln_f, record the change in the final residual.
Earlier positions are causally unaffected, so their per-block K/V come from one
clean forward (exactness gate G0 verifies this harness at alpha=0).

S_ruler(i,j) = cos of concatenated per-context deflated responses.
Function     = co-firing on DISJOINT text (docs 500+), as in cofire.py.
Null         = N random orthogonal rotations of the decoder set before injection
               (preserves all pairwise cosines; destroys feature-weight alignment).
Verdict per PREREG_ruler.md decision rule. Checkpoints after every condition.
"""

import argparse
import os
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY

LAYER = 8
ALPHA = 6.0
SEQ = 48
torch.set_num_threads(8)


# ---------- data ----------

def pile_contexts(tok, n_ctx, seq, doc_start):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    toks, i = [], doc_start
    while len(toks) < n_ctx * seq and i < len(ds):
        toks.extend(tok(ds[i]["text"])["input_ids"])
        i += 1
    toks = toks[: n_ctx * seq]
    return torch.tensor(toks).reshape(n_ctx, seq)


# ---------- the ruler harness ----------

def split_heads(t, nh, hd):
    return t.reshape(t.shape[0], nh, hd)


def clean_pass(model, ids):
    """One clean forward; returns per-layer inputs hs[0..12] (hs[12]=ln_f output)."""
    with torch.no_grad():
        out = model(ids.unsqueeze(0), output_hidden_states=True)
    return [h[0] for h in out.hidden_states]          # each (seq, 768)


def prefix_kv(model, hs, layer):
    """Clean per-position K,V for `layer` from its block input hs[layer]."""
    blk = model.transformer.h[layer]
    nh, hd = blk.attn.num_heads, blk.attn.head_dim
    with torch.no_grad():
        qkv = blk.attn.c_attn(blk.ln_1(hs[layer]))     # (seq, 3*768)
    _, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
    return split_heads(k, nh, hd), split_heads(v, nh, hd)   # (seq, nh, hd)


def propagate(model, hs, kvs, X):
    """Run injected last-position states X (K,768) through blocks LAYER..11 + ln_f.
    kvs[l] = clean (K_prefix, V_prefix) for positions 0..p-1 at block l."""
    p = hs[0].shape[0] - 1
    x = X
    for l in range(LAYER, len(model.transformer.h)):
        blk = model.transformer.h[l]
        nh, hd = blk.attn.num_heads, blk.attn.head_dim
        kp, vp = kvs[l]                                # (seq, nh, hd) clean
        a = blk.ln_1(x)
        qkv = blk.attn.c_attn(a)
        q, k, v = qkv.split(qkv.shape[-1] // 3, dim=-1)
        q, k, v = (split_heads(t, nh, hd) for t in (q, k, v))     # (K, nh, hd)
        sc_pre = torch.einsum("khd,phd->khp", q, kp[:p]) / hd ** 0.5
        sc_self = (q * k).sum(-1, keepdim=True) / hd ** 0.5       # (K, nh, 1)
        w = torch.softmax(torch.cat([sc_pre, sc_self], dim=-1), dim=-1)
        out = torch.einsum("khp,phd->khd", w[..., :p], vp[:p]) + w[..., p:] * v
        out = blk.attn.c_proj(out.reshape(out.shape[0], -1))
        x = x + out
        x = x + blk.mlp(blk.ln_2(x))
    return model.transformer.ln_f(x)                   # (K, 768)


def responses(model, contexts, D):
    """Delta ln_f-residual per feature per context. D: (K,768) directions.
    Returns (K, C, 768) AFTER per-context mean-over-features removal."""
    K = D.shape[0]
    R = np.empty((K, contexts.shape[0], D.shape[1]), dtype=np.float64)
    for c in range(contexts.shape[0]):
        hs = clean_pass(model, contexts[c])
        kvs = {l: prefix_kv(model, hs, l) for l in range(LAYER, len(model.transformer.h))}
        base = hs[LAYER][-1]                            # clean block-8 input, last pos
        with torch.no_grad():
            X = base.unsqueeze(0) + ALPHA * D           # (K, 768)
            fin = propagate(model, hs, kvs, X)          # (K, 768)
        d = (fin - hs[-1][-1]).double().numpy()
        R[:, c, :] = d - d.mean(0, keepdims=True)       # shared-mode removal
    return R


def g0_check(model, contexts):
    """alpha=0 must reproduce the clean final residual."""
    hs = clean_pass(model, contexts[0])
    kvs = {l: prefix_kv(model, hs, l) for l in range(LAYER, len(model.transformer.h))}
    with torch.no_grad():
        fin = propagate(model, hs, kvs, hs[LAYER][-1].unsqueeze(0))
    return float((fin[0] - hs[-1][-1]).abs().max())


# ---------- statistics ----------

def cos_rows(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Xn @ Xn.T


def ranks(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def spearman(a, b):
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial_spearman(a, b, ctrl):
    ra, rb, rc = ranks(a), ranks(b), ranks(ctrl)
    A = np.stack([rc, np.ones_like(rc)], 1)
    res_a = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    res_b = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return float(np.corrcoef(res_a, res_b)[0, 1])


def rand_orth(d, rng):
    Q, Rm = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(Rm))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=400)
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--nulls", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.k, args.contexts, args.nulls = 20, 4, 2
    tag = "smoke" if args.smoke else "full"

    hook = f"blocks.{LAYER}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    D_np = W_dec[idx].astype(np.float64)

    tok, model = load_gpt2()
    ctx_ruler = pile_contexts(tok, args.contexts, SEQ, doc_start=0)

    # G0 harness exactness
    g0 = g0_check(model, ctx_ruler)
    print(f"[G0] alpha=0 max|delta| = {g0:.2e}  ({'PASS' if g0 < 1e-3 else 'FAIL'})",
          flush=True)

    # co-firing on DISJOINT text (docs 500+)
    from cofire import harvest
    ctx_fn = pile_contexts(tok, 8000 // 64, 64, doc_start=500)
    A = harvest(LAYER, idx, ctx_fn, model, W_enc, b_enc, b_dec)
    fires = (A > 0).sum(0)
    keep = fires >= 10
    print(f"[fn] co-firing: {keep.sum()}/{idx.size} features fire>=10 on disjoint text",
          flush=True)

    # real ruler
    D_t = torch.tensor(D_np, dtype=torch.float32)
    R = responses(model, ctx_ruler, D_t)               # (K, C, 768) deflated
    np.save(f"ruler_{tag}_real.npy", R.astype(np.float32))
    print("[ruler] real responses done", flush=True)

    # matrices on kept features
    Rk = R[keep]
    K = int(keep.sum())
    iu = np.triu_indices(K, 1)
    s_geo = cos_rows(D_np[keep])[iu]
    s_fn = cos_rows(A.T[keep].astype(np.float64))[iu]
    flat = Rk.reshape(K, -1)
    s_ruler = cos_rows(flat)[iu]

    # gates G1, G2
    h = Rk.shape[1] // 2
    g1 = spearman(cos_rows(Rk[:, :h].reshape(K, -1))[iu],
                  cos_rows(Rk[:, h:].reshape(K, -1))[iu])
    g2 = spearman(s_ruler, s_geo)
    # descriptive: shared energy pre-deflation is removed at source; report resid top-1
    sv = np.linalg.svd(flat, compute_uv=False)
    print(f"[G1] split-half rho = {g1:+.3f} ({'PASS' if g1 > 0.3 else 'FAIL'})",
          flush=True)
    print(f"[G2] ruler-vs-cosine rho = {g2:+.3f} ({'PASS' if g2 > 0.2 else 'FAIL'})",
          flush=True)
    print(f"[desc] deflated-response top-1 SVD energy = {sv[0]**2/(sv**2).sum():.3f}",
          flush=True)

    real_rho = spearman(s_ruler, s_fn)
    real_part = partial_spearman(s_ruler, s_fn, s_geo)
    geo_rho = spearman(s_geo, s_fn)
    print(f"[real] rho(ruler,fn)={real_rho:+.4f}  partial|geo={real_part:+.4f}  "
          f"rho(geo,fn)={geo_rho:+.4f}", flush=True)

    # rotation nulls
    null_rho, null_part = [], []
    for t in range(args.nulls):
        Q = rand_orth(D_np.shape[1], np.random.default_rng(t + 1))
        Dq = torch.tensor(D_np @ Q.T, dtype=torch.float32)
        Rn = responses(model, ctx_ruler, Dq)[keep]
        s_n = cos_rows(Rn.reshape(K, -1))[iu]
        null_rho.append(spearman(s_n, s_fn))
        null_part.append(partial_spearman(s_n, s_fn, s_geo))
        np.savez(f"ruler_{tag}_null{t}.npz", rho=null_rho[-1], part=null_part[-1])
        print(f"[null {t}] rho={null_rho[-1]:+.4f} partial={null_part[-1]:+.4f}",
              flush=True)

    nr, np_ = np.array(null_rho), np.array(null_part)
    z = (real_rho - nr.mean()) / (nr.std() + 1e-12)
    zp = (real_part - np_.mean()) / (np_.std() + 1e-12)
    print(f"\n=== VERDICT INPUTS (PREREG_ruler.md) ===")
    print(f"z        = {z:+.2f}   (real {real_rho:+.4f} vs null {nr.mean():+.4f}"
          f" ± {nr.std():.4f})")
    print(f"z_partial= {zp:+.2f}   (real {real_part:+.4f} vs null {np_.mean():+.4f}"
          f" ± {np_.std():.4f})")
    gates = (g0 < 1e-3) and (g1 > 0.3) and (g2 > 0.2)
    if not gates:
        v = "INDETERMINATE (instrument gate failure)"
    elif z >= 4 and zp >= 4:
        v = "CONFIRM H1"
    elif z <= 2 or zp <= 2:
        v = "REFUTE H1"
    else:
        v = "INDETERMINATE"
    print(f"VERDICT: {v}")


if __name__ == "__main__":
    main()
