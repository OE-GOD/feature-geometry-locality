"""
Activation-based (non-tautological) scale test of the 'feature-agnostic response'.

The blog's core claim: a feature's functional response is ~99% one shared mode
that ignores which feature you touched. That number came from GPT-2-small. Co-firing
tests it on real activations (no ablation, no linear tautology): does the SAE
activation matrix collapse to one shared 'general activity' axis, and does decoder
geometry predict co-firing above a rank-matched null?

Run GPT-2-small and Gemma-2-2b through the SAME analysis core:
  - shared-activity axis = top-1 SVD energy of the activation matrix
  - geometry->co-firing coupling z vs rank-matched null (deflate 0/1/3)
Gemma harvest uses a pre-hook on layer 12 that captures the residual (hs[12],
validated L0~82) and stops the forward early -> only 12 of 26 layers computed.
"""

import glob
import time
import numpy as np
import torch

from nlcp_compare import cos_rows, coupling_r2, rank_matched_null

HOME = glob.os.path.expanduser("~")
torch.set_num_threads(8)


def _top1(M):                                       # top-1 SVD energy of rows
    sv = np.linalg.svd(M, compute_uv=False)
    return sv[0] ** 2 / (sv ** 2).sum()


def analyze(name, A, D, rng, rot=10, k_target=400):
    fires = (A > 0).sum(0)
    alive = np.where(fires >= 10)[0]
    if k_target and alive.size > k_target:          # match K across models
        alive = np.sort(rng.choice(alive, k_target, replace=False))
    A, D = A[:, alive], D[alive]
    K = D.shape[0]
    F = A.T.astype(np.float64)                       # K x positions (magnitudes)
    Fbin = (F > 0).astype(np.float64)               # K x positions (co-occurrence)
    top1 = _top1(F)                                  # raw (magnitude-weighted)
    top1b = _top1(Fbin - Fbin.mean(1, keepdims=True))  # binary, mean-removed
    iu = np.triu_indices(K, 1)
    sgeo = cos_rows(D)[iu]
    print(f"\n{name}: K={K} (matched), positions={A.shape[0]}, "
          f"mean fires/feat={fires[alive].mean():.0f}")
    print(f"  shared-activity axis: raw top-1 SVD = {top1:.4f} | "
          f"binary(mean-removed) top-1 = {top1b:.4f}")
    print(f"  {'defl':>4} {'coupling_R2':>12} {'rank-null':>18} {'z':>7}")
    for r in (0, 1, 3):
        real = coupling_r2(F, sgeo, iu, r)
        nm, ns = rank_matched_null(D, F, sgeo, iu, r, rot, rng)
        z = (real - nm) / (ns + 1e-9)
        print(f"  {r:>4} {real:>12.4f} {nm:>11.4f}±{ns:>6.4f} {z:>7.1f}")
    return top1, top1b


# ---------- GPT-2 harvest (fast) ----------
def run_gpt2(k=400, docs=400, seq=64, max_pos=8000, layer=8, seed=0):
    from real_effect import fetch_sae, load_gpt2
    from run_layer import DEAD_LOG_SPARSITY
    from cofire import build_pile_contexts, harvest
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(f"blocks.{layer}.hook_resid_pre")
    alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(alive, min(k, alive.size), replace=False))
    tok, model = load_gpt2()
    ctx = build_pile_contexts(tok, docs, seq, max_pos)
    A = harvest(layer, idx, ctx, model, W_enc, b_enc, b_dec)
    return analyze("GPT-2-117M L8", A, W_dec[idx].astype(np.float64),
                   np.random.default_rng(seed))


# ---------- Gemma harvest (forward passes, early-stop at layer 12) ----------
class _Stop(Exception):
    pass


def gemma_contexts(tok, docs, seq, max_pos):
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    bos = tok.bos_token_id
    toks = []
    for i in range(min(docs, len(ds))):
        toks.extend([t for t in tok(ds[i]["text"])["input_ids"] if t != bos])
        if len(toks) >= max_pos:
            break
    toks = toks[:max_pos]
    toks = toks[:(len(toks) // seq) * seq]
    return torch.tensor(toks).reshape(-1, seq)


def run_gemma(k=400, docs=400, seq=48, max_pos=6000, layer=12, seed=0, batch=8, width="16k"):
    p = glob.glob(f"{HOME}/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/"
                  f"snapshots/*/layer_{layer}/width_{width}/*/params.npz")[0]
    sae = np.load(p)
    W_dec = sae["W_dec"].astype(np.float64)
    cache = f"gemma_acts_L{layer}_w{width}_{max_pos}.npy"
    if glob.os.path.exists(cache):
        A = np.load(cache)
        print(f"  [gemma] loaded cached acts {A.shape} from {cache}", flush=True)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        W_enc = torch.tensor(sae["W_enc"]).float()
        b_enc = torch.tensor(sae["b_enc"]).float()
        thr = torch.tensor(sae["threshold"]).float()
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32).eval()
        ctx = gemma_contexts(tok, docs, seq, max_pos)
        cap = {}

        def pre(_m, args):
            cap["h"] = args[0].detach()
            raise _Stop()

        handle = model.model.layers[layer].register_forward_pre_hook(pre)
        acts_all = []
        t0 = time.time()
        for bi, i in enumerate(range(0, ctx.shape[0], batch)):
            try:
                with torch.no_grad():
                    model(ctx[i:i + batch])
            except _Stop:
                pass
            h = cap["h"].reshape(-1, cap["h"].shape[-1])
            with torch.no_grad():
                pre_acts = h @ W_enc + b_enc
                a = (pre_acts * (pre_acts > thr)).float().numpy()
            acts_all.append(a)
            if bi == 0:
                print(f"  [gemma] first batch {time.time()-t0:.1f}s "
                      f"({batch}x{seq} tok, 12/26 layers) -> {ctx.shape[0]//batch} batches",
                      flush=True)
        handle.remove()
        del model                                             # free ~10GB before analysis
        A = np.concatenate(acts_all, 0)
        np.save(cache, A)
        print(f"  [gemma] harvested + cached {A.shape} -> {cache}", flush=True)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)     # Gemma massive-acts safety
    return analyze(f"Gemma-2B L{layer} w{width}", A, W_dec, np.random.default_rng(seed))


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("gpt2", "both"):
        run_gpt2()
    if which in ("gemma", "both"):
        run_gemma()
    if which == "gemma65k":
        run_gemma(width="65k")
