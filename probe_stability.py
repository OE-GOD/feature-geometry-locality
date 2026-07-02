"""Per-context stability + common-component check for the causal proxy."""
import argparse, numpy as np, torch
from run_layer import fetch, DEAD_LOG_SPARSITY
from causal_proxy import SAMPLE_TEXT, cos_pairs, binned
from probe_context import DIVERSE_TEXT, build_ctx

def effects_per_ctx(layer, idx, W_dec, alpha, contexts, model):
    """Return F of shape (n_ctx, len(idx), V): per-context mean-over-pos shift."""
    block = model.transformer.h[layer]
    state = {"dir": None}
    def hook(m, a):
        if state["dir"] is None: return None
        return (a[0] + alpha*state["dir"],) + a[1:]
    h = block.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            base = model(contexts).logits.mean(dim=1)  # (C,V)
            C, V = base.shape
            out = np.empty((C, len(idx), V))
            for r, fi in enumerate(idx):
                state["dir"] = torch.tensor(W_dec[fi], dtype=torch.float32)
                inj = model(contexts).logits.mean(dim=1)  # (C,V)
                out[:, r, :] = (inj - base).double().numpy()
                state["dir"] = None
    finally:
        h.remove()
    return out

def dr2(F, Sgeo, sgeo, iu, rng, nperm=20):
    Sfn = cos_pairs(F); sfn = Sfn[iu]
    _, real = binned(sgeo, sfn)
    perm = [binned(sgeo, cos_pairs(F[rng.permutation(F.shape[0])])[iu])[1] for _ in range(nperm)]
    Sg = Sgeo.copy(); np.fill_diagonal(Sg, -np.inf)
    nn = np.argsort(-Sg, axis=1)[:, 0]
    rank1 = float(Sfn[np.arange(F.shape[0]), nn].mean())
    return real, np.mean(perm), real-np.mean(perm), rank1, float(sfn.mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--contexts", type=int, default=24)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text", default="orig")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    W_dec, ls, _, _ = fetch(f"blocks.{args.layer}.hook_resid_pre")
    alive = np.where(ls > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    text = SAMPLE_TEXT if args.text=="orig" else DIVERSE_TEXT
    ctx = build_ctx(tok, text, args.contexts, args.seq)
    D = W_dec[idx]; Sgeo = cos_pairs(D); iu = np.triu_indices(len(idx),1); sgeo=Sgeo[iu]
    Fpc = effects_per_ctx(args.layer, idx, W_dec, args.alpha, ctx, model)  # (C,k,V)
    C = Fpc.shape[0]
    Fmean = Fpc.mean(0)
    print(f"text={args.text} layer={args.layer} k={len(idx)} C={C}")
    # 1) full mean-over-context result
    prng = np.random.default_rng(args.seed+999)
    r = dr2(Fmean, Sgeo, sgeo, iu, prng); print(f"  ALL-CTX-MEAN     real={r[0]:+.4f} perm={r[1]:+.4f} delta={r[2]:+.4f} rank1={r[3]:+.4f} base={r[4]:+.4f}")
    # 2) common component: subtract per-feature the across-feature mean shift (shared dir)
    Fcc = Fmean - Fmean.mean(0, keepdims=True)
    prng = np.random.default_rng(args.seed+999)
    r = dr2(Fcc, Sgeo, sgeo, iu, prng); print(f"  MINUS-COMMON     real={r[0]:+.4f} perm={r[1]:+.4f} delta={r[2]:+.4f} rank1={r[3]:+.4f} base={r[4]:+.4f}")
    # 3) single-context results (does one paragraph dominate?)
    deltas=[]; rank1s=[]
    for c in range(C):
        prng = np.random.default_rng(args.seed+999)
        rr = dr2(Fpc[c], Sgeo, sgeo, iu, prng); deltas.append(rr[2]); rank1s.append(rr[3])
    print(f"  SINGLE-CTX delta: mean={np.mean(deltas):+.4f} sd={np.std(deltas):.4f} min={np.min(deltas):+.4f} max={np.max(deltas):+.4f}")
    print(f"  SINGLE-CTX rank1: mean={np.mean(rank1s):+.4f} sd={np.std(rank1s):.4f} min={np.min(rank1s):+.4f} max={np.max(rank1s):+.4f}")
    # 4) split-half context stability of F (do two disjoint ctx halves agree?)
    h1 = Fpc[:C//2].mean(0); h2 = Fpc[C//2:].mean(0)
    # cosine between the two half-estimates of each feature's shift vector
    def rowcos(A,B):
        an=A/np.clip(np.linalg.norm(A,axis=1,keepdims=True),1e-12,None)
        bn=B/np.clip(np.linalg.norm(B,axis=1,keepdims=True),1e-12,None)
        return (an*bn).sum(1)
    hc = rowcos(h1,h2)
    print(f"  SPLIT-HALF F self-cos: mean={hc.mean():+.4f} sd={hc.std():.4f} (1=stable across contexts)")

if __name__=="__main__":
    main()
