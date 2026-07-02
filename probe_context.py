"""Adversarial context/injection robustness probe for the causal proxy."""
import argparse, json, numpy as np, torch
from run_layer import fetch, DEAD_LOG_SPARSITY
from causal_proxy import SAMPLE_TEXT, cos_pairs, binned

# A much more diverse corpus: different domains, registers, syntax, lengths.
DIVERSE_TEXT = """Hey, are you coming to the party tonight or not? I honestly can't tell.
def merge(a, b): return sorted(a + b) # naive but fine for small lists.
The quarterly earnings fell short of analyst expectations, sending shares down 4%.
Once upon a time, in a kingdom by the sea, there lived a lonely lighthouse keeper.
BREAKING: officials confirm the bridge will reopen Monday after months of repairs.
sudo apt-get update && sudo apt-get install -y build-essential cmake ninja.
The patient presented with acute abdominal pain, nausea, and a low-grade fever.
Whereas the parties hereto agree to the terms set forth in Section 3.2 below,
lol no way did that actually happen?? send pics or it didn't happen fr fr.
The derivative of sin(x) is cos(x), and the integral of 1/x is the natural log.
Preheat the oven to 425 degrees and roast the vegetables for twenty-five minutes.
"To be, or not to be, that is the question," the actor whispered to the empty hall.
Traffic on the interstate is backed up for six miles due to an overturned truck.
Our results suggest that sparse autoencoders recover interpretable linear features.
The dog chased the ball across the wet grass, barking with pure delight."""

def build_ctx(tok, text, n_ctx, seq_len):
    ids = tok(text.replace("\n", " "), return_tensors="pt")["input_ids"][0]
    chunks, step = [], max(1, (ids.shape[0] - seq_len) // n_ctx)
    for i in range(0, ids.shape[0] - seq_len, step):
        chunks.append(ids[i:i+seq_len])
        if len(chunks) >= n_ctx: break
    return torch.stack(chunks)

def causal_effects(layer, idx, W_dec, alpha, contexts, model, readout):
    block = model.transformer.h[layer]
    state = {"dir": None}
    def pre_hook(m, args):
        if state["dir"] is None: return None
        return (args[0] + alpha * state["dir"],) + args[1:]
    h = block.register_forward_pre_hook(pre_hook)
    def reduce(lg):
        if readout == "mean": return lg.mean(dim=(0,1))
        elif readout == "last": return lg[:, -1, :].mean(dim=0)
        else: raise ValueError(readout)
    try:
        with torch.no_grad():
            base = reduce(model(contexts).logits)
            V = base.shape[-1]; out = np.empty((len(idx), V))
            for r, fi in enumerate(idx):
                state["dir"] = torch.tensor(W_dec[fi], dtype=torch.float32)
                out[r] = (reduce(model(contexts).logits) - base).double().numpy()
                state["dir"] = None
    finally:
        h.remove()
    return out

def analyze(F, Sgeo, sgeo, iu, rng, nperm):
    Sfn = cos_pairs(F); sfn = Sfn[iu]
    _, real = binned(sgeo, sfn)
    perm = []
    for _ in range(nperm):
        pp = rng.permutation(F.shape[0])
        _, r2p = binned(sgeo, cos_pairs(F[pp])[iu]); perm.append(r2p)
    pm, ps = float(np.mean(perm)), float(np.std(perm))
    Sg = Sgeo.copy(); np.fill_diagonal(Sg, -np.inf)
    nn = np.argsort(-Sg, axis=1)[:, 0]
    rank1 = float(Sfn[np.arange(F.shape[0]), nn].mean())
    return dict(real_r2=round(real,4), perm=round(pm,4), perm_sd=round(ps,4),
                delta=round(real-pm,4), z=round((real-pm)/(ps+1e-9),2),
                rank1=round(rank1,4), baseline=round(float(sfn.mean()),4))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--contexts", type=int, default=24)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    W_dec, ls, _, _ = fetch(f"blocks.{args.layer}.hook_resid_pre")
    alive = np.where(ls > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    D = W_dec[idx]; Sgeo = cos_pairs(D)
    iu = np.triu_indices(len(idx), k=1); sgeo = Sgeo[iu]

    conds = [
        ("ORIG_mean", SAMPLE_TEXT, "mean", args.contexts),
        ("DIVERSE_mean", DIVERSE_TEXT, "mean", args.contexts),
        ("ORIG_last", SAMPLE_TEXT, "last", args.contexts),
        ("DIVERSE_last", DIVERSE_TEXT, "last", args.contexts),
        ("DIVERSE_mean_more", DIVERSE_TEXT, "mean", args.contexts*2),
    ]
    print(f"layer={args.layer} k={len(idx)} alpha={args.alpha} seq={args.seq}")
    for name, text, ro, nc in conds:
        # fresh rng for permutations so perm draws identical across conds
        prng = np.random.default_rng(args.seed + 999)
        ctx = build_ctx(tok, text, nc, args.seq)
        F = causal_effects(args.layer, idx, W_dec, args.alpha, ctx, model, ro)
        r = analyze(F, Sgeo, sgeo, iu, prng, args.perm)
        print(f"{name:20s} nctx={ctx.shape[0]:3d} real={r['real_r2']:+.4f} "
              f"perm={r['perm']:+.4f}±{r['perm_sd']:.4f} delta={r['delta']:+.4f} "
              f"z={r['z']:7.2f} rank1={r['rank1']:+.4f} base={r['baseline']:+.4f}")

if __name__ == "__main__":
    main()
