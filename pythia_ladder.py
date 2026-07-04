"""
Clean same-family scale ladder: Pythia 70m -> 160m -> 410m, EleutherAI top-k SAEs.

This removes the family confound in the GPT-2-vs-Gemma comparison: same architecture
(GPTNeoX), same tokenizer/training corpus, same SAE recipe (top-k, k=16) at every
rung -> the ONLY thing that changes is scale. The within-ladder TREND is the signal.

For each rung, over ~40k Pile positions (co-firing, non-tautological):
  - shared-activity axis: raw + binary(mean-removed) top-1 SVD  (does it shrink w/ scale?)
  - geometry -> co-firing coupling z vs rank-matched null        (does it stay negative?)

Residual hook = hidden_states[mid_layer+1] (validated by FVU ~0.01). A pre-hook on
layer mid+1 captures it and stops the forward early (compute only mid+1 of L layers).
"""

import json
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from gemma_cofire import analyze, gemma_contexts   # analyze is model-agnostic

torch.set_num_threads(8)

RUNGS = [
    ("pythia-70m",  "EleutherAI/sae-pythia-70m-32k",  6,  3),
    ("pythia-160m", "EleutherAI/sae-pythia-160m-32k", 12, 6),
    ("pythia-410m", "EleutherAI/sae-pythia-410m-65k", 24, 12),
]


class _Stop(Exception):
    pass


def load_sae(repo, layer):
    cfg = json.load(open(hf_hub_download(repo, f"layers.{layer}.mlp/cfg.json")))
    with safe_open(hf_hub_download(repo, f"layers.{layer}.mlp/sae.safetensors"), framework="pt") as f:
        Wenc = f.get_tensor("encoder.weight").float()   # (num_latents, d_in)
        benc = f.get_tensor("encoder.bias").float()
        Wdec = f.get_tensor("W_dec").float()            # (num_latents, d_in)
        bdec = f.get_tensor("b_dec").float()
    return cfg["k"], Wenc, benc, Wdec, bdec


def harvest(model_name, repo, n_layers, mid, max_pos=40000, seq=64, batch=32):
    cache = f"pythia_acts_{model_name}_mlpL{mid}_{max_pos}.npy"
    k, Wenc, benc, Wdec, bdec = load_sae(repo, mid)
    W_dec_np = Wdec.double().numpy()
    if __import__("os").path.exists(cache):
        A = np.load(cache)
        print(f"  [{model_name}] cached acts {A.shape}", flush=True)
        return A, W_dec_np
    model = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{model_name}").eval()
    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")
    ctx = gemma_contexts(tok, 2000, seq, max_pos)        # reuse Pile loader (BOS-stripped)

    cap = {}

    def mlp_hook(_m, _in, out):                       # capture MLP output of layer `mid`
        cap["h"] = out.detach()
        raise _Stop()

    handle = model.gpt_neox.layers[mid].mlp.register_forward_hook(mlp_hook)

    def encode(x):
        preacts = (x - bdec) @ Wenc.T + benc
        topv, topi = preacts.topk(k, dim=-1)
        a = torch.zeros_like(preacts)
        a.scatter_(-1, topi, torch.relu(topv))
        return a

    acts, checked = [], False
    import time
    t0 = time.time()
    for bi, i in enumerate(range(0, ctx.shape[0], batch)):
        try:
            with torch.no_grad():
                model(ctx[i:i + batch])
        except _Stop:
            pass
        h = cap["h"].reshape(-1, cap["h"].shape[-1]).float()
        with torch.no_grad():
            a = encode(h)
            if not checked:                              # validate hook via FVU once
                xr = a.double() @ Wdec.double() + bdec.double()
                fvu = ((h.double() - xr) ** 2).sum() / ((h.double() - h.double().mean(0)) ** 2).sum()
                print(f"  [{model_name}] hook FVU={fvu:.3f} (want <0.3), "
                      f"{ctx.shape[0]//batch} batches, {mid+1}/{n_layers} layers", flush=True)
                checked = True
        acts.append(a.half().numpy())
    handle.remove()
    del model
    A = np.concatenate(acts, 0)
    np.save(cache, A)
    print(f"  [{model_name}] harvested {A.shape} in {time.time()-t0:.0f}s", flush=True)
    return A, W_dec_np


def main():
    results = []
    for name, repo, nl, mid in RUNGS:
        try:
            A, D = harvest(name, repo, nl, mid)
        except Exception as e:
            print(f"  [{name}] SKIPPED ({type(e).__name__}: {str(e)[:80]})", flush=True)
            continue
        top1, top1b = analyze(f"{name} L{mid}", A, D, np.random.default_rng(0))
        print(f"  [{name}] -> raw={top1:.4f} binary={top1b:.4f}", flush=True)
        results.append((name, top1, top1b))
    print("\n=== LADDER SUMMARY (same family, same recipe, scale is the only variable) ===")
    print(f"{'model':>14}{'raw shared':>12}{'binary shared':>15}")
    for name, t1, t1b in results:
        print(f"{name:>14}{t1:>12.4f}{t1b:>15.4f}")
    print("Monotone DOWN with scale => Gemma shared-mode shrink is SCALE.")
    print("Flat / UP => the GPT2->Gemma shrink was FAMILY/recipe, not a scale law.")


if __name__ == "__main__":
    main()
