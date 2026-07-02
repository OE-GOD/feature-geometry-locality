"""
Measure real on-distribution SAE feature ablation effects on GPT-2 logits.

For each sampled alive SAE feature, this script finds real corpus positions
where the feature fires, subtracts its actual decoder contribution at the
residual stream hook, and stores the mean clean-minus-ablated logit vector.
"""

import argparse

import numpy as np
import torch

from causal_proxy import build_contexts
from run_layer import DEAD_LOG_SPARSITY, GPT2_REPO, SAE_REPO, load_np


def fetch_sae(hook):
    from huggingface_hub import hf_hub_download

    w_path = hf_hub_download(SAE_REPO, f"{hook}/sae_weights.safetensors")
    s_path = hf_hub_download(SAE_REPO, f"{hook}/sparsity.safetensors")
    W_dec, W_enc, b_enc, b_dec = load_np(
        w_path, "W_dec", "W_enc", "b_enc", "b_dec"
    )
    (log_sparsity,) = load_np(s_path, "sparsity")
    return W_dec, W_enc, b_enc, b_dec, log_sparsity


def load_gpt2():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(GPT2_REPO)
    model = GPT2LMHeadModel.from_pretrained(GPT2_REPO).eval()
    return tok, model


def encode_targets(H, W_enc, b_enc, b_dec, idx):
    W_sub = torch.from_numpy(W_enc[:, idx])
    b_sub = torch.from_numpy(b_enc[idx])
    centered = H - torch.from_numpy(b_dec)
    return torch.relu(centered @ W_sub + b_sub)


def reconstruction_cosine(H, W_dec, W_enc, b_enc, b_dec, n_sample=32):
    n = min(n_sample, H.shape[0])
    sample = H[:n]
    W_enc_t = torch.from_numpy(W_enc)
    W_dec_t = torch.from_numpy(W_dec)
    b_enc_t = torch.from_numpy(b_enc)
    b_dec_t = torch.from_numpy(b_dec)

    with torch.no_grad():
        a = torch.relu((sample - b_dec_t) @ W_enc_t + b_enc_t)
        recon = a @ W_dec_t + b_dec_t
        cos = torch.nn.functional.cosine_similarity(sample, recon, dim=1)
    return float(cos.mean().item())


def collect_clean(layer, contexts, model):
    block = model.transformer.h[layer]
    recorded = []

    def pre_hook(module, args):
        recorded.append(args[0].detach().clone())
        return None

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            clean_logits = model(contexts).logits.detach().clone()
    finally:
        handle.remove()

    if len(recorded) != 1:
        raise RuntimeError("expected exactly one recorded residual tensor")
    return clean_logits, recorded[0]


def real_effects(layer, idx, W_dec, activations, firing_counts, clean_logits,
                 contexts, model, min_fires, mode="zero", n_rand=3, seed=0):
    block = model.transformer.h[layer]
    vocab = clean_logits.shape[-1]
    d = W_dec.shape[1]
    F = np.zeros((len(idx), vocab), dtype=np.float32)
    clean_flat = clean_logits.reshape(-1, vocab)
    acts = activations.reshape(-1, len(idx))
    dec = torch.from_numpy(W_dec[idx])
    state = {"row": None, "dir": None}

    def pre_hook(module, args):
        row = state["row"]
        if row is None:
            return None
        hidden = args[0]
        # ablate along state["dir"] (the real feature dir, or a control dir),
        # scaled by the REAL feature's per-position activation (magnitude match).
        delta = activations[:, :, row].unsqueeze(-1) * state["dir"].view(1, 1, -1)
        ablated = hidden - delta
        if mode == "normmatch":
            scale = hidden.norm(dim=-1, keepdim=True) / ablated.norm(
                dim=-1, keepdim=True).clamp_min(1e-6)
            ablated = ablated * scale
        return (ablated,) + args[1:]

    g = torch.from_numpy(
        np.random.default_rng(seed).standard_normal((n_rand, d)).astype(np.float32))
    rand_dirs = g / g.norm(dim=1, keepdim=True)

    def effect(row, fire_pos, direction):
        state["dir"] = direction
        abl = model(contexts).logits.reshape(-1, vocab)
        state["dir"] = None
        return (clean_flat.index_select(0, fire_pos)
                - abl.index_select(0, fire_pos)).mean(dim=0)

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            for row in range(len(idx)):
                if firing_counts[row] < min_fires:
                    continue
                fire_pos = torch.nonzero(acts[:, row] > 0, as_tuple=False).squeeze(1)
                state["row"] = row
                eff_real = effect(row, fire_pos, dec[row])
                if mode == "randctrl":
                    # subtract the magnitude-matched random-direction ablation:
                    # cancels the generic "perturb -> confidence collapse" axis,
                    # leaving what's specific to the feature's real direction.
                    eff_rand = torch.stack(
                        [effect(row, fire_pos, rand_dirs[j]) for j in range(n_rand)]
                    ).mean(dim=0)
                    eff_real = eff_real - eff_rand
                F[row] = eff_real.detach().cpu().numpy().astype(np.float32)
                state["row"] = None
    finally:
        state["row"] = None
        handle.remove()
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--contexts", type=int, default=48)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-fires", type=int, default=5)
    ap.add_argument("--ablation", choices=["zero", "normmatch", "randctrl"],
                    default="zero",
                    help="normmatch = preserve norm; randctrl = subtract "
                         "magnitude-matched random-direction ablation (shared-axis-free)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sparsity = fetch_sae(hook)

    alive = np.where(log_sparsity > DEAD_LOG_SPARSITY)[0]
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(alive, size=min(args.k, alive.size), replace=False))

    tok, model = load_gpt2()
    contexts = build_contexts(tok, args.contexts, args.seq)

    clean_logits, hidden = collect_clean(args.layer, contexts, model)
    H = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    activations_flat = encode_targets(H, W_enc, b_enc, b_dec, idx)
    firing_counts = (activations_flat > 0).sum(dim=0).cpu().numpy().astype(np.int64)
    activations = activations_flat.reshape(hidden.shape[0], hidden.shape[1], len(idx))

    recon_cos = reconstruction_cosine(H, W_dec, W_enc, b_enc, b_dec)
    F = real_effects(
        args.layer, idx, W_dec, activations, firing_counts, clean_logits,
        contexts, model, args.min_fires, mode=args.ablation, seed=args.seed
    )

    default_out = f"real_effect_L{args.layer}_k{args.k}_s{args.seed}"
    if args.ablation != "zero":
        default_out += f"_{args.ablation}"
    out_path = args.out or f"{default_out}.npz"
    np.savez(
        out_path,
        F=F.astype(np.float32, copy=False),
        idx=idx.astype(np.int64, copy=False),
        firing_counts=firing_counts.astype(np.int64, copy=False),
    )

    enough = firing_counts >= args.min_fires
    row_norms = np.linalg.norm(F, axis=1)
    nonzero_norms = row_norms[row_norms > 0]
    mean_l2 = float(nonzero_norms.mean()) if nonzero_norms.size else 0.0
    print(f"reconstruction sanity mean_cos={recon_cos:.4f}")
    print(f"features with firing_count >= min_fires: {int(enough.sum())}/{len(idx)}")
    print(
        f"firing_count mean={float(firing_counts.mean()):.2f} "
        f"median={float(np.median(firing_counts)):.2f}"
    )
    print(f"mean L2 norm of nonzero F_i rows={mean_l2:.6f}")
    nz = F[row_norms > 0]
    if nz.shape[0] > 1:
        sv = np.linalg.svd(nz.astype(np.float64), compute_uv=False)
        print(f"top-1 SVD energy (shared axis) = {sv[0]**2/(sv**2).sum():.4f}  "
              f"[ablation={args.ablation}]")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
