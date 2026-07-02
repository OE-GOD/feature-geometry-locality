"""
Honest linearity check for the blog: how close are the MEASURED nonlinear
forward-pass causal effects to the linear-J prediction, and what is the actual
residual-stream norm at layer 8? (Replaces the tautological corr=1.0000.)
"""
import numpy as np
import torch
from run_layer import fetch, DEAD_LOG_SPARSITY
from causal_proxy import build_contexts, causal_effects

HOOK, LAYER, K, SEED = "blocks.8.hook_resid_pre", 8, 200, 0

W_dec, log_sp, _, _ = fetch(HOOK)
alive = np.where(log_sp > DEAD_LOG_SPARSITY)[0]
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(alive, size=K, replace=False))
D = W_dec[idx].astype(np.float64)
Jt = np.load("Jt_L8_c8.npy")  # (768, V) linear response

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
contexts = build_contexts(tok, 16, 32)

# --- residual norm at layer 8 input, over the contexts ---
norms = {}
def grab(mod, args):
    h = args[0]
    norms["v"] = h.norm(dim=-1).flatten().double().numpy()
    return None
hd = model.transformer.h[LAYER].register_forward_pre_hook(grab)
with torch.no_grad():
    model(contexts)
hd.remove()
n = norms["v"]
print(f"resid_pre L8 norm over {n.size} positions: mean {n.mean():.1f}  "
      f"median {np.median(n):.1f}  p10 {np.percentile(n,10):.1f}  "
      f"p90 {np.percentile(n,90):.1f}")

def cos_off(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T
    return S[np.triu_indices(X.shape[0], 1)]

F_lin = D @ Jt                      # linear prediction
lin_cos = cos_off(F_lin)
for a in (2.0, 6.0, 12.0):
    F_meas = causal_effects(LAYER, idx, W_dec, a, contexts, model)  # true forward
    r = np.corrcoef(cos_off(F_meas), lin_cos)[0, 1]
    print(f"alpha={a:>4}: corr(cos(measured causal F), cos(linear-J prediction)) "
          f"= {r:.4f}")
