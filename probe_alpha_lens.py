"""
Alpha-linearization lens probe.

Two attacks on the causal proxy claim:
  (A) linearity: is F(alpha) ~ alpha * J D for the tested alphas? (cos-invariance)
  (B) spectrum-matched null (the null that KILLED the direct proxy):
      linearize the causal map to J (V x d) by basis injection, form M=J^T J,
      and ask whether the Sgeo->Sfn halo exceeds a random-eigenvector map with
      the SAME spectrum. If delta_r2_spectrum ~ 0, the causal halo is the same
      proxy tautology, just with J in place of wte.
"""
import numpy as np, torch, json, sys
from run_layer import fetch, DEAD_LOG_SPARSITY
import causal_proxy as cp
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

LAYER = 8
K = 200
SEED = 0
NCTX = 16
SEQ = 32

W_dec, ls, wte, ln_w = fetch(f'blocks.{LAYER}.hook_resid_pre')
alive = np.where(ls > DEAD_LOG_SPARSITY)[0]
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(alive, size=min(K, alive.size), replace=False))

tok = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
ctx = cp.build_contexts(tok, NCTX, SEQ)

D = W_dec[idx].astype(np.float64)   # (K, 768)
d_model = D.shape[1]

# ---- (A) F at several alphas for the subset, check cosine invariance ----
def sfn_at(alpha):
    F = cp.causal_effects(LAYER, idx, W_dec, alpha, ctx, model)
    return cp.cos_pairs(F), F

Sgeo = cp.cos_pairs(D)
iu = np.triu_indices(K, k=1)
sgeo = Sgeo[iu]

alphas = [0.25, 1.0, 2.0, 6.0, 12.0, 24.0]
sfns = {}
Fs = {}
for a in alphas:
    S, F = sfn_at(a)
    sfns[a] = S[iu]
    Fs[a] = F
    print(f'computed F alpha={a}')

# cosine invariance: correlation of sfn vectors across alpha
print('\n(A) cosine-invariance of Sfn across alpha (corr with alpha=0.25):')
ref = sfns[0.25]
for a in alphas:
    c = np.corrcoef(ref, sfns[a])[0,1]
    print(f'   alpha={a:5}: corr={c:.4f}  mean|sfn|={np.abs(sfns[a]).mean():.4f}')

# delta_r2 vs permutation null at each alpha (reproduce claim metric)
def binned_r2(sg, sf):
    _, r2 = cp.binned(sg, sf)
    return r2
def perm_null(F, nperm=20):
    r=[]
    for _ in range(nperm):
        pp = rng.permutation(K)
        r.append(binned_r2(sgeo, cp.cos_pairs(F[pp])[iu]))
    return np.mean(r), np.std(r)
print('\n(A2) delta_r2 vs PERMUTATION null (claim metric) across alpha:')
for a in alphas:
    real = binned_r2(sgeo, sfns[a])
    pm, ps = perm_null(Fs[a])
    print(f'   alpha={a:5}: real_r2={real:.4f} perm={pm:.4f}+/-{ps:.4f} delta={real-pm:.4f} z={(real-pm)/(ps+1e-9):.1f}')

# ---- (B) linearized Jacobian J via basis injection ----
print('\n(B) building linearized Jacobian J via 768 basis injections (alpha_probe=1.0)...')
ALPHA_PROBE = 1.0
block = model.transformer.h[LAYER]
state = {'dir': None}
def pre_hook(m, args):
    if state['dir'] is None: return None
    return (args[0] + ALPHA_PROBE * state['dir'],) + args[1:]
handle = block.register_forward_pre_hook(pre_hook)
with torch.no_grad():
    base = model(ctx).logits.mean(dim=(0,1))  # (V,)
    V = base.shape[-1]
    Jt = np.empty((d_model, V), dtype=np.float64)  # J^T : d_model x V, J[:,j] col
    e = torch.zeros(d_model, dtype=torch.float32)
    for j in range(d_model):
        e.zero_(); e[j] = 1.0
        state['dir'] = e.clone()
        shift = (model(ctx).logits.mean(dim=(0,1)) - base).double().numpy()
        Jt[j] = shift / ALPHA_PROBE
        state['dir'] = None
        if j % 128 == 0: print(f'   basis {j}/{d_model}')
handle.remove()

# M = J^T J  (d_model x d_model), J is (V, d_model) => M[a,b]=sum_v J[v,a]J[v,b]
# Jt is (d_model, V) = J^T, so M = Jt @ Jt.T
M = Jt @ Jt.T
print('M shape', M.shape, 'symmetric err', np.abs(M-M.T).max())

# verify linearization: F_pred = D @ J^T ; cos should match cos of measured F at small alpha
F_lin = D @ Jt.T  # (K, V)  == predicted causal effect (linear)
S_lin = cp.cos_pairs(F_lin)[iu]
for a in [0.25, 1.0, 2.0, 6.0, 12.0]:
    c = np.corrcoef(S_lin, sfns[a])[0,1]
    print(f'   linear-J Sfn vs measured alpha={a}: corr={c:.4f}')

# ---- spectrum-matched null on M (null_control style) ----
def cos_under_metric(y, Mm):
    with np.errstate(all='ignore'):
        G = y @ Mm @ y.T
        nrm = np.sqrt(np.clip(np.diag(G), 1e-12, None))
        return G / np.outer(nrm, nrm)

sfn_real = cos_under_metric(D, M)[iu]
real_r2 = binned_r2(sgeo, sfn_real)
evals = np.clip(np.linalg.eigvalsh(M), 0, None)
null_r2 = []
for r in range(10):
    Q,_ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    M_rand = (Q*evals) @ Q.T
    sfn_r = cos_under_metric(D, M_rand)[iu]
    null_r2.append(binned_r2(sgeo, sfn_r))
null_r2 = np.array(null_r2)
print('\n(B) SPECTRUM-MATCHED null on linearized causal map J:')
print(f'   real_r2(J)={real_r2:.4f}  spectrum_null_r2={null_r2.mean():.4f}+/-{null_r2.std():.4f}')
print(f'   delta_r2_spectrum = {real_r2-null_r2.mean():.4f}  z={(real_r2-null_r2.mean())/(null_r2.std()+1e-9):.1f}')

json.dump({
 'resid_norm_L8': 198.0,
 'alphas': alphas,
 'sfn_corr_vs_a0.25': {str(a): float(np.corrcoef(ref,sfns[a])[0,1]) for a in alphas},
 'delta_r2_perm': {str(a): float(binned_r2(sgeo,sfns[a])-perm_null(Fs[a])[0]) for a in alphas},
 'linJ_corr': {str(a): float(np.corrcoef(S_lin,sfns[a])[0,1]) for a in [0.25,1.0,2.0,6.0,12.0]},
 'real_r2_J': float(real_r2),
 'spectrum_null_r2_mean': float(null_r2.mean()),
 'spectrum_null_r2_sd': float(null_r2.std()),
 'delta_r2_spectrum': float(real_r2-null_r2.mean()),
}, open('probe_alpha_lens_out.json','w'), indent=2)
print('\nwrote probe_alpha_lens_out.json')
