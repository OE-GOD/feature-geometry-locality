"""Part B: spectrum-matched null on the linearized causal map at layer 8."""
import numpy as np, torch, json, os
from run_layer import fetch, DEAD_LOG_SPARSITY
import causal_proxy as cp
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

LAYER=8; K=200; SEED=0; NCTX=8; SEQ=32; ALPHA_PROBE=1.0
W_dec, ls, wte, ln_w = fetch(f'blocks.{LAYER}.hook_resid_pre')
alive=np.where(ls>DEAD_LOG_SPARSITY)[0]
rng=np.random.default_rng(SEED)
idx=np.sort(rng.choice(alive,size=min(K,alive.size),replace=False))
tok=GPT2TokenizerFast.from_pretrained('gpt2')
model=GPT2LMHeadModel.from_pretrained('gpt2').eval()
ctx=cp.build_contexts(tok,NCTX,SEQ)
D=W_dec[idx].astype(np.float64); d_model=D.shape[1]

JT_PATH=f'Jt_L{LAYER}_c{NCTX}.npy'
if os.path.exists(JT_PATH):
    Jt=np.load(JT_PATH); print('loaded',JT_PATH)
else:
    block=model.transformer.h[LAYER]; state={'dir':None}
    def pre(m,a):
        if state['dir'] is None: return None
        return (a[0]+ALPHA_PROBE*state['dir'],)+a[1:]
    h=block.register_forward_pre_hook(pre)
    with torch.no_grad():
        base=model(ctx).logits.mean(dim=(0,1)); V=base.shape[-1]
        Jt=np.empty((d_model,V),dtype=np.float64)
        e=torch.zeros(d_model)
        for j in range(d_model):
            e.zero_(); e[j]=1.0; state['dir']=e.clone()
            Jt[j]=((model(ctx).logits.mean(dim=(0,1))-base).double().numpy())/ALPHA_PROBE
            state['dir']=None
            if j%128==0: print('basis',j,flush=True)
    h.remove(); np.save(JT_PATH,Jt); print('saved',JT_PATH)

iu=np.triu_indices(K,k=1)
Dn=D/np.clip(np.linalg.norm(D,axis=1,keepdims=True),1e-12,None)
Sgeo=Dn@Dn.T; sgeo=Sgeo[iu]
def r2(sf):
    _,v=cp.binned(sgeo,sf); return v

# real linearized causal function
F_lin=D@Jt                      # (K,V)
S_lin=cp.cos_pairs(F_lin)[iu]
real_r2=r2(S_lin)

# spectrum-matched null on M=J^T J
M=Jt@Jt.T
evals=np.clip(np.linalg.eigvalsh(M),0,None)
def cum(y,Mm):
    with np.errstate(all='ignore'):
        G=y@Mm@y.T; nrm=np.sqrt(np.clip(np.diag(G),1e-12,None)); return G/np.outer(nrm,nrm)
# sanity: cum(D,M) cos should equal cos of F_lin
S_M=cum(D,M)[iu]
print('corr(cos F_lin, cos under M):',round(float(np.corrcoef(S_lin,S_M)[0,1]),4))
null=[]
for _ in range(10):
    Q,_=np.linalg.qr(rng.standard_normal((d_model,d_model)))
    Mr=(Q*evals)@Q.T
    null.append(r2(cum(D,Mr)[iu]))
null=np.array(null)
print(f'\n=== SPECTRUM-MATCHED NULL on linearized causal map (L8) ===')
print(f'real_r2(linear causal) = {real_r2:.4f}')
print(f'spectrum_null_r2       = {null.mean():.4f} +/- {null.std():.4f}')
print(f'delta_r2_spectrum      = {real_r2-null.mean():.4f}  z={(real_r2-null.mean())/(null.std()+1e-9):.2f}')

# also: compare to the DIRECT proxy metric M_wte = wte^T wte on same D subset
y=(D-D.mean(axis=1,keepdims=True))*ln_w
Mwte=wte.astype(np.float64).T@wte.astype(np.float64)
ew=np.clip(np.linalg.eigvalsh(Mwte),0,None)
dir_real=r2(cum(y,Mwte)[iu])
dnull=[]
for _ in range(10):
    Q,_=np.linalg.qr(rng.standard_normal((d_model,d_model)))
    dnull.append(r2(cum(y,(Q*ew)@Q.T)[iu]))
dnull=np.array(dnull)
print(f'\n=== DIRECT proxy (wte) same subset for reference ===')
print(f'real_r2(direct)={dir_real:.4f} spectrum_null={dnull.mean():.4f}+/-{dnull.std():.4f} delta={dir_real-dnull.mean():.4f}')

json.dump({'layer':LAYER,'K':K,'nctx':NCTX,
  'causal_linear_real_r2':float(real_r2),
  'causal_spectrum_null_r2':float(null.mean()),'causal_spectrum_null_sd':float(null.std()),
  'causal_delta_r2_spectrum':float(real_r2-null.mean()),
  'direct_real_r2':float(dir_real),'direct_spectrum_null_r2':float(dnull.mean()),
  'direct_delta_r2_spectrum':float(dir_real-dnull.mean())},
  open('probe_B_out.json','w'),indent=2)
print('\nwrote probe_B_out.json')
