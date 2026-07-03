"""
ADVERSARIAL: is the SAE decoder circle a DISCOVERY or a foregone conclusion?

The positive claim: decoder directions of per-concept SAE features recover the
family's circular order (months/days), matching the model residual.

Attack decomposition:
 (1) DELTA over model: is decoder-vs-circle beyond model-vs-circle? Does the SAE
     ever recover structure the MODEL does NOT (add signal), or only inherit it?
     Also compare decoder<->model AGREEMENT (corr of the two cosine matrices).
 (2) IDENTITY: does each feature's decoder direction ~= its concept's model
     residual direction? cos(W_dec[sel_m], R_m) and the recovered-only
     alternative: build the "SAE circle" purely from picking features and see if
     it is just a re-embedding of R.
 (3) ENCODER: does the encoder row give the same circle (encoder ~ decoder)?
     If enc==dec==model, the geometry result is the linear-rep hypothesis
     restated, not a property of SAE decoder geometry.

Prints per (family, layer): model corr, decoder corr, encoder corr, delta,
mean cos(dec_i, R_i), mean cos(enc_i, R_i), mat-agreement(dec,model).
"""
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY

FAMILIES = {
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"],
}
TEMPLATES = [
    "The month is {}", "My favorite is {}", "It happened in {}",
    "The event took place in {}", "She was born in {}", "We met in {}",
]


def concept_residual(layer, word, tok, model):
    block = model.transformer.h[layer]
    vecs = []
    for t in TEMPLATES:
        ids = tok(t.format(word), return_tensors="pt")["input_ids"]
        rec = []

        def pre(_m, args):
            rec.append(args[0].detach())
            return None

        h = block.register_forward_pre_hook(pre)
        with torch.no_grad():
            model(ids)
        h.remove()
        vecs.append(rec[0][0, -1].double().numpy())
    return np.mean(vecs, axis=0)


def circular_target(n):
    ang = 2 * np.pi * np.arange(n) / n
    pts = np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts @ pts.T


def cos_mat(X):
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    return Xn @ Xn.T


def offdiag(S):
    iu = np.triu_indices(S.shape[0], 1)
    return S[iu]


def corr_to(S, T):
    return float(np.corrcoef(offdiag(S), offdiag(T))[0, 1])


def perm_p(S, T, rng, n=2000):
    iu = np.triu_indices(S.shape[0], 1)
    t = T[iu]
    obs = np.corrcoef(S[iu], t)[0, 1]
    K = S.shape[0]
    draws = np.empty(n)
    for i in range(n):
        p = rng.permutation(K)
        Sp = S[np.ix_(p, p)]
        draws[i] = np.corrcoef(Sp[iu], t)[0, 1]
    ge = int((draws >= obs).sum())
    return obs, (ge + 1) / (n + 1), (obs - draws.mean()) / (draws.std() + 1e-9)


def run(family, layer, tok, model, rng):
    words = FAMILIES[family]
    n = len(words)
    hook = f"blocks.{layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)

    R = np.stack([concept_residual(layer, w, tok, model) for w in words])
    A = np.maximum((R - b_dec) @ W_enc + b_enc, 0.0)
    alive = log_sp > DEAD_LOG_SPARSITY
    sel = []
    for m in range(n):
        others = A[np.arange(n) != m].mean(0)
        score = A[m] - others
        score[~alive] = -np.inf
        sel.append(int(np.argmax(score)))
    sel = np.array(sel)

    T = circular_target(n)
    Smodel = cos_mat(R)
    Sdec = cos_mat(W_dec[sel])
    Senc = cos_mat(W_enc[:, sel].T)   # encoder rows for the selected features

    mc, mp, mz = perm_p(Smodel, T, rng)
    dc, dp, dz = perm_p(Sdec, T, rng)
    ec, ep, ez = perm_p(Senc, T, rng)

    # (1) does decoder recover structure the MODEL lacks?
    #     compare decoder->circle to model->circle; and residualize:
    #     how much of decoder->circle survives after regressing out model->circle?
    dm_off, dcirc_off, mcirc_off = offdiag(Sdec), offdiag(T), offdiag(Smodel)
    # partial corr: corr(Sdec, T | Smodel)
    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - (b[0] * x + b[1])
    partial = float(np.corrcoef(resid(dm_off, mcirc_off),
                                resid(dcirc_off, mcirc_off))[0, 1])

    # (2) identity: is decoder dir ~ model residual dir for the SAME concept?
    def rownorm(X):
        return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    Rn = rownorm(R)
    Dn = rownorm(W_dec[sel])
    En = rownorm(W_enc[:, sel].T)
    cos_dec_R = float(np.mean(np.sum(Dn * Rn, axis=1)))
    cos_enc_R = float(np.mean(np.sum(En * Rn, axis=1)))
    cos_dec_enc = float(np.mean(np.sum(Dn * En, axis=1)))
    # matrix agreement decoder-cos vs model-cos (how much decoder geometry just IS model geometry)
    agree_dec_model = corr_to(Sdec, Smodel)
    agree_enc_model = corr_to(Senc, Smodel)

    return dict(family=family, layer=layer, n=n,
                model_corr=mc, model_z=mz, model_p=mp,
                dec_corr=dc, dec_z=dz, dec_p=dp,
                enc_corr=ec, enc_z=ez, enc_p=ep,
                delta_dec_minus_model=dc - mc,
                partial_dec_circle_given_model=partial,
                cos_dec_R=cos_dec_R, cos_enc_R=cos_enc_R, cos_dec_enc=cos_dec_enc,
                agree_dec_model=agree_dec_model, agree_enc_model=agree_enc_model)


def main():
    tok, model = load_gpt2()
    rng = np.random.default_rng(0)
    rows = []
    for family in ["months", "days"]:
        for layer in [4, 6, 8, 10]:
            rows.append(run(family, layer, tok, model, rng))

    hdr = ("fam    L  | modelC  decC   encC  | delta  partial | "
           "cos(dec,R) cos(enc,R) cos(dec,enc) | agree(dec,model) agree(enc,model)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['family']:<6} {r['layer']:>2} | "
              f"{r['model_corr']:+.3f} {r['dec_corr']:+.3f} {r['enc_corr']:+.3f} | "
              f"{r['delta_dec_minus_model']:+.3f} {r['partial_dec_circle_given_model']:+.3f} | "
              f"  {r['cos_dec_R']:+.3f}     {r['cos_enc_R']:+.3f}      {r['cos_dec_enc']:+.3f}    | "
              f"     {r['agree_dec_model']:+.3f}          {r['agree_enc_model']:+.3f}")
    # summary
    print("\nSUMMARY")
    dm = np.array([r['delta_dec_minus_model'] for r in rows])
    pa = np.array([r['partial_dec_circle_given_model'] for r in rows])
    cdr = np.array([r['cos_dec_R'] for r in rows])
    cde = np.array([r['cos_dec_enc'] for r in rows])
    ag = np.array([r['agree_dec_model'] for r in rows])
    print(f"  mean delta(dec-model corr) = {dm.mean():+.3f}  (>0 means SAE adds; ~0/<0 means inherits)")
    print(f"  mean partial(dec->circle | model) = {pa.mean():+.3f}  (residual circle signal after removing model)")
    print(f"  mean cos(decoder_i, residual_i) = {cdr.mean():+.3f}")
    print(f"  mean cos(decoder_i, encoder_i)  = {cde.mean():+.3f}")
    print(f"  mean matrix-agreement(dec cos, model cos) = {ag.mean():+.3f}")


if __name__ == "__main__":
    main()
