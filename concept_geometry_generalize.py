"""
ADVERSARIAL GENERALIZATION TEST for the claim:
  "SAE decoder geometry reflects function: per-concept SAE features recover a
   concept family's known structure (months/days -> circle)."

The two positive families (months, days) were hand-picked from prior work that
ALREADY showed them geometric (Engels et al. 2024). This script tests whether
the geometry-reflects-function result GENERALIZES to families with no
pre-established circular geometry:

  - NON-CYCLIC but ordered (linear/ordinal): number-words, planets(by distance),
    rainbow colors(by wavelength). Target = linear/ordinal, NOT circular.
  - ARBITRARY categorical (no ground-truth order): animals, fruits, emotions,
    countries. No 1-D target is legitimate; we instead test the TARGET-FREE
    generalization: does the SAE decoder geometry reflect the MODEL's own
    residual geometry (corr(Sdec, Smodel)) above a label-permutation null?

For every family we report, per requested structure target:
  MODEL residual  vs target   (positive control: does the model even have it?)
  SAE decoder     vs target   (does the SAE recover it?)
  SAE decoder     vs MODEL    (target-free: does SAE geometry mirror the model?)

Each against the label-permutation null (shuffle which feature is which concept),
which holds real decoder directions + spectrum fixed and only breaks the
concept<->feature correspondence.
"""

import argparse
import numpy as np
import torch

from real_effect import fetch_sae, load_gpt2
from run_layer import DEAD_LOG_SPARSITY
from concept_geometry import (concept_residual, cos_mat, circular_target,
                              perm_null)

# ---- families -------------------------------------------------------------
# cyclic (known circular geometry) -- the ORIGINAL hand-picked positives
CYCLIC = {
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"],
    "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"],
}
# ordered but LINEAR / ordinal (not circular) -- structure is real but new
LINEAR = {
    # counting words, genuine 1-D order, not in the cyclic-geometry literature
    "numbers": ["one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten"],
    # planets by distance from the sun -- ordinal
    "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
                "Uranus", "Neptune"],
    # rainbow spectrum by wavelength -- ordinal (could arguably be a color wheel,
    # so we also score it against the circular target)
    "colors": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"],
    # size-ordered animals (small->large) -- a WEAK/subjective order, good stress
    "animals_size": ["mouse", "cat", "dog", "wolf", "horse", "elephant", "whale"],
}
# ARBITRARY categorical -- no defensible 1-D order. Target-free test only.
ARBITRARY = {
    "animals": ["cat", "dog", "cow", "horse", "pig", "sheep", "goat", "duck"],
    "fruits": ["apple", "banana", "orange", "grape", "cherry", "lemon",
               "peach", "mango"],
    "emotions": ["happy", "sad", "angry", "afraid", "surprised", "disgusted",
                 "calm", "excited"],
    "metals": ["iron", "gold", "silver", "copper", "lead", "tin", "zinc",
               "nickel"],
}

ALL = {**CYCLIC, **LINEAR, **ARBITRARY}

# GENERIC templates (not month-specific) so no family is advantaged.
TEMPLATES = [
    "The word is {}", "I was thinking about {}", "She said {}",
    "The next one is {}", "They mentioned {}", "Here we have {}",
]


def concept_residual_generic(layer, word, tok, model):
    """Same as concept_geometry.concept_residual but with generic templates."""
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


def linear_target(n):
    """Ordinal target: similarity decreases with index distance |i-j|.
    Centered/normalized so corr is scale-free; T_ij = -|i-j|."""
    idx = np.arange(n)
    return -np.abs(idx[:, None] - idx[None, :]).astype(float)


def select_features(R, W_enc, b_enc, b_dec, alive, n):
    A = np.maximum((R - b_dec) @ W_enc + b_enc, 0.0)
    sel = []
    for m in range(n):
        others = A[np.arange(n) != m].mean(0)
        score = A[m] - others
        score[~alive] = -np.inf
        sel.append(int(np.argmax(score)))
    sel = np.array(sel)
    n_distinct = len(set(sel.tolist()))
    selectivity = np.mean([A[m, sel[m]] - A[np.arange(n) != m, sel[m]].mean()
                           for m in range(n)])
    return sel, n_distinct, selectivity


def run_family(name, words, layer, tok, model, W_dec, W_enc, b_enc, b_dec,
               alive, rng):
    n = len(words)
    R = np.stack([concept_residual_generic(layer, w, tok, model) for w in words])
    sel, n_distinct, selectivity = select_features(R, W_enc, b_enc, b_dec,
                                                    alive, n)
    Smodel = cos_mat(R)
    Sdec = cos_mat(W_dec[sel])

    # which targets to score against
    targets = {}
    if name in CYCLIC:
        targets["circular"] = circular_target(n)
    if name in LINEAR:
        targets["linear"] = linear_target(n)
        if name == "colors":                       # color wheel is plausible too
            targets["circular"] = circular_target(n)
    # arbitrary families: no 1-D target (target-free only)

    out = {"name": name, "n": n, "n_distinct": n_distinct,
           "selectivity": selectivity, "rows": []}

    # structured-target scores (model control + SAE recovery)
    for tname, T in targets.items():
        for who, S in [("MODEL", Smodel), ("SAE_dec", Sdec)]:
            obs, p, nmean, nsd = perm_null(S, T, rng)
            z = (obs - nmean) / (nsd + 1e-9)
            out["rows"].append((f"{who} vs {tname}", obs, p, z))

    # TARGET-FREE: does SAE decoder geometry mirror the MODEL's own geometry?
    obs, p, nmean, nsd = perm_null(Sdec, Smodel, rng)
    z = (obs - nmean) / (nsd + 1e-9)
    out["rows"].append(("SAE_dec vs MODEL (target-free)", obs, p, z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=list(ALL))
    args = ap.parse_args()

    hook = f"blocks.{args.layer}.hook_resid_pre"
    W_dec, W_enc, b_enc, b_dec, log_sp = fetch_sae(hook)
    alive = log_sp > DEAD_LOG_SPARSITY
    tok, model = load_gpt2()
    rng = np.random.default_rng(args.seed)

    print(f"\n=== layer {args.layer}  (generic templates, seed {args.seed}) ===")
    print(f"{'family':<14} {'n':>2} {'dist':>4} {'selec':>6}   "
          f"{'measure':<32} {'corr':>7} {'perm_p':>7} {'z':>6}")
    for name in args.families:
        r = run_family(name, ALL[name], args.layer, tok, model,
                       W_dec, W_enc, b_enc, b_dec, alive, rng)
        first = True
        for label, obs, p, z in r["rows"]:
            head = (f"{r['name']:<14} {r['n']:>2} {r['n_distinct']:>4} "
                    f"{r['selectivity']:>6.1f}") if first else " " * 30
            tag = "*" if p < 0.05 else " "
            print(f"{head}   {label:<32} {obs:+.3f} {p:>7.3f} {z:>+6.1f} {tag}")
            first = False
        print()


if __name__ == "__main__":
    main()
