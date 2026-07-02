Implemented [real_effect.py](/Users/oe/feature-geometry-locality/real_effect.py).

It loads the jbloom SAE weights including encoder/biases, samples alive features, records clean residuals/logits at `blocks.{layer}.hook_resid_pre`, computes SAE activations, ablates `a_i * W_dec[i]` via a GPT-2 block `forward_pre_hook`, and writes `F`, `idx`, and `firing_counts` to `.npz`.

Verification run completed:

```text
python3 real_effect.py --k 40 --contexts 16
reconstruction sanity mean_cos=0.8637
features with firing_count >= min_fires: 38/40
firing_count mean=22.93 median=19.50
mean L2 norm of nonzero F_i rows=4126.724609
wrote real_effect_L8_k40_s0.npz
```

Artifact shape check passed: `F` is `(40, 50257)` `float32`, and 38 rows are nonzero. The expected urllib3/HuggingFace cache warnings appeared but did not affect the run.