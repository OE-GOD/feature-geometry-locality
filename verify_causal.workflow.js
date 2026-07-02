export const meta = {
  name: 'verify-causal-proxy',
  description: 'Adversarially verify the causal-proxy positive locality signal against its own confounds',
  phases: [
    { title: 'Verify', detail: '3 skeptics: alpha/linearization, common-component, context/injection validity' },
    { title: 'Judge', detail: 'synthesize surviving claim' },
  ],
}

const DIR = '/Users/oe/feature-geometry-locality'

const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['REFUTED', 'WEAKENED', 'SURVIVES'] },
    strongest_objection: { type: 'string' },
    evidence: { type: 'string' },
    quantitative_impact: { type: 'string' },
    recommended_fix: { type: 'string' },
  },
  required: ['lens', 'verdict', 'strongest_objection', 'evidence',
             'quantitative_impact', 'recommended_fix'],
}

const HEADLINE = `CLAIM UNDER TEST: "Under a CAUSAL function proxy (inject alpha*decoder_dir into the residual stream at the feature's layer, run the rest of GPT-2, measure mean output-logit shift over contexts), GPT-2-small SAE feature geometry DOES predict downstream function above a label-permutation null: at layer 8, null-subtracted delta_R2 > 0 at several sigma, the OPPOSITE sign from the refuted direct-logit proxy. This suggests mid-layer SAE features have genuine (not tautological) local functional structure that the linear proxy was blind to."`

const skepticPrompt = (lens, detail) => `You are an adversarial reviewer. REFUTE the claim if you can; default to skepticism. You may run python3 in ${DIR} (files: causal_proxy.py, null_control.py, run_layer.py, locality.py, worlds.py; causal result JSONs causal_L*.json; causal_sweep.log). Read causal_proxy.py FIRST to understand exactly what is computed. Python stderr warnings are noise; model/SAE weights are cached.

${HEADLINE}

YOUR LENS: ${lens}. ${detail}

Attack ONLY through your lens; run real experiments (modify/extend causal_proxy.py or write a probe script). Return structured output: lens; verdict (REFUTED=claim wrong through this lens / WEAKENED=survives with caveats, state them / SURVIVES=attack failed, say what you tried); strongest_objection; evidence (what you actually ran + numbers); quantitative_impact (how much of the delta_R2 signal your objection removes); recommended_fix.`

const LENSES = [
  ['alpha-linearization',
   `The injection scale alpha is a free parameter. If alpha is small the model responds ~linearly and the causal proxy degenerates toward a (different) linear map of d -> a MILDER version of the same tautology that killed the direct proxy. If alpha is huge the model is pushed off-distribution and the "function" is meaningless saturation. Check: (a) does delta_R2 stay positive and significant across alpha (compare the alpha=2 vs 6 vs 12 runs, run more if needed)? (b) Is the signal in the near-linear regime just reproducing the spectrum-matched null? Build the analog of null_control.py for the causal proxy at small alpha and see if a spectrum-matched-random injection reproduces the curve. Determine the alpha range, if any, where the signal is both real and on-distribution.`],
  ['common-component-artifact',
   `Injecting ANY direction into the residual stream may push the same set of high-frequency tokens (a shared "logit-shift" component driven by GPT-2's unigram/frequency structure, not the specific feature). If so, all causal-effect vectors share a large common component, inflating Sfn baseline. The label-permutation null is claimed to control for this. VERIFY that claim: does subtracting the mean causal-effect vector before computing Sfn change delta_R2? Does the permutation null actually preserve the common component (it should, since it reuses the same vectors)? Quantify how much of Sfn is the shared component vs feature-specific, and whether the geometry->function signal survives removing it.`],
  ['context-and-injection-validity',
   `The proxy injects at ALL positions of only ~8-24 token positions from ONE hardcoded paragraph, then averages logit shifts over all positions. Objections: (1) tiny, non-representative context set -> effect vectors are estimated from too little data and could be dominated by the specific text; (2) injecting at every position and averaging conflates position-specific effects; (3) mean-over-context may wash out exactly the functional distinctions that matter. Check robustness: does delta_R2 survive with a DIFFERENT/larger context set (write your own more diverse text), with last-position-only readout, and with more contexts? Does the rank-1-neighbor causal similarity hold up? Report which parts of the claim are context-robust.`],
]

const verdicts = (await parallel(LENSES.map(([lens, detail]) => () =>
  agent(skepticPrompt(lens, detail), { label: `skeptic:${lens}`, phase: 'Verify', schema: SKEPTIC_SCHEMA, effort: 'high' }))
)).filter(Boolean)

const judgment = await agent(
  `You are the meta-reviewer. Three adversarial skeptics attacked this claim, each through one lens:\n\n${HEADLINE}\n\nTheir verdicts (JSON):\n${JSON.stringify(verdicts, null, 2)}\n\nSynthesize: (1) Does the causal-proxy positive result SURVIVE, survive-with-caveats, or die? (2) State the strongest surviving version of the claim, with every caveat the skeptics established. (3) List the concrete next experiments needed to make it publishable. Be brutally honest — a wrong positive result is worse than an honest negative one. Return prose.`,
  { label: 'judge', phase: 'Judge', effort: 'high' })

return { verdicts, judgment }
