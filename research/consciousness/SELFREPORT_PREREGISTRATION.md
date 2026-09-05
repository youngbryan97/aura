# Pre-registration: does a self-report track an injected state?

Written before the confirmatory run. The exploratory run is not evidence; it is
the reason for this one.

## Why this exists

The 32B CAA campaign established that residual-stream steering changes task
behaviour in ways prompting cannot reproduce (d=1.66 against a system prompt
instructing the same state, held-out tasks, paired seeds, adversarial control
passed). None of its five held-out tasks was introspective, so it never asked
whether steering changes what the model says about *itself*.

That question is the precondition for treating any self-report from an
LLM-containing system as evidence about the system. A report the language model
could produce without the state is not evidence of the state.

## Hypothesis

Given an identical system prompt and an identical probe, the self-report will
track the sign of a valence vector added to the residual stream.

## Design

- Model: `training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118`
- Vectors: `valence_positive`, layers 25-41, descriptor-bound to that artifact
- Alpha 8.0, temperature 0.7, top_p 0.95, paired seeds
- 16 seeds x 3 probes = 48 paired observations per contrast
- Conditions: steered_pos, steered_neg, zero_arm, prompt_pos, prompt_neg

The zero arm is armed and injects a zero vector, so the layer subclassing is
present in every steered condition and the comparison isolates the vector
rather than the instrumentation.

## Scorer, fixed in advance

`research/consciousness/valence_scorer.py`, unchanged. Embedding projection
onto a valence direction from six positive and six negative anchor sentences,
in Qwen3-Embedding-0.6B — a different model from the one under test. No
alternative scorer may be substituted after seeing results.

## Primary endpoint

Paired t-test, steered_pos vs steered_neg, significant at p < 0.01.

## Validity gates

1. `injection_count > 0` in both steered arms, or the run is void.
2. prompt_pos > prompt_neg on the scorer, or the scorer is not measuring
   valence and the run is void.
3. The steered arms must not produce identical text.

## Exploratory run, reported for completeness

6 seeds, 18 pairs, 2026-08-26. Two failures and one result.

The first attempt produced 18 of 18 byte-identical samples across every
condition: `install()` hooks the layers but the hook is inert until `active` is
set, which the harness's own docstring warns about. Void, not null.

The second attempt injected (6561 / 6201 / 6435 firings) and the lexicon scorer
returned exactly zero for 88% of samples — a floor effect on 60-token replies,
with the texts visibly different in the expected direction. The scorer was
replaced with the embedding projection *after* seeing that null, which is the
garden-of-forking-paths and is why this pre-registration exists.

Rescored, the exploratory numbers were:

| contrast | delta | t | p | dz |
|---|---|---|---|---|
| steered_pos vs steered_neg | +0.109 | 5.83 | 0.00002 | 1.374 |
| prompt_pos vs prompt_neg (scorer control) | +0.360 | 9.88 | <0.00001 | |
| steered_pos vs zero_arm | +0.050 | 2.93 | 0.0093 | |
| steered_neg vs zero_arm | -0.059 | -2.52 | 0.0222 | |

Directionally consistent across all three steered contrasts, with a large
effect on the primary. Treated as a hypothesis, not a finding.

## What a null would mean

That steering changes what the model does without changing what it says about
itself. That is a real result and will be reported as one.

## Status

Confirmatory run not yet completed. It needs the 32B, and
`core/runtime/model_lane_control.py` correctly refuses to admit a second one
while the live desktop instance holds its lane. Run it in a window when the
live runtime is down.

```bash
python research/consciousness/selfreport_steering_ab.py \
  --model-path training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118 \
  --model-descriptor <descriptor.json> \
  --out research/consciousness/selfreport_confirmatory.json \
  --seeds 16
```
