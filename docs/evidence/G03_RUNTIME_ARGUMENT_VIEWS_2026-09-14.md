# G03: runtime operation boundaries are a measured null

The source-only argument refit decoded all 728 training examples with the
frozen atomic-literal parent. Twelve had unambiguous changed operation
boundaries; 716 were unchanged. Every original example was retained. Added
views changed only operation spans, never the gold argument identities.

Both pairwise fits converged. On all 500 source-validation examples:

| Measurement | Incumbent | Refit |
| --- | ---: | ---: |
| Exact program | 436 | 436 |
| Equivalent program | 456 | 456 |
| Paired gains | 0 | 0 |
| Paired regressions | 0 | 0 |

The incumbent remains selected. This result does not measure answer equality;
the earlier 473/500 result did. Neither metric is substituted for the other.
These exposed development cases are not fresh transfer evidence.

Parent identity: `df295e6c56caf7427974fac2cc8e88eecbad4593de1ed00d7790e4907a59a7ea`.
Refit identity: `f38e96b7a38482506d2efa86ea157cb80c5e09b373237c4e31a36a47e80631c2`.
The candidate remains at
`~/.aura/rlc-evidence/semantic-runtime-argument-views-dev-20260914/candidate.json`.

The [paired report](../../artifacts/rlc/semantic_runtime_argument_views_dev_20260914/validation.json)
and [supervisor receipt](../../artifacts/rlc/semantic_runtime_argument_views_dev_20260914/detached_receipt.json)
retain the completed run. The supervisor had already passed when a stop was
requested; that request stopped nothing. No evaluation rows were lost.

The long comparison exposed missing per-case progress and recovery in the
selection runner. The follow-up implementation checkpoints completed cases
against the candidate and observation identities. It does not change scoring
or confer serving authority. G03 remains open.

## Failure localization

On the atomic parent, supplying gold operation spans to the existing argument
chart recovers 23 of its 44 non-equivalent programs. In the other 21, the target
is feasible but another graph scores higher. No target is excluded by the
joint constraints in this diagnostic. This is an intervention, not autonomous
success: gold operations and gold input spans were supplied.

The operation-only diagnostic on those 23 cases finds nine gold charts outside
the beam, four missing gold spans, and ten gold charts inside the beam. Its
label-limit and feasibility settings match this parent. It still uses gold
input grounding, so it does not isolate errors caused by autonomous grounding.
The [argument attribution](../../artifacts/rlc/semantic_runtime_argument_views_dev_20260914/argument-attribution.json)
and [operation attribution](../../artifacts/rlc/semantic_runtime_argument_views_dev_20260914/operation-attribution.json)
are retained as diagnostic evidence only.
