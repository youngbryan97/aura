# G03 development: prefix-feasible argument search

This is an exposed-cohort diagnostic, not fresh replication. G03 remains open.

## Defect and candidate

The existing typed argument search truncates each operation's candidates to
128 before applying previous operations' register-use and span constraints.
Consequently, infeasible high-scoring candidates can consume the entire beam.
A focused test reproduces this with a beam of one: a used register eliminates
the only retained option even though a lower-scoring valid continuation exists.

`with_prefix_feasible_arguments()` creates an opt-in candidate that applies
prefix overlap, register-use, and cycle constraints before truncation. Complete
graphs still require connectedness and the original learned use contract.
No coefficients are refitted. Legacy receipts retain the original search;
the candidate has a different receipt:
`4fa9c670f0cec6126b80d8ef70bba1429c807c58c421ecb7f24379731f092df4`.

The existing lesion CLI accepts `--prefix-feasible-arguments`; it does not
modify the transducer file or grant serving authority. It now also accepts
verified held-out-only bundles instead of requiring a training split in them.

## Measured rejection

The diagnostic loaded the independently verified feature bundle
`~/.aura/rlc-evidence/semantic-natural-weave-replication-v2`, manifest
`2f403cdc3f4a435c2b9d95ec7984308989a87cc0688c46c546a4984fd5cdc75e`,
and used the frozen package's explicit representation-compatibility receipt.
An initial attempt with the v1 bundle was rejected for a different worker
session; no identity was forced or rewritten.

The run visited examples in bundle order. A four-minute between-example bound
stopped it after example 28 at 248.55 seconds. The bound does not interrupt an
in-flight decode. This is a partial run; the remaining 20 examples were not
evaluated by this diagnostic.

- Candidate: 16/28 answer-exact, 14/28 program-exact, two refusals.
- Frozen incumbent on those same rows: 13/28 answer-exact.
- Paired differences: four gains, one regression.
- No live-model load, training, ordinary decode, or new lesion campaign.

The candidate is rejected for promotion: it regresses a previously correct
answer and does not establish acceptable end-to-end cost. Making more graphs
reachable does not establish that the learned ranking selects the right one.
Neither the frozen activation nor the running application was changed.

## Existing substrate diagnosis

`diagnose_compositional_definition_relations` was run over all 48 examples:

| Split | References | Runtime definition top-1 | Gold definition top-1 |
| --- | --- | --- | --- |
| Validation | 240 | 229 | 109 |
| Test | 240 | 221 | 106 |

Diagnostic receipt:
`46426810d14a4d2cb0a513ce6ff15d9595870c9f0f3198a17a39825a8120b9f7`.
Gold reference spans are supplied in this diagnostic; expected answers are
not. Therefore 450/480 is a binding diagnostic, not end-to-end accuracy.
Replacing runtime definitions with gold spans does not improve this learned
representation. Reference proposal and graph composition remain the next
targets rather than assuming that a new definition encoder is necessary.

The universal floor is already the execution and type-signature owner through
`semantic_program_floor.py`. The language substrate's learned matchers and
resident feature extraction serve sentence decisions; those binary boundaries
must not be mistaken for register-binding correctness. The connectome's
corroboration work compares independent influence measurements. The subject
snapshot work now restores persistent stores between experimental arms.
These are reusable integration and experiment-isolation components, not
evidence that semantic selection is already solved.

## Validation

43 focused tests passed, including legacy transducer replay, new prefix
constraints, candidate receipt round-trip and existing qualification checks.
Smoke: 164 passed, one skipped. Touched Python paths pass Ruff.
Compile passed. Aggregate layering remains red on two imports in
`core/verify/influence_turn_probe.py`; governance lint remains red on new store
restore writes in `core/subject/snapshot.py`. Neither file is changed by this
candidate. These release-gate failures are not counted as passing.
