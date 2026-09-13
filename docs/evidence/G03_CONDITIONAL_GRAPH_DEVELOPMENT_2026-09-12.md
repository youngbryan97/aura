# G03 conditional graph development

The development scorer can now condition binary argument labels on selecting
one mention per slot. In that conditional objective the shared negative-label
term cancels, leaving weighted log odds. The old sum of positive-label log
probabilities saturated strong correct evidence near zero. The new candidate
uses the same coefficients, source-learned register contracts and constrained
graph solver. It does not change the frozen parent or a serving package.

## Exposed weave results

| Candidate | Exact programs | Exact answers |
| --- | ---: | ---: |
| Frozen parent | 19/48 | 21/48 |
| Constrained graph, positive-label scores | 41/48 | 41/48 |
| Constrained graph, conditional log odds | 43/48 | 43/48 |
| Conditional candidate, coefficient lesion | 0/48 | 1/48 |
| Conditional candidate, hidden-token shuffle | 0/48 | 0/48 |
| Source-trained pairwise argument ranker | 37/48 | 38/48 |

The conditional candidate gains 22 answers and regresses none against the
frozen parent on these 48 exposed examples. Against the prior constrained
candidate it gains three answers and regresses one. Five errors remain.
These are development comparisons after inspecting failures, not fresh
replication, broad transfer, neural recurrent computation or frontier evidence.

The pairwise experiment uses the exact parent source cohort: 728 training and
500 validation examples. Only training examples enter its optimizer. It learns
from positive/negative mention pairs, preserves other coefficient groups and
records its combined-score parameterization. Its training loss falls, but it
regresses four parent answers on the weave evaluation. It is rejected.

## Source and execution order

All 144 examples in the retained cataphoric source bundle have a different
textual and execution order. The shared argument-proposal builder formerly
used caller order to find neighboring clauses. Training supplied execution
order; runtime charts supplied source order. The builder now finds textual
neighbors and returns proposal banks aligned to the caller's node identities.
Permutation tests cover both global and clause-local proposal modes.

A full proposal refit after that repair is a measured null: its coefficient
hash remains the parent's hash. The repair is necessary for correct proposal
construction but has not established a learned capability gain.

There was a separate runtime exclusion: references to a later textual operation
were dropped whenever their relation score was non-positive. With gold operation
anchors and mentions, that rule excludes 47 of 48 correct forward edges in the
cataphoric validation set. This diagnostic supplies gold annotations and is not
itself a runtime result.

The opt-in `joint_graph_v1` policy removes that local sign exclusion and leaves
acyclicity, connectivity, type, register-use and dependency-lesion checks to
the complete graph solver. It is valid only with the constrained graph search.
On the first twelve source validation cases per family, it recovers five
cataphoric programs from a parent score of zero and retains 43/48 weave programs.
The other pilot families retain the conditional scorer's outcomes. This is an
84-example pilot, not full-source admission.

The definition locator has a related register-order assumption. A separately
identified source-neighbor candidate bounds both local spans and old envelopes
using textual neighbors. Its full evaluation is not established by this record.

## Evaluation semantics

The selector now reports a secondary structural-equivalence diagnostic. It
recognizes independent instruction schedules and integer add/multiply operand
exchange while retaining input identities, primitive multiplicity and the
connected computation. It rejects directional operand exchanges, invalid
references, dead computation and coincidentally equal example answers.

That diagnostic explains the conditional scorer's one strict-program regression
in the source role-binding pilot: the multiplication operands were exchanged,
with the same computation and answer. The selector still uses strict program
gains with zero strict regressions. Equivalence does not silently relax selection.

## Retained evidence

Rows, controls, source pilots, failed ranking results and the exact-source null
are retained under `artifacts/rlc/semantic_program_conditional_graph_dev_20260912/`.
The receipt binds file hashes and the unpromoted ranker's local artifact hash.
The conditional scorer receipt is
`64336020303a482c5dcac530fcc4b58aef25cb125d39fcd5438fc9f582828ff7`;
the order-invariant graph candidate is
`2b6b5683dc48989df2d3067e3f4201f9984569e252c601ded6963009bbdc0b6e`.

Focused checks separately passed 51 structural-equivalence/selector/floor
tests, 42 clause-order/shared-transducer tests, and three forward-reference
tests. The prior optimizer run passed 34 solver tests; two test-fixture errors
in the accompanying forward-reference suite were fixed and rerun. Final
combined gates are recorded in the checkpoint's subsequent verification record.
G03 remains open; no G04-G12 requirement is closed by these results.
