# Complete-request operation evidence: development pilot

An operation head using only the mean of causal hidden states inside a span
cannot depend on words following that span. The two expanded-search regressions
already recorded select a short subtraction fragment over a complete division
phrase. This motivates testing request-conditioned evidence, not a claim that
missing suffix context accounts for every error.

The opt-in `contextual_span_request_interaction` view uses the existing frozen
request states. Let `u` be the normalized mean of the candidate span's final
hidden channel, and `v` the normalized final state of the request. Its feature
is `normalize(concat(u, v, normalize(u * v)))`, with elementwise multiplication.
Zero vectors remain zero. Each component has a declared width; no new model,
prompt, answer annotation, or runtime target is supplied to feature extraction.

The product permits context-dependent distinctions that an additive linear
head cannot express. A focused test fits the four sign pairs of a one-dimensional
span and request, where the class depends on their product. That finite example
establishes representational capacity for that example only. It does not prove
the 27B state contains every needed distinction or establish general transfer.

The view is available through the existing refit CLI's repeatable
`--operation-view-mode` option. Defaults and existing coefficient identities
retain their old feature definitions. Runtime decoding, graph-score replay,
gradient computation, serialization, and lesions consume the same declared view.

## Refit integrity repair

The operation-view refitter previously trained only positive spans even when
its parent had a learned background class. It then retained that parent's
background receipt, which no longer described the fitted head.

The refitter now retains the operation vocabulary and source-only background
supervision, records the new head's actual training spans, and calibrates the
same background scoring policy used by runtime. Background remains in the
probability denominator and never becomes an executable opcode. Both ordinary
log probability and background log odds have runtime-score regression tests.

## Measured pilot

Training: 64 source-identity/geometry-selected training examples, 160 annotated
operations. Evaluation: 16 exposed validation examples chosen by the same
source-independent selection policy. Validation calibrates chart length and is
also measured here; these are development results, not fresh replication.
No test examples were used. Arm order rotates per example; each decode has the
same 20-second search allowance. The unchanged incumbent is a third arm.

| Arm | Semantically equivalent | Witnessed different |
|---|---:|---:|
| Incumbent | 16/16 | 0 |
| Span-only refit | 13/16 | 3 |
| Complete-request refit | 12/16 | 4 |

Both refits are rejected for promotion. The new view regresses on one additional
case relative to the equally trained span-only head. The incumbent was trained
on more data; this pilot alone cannot separate that loss of training coverage
from representation quality. A separately recorded follow-up uses all 764
training examples and retains the paired span-only control.

Report: `~/.aura/rlc-evidence/semantic-request-context-pilot-20260921/report.json`.
Receipt: `6533f2049afe343a51e7504ad04571cacb1da7b7b2d5d557041b8cbfb12c4ea0`.
Implementation: `cd39edd5040d1f589ad69ad87c06a1637f6e161ac474f78c5f45df12bdafd76b`.
The run's script is retained beside its plan and individual row receipts.

Tests: 86 focused passed; smoke 164 passed, one skipped. Lint, compile,
governance, layering, and writing passed. G03 remains open; serving is unchanged.
