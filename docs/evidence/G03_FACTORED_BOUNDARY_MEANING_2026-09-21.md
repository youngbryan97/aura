# Separate boundary evidence from operation meaning

The middle-mean labeled-span candidate completed the declared 54-case
development cohort. It gets 38/54, compared with 30/54 for the incumbent and
33/54 for the runtime-coupled transition candidate. All sixteen controls
pass, but all three source-training retention cases regress. The cohort
contains the union of failures from the two complete prior runs, plus the
fixed controls and training cases. It is not a random sample or a full replay.
No candidate is promoted.

The report is `semantic-middle-joint-pilot-20260921/report.json` under
`~/.aura/rlc-evidence/`, receipt
`4e9fb7f860e73cf495b53e4af0c97a6875f767cf37ee2c59ea4b4ac35da6e94a`.
The detached run finished in 395.73 seconds with exit zero and empty lineage.

## Ordered evidence diagnostic

Eight feature views use the same 1,720 source operation labels, the same
three construction-held-out training folds and the same classifier. The
1,192 exposed validation labels do not fit coefficients. Operation boundaries
are supplied for this diagnostic; no test examples are used.

| View | Source construction cross-validation | Validation labels |
|---|---:|---:|
| Middle mean | 1700/1720 | 1174/1192 |
| Middle endpoint | 1716/1720 | 1192/1192 |
| Middle mean and transition | 1714/1720 | 1184/1192 |
| Middle mean and endpoints | 1716/1720 | 1172/1192 |
| Balanced middle and final mean | 1696/1720 | 1174/1192 |
| Balanced middle and final transition | 1652/1720 | 1183/1192 |
| Middle and lexical mean | 1668/1720 | 1144/1192 |
| Middle mean and request interaction | 1716/1720 | 1178/1192 |

Every view fits 1720/1720 training labels. The endpoint view ties for best
source cross-validation with fewer feature coordinates. Its perfect
validation classification is conditional on supplied boundaries. It is not
500/500 autonomous semantic recovery or evidence of broad reasoning gain.

The diagnostic report is `semantic-ordered-view-capacity-20260921/report.json`,
receipt `076e6808056841bad0a11730fd76070bbf663c3e688c93acb7a7a02cfb38cfa1`.
All eight modes completed in 59.81 seconds with empty child lineage.

## Endpoint-only runtime failure

Refitting the existing joint operation/background head to `middle_last`
produces only 16/54 correct programs, including 8/16 controls. This candidate
is rejected. Its report is `semantic-endpoint-pilot-20260921/report.json`,
receipt `0f7cfba43f0f925690809072aa301f43e161fcfee1693658e43d26af057baa31`.
The complete detached run finished in 179.53 seconds with empty lineage.

An endpoint feature has an exact limitation: two spans with the same endpoint
have the same feature, irrespective of their starts. It cannot simultaneously
label one such span as an operation and another as background when training
demands both. This is a limitation of that feature/objective pairing, not a
proof that the full model observation lacks the needed information.

## Existing components, separate responsibilities

The operation-view refitter now has an opt-in conditional-label mode. It
learns operation identity from source-positive spans and leaves boundary
evidence in the existing pointer. Runtime uses its existing score
`boundary_score(span) + log P(operation | span)`. The default refit retains
its parent's background supervision. Dropping a background class also updates
the complete-search label inventory and records the new supervision contract.

For the recognition distribution, let `b(s)` score a span and `q(l | s)` be
a normalized distribution over operation labels. Summing
`exp(b(s)) * q(l | s)` over labels gives `exp(b(s))`. Therefore the partition
over non-overlapping labeled span sets reduces to the existing boundary-set
partition. The target loss separates into boundary-set likelihood and
conditional label likelihood. This identity does not prove correct argument
binding or generalization; type and graph constraints introduce further
dependencies. Runtime probability clipping remains a numerical approximation.

The boundary trainer also now excludes the public input spans that runtime
excludes. Previously its negative inventory included candidates the decoder
could never select. Exclusions are bound into the training receipt; a target
overlapping an excluded span is rejected. Tests compare the training inventory
with the runtime's complete span inventory, not only a synthetic mask.

The conditional candidate still requires autonomous measurement. Defaults,
serving and release thresholds are unchanged. G03 remains open.

Verification: 111 focused tests passed in 32.76 seconds; smoke passed 164
with one skip in 79.72 seconds. Ruff, compilation, layering, governance lint
and writing passed without changing baselines.
