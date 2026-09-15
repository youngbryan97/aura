# Full-mention pointer-margin result

The source-trained argument ranker saw sixteen pointer negatives per operation,
but the deployed candidate pool could contain many more mentions. Its pairwise
objective also omitted the fixed pointer term used when selecting an argument.
This trial fits all runtime proposals plus source semantic spans and includes
that fixed pointer term in each training margin. Definition, operation and
graph-relation coefficients remain unchanged. It is not full-graph training.

The fit uses 728 source-training examples and no validation or test labels.
The two role objectives converged in 11 and 8 iterations over 294,609 and
278,931 pairs. Exact batch expansion of shared span vectors avoids allocating
the full 18.2 GB and 17.2 GB feature matrices. Stored vectors occupy 3.88 GB
and 3.56 GB respectively; no precision reduction is used. Dense and factorized
fits are bit-identical in the focused tests.

The complete paired development comparison contains 500 rows:

| Model | Exact programs | Equivalent programs | Exact gains | Exact regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 436 | 456 | 0 | 0 |
| Full-mention margin refit | 435 | 454 | 0 | 1 |

Equivalent-program scoring records zero gains and two regressions. Selection
retains the incumbent. The candidate receives no serving authority.

The detached run completed in 1,737.176 seconds, exit zero, with no timeout,
no restart and an empty process group after completion. Twenty focused tests
passed, as did smoke (164 passed, one skipped), lint, compile, governance and
layering before the run. The executable source was held unchanged during it.

Evidence is under `artifacts/rlc/semantic_runtime_margin_dev_20260914/`.
The validation digest is
`8b93e110104a0cdd3ac76bd2c124f532433ed35e48dfc39abdd6cc9fd5b8e808`.
The parent receipt is
`df295e6c56caf7427974fac2cc8e88eecbad4593de1ed00d7790e4907a59a7ea`;
the candidate receipt is
`69089d312d856eca7a1b83841f7b7f57da02d0e85fdb4e596f38d90d97c6d667`.

The candidate and resumable validation checkpoint remain in
`~/.aura/rlc-evidence/semantic-runtime-margin-dev-20260914/`. This negative
result does not close G03. Remaining attribution concerns the full operation,
definition and dependency graph, not an unmeasured shortage of pointer negatives.
