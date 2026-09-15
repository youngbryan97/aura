# Source graph-factor calibration

The completed trial fits three positive scales over the existing role,
relation and pointer factors. Proposal scale and all neural heads stay fixed.
Each of 728 training cases contributes its best source-labeled argument graph
and its best competing register assignment under the retained runtime pool.
Alternative mention realizations of the same assignment are excluded together.
Operation boundaries come from source annotations during fitting. Autonomous
validation receives neither those annotations nor expected answers.

The fit converged in nine iterations. The objective fell from 0.295069 to
0.158806; scales changed from `[1, 1, 0.5]` to approximately
`[0.000001, 0.657795, 0.752178]`. This is one fixed-pool contrast fit, not
iterative full-graph training. Exact register-assignment contrasts can include
algebraically equivalent programs, so the independent equivalence score below
is also retained.

| Candidate | Exact programs | Equivalent programs | Exact gains | Exact regressions |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 436/500 | 456/500 | 0 | 0 |
| Graph-factor refit | 435/500 | 454/500 | 1 | 2 |

The equivalent-program comparison has zero gains and two regressions.
Selection retains the incumbent. No model or serving manifest changed.
These are program scores; earlier answer-execution scores are different
measurements and must not be substituted for them.

The detached run completed in 621.360 seconds, with return code zero, no
restart or timeout, and an empty process group and descendant lineage.
Before the fixed-source run, 150 focused tests and smoke (164 passed, one
skipped), lint, compile, governance and layering passed.

Raw validation and completion receipts are retained under
`artifacts/rlc/semantic_graph_factor_dev_20260915/`. Validation digest:
`1e209e2ad7103f51bc7dadaebccf25318076f590d43f4cbb04d6dfa4a83fb0fa`.
Candidate receipt:
`7eb215e31eb31e4f446ad7b8a3c21c950e7c2d60135934b6323fd277ea126618`.
The model file remains at
`~/.aura/rlc-evidence/semantic-graph-factor-dev-20260915/candidate.json`.

No validation labels entered fitting; all 500 validation rows are exposed
development data and zero test examples were used. G03 remains open. This
result does not support further promotion by rescaling the same factors.
Remaining diagnosis must separate operation-chart support, argument support,
and incorrect learned ranking under the actual decoded operation context.
