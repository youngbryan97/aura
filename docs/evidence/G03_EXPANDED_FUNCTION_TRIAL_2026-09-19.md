# Expanded function-coordinate trial

The source-only trial selected 32 training examples from a 256-example scan,
retained operation labels and boundaries across those 256 examples, and replayed
100 validation examples. Validation outcomes did not select training examples.

| Cohort | Before | After | Gains | Regressions |
| --- | ---: | ---: | ---: | ---: |
| Training | 28/32 | 30/32 | 2 | 0 |
| Validation | 91/100 | 91/100 | 0 | 0 |

All 10,124 retained inequalities reached their required margin. The two remaining
training failures still selected their original incorrect programs. Passing a
fixed collection of latent graph comparisons therefore did not establish
correct autonomous decoding. Operation-chart retention search was incomplete.
There is no measured validation gain and no promotion.

Receipt: `13c8818089808281bbf1f4eba78b3636e9033dde64884fc25a584b7ac89da9da`.
Artifact: `~/.aura/rlc-evidence/semantic-functional-expanded-20260919.json`.
Elapsed time through the last replay: 1,876.47 seconds.

The existing iterative trainer re-mines predictions after each fit. It now
accepts a separate source-retention training cohort, so active mining need not
discard broader operation and boundary supervision. The retention cohort must
include unchanged mining observations, match the model geometry, and contain
neither validation nor test examples. Seven new cohort tests and twenty existing
joint-learning tests passed. Iterative outcome measurements remain separate.

Trial reporting also counts unresolved equivalence (`unknown`) as unmeasured,
alongside missing observations. Finite probes without a distinction do not prove
equivalence. Fifteen trial tests passed, including both unresolved states.

G03 remains open; neither source fitting nor unchanged validation establishes
fresh transfer, broad reasoning gain, or frontier performance.
