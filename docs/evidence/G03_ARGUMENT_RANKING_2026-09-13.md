# Source argument ranking development

The existing pairwise trainer was applied to the feasible-operation candidate
using 728 source-training examples. The 500 validation cases were excluded
from fitting. No new model load or live serving was involved.

The original pairwise objective converged in 30.95 seconds. It scored 472/500
source answers: ten gains and two regressions against the 464/500 parent.
Source evaluation took 119.10 seconds. Exposed weave remained 47/48 in 36.67
seconds. Coefficient and hidden-shuffle controls each scored 0/48.

Inspection found that alternative source-labeled mentions of the same register
were included as negatives. An opt-in correction excludes those aliases from
negative rows. Identity comes from labeled register references, not equality
of values. A span with conflicting register labels is not treated as an alias.
Legacy behavior remains the default for replay; the new receipt names the
changed negative-source policy.

The corrected objective removed 376 negative pairs and converged in 47.99
seconds. It scored 470/500: ten gains and four regressions against the same
parent. Source evaluation took 141.31 seconds. Exposed weave remained 47/48
in 58.82 seconds. Neither candidate is promoted: both lose prior successes.

The focused ranking, feasibility and chart suites passed 45 tests. Lint,
compile, governance and layering passed. Smoke reported 163 passed, one
skipped and the resident-manifest drift failure. G03 remains open.

Reports are preserved under
`artifacts/rlc/semantic_argument_ranking_dev_20260913/`.
These are exposed development results, not fresh replication or broad gain.
