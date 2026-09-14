# Arity-state search correction

Operation feasibility was checked after partial interval charts had already
been truncated. With mixed unary and binary primitives, higher-scoring
infeasible prefixes could remove every feasible chart at a cardinality.

The opt-in v3 policy retains top-k prefixes separately by total argument
edges and maximum primitive arity. Together with cardinality, these are the
sufficient statistics used by the existing edge-count feasibility bounds.
The final feasible beam remains the same size. Unknown primitives are excluded
from this policy, and older receipts retain their original search behavior.

65 focused tests passed, including exhaustive feasible top-k comparisons with
overlapping spans. Lint, compile, governance and layering passed. Smoke had
163 passes, one skip and the unchanged resident-manifest drift failure.

The source evaluation took 115.23 seconds and scored 464/500. Exposed weave
took 42.08 seconds and scored 47/48. Both have zero gains and zero regressions
against the v2 parent. This is a measured null, not a reasoning improvement.
Controls were not rerun for this unpromoted null candidate.

Receipts and reports: `artifacts/rlc/semantic_arity_state_dev_20260913/`.
G03 remains open. No live activation or broad-transfer claim is made.
