# Conditional fit search interruption

The first conditional source-fit attempt decoded its 25 source acquisition
rows and mined one complete row. It was then interrupted while mining the
second row. No fit, candidate, or validation result was completed.

The captured stack ends at `_dual_bound_screen -> scipy.optimize.linprog ->
highs.run`. The miner supplied a three-second solver allowance, but this optional
linear relaxation received no time limit. The shortlist integer solve and final
integer solve also each received a fresh allowance rather than sharing one.
The missing allowance was an orchestration defect, not evidence that the graph
was infeasible or the selection mechanism had failed.

`optimize_argument_chart` now shares the caller's explicit remaining allowance
across shortlist, linear screening, and final integer search. An unsuccessful
optional screen preserves all original choices. Exhaustion raises the existing
incomplete-search exception; a shortlist result never becomes a false optimum.
Calls without a declared allowance still receive no invented timeout.

Focused verification: 122 tests pass, including exhaustive small-chart
comparisons and regression tests for each phase's remaining allowance. The
source fit must restart from the repaired implementation and retain a new
identity. The interrupted attempt remains at
`~/.aura/rlc-evidence/semantic-conditional-fit-20260920/`.

G03 remains open. This repair makes the diagnostic bounded; it does not establish
semantic accuracy, fresh transfer, public answer correctness, or serving authority.
