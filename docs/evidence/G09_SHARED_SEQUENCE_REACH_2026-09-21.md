# Shared retained-sequence reach

Developmental evaluation and ordinary sequence answering now consult the same
retained executable meanings: invented kinds, unshaped rules and invented
floor operators. Previously a retained floor operator could remain invisible
to the evaluator's narrower induction grammar.

One executable meaning must explain every supplied transition. Separate rules
matching separate examples do not establish a solution. If retained meanings
disagree on the requested input, their agreement on earlier examples does not
justify choosing the first answer.

Developmental comparison records both solved status and search work. A faster
failed search receives no positive utility. Losing a previously solved family
rejects the change even if another family opens. For fixed cohort size n,
the utility is solved_count + sum(solved / (1 + work)) / (n + 1).
The cost contribution is below one, so it cannot outweigh a solved-count loss.
The separate per-family check also prevents exchanging one solved family for
another. Measurements run in reversible trials to preserve retained state.

Focused verification: 22 tests passed with randomized seeds 4012339692 and
12345. These include ordinary sequence answering from an installed operator,
withdrawal, conflicting meanings, non-vacuous evidence, state restoration,
and correctness-before-cost comparisons. Targeted Ruff passed.

This establishes integration and bounded synthetic behavior, not fresh
cross-domain transfer, calibrated semantic understanding or G09 closure.
