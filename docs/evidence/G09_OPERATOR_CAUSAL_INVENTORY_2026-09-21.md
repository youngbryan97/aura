# Operator interventions and trial restoration

The shared causal inventory omitted retained kernel operators. Comparisons
also allowed baseline mutations and unsuccessful interventions to survive into
later arms. A custom undo failure was caught instead of invalidating the result.

The inventory now includes invented operators and their declared dependents.
Withdrawal removes the dependent closure while preserving unrelated later
inventions. It also removes withdrawn entries from rollback snapshots so that
a later rollback cannot resurrect them. Invalid declared lineage is refused.

Baseline and intervention arms both use the existing reversible trial. Its
snapshot now includes the spelling-search sets and dictionaries. Partial,
failed, and raising interventions restore shared state. Custom undo failures
propagate after shared restoration instead of producing a clean comparison.

The focused regression run passed 118 tests. Twelve tests in
`tests/test_operator_causal_inventory.py` cover these cases. One executes
bounded floor search with a retained offset operator, withdraws it, measures
the increased search cost on different inputs, and checks rescue. This is a
same-function input generalization test, not a new-family reasoning result.

Remaining boundaries: lineage is declared, not extracted from arbitrary
functions; snapshots cover registered state, not every external side effect;
the developmental cost consumer still uses its narrower induction grammar.
No broad gain, G09 closure, live deployment, or new-family transfer is claimed.
