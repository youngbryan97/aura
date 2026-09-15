# Joint semantic refinement workplan

This work implements the six mechanisms requested on 2026-09-15. It extends
the existing semantic program substrate, procedure registry and proof kernel.
It does not replace or close the master G03-G12 obligations.

## Implementation and acceptance

- [x] S01 Capacity certificates. Convert frozen score comparisons into linear
  constraints. Reuse the exact arithmetic kernel to check witnesses. Separate
  proved infeasibility, verified feasible weights and unresolved numerical
  search. Retain the comparison identities and assumptions with the result.
  [Checked evidence](evidence/G03_CAPACITY_AND_SEARCH_2026-09-15.md).
- [x] S02 Complete bounded operation search. Retain every interpretation
  inside the declared candidate grammar, with lazy search and sound bounds.
  Compare to exhaustive enumeration on small cases. Report an interrupted
  search as incomplete, never as proof that no interpretation exists.
  [Implementation and exhaustive checks](evidence/G03_CAPACITY_AND_SEARCH_2026-09-15.md).
  Complete only for the declared operation grammar; S03/S07 remain open.
- [ ] S03 Joint semantic learning. Train operation, reference, scope and
  dependency decisions against complete incorrect interpretations. Preserve
  equivalent correct programs and use the same candidate construction at
  training and inference. A three-scalar calibration is not this task.
  [Witnessed graph negatives and algebraic positives](evidence/G03_SEMANTIC_COUNTEREXAMPLES_2026-09-15.md)
  are implemented; joint neural-head learning remains open.
  [Graph-supervised relation tissue](evidence/G03_GRAPH_RELATION_CANARY_2026-09-15.md)
  repairs the measured relation conflict and selects equivalent programs on
  four training canaries. Full-cohort measurement and operation learning remain.
- [ ] S04 Counterexample refinement. Repeatedly find new wrong complete
  programs, retain their distinguishing evidence and refit against the whole
  retained set. Detect unsupported or inseparable distinctions and request
  representation expansion rather than declaring optimizer convergence done.
  The relation refit retains witnessed pairs across rounds; full-cohort
  iterative qualification is still required.
- [ ] S05 Counterfactual training. Generate meaning-preserving renamings,
  reorderings and recompositions plus minimal meaning-changing contrasts.
  Validate their IR independently. Freeze fresh evaluation families outside
  the generation/training inventory.
- [ ] S06 Reusable verified abstractions. Store parameterized procedures with
  scope, dependencies and evidence in the existing registry. Demonstrate
  reuse under new names and values without answer lookup or task-label access.
- [ ] S07 Integrated measurement. Run the unchanged 500-row development
  comparison, freeze a selected candidate, publish prospective fresh transfer
  and matched-arm plans, then perform G04-G08 measurement and verification.
- [ ] S08 Runtime and broad transfer. Qualify current-model materialization,
  validate live use and evaluate broad tasks and named frontier comparisons
  under the original G09-G12 requirements.

## Correctness boundary

For a bounded, unambiguous task, correct selection follows if the correct
semantic class is reachable, its best score exceeds every incorrect class,
search is complete, and execution preserves the selected meaning. A proof
about this finite score model does not prove language understanding, future
generalization, or broad frontier performance.

Training annotations and diagnostic targets remain outside runtime inputs.
The independent verifier must distinguish interpreting the requested task
correctly from merely executing a well-typed but incorrect program.
