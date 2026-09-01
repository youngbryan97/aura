# Failures that predate this work

Run at `ab58a159c` — the audited HEAD, before any of the endogenous-expansion
work — in a detached worktree, 2026-09-01.

```
FAILED tests/test_a_relation_learned_in_one_world_helps_in_another.py::test_refactoring_reaches_a_world_the_winners_could_not
FAILED tests/test_a_relation_learned_in_one_world_helps_in_another.py::test_a_restarted_library_still_reaches_what_a_blank_one_cannot
FAILED tests/test_a_concept_formed_from_the_failures.py::test_the_violations_only_go_down
FAILED tests/test_a_god_object_only_shrinks.py::test_the_tree_is_within_its_baseline
FAILED tests/test_a_god_object_only_shrinks.py::test_the_decomposition_this_gate_exists_to_encourage_passes
FAILED tests/test_a_god_object_only_shrinks.py::test_the_budget_is_the_measured_total
FAILED tests/test_a_god_object_only_shrinks.py::test_a_file_that_shrank_must_be_re_recorded
7 failed, 52 passed
```

`tests/governance/test_governance_lint.py::test_lint_passes_on_repo` passed at
this commit and fails on current `main`. The finding names
`core/knowledge/atomspace_persistence.py` and a write through
`get_file_write_gateway().write_text_async`, which arrived with the atomspace
refactor rather than with this work.

The god-object gate is over its budget by 29,171 lines across some thirty
files. None of them are files this work touches, and every new module here is
well under the two-thousand-line ceiling.

Recorded because "no regression" is a claim, and a claim needs somewhere it can
be checked rather than believed.
