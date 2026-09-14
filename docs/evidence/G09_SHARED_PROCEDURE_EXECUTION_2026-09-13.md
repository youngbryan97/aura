# Shared procedure execution, 2026-09-13

## Boundary repaired

The common procedure registry already composed signatures across backends.
It did not execute those compositions. A signature's `apply` method writes
declared effects, including a default `True`; that is a planning projection,
not a computed result.

`core/cognition/procedure_execution.py` now lowers registered compositions
into the existing `tool_plan.Executor`. It resolves the complete procedure
graph and backend availability before execution. It checks preconditions at
each step and validates explicitly returned outputs against their declared
types and constant values. It does not use `Signature.apply` to manufacture
execution results.

Nested compositions and repeated procedure uses execute in order. Cycles,
missing or retired procedures, inconsistent composition metadata and missing
backend adapters are rejected before side effects. Callback authority remains
with the caller's backend gateway. Async backends require an explicit adapter;
this executor does not silently drop an unawaited coroutine.

The call count is derived from the finite program, not an arbitrary elapsed
deadline. Floor fuel and backend cancellation/authority remain in force.
Failure preserves the completed local state and records the failed step. No
external rollback is claimed. Structural completion is not automatically
recorded as task correctness in the procedure value learner.

Typed boolean `False` now satisfies a boolean port. Legacy untyped false-as-
absence semantics are unchanged.

## Runtime connection

`semantic_program_runtime.execute_compositional_semantic_observation` uses the
shared executor when the resident caller supplies its procedure registry.
`semantic_procedure_backend` adapts the source-independent program to the
existing universal floor. The result is computed once and its floor receipts
are preserved. The runtime test forbids the previous direct-execution bypass
and observes exactly one backend call.

The executable cross-backend test supplies a sequence through a tool adapter,
selects its last element through one real floor program, and multiplies that
value through another. It computes 33 without source wording or an expected
answer available to the execution path. This is an integration contract test,
not a held-out reasoning or live capability measurement.

## Verification

The final combined pass passed 156 procedure, semantic-runtime, shadow,
agent-surface, procedure-economy and conversion tests, including the executable
output invariant and async-adapter rejection.

Lint, compile, governance-lint, layering and writing passed. Smoke passed 163
tests, skipped one and failed the existing installation alarm:
`semantic_neural_activation_invalid:resident_manifest_drift`, with no drifted
bound source files. The changed resident manifest was not re-sealed against
historical evidence.

G03 remains open. G09 still needs task-grounded selection, independently
assessed outcomes and broad comparisons. No candidate was promoted, no model
was loaded or fused, and no live serving claim follows from these tests.
