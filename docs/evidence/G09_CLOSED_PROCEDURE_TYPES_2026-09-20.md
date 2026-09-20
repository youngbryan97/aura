# Closed structural procedure execution

The reviewed reference's proof-type registry admitted values that failed a
declared parent predicate. Aura already has shared procedure signatures and
runtime value checks, so this change extends that owner instead of importing
another registry.

`execute_procedure` and `execute_procedure_plan` now accept explicit
`closed_types=True`. The executor checks every nested procedure declaration
before invoking any backend. The planner also checks goal declarations when
the goal is already satisfied and no backend is needed. Unknown structural
types and invalid typed constants are rejected before effects. Boolean values
do not satisfy integer ports.

The existing semantic-program runtime requests this mode on its shared
procedure path. Its execution receipt records `closed_types_checked`; task
correctness still requires independent assessment. Legacy nominal labels retain
their existing presence semantics unless the caller requests closed checking.
The closed vocabulary reuses the existing value predicates. It does not add
user-defined subtyping, certify a task's intended meaning, or authorize effects.

## Verification

- 97 focused procedure, planning, semantic-currency and runtime tests passed.
- Smoke: 164 passed, one skipped.
- Lint, compile, governance ownership, layering and writing gates passed their
  existing baselines. Those baselines still include recorded migration debt.
- `procedure.closed_structural_declarations` registers the new invariant.

The checks include unknown nested declarations before any backend call,
invalid constants, already-satisfied unknown goals, explicit boolean mode,
legacy nominal compatibility, real typed computation, and absence of implicit
success credit. The semantic runtime test verifies the integrated receipt.

This implements the closed built-in-type portion of report item F. Extensible
subtypes and broad task interpretation remain separate work. G09 remains open:
these contracts are neither a broad reasoning measurement nor live proof.
