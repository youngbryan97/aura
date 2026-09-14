# G09: measured task outcomes can change procedure selection

The shared procedure path now has an outcome/value adapter. It uses the
existing OutcomeLedger and ActionValueModel; it does not introduce another
learner or knowledge store.

A plan's learning key binds its ordered backend contracts and requested
goal. Process-local registry IDs are excluded. Compositions inherit stable
identity from their leaves; legacy procedures without executable contracts
cannot receive durable learning identity through this adapter.

Ranking is an in-memory read over caller-eligible plans. Execution opens one
receipt for the complete task through the shared procedure executor. A later
explicit assessment resolves it and retains evaluator/evidence provenance.
Returning an integer or satisfying a structural requirement does not resolve
correctness. Separate executions do not collapse into the same receipt. The
task reward is not distributed across leaves that were never individually
assessed.

The test executes two read-and-sum plans through TOOL and RLC backend
callbacks. Both complete and return integers; one result is wrong. An
independent fixture check records failure and success. After reopening the
SQLite ledger and rebuilding the registry in reverse registration order,
the shared learner selects the successful composition for new inputs. The
new ungraded execution adds no successful observation.

Ninety-five procedure, planning, semantic runtime and execution tests passed
in 12.99 seconds. This is CPU component evidence with supplied candidate
plans and a fixture evaluator. It does not demonstrate that the live language
front end proposes those plans, chooses an independent evaluator, or gains
on broad held-out tasks. G09 remains open.

The gate initially mistook an annotated OutcomeLedger.open call for a raw
file open because its receiver was named ledger. The audit now resolves the
declared receipt API and Path parameter types. Unknown annotations and
rebound parameters keep conservative path checks. Thirty-three audit and
feedback tests passed in 30.56 seconds without changing the audit baseline.
The expanded audit/feedback run passed 35 tests in 55.44 seconds, including
variadic parameter shadowing and import rebinding. Lint, compile, governance,
layering and writing passed. Smoke passed 163 tests and skipped one, with the
existing resident-manifest qualification alarm still failing. That alarm is
not waived by this component evidence.
