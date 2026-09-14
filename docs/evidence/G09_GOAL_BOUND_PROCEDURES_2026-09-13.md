# Goal-bound common procedures

The registry could rank procedures whose inputs were present, but that did
not establish that their effects served the current task. A procedure that
accepts two integers may add them or subtract them. Its value score cannot
decide which meaning the user requested.

`procedure_planning.py` now searches backward from typed task requirements
through existing procedure signatures. It uses state keys supplied by the
task's grounding, checks constant effects and all required inputs, and rejects
steps that overwrite a still-required result with an incompatible effect.
An effect-key index removes irrelevant procedures from expansion. Search has
explicit depth and expansion accounting; it does not run tools or manufacture
observed values. Computed outputs do not establish an unobserved constant.

The caller can restrict eligible computation identities. The semantic runtime
does so using the learned IR's interned procedure, then checks its actual typed
result through the existing executor. This preserves the interpretation
boundary: another program with identical port types cannot replace the decoded
one. The runtime receipt names this selection basis and keeps semantic
correctness unmeasured. Identity-bound calls do not refresh unrelated learners.

Cross-backend tasks can use the broader dependency search. Repeated execution
interns a composition once rather than accumulating duplicate registry entries.
Actual backend writes, gateway authority and failed-step handling remain owned
by the shared executor. No procedure receives a successful learning observation
merely because its code returned.

## Checks

The tests compose a local CSV reader, a real universal-floor multiplication
program and a report formatter without supplying a prebuilt chain. Three input
cases include zero quantity and different magnitudes. They check the final
value and source hash; removing the required computation removes the plan.
This is a CPU integration test with supplied semantic IR, not natural-language
transfer or freely decoded public-answer evidence.

Twenty randomized finite Boolean problems compare dependency search with
exhaustive enumeration through depth three. Other cases cover opposite goals,
typed false, unknown constants, cycles, resource bounds, missing inputs, changed
state, retired procedures and repeated composition. The task-goal invariant is
executable. The final focused pass passed 133 tests, including typed-constant
identity and identity-bound refresh checks. Lint, compile, governance, layering and writing
passed. Smoke passed 163, skipped one and retained the current-manifest alarm.

G09 remains open. General interpretation of arbitrary requests, independently
assessed outcome learning and broad live comparisons still need evidence.
G10/G11 retain the resident-manifest qualification alarm; no historical
activation was re-sealed.
