# R09 pending-window reconciliation

The live failure is recorded in `R09_DIALOGUE_BUDGET_CUSTODY_2026-09-09.md`:
a second window restored an in-flight question but did not add the answer
after completion. A manual reload recovered it from durable history.

The UI discarded the exchange ID during conversion to messages, then refused
all history hydration once the transcript contained any element. Reconnect
also explicitly disabled history after the first bootstrap.

Restored messages now retain the backend exchange ID and role. A passive
window can insert a completed answer beside its known question. It does not
clear the transcript, match on text, replay unknown exchanges, or take over
an active local delivery. Repeated hydration skips an answer already bound
to that exchange. An unbound live bubble is left to its delivery owner.

The JavaScript regression failed before repair with only the question present.
It now checks pending-to-complete recovery, repeated hydration, repeated
identical questions with different IDs, late completion before a later pair,
unknown exchanges, and active-delivery ownership. The production conversion
function is exercised, including original timestamps. The focused Python/Node
selection passed 16 tests in 9.34 seconds. JavaScript syntax checking passed.
Smoke passed 164 tests with one skipped in 116.78 seconds. Lint and compilation
passed.

Live replay of this repair is pending. R09 is not closed by these checks.
