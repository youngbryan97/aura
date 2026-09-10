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

## Live replay, 17:48 onward

The shell repair at ad42ee1f1 was deployed to the canonical checkout and loaded
by browser reload. PID 77385 continued running its earlier Python revision;
this isolates shell validation and does not validate the newer allocator.

Turn `aura-chat-47fb6801-06f7-4e7b-90c0-5167f5a670f3` requested a short
explanation of redo after a committed transaction. A second window opened
during generation restored the pending question. Without a manual reload,
it later displayed the full answer beside that question. A subsequent DOM
inspection still contained exactly one answer. The answer correctly explained
that a durable commit log can precede flushing modified pages to disk.

The worker measured 2336 prefill tokens in 25.55 seconds and 247 generated
tokens in 52.74 seconds. Total request time was 266454.62 ms, including
51006.04 ms of reply stabilization. Event-loop freezes occurred during the
turn. The stall trace at `data/error_logs/stalls/stall_1789001530.txt` showed
reply validation rebuilding the source skill catalog synchronously. The
watchdog remained alive and reported its unhealthy state rather than being
restarted by the controller.

The pending-window subcase is live-verified. Latency and the broader R09
obligations remain open.
