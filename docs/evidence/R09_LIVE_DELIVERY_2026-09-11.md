# R09 live delivery replay

## Source and process

The supported signed-app launch started runtime PID 32237. Boot verified the
dirty workspace snapshot
`d0d8522c31b63555e98f512fb1945b6985cedfb985cd120c946a1aaa15b94e15`
with source_current/source_verified true and no revision issues. The launch
checkout's MLX client, worker, progress channel and chat-format files matched
checkpoint `c5b8ca7e9` byte for byte. No other desktop runtime, training
campaign or soak owned the model at launch.

The resident model was
`Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`. Chat dependencies
finished warming at 01:31:54 PDT, 60.74 seconds after cortex readiness.

## Original failure replay

The exact recurrent-reasoning question from the failed September 10 replay
was submitted through the desktop chat at 01:32:19. Delivery
`aura-chat-b88d171f-aaab-43a5-9d0d-8f626ed43906`, turn
`646958f664d044fcba89d6029096c886`, completed at 01:36:39 with one attempt,
one generation and one displayed answer. The worker measured 6,996 prefill
tokens in 38.38 seconds and 2,035 decode tokens in 199.67 seconds, with
38.64 seconds to first token. It was not falsely cancelled for lack of
progress. No private reasoning appeared in the public answer.

The answer addresses recurrence, state, verification and comparative
measurement and ends in a complete sentence. This is delivery evidence, not
semantic certification: its account incorrectly treats ordinary autoregressive
generation as having no dependence on prior generated tokens. Its architectural
claims also need independent verification. R10 retains those obligations.

## Pending transcript omission

A second browser window opened while that turn was generating. Its bootstrap
returned 100 prior exchanges but omitted the active question. This was not a
text-deduplication defect: the server had not opened the new transcript
exchange. The early self-evidence recovery branch began generation before the
normal branch's prelogging call.

The repair opens transcript custody at scoped API ingress, before answer
routing. Later begin/log calls reuse that delivery's primary exchange and
preserve the original visible user text. The terminal journal still owns the
final answer. Passive windows extend their history from a shared exchange ID;
identical wording does not identify an exchange, and an active delivery's
unbound bubbles cannot be adopted by history polling.

The focused transcript and history suites passed 26 tests in 6.34 seconds.
The delivery journal, Stop, terminal custody/recovery, UI history and
shared-history regressions passed 113 tests in 16.73 seconds. The Node
pending-window regression passed. Live deployment of this transcript repair
and the distinct latent-worker Stop replay remain pending at this entry.

Smoke passed 164 tests with one skipped in 67.94 seconds. Lint, compile,
governance-lint and layering passed; existing governance and layering debt
did not increase.

## Preflight replay and audit repair

The transcript repair was deployed through supported UI reboot. The old
runtime exited; a transient successor received SIGTERM before PID 41183
became the sole serving process. Boot verified dirty-source snapshot
`ba323f906b0428a37569806df567b48e30b4ccfebe0a53b08be457d745208e93`.
The transient successor is a separate lifecycle observation, not proof of
exactly one attempted spawn.

The Ship of Theseus question at 01:44:43 failed before cognition. Delivery
`aura-chat-1a8262b2-570b-4e91-8bea-630a25698406` produced a conscience
unavailable response. The word `paradox` matched the raw substring `dox`.
The resulting audit append then raised GovernanceViolationError because it
had no internal file-write authority. A status poll also emitted a conflicting
immutable receipt: its HTTP 200 was substituted for the turn's HTTP 503.

Catalog triggers now match whole lexical phrases; the dox word family remains
covered explicitly. Audit writes use a scoped internal file-write authority
for the fixed violations ledger. Chat runs this synchronous preflight off
the event loop. I/O failure remains recorded and does not erase a refusal.
Answer receipts retain the sealed turn's HTTP status on both original delivery
and polling. They do not reinterpret a successful poll as a successful turn.

The focused tests passed 32 cases in 3.06 seconds. Delivery, cancellation,
terminal custody, social receipts and conscience regressions passed 76 cases
in 6.11 seconds. Smoke passed 164 with one skipped in 42.05 seconds. The
aggregate lint found an inherited import-order defect in bonding_phase;
its imports were sorted without changing behavior. Live replay remains due.

## Completed preflight replay and passive-window defects

Supported reboot loaded PID 52242 with verified source snapshot
`d059007f9f8a0ec5381f3a20ac3c3b1c19e92b7988e4329f45045ececb7b44d1`.
The unchanged Ship of Theseus question at 01:58:40 passed conscience and
completed one generation. Delivery
`aura-chat-f5857e9a-c720-4567-bb64-11773526c032` opened transcript exchange
`c18d566eeb6a4166946c6e5ca96351d9` before generation. The initial pending
question was present in the bootstrap API. This closes that preflight replay,
not R09.

The delivered answer contained an unrelated intention-store sentence. Its
receipt identified one measured_sentence_replacement at
`chat.capability_claim_reconciliation`, for deferred_action. The impersonal
claim that the ship does not persist shared a predicate with the capability
catalog. It did not deny Aura's ability. Catalog entries now separate topic
cues from the entity names required to ground impersonal denials; direct
self-predicates remain eligible. Tests cover several unrelated subjects and a
new capability declared through the same interface. This is a bounded repair
of predicate/entity confusion, not proof of complete semantic attribution.

A passive window remained stale despite the current bootstrap API. Code and
deterministic timer tests exposed refresh starvation: workload changes reset
the next poll. An earlier scheduled refresh now retains its deadline. A live
unsolicited message also had no exchange identity and prevented transcript
suffix reconciliation. Such events now carry a distinct transcript-event
marker. They remain visible without being mistaken for unbound turn replies.
The active-delivery protection is unchanged.

Focused capability, transcript and cancellation checks passed 96 tests with
three skips in 11.18 seconds. Smoke passed 164 tests with one skip in 69.50
seconds. An earlier smoke attempt was interrupted after 282.59 seconds with
73 passed and one skipped while calibration tests were computing under host
load; it is not counted as a pass. Lint, compile, governance-lint and layering
passed. Deployment and live replay of these latest repairs remain pending.

The neural feed also recorded event-loop lag and lease-renewal delays during
concurrent tests and two rendered chat windows. Health recovered after the
extra windows closed. This observation is retained for R06/R08; it does not
establish that rendering was the sole cause.
