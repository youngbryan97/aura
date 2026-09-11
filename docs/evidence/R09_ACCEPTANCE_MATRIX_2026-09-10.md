# R09 acceptance matrix

R09 requires all rows below. A model response, a successful HTTP status, and a
passing offline regression are distinct evidence. This matrix consolidates
the existing R09 receipts rather than replacing their historical outcomes.

| Requirement | Existing evidence | Remaining acceptance |
| --- | --- | --- |
| Complete public response | Complete redo explanation in pending-window replay | Current-source complete answer, no missing requested part or private reasoning |
| Durable reconnect | Pending second window received the terminal answer without manual reload | Repeat on current source; verify exact answer and one bubble after reload |
| One final answer per turn | Pending-window reconciliation and delivery-journal regression | Current-source final response count, stable exchange identity |
| Cancellation | Ordinary prefill stopped without model reload | Current-source Stop, terminal acknowledgement, next successful turn; latent cleanup replay |
| Follow-up semantics | Corrected title recalled; paraphrased explanation failed | Original reading-group correction and reason answered without title hints |
| Multi-turn retention | Authorized chronological reader recovers 40 exchanges | Correct use across intervening topic and restart, without invented history |
| Display capacity | Live endpoint 100 exchanges; DOM 200 messages | Passed subcase; no increase to model admission budget |

## Referenced evidence

- [Pending-window live recovery](R09_PENDING_WINDOW_RECONCILIATION_2026-09-09.md)
- [Ordinary prefill Stop and latent cleanup](R09_BOUND_LATENT_STOP_2026-09-09.md)
- [Chronological history and failed recall](R09_CHRONOLOGICAL_HISTORY_2026-09-10.md)
- [Continuation history](R09_CONTINUATION_HISTORY_2026-09-10.md)
- [Quoted recall](R09_QUOTED_RECALL_2026-09-10.md)

## Current-source preparation

Checkpoint b2ae1fe89 removes the partial belief reader's global-ignorance
assertion. The production file was applied to the canonical launch checkout
without replacing unrelated work. A supported UI reboot replaced PID 89247
with PID 98193. Boot reports source_current and source_verified true, no
revision issues, and expected/actual workspace identity
`d0f2f0bee22161b63aa34795b8de27cb45b00becb112c92505a707e1a310200a`.
This is a dirty-source snapshot, not a clean-commit deployment.

The selected delivery-journal, cancellation, live-persistence, UI-history and
personal-fact suites passed 83 tests with one skipped in 12.82 seconds. This
does not replace the remaining live rows above.

## Successful recall and reconnect

Turn `f870f2065b1d4bf780d7505b154ee8dc`, delivery
`aura-chat-2e2df92b-4e47-4393-876a-5ac1bf0bd3fe`, completed in 159.54 seconds.
It correctly recalled The Tell-Tale Heart, the short-story/novel correction,
and Gone Girl. Forty exchanges reached the turn. A passive window opened
during generation restored the pending question, then received exactly one
answer through bootstrap reconciliation. Reloading the initiating window
preserved the same answer. Neither window created a second generation.

The answer contained a redundant final sentence. The original 285-character
answer already explained the correction, but literal coverage comparison
failed to match correcting to corrected and requested the reason again.
The shared language layer now uses Snowball word-form normalization when
literal anchors do not overlap; structural obligation checks remain intact.
This is lexical engagement evidence, not a correctness or entailment proof.
The upstream implementation is documented at
https://pypi.org/project/snowballstemmer/3.0.1/ and uses thread-local stemmers
because shared instances are not thread-safe. Dependency is pinned in both
installation manifests. The focused coverage/projection suites passed 157
tests in 13.93 seconds. Live deployment of this follow-up is pending.

## Current-source ordinary cancellation

The database-recovery request began at 20:21:48 PDT. Stop requested worker
job 5 cancellation at 20:22:13.116; worker ownership cancellation completed
at 20:22:13.886. The desktop displayed one acknowledgement at 20:22:14.
The next reading-group follow-up was admitted on the same runtime at
20:23:32. Its answer remains pending at this recording.

At 20:24:24 that follow-up completed: "We settled on Gone Girl by Gillian
Flynn." The runtime remained PID 98193. This closes the ordinary Stop/next
turn subcase, not the latent cancellation row.

The word-form change passed smoke (164 passed, one skipped, 105.24 seconds),
lint, compile, governance-lint and layering. Existing governance debt and
37 grandfathered layering edges were unchanged.

## Clean coverage replay

The supported restart replaced PID 98193 with PID 4082 after a slow runner
shutdown. A diagnostic stack showed asyncio runner closure and a background
language-matcher warmup; the predecessor exited before a targeted termination
attempt, which returned no-such-process and had no effect. The waiter launched
one replacement. Boot verified workspace
`fb17ca8190b5e6ba12d968e4fa57c9f95589eb59e27f78ad97f46ac6cb7155f8`.

Delivery `aura-chat-3e574c22-1b3a-44f9-a12b-4b51bd8c91a2`, turn
`517634cad8584c4fa16f582c1ba99e07`, answered the identical reading-group
question correctly in about 83 seconds. It used one foreground generation,
zero text mutations, and no repair retry. The answer explained that the
original recommendation was a short story rather than the requested novel,
then named Gone Girl as the replacement. This validates the coverage repair
on the resident model without inserting the expected answer into the request.

## Multipart reconnect and stream ownership audit

Delivery `aura-chat-b2bd44c3-ed1d-4174-b89a-9110f71fa2b4`, turn
`ef46cc4c8db54b60b24a2cb8b9d91acd`, completed in 126.79 seconds with one
attempt and generation. Reloading the initiating window during generation
restored the pending request. The completed transcript contained one answer,
three numbered items, and the requested example. No private reasoning was
displayed. This proves complete delivery, not factual correctness: the answer
incorrectly says WAL must flush before modifying an in-memory page and refers
to the example's single transfer as if it were two transactions. Those are
recorded semantic defects, not a passing database explanation.

The stream audit found two independent ownership bugs. State-machine draft
events carried no delivery identity on the global telemetry topic; desktop
windows accepted any such event. Final HTTP delivery also deduplicated by
answer prefix across the conversation, suppressing valid repeated answers.
The repair attaches the existing delivery binding to public draft events,
keeps unbound voice chunks on their private channel, and matches desktop
events to the active request. Terminal delivery replaces that request's draft
and deduplicates by turn identity instead of words. Live deployment is pending.

The neural feed at 20:39 also showed event-loop lag (1.123 seconds), deferred
background model admission, a stale circadian refresh, and an unrated
autonomous refactor refusal. These are not evidence of an R09 delivery loss;
they remain observations for the runtime-warning and scheduler obligations.

## Repeated-answer live result

Checkpoint `5dae3420b` was applied to the launch checkout and the supported
restart replaced PID 4082 with PID 10940. The predecessor completed service
teardown, waited in asyncio runner closure, and exited without forced
termination. Boot verified workspace
`1e4d7251003b9edec3175adfadef4271157a51963951cb18a25036ce34d9edae`.

Two consecutive identical desktop requests, "Reply with just OK.", both
returned `OK`. The DOM retained both answers, each beside its own request.
Deliveries `aura-chat-32b22782-ce68-4038-94e1-edba6908d8c0` and
`aura-chat-bbfb9792-e117-4742-b377-c9a07e9cf280` have distinct turns
`33cdcd953d494664a6aabd5494135fe3` and
`98dc0ded7be649b4aab4cc78966cea8b`. The second took about 21 seconds.
This passes repeated-answer delivery on the resident cortex.

The follow-up `d7d916288` separates delivery IDs from saved history exchange
IDs. They are distinct server namespaces, so the UI must not label a delivery
as an exchange. The Node regression verifies that separation as well as same
turn replay, different-turn identical text, draft replacement, and late
confidence. Ninety-four confidence/runtime regressions passed in 18.59 seconds.
This follow-up is pushed but not yet loaded by the live window.

## Stop must cover fallback generation

The next reasoning request selected the tool-capable ordinary route, not the
latent route. Delivery `aura-chat-9c61053a-0277-4b9a-84c4-3ac1e277efcb`
accepted Stop. Worker job 20 acknowledged at 20:56:40.495, but a new ordinary
generation started at 20:56:40.707. The desktop did not receive its terminal
Stop acknowledgement until 20:58:01.

The tool-answer wrapper caught CancelledError with TimeoutError and returned
None, authorizing ordinary fallback after the owner had stopped. Cancellation
now propagates separately. The MLX generation entry also checks the existing
execution owner's stopping token before any worker or client mutation, so
other fallback callers cannot admit work for a stopped owner. The focused
tool/client suites passed 113 tests in 31.96 seconds. This is a new live
cancellation defect with a tested repair; it is not yet a passing live replay.
