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

## Repaired tool Stop replay

Checkpoint `9a64751d5` was applied with the UI identity follow-up. Supported
reboot replaced PID 10940 with PID 16694. Boot verified workspace
`9fe8f017b7352d55c657c73424789be38c1955438319f18b4b2badd8e579d888`,
with source_current/source_verified true and no issues.

The identical switches-and-bulbs request took the same tool-capable path.
Stop interrupted worker job 3 at 448/1619 prefill tokens at 21:03:32.077.
Delivery `aura-chat-278d89c5-808c-40c3-b2c6-cbf21c91f12a`, turn
`4614b66141b64cf59e4ad22f331a7f0b`, sealed at 21:03:32.449. The desktop
displayed one acknowledgement in that same second. No fallback generation
started. The next turn, delivery
`aura-chat-d7502048-b2ce-461b-a049-c9ef18bdcf2d`, correctly recalled
"Gone Girl by Gillian Flynn" at 21:05:31 on the same process.

The delivery-journal, cancellation, and live-persistence suites passed
59 tests in 13.20 seconds. Ordinary and tool-path cancellation are live-passed;
the distinct latent-worker cancellation replay remains outstanding.

## Recovery ownership and a slow reconnect reader

Delivery `aura-chat-49873e39-96d6-4a1d-baf4-04c42e7b344b`, turn
`5ed574b436d445f9b86736b4119e3637`, failed after the primary model produced
2,150 characters. A short runtime self-evidence projection was incomplete;
recovery classified that text as a prior model segment even though the turn
had no consumed generation or transaction identity. The resulting completion
could not prove ownership. Recovery now resumes only consumed model work;
otherwise it submits the original question as a new CognitiveEngine turn.
It no longer replaces that question with a repair directive. The production
closure has executable tests for both cases. Five focused checks and 22
continuation/ownership regressions passed. Smoke passed 164 with one skipped;
lint, compile and layering passed. Live replay is pending.

Reload retained one answer per current-runtime exchange with separate saved
history IDs, but restored only four exchanges. The neural log identifies the
cause: conversation_history_read exceeded its 1,500 ms wait and discarded the
result. A read-only query of the 3,002-row conversation database returned the
requested 600 rows in 9.8 ms outside the live executor. Older history is still
stored; the saturated read path, not retention, is the remaining defect.

## Retained read ownership and late history merge

The reader now owns one in-flight load per store/principal/surface/session/
window scope. A caller timeout or cancellation leaves that load intact. A
later poll consumes its result, retained for 30 seconds; at most 16 scopes
can be retained, and an in-flight owner cannot be evicted for another read.
Failures are reported even after the initiating waiter has gone away.

The browser also needed a merge repair. A bootstrap that first restored four
RAM exchanges ignored older durable rows on subsequent polls. Passive panes
now prepend missing older exchanges before their first known turn, without
replacing existing bubbles or adopting an unbound active delivery. Insertion
happens before pruning, and restored messages render whole without animation.

The delayed-read, scope, UI merge and rendering suites passed 31 tests in
11.62 seconds before the late-failure reporting test was added. Smoke passed
164 with one skipped in 191.47 seconds; lint, compile and layering passed.
A fresh live window on the prior revision restored all 100 exchanges at
21:47, confirming the reported failure is intermittent rather than data loss.
These new changes still require deployment and live replay.

The final focused run, including late-failure reporting and both recovery
ownership cases, passed 34 tests in 15.14 seconds.

## Worker activity across parent-loop stalls

The 22:14:58 desktop recurrence question reached the resident worker. Its
receipt measured 1,897 prefill tokens in 25.88 seconds and 116 decoded tokens
in 65.81 seconds. The parent declared a 48-second token stall at 22:18:44;
the worker then honored that cancellation at 22:18:49. The decoded segment
contained no public answer, and the desktop received a fallback at 22:20:35.
Parent-loop delays had prevented consumption of progress already emitted by
the worker. This is a failed live replay.

The repair publishes the existing inference watchdog's activity through a
small shared record, bound to the job sequence and allocated anew for each
worker process. Parent stall decisions read this record directly. Heartbeat
receipts retain their worker emission time and cannot renew another request.
Queued stall reports yield to more recent inference evidence. Prefill and
terminal completion use the same channel; heartbeat generation alone never
refreshes it. Reads and writes acquire the shared lock without waiting, so a
worker death during publication cannot block its parent.

The combined lifecycle, progress, cancellation and IPC run passed 433 checks
in 140.50 seconds. The final first-token watchdog and memory-focused run
passed 122 checks in 37.49 seconds. Smoke passed 164 with one skipped in
57.94 seconds; lint, compile, governance-lint and layering passed. A real
spawned-child test continued publishing while the parent's event loop was
deliberately blocked. Live deployment and R09 acceptance remain pending.

## R09 closeout, September 12

R09 is complete for the acceptance cases listed below. This table supersedes
the open dispositions earlier in this record without changing those dated
observations. The final live sequence ran on the resident 27B, PID 7761,
workspace snapshot
`a81ad0fedd37274043d0144e8220ade32e03b31274497270025f3931c7e10bf7`.
It was a verified dirty-source deployment, not a clean-commit deployment.

| Requirement | Result and retained evidence | Executable regression |
| --- | --- | --- |
| Complete public response | PASS. Delivery `aura-chat-24419db2-9302-495b-b6f4-310909c584c8` delivered three numbered points and two examples, without private reasoning or PARTIAL. One generation, zero retries or mutations. | `tests/test_chat_render_completeness.py`; `tests/js/chat_numbered_answers.mjs` |
| Durable reconnect | PASS. Reload during that delivery restored the pending request, then one complete answer without another reload. Final reload retained all three September 12 turns. The earlier passive-window replay retained one owner across windows. | `tests/test_ui_durable_conversation_history.py`; `tests/test_chat_delivery_journal.py` |
| One final answer per turn | PASS. The current sequence retained one bubble per identity after reload. The earlier consecutive OK deliveries remained two distinct answers, not one text-deduplicated answer. | `tests/test_chat_stream_identity.py`; `tests/test_chat_delivery_journal.py` |
| Ordinary Stop and next turn | PASS. Worker job 5 acknowledged Stop at 20:22:13 on September 10; the next turn answered Gone Girl at 20:24:24 on PID 98193. | `tests/test_chat_delivery_cancellation.py` |
| Tool-path Stop and next turn | PASS. Delivery `aura-chat-278d89c5-808c-40c3-b2c6-cbf21c91f12a` sealed Stop at 21:03:32 with no new fallback generation. Delivery `aura-chat-d7502048-b2ce-461b-a049-c9ef18bdcf2d` then answered correctly on the same process. | `tests/test_chat_delivery_cancellation.py`; `tests/test_chat_lane_can_run_a_tool.py` |
| Latent Stop and next turn | PASS. Delivery `aura-chat-ca9c69dd-ddc2-4966-9d07-4817c503559f` received one Stop acknowledgement and job 5 recorded latent_reason_caller_cancelled. Delivery `aura-chat-86c2700a-f0c1-445f-8c1e-9cd1b34e5dd4` then recalled the corrected title and reason on the same process. | `tests/test_chat_delivery_cancellation.py`; `tests/test_latent_cortex_wiring.py`; `tests/test_mlx_soft_cancel.py` |
| Follow-up semantics | PASS. The exact title-free paraphrase `aura-chat-c32f4d8b-5235-4761-bee1-0967cc3018a3` explained the earlier correction with no mutation or retry. September 12 delivery `aura-chat-24c34582-e132-4fdd-97f1-b3f11f8959d1` resolved "those processes" and retained its full example. | `tests/test_a_correction_answers_what_was_asked.py`; `tests/test_capability_ledger.py` |
| Multi-turn context across topics and restart | PASS. September 12 delivery `aura-chat-5f8633e4-060a-45a8-b050-6842947ced9d` recovered the agreed title, original title, and correction after the science discussion and runtime restart. It used 40 admitted exchanges, one generation, and no retry or mutation. | `tests/test_ui_durable_conversation_history.py`; `tests/test_live_conversation_persistence.py` |
| Display capacity | PASS. Bootstrap restores 100 exchanges and the browser displays 200 messages, including late durable history after an initial RAM-only response. This does not enlarge the model admission budget. | `tests/test_ui_durable_conversation_history.py` |

The [September 11 record](R09_LIVE_DELIVERY_2026-09-11.md) contains the exact
paraphrased and latent-Stop receipts. The
[September 12 record](R09_DELIVERY_REPLAY_2026-09-12.md) contains the final
sequence, source identity, measurements, and neural observations. All checks
above exercise shared mechanisms; no expected title or canned science answer
was added to the production system.

Final focused validation passed 119 checks; the rebased delivery, cancellation,
identity, history, rendering, and capability run passed 153. Smoke passed 164
with one skipped. Compile and layering passed. Aggregate lint remained red
on 69 unrelated integration findings; this is not an all-repository green
certificate. R06 retains event-loop blocking, R08 retains incorrect telemetry
meaning and other warnings, and R11 retains latency work. The earlier worker
death and restart races remain recorded. This bounded delivery closeout does
not prove failure-free endurance, arbitrary factual accuracy, or RLC gains.
