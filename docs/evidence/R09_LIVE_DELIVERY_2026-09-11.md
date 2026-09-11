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

## Afternoon replay and cancelled-latent follow-up

Supported reboot loaded PID 32432 at 14:55:46 PDT with source snapshot
`4f87d82184345f6e2ba8073f5ab7f517261012f4a6bd2611987e1eb051299432`.
Boot reported source_current/source_verified true and no revision issues.
The deployed capability and browser files matched checkpoint `e378f2e18`.

The unchanged Ship of Theseus question, delivery
`aura-chat-1b048c6f-a8fb-4ca6-b685-2e1b80b1c425`, turn
`439b1e0de010487ca2a0440b4bf412e7`, produced a complete 2,980-character
answer with no capability-text replacement. The first generation measured
667 decode tokens, 45.31 seconds of prefill and 65.95 seconds of decode.
It was nevertheless rejected as an unsupported screen reading because the
sentence put a quoted ship in a dock. Mere co-occurrence of a quotation and
a display-related noun is not attribution. Screen-reading detection now
requires an actual display reading relation; attributed quotes still require
capture evidence. Its 49 focused tests passed in 5.54 seconds.

The unnecessary retry failed with no public answer; the original draft was
preserved and delivered at 15:01:17. A reconnected window recovered the pending
question and subsequently received exactly one complete answer without a
manual reload. This is a passing reconnect subcase, not a clean-generation
pass. A separate passive window opened earlier was closed by the browser
session boundary and is not counted as observed live completion.

The shorter recurrent-verification question reached the actual latent worker.
Delivery `aura-chat-ee07f429-5fd6-4c3b-b266-2d0202993856`, turn
`bfaa9f403a6b4bc8b7ae6f5b5c449ea4`, requested Stop at about 15:05:40.
Job sequence 8 received `latent_reason_caller_cancelled`; request
`2873b71e4abb48e38bc70a95509fb263` reported verified cleanup with the resident
lane preserved. The UI displayed one cancellation acknowledgement at 15:05:41.

The following reading-group question completed at 15:06:54, but incorrectly
denied any record of the selected novel. Delivery
`aura-chat-b563f056-8df7-4c8a-bc5d-df250ccf88df`, turn
`9432fe63bbfa44868931c04bb76646f8`, is therefore a semantic failure despite
its delivery contract reporting success. It does not close Stop/next-turn
acceptance.

The worker's 10,240-token window was not evidence of leaked latent settings:
it was the qualified simple lane's 8,192-token input limit plus 2,048 reserved
output tokens. The short question carried an 84-message prompt, but lane
selection considered only the question. Admission rejected the remaining
9,024 prompt tokens; the fallback silently retained only five dialogue
messages. A shared-history probe also supplied an unsupported no-agreement
assertion. Foreground lane selection now considers assembled input against
existing qualified envelopes, without expanding explicit caller contracts or
requested output. Retries retain the already admitted dialogue, exact current
input, and evidence rather than applying a second count/character window.
The context/inference regressions passed 255 tests in 65.70 seconds.
Live validation of these repairs remains required.

The shared-history reader now uses the foreground turn's existing evidence
custody, including its scoped durable transcript. An empty or unread admitted
snapshot cannot fall through to another RAM conversation. Topic overlap only
selects excerpts; it cannot establish agreement or absence. Corrections and
adjacent replies remain together. The shared-history, custody and screen
regressions passed 89 tests in 7.03 seconds. Final smoke passed 164 tests with
one skip in 67.60 seconds; lint, compile, governance-lint and layering passed.

Checkpoint `da2fe8952` was pushed and its three production files applied to
the launch checkout. UI reboot at 15:30:12 saved state and completed service
teardown, but PID 32432 lingered. A native process sample was retained at
`/tmp/aura-r09-0911/shutdown-sample.txt`. A targeted SIGTERM was observed;
the process subsequently exited with code zero. Later SIGUSR1 and SIGKILL
attempts both returned no-such-process and had no effect. Relaunch raced the
desktop launcher: PID 59539 stopped before event-loop boot, and orphan cleanup
reaped waiter 58765. Launcher-owned PID 59679 became the sole live instance.
This is recovery, not a clean restart proof; the launch ownership race remains
an R01/R02 observation.

Boot verified source snapshot
`e8dacf2f79359f5414c421985a36c3a9b3180876f28e1fd16e797f0e388eae0f`,
with readiness true and no revision issues. The unchanged reading-group
question was submitted at 15:34:54, delivery
`aura-chat-61e0334a-b8f2-450a-92ad-a5e22d7fbfb4`, turn
`eb879aff0aff4c42abbdae4e53f4c9b0`. The 83-message conversation reached
the resident worker as 9,572 tokens without the previous context rejection.
Its answer is pending at this recording.

At 15:36:13 it answered "Gone Girl by Gillian Flynn." The receipt reports
one attempt, one generation, zero text mutations, and 40 retained exchanges.
This passes the identical title-recall replay after restart.

The next paraphrase, "What did you get wrong before that, and what was my
reason for correcting you?", delivery
`aura-chat-f1b39e13-ee70-4588-b887-9386a5525f84`, exposed two more false
rejections. Coverage marked the first part missing. A subsequent accurate
short-story/novel correction beginning "I called The Tell-Tale Heart a novel"
was rejected as `unfounded_tool_execution_claim`: the detector treated any
first-person "called" as invocation, regardless of its object. Repeated
generation was stopped at 15:41:46 and the one cancellation acknowledgement
was delivered. This replay fails R09; the retained title result does not
override it. Repairs and focused regressions for the shared validators are
in progress, with 50 direct tests and 32 related reliability tests passing.
# Exact paraphrased follow-up passes after validator repair

The restored browser displayed the successful answer. A subsequent latent
request, delivery `aura-chat-ca9c69dd-ddc2-4966-9d07-4817c503559f`, was
stopped at 15:59:07. The worker logged soft cancellation of job 5 with
`latent_reason_caller_cancelled`; the desktop displayed one Stop acknowledgement.
The next delivery `aura-chat-86c2700a-f0c1-445f-8c1e-9cd1b34e5dd4`
completed at approximately 16:00:38, correctly recalling Gone Girl and the
short-story correction. This validates post-latent-Stop history continuity.

The supported reboot replaced PID 59679 with PID 78712. Competing native
launcher and detached waiter attempts again appeared before the sole surviving
replacement; this remains a restart-ownership defect, not clean R02 evidence.
The surviving runtime verified workspace
`d22545ad3642e755da84469375b999d0c1a1aa6e6ab47d257bdb01091df00b6d`.
Both changed validators matched checkpoint `5e601d4f6` before launch.

At 15:55:41 PDT the unchanged desktop question was submitted: "What did you
get wrong before that, and what was my reason for correcting you?" Delivery
`aura-chat-c32f4d8b-5235-4761-bee1-0967cc3018a3` completed with the correct
short-story versus novel correction, original title and replacement title.
The receipt records 40 context exchanges, one generation, 63 generated tokens,
zero completion or repair retries, zero text mutations, and cognitive_engine
delivery. No answer hint was inserted into the question. The initiating
browser tab was subsequently closed; restored-window inspection is separate.
