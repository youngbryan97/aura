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
