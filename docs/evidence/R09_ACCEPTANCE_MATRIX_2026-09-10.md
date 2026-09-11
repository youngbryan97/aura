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
