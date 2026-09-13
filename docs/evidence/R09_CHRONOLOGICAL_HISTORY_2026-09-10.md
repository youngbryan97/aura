# R09 chronological history and display capacity

## Live failure

At 19:33 PDT, the desktop request asked what Aura had got wrong in the
reading-group discussion and why Bryan corrected it. Delivery
`aura-chat-5018b04b-577c-4b2e-a3cc-a6ee95ac3782` completed at 19:35 with an
incorrect denial that the discussion was available. The UI displayed it.
The receipt counted only two prior exchanges reaching cognition.

The durable loader scanned three nonempty sessions, including the current
pending session. Two intervening topics pushed the reading-group exchange
outside that scan. This was a history-selection failure, not evidence that
the resident model could not understand the supplied conversation.

## Repair and checks

The persistence service now reads authorized turns in chronological order
across sessions. Authorization precedes the row limit. The current-session
reader and cross-session reader share revision decoding and stable ordering.
The route retains its existing completed-exchange pairing and attestation.
Paired-device UI history remains restricted to its own session.

The desktop restores 100 exchanges and retains 200 message bubbles. Previously
it restored only 12 exchanges and pruned at 40 bubbles. The model's 40-exchange
admission window and token allocator are unchanged. Stored history is not
deleted by display pruning.

64 focused tests passed, including 105-session restoration, pending turns,
interleaved sessions, principal and surface isolation, revisions, and actual
JavaScript hydration/pruning. After the chronological index addition, the
64 focused tests passed again in 9.18 seconds. Smoke: 164 passed, one skipped
in 57.01 seconds. Lint, compile, governance-lint and layering passed.

A read-only probe of the live database returned 240 turns across 85 sessions
in 29.99 ms and 600 turns across 208 sessions in 16.34 ms. Both windows included
the reading-group discussion. These are local retrieval measurements, not
generation latency or live semantic proof.
The complete route reader then recovered 40 attested pairs in 13.82 ms,
including four reading-group exchanges, from the same read-only database.

## Remaining live findings

The failed reply also received an unrelated code-execution capability sentence.
The personal-fact reader treated a partial belief lookup as proof of global
ignorance. Those mechanisms need separate correction; extending history does
not establish that they are fixed.

The actual neural feed at 19:43 showed reimplementation admission deferral
reported as a code-generation failure, blocking writes from blinded-workspace
creation, and verifier errors. R06/R08 retain those obligations. No R item is
closed by this checkpoint alone.

## Live chronological-window replay, 20:01-20:07 PDT

The controlled restart produced exactly one replacement runtime, PID 89247.
Boot reported matching expected and actual workspace identity
`bbe857ab566ddb175021f4a5ad3c560e18bc0398bef4e181bf57cf6ab8dc13d9`,
with source_current/source_verified true and no identity issues. This was a
verified dirty workspace snapshot, not a clean-commit deployment; unrelated
work was preserved. The resident model remained the 27B persona/CRSM model.

The live history endpoint returned 100 exchanges across 63 sessions. The
actual desktop DOM contained 200 message bubbles. The repeated reading-group
question received 40 context exchanges, versus two before this repair.

Delivery `aura-chat-0b42c011-4e00-4f04-b61b-e2bb8f57c73d`, turn
`fe913752b0ff4d749f3929a6e0d7d74d`, failed after 344.52 seconds. It used two
foreground generations and one completion retry, then served the degraded
fallback. HTTP 200 did not indicate semantic success. The first 201-character
draft was marked unanswered_question_part; the repair requested the remaining
question. The worker later reported fabricated_shared_history and
internal_task_prompt_leak on its candidate. This is failed recall evidence.

The retry's system context included the personal-fact reader's global
ignorance instruction. That reader samples only 40 belief entries and never
reads the transcript or episodic store. Its no-match path now abstains instead
of asserting that all knowledge sources lack the answer. Positive matches
remain available. No new prompting or phrase-specific exception was added.
This follow-up fix has 15 passing focused tests; smoke 164 passed, one skipped
in 80.56 seconds; lint, compile, governance-lint and layering passed. It is
not yet deployed or live-validated.

The neural feed also reported conversation_persistence degradation events
during startup and the live turn. History display eventually recovered, but
that does not close the separate receipt-I/O latency investigation.
