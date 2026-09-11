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
