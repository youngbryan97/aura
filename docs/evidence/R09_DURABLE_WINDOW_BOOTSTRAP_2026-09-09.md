# R09 durable window bootstrap

## Defect and repair

A new desktop process rebuilt the UI transcript from its empty in-memory log,
although completed exchanges remained in ConversationPersistence and the chat
context reader could recover them. Bootstrap now uses that same scoped durable
reader, then reconciles with a fresh live snapshot by exchange identity. A turn
that finishes during the disk read wins over the older durable snapshot. Equal
text in distinct turns is not deduplicated. Full live windows avoid disk reads.

Paired-device reads remain restricted to their authenticated principal, surface,
and device session. Only public exchange fields leave the durable store for the
UI; runtime attestations and action metadata do not. Cancellation restores both
request identity context variables.

## Verification

- Seven new tests, including real SQLite restoration and the real bootstrap
  route, pass with the persistence suite: **19 passed in 4.53s**.
- Smoke: **164 passed, one skipped in 78.04s**.
- Lint, compile, governance-lint and layering pass after removing the obsolete
  paired-session import from the route.
- Resident restart and fresh-window restoration are pending at this checkpoint.

## Related live observations before deployment

Resident PID 20867 at revision e56825a26 survived the database recovery question
and delivered one complete, correct answer on 2026-09-09 at 18:33. The worker
decoded 75 tokens with no repair. Request time was 19.60s; stabilization was
108.23ms, versus the earlier catalog-rescan case's 51.01s. These are individual
observations, not a controlled general latency estimate.

The next question, "But how can it redo the changes safely if some of them
already reached the data files?", was intercepted by verified_answer_provenance
and answered with Aura's knowledge provenance instead of database semantics.
That is a separate routing failure, not successful context retention.

At 18:47 the visible neural stream also reported "A scan stopped early - unknown
error" beside a foreground deferral. Cause and rendering remain to be traced;
the stream's summary alone does not establish a failed scan.

R09 remains open. This checkpoint does not close cancellation, all follow-up
semantics, or the entire release.
