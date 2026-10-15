# The frames the actor used — 15 October 2026

The prior CPU case-18 attempts did not identify a body. Screenshots taken
twenty seconds apart did not retain the frames from their short fast-play
intervals. That left the failure impossible to replay from those artifacts.

An opt-in recorder now wraps the actual `play_as_it_happens` look and hand
ports. It copies each returned uint8 RGB/RGBA frame without resizing or
changing channels. A bounded asynchronous writer publishes lossless PNGs,
raw-byte hashes, capture receipts and an ordered event journal. Input call
requests, completion and failure remain separate events. Completion does
not prove that a game accepted an input or that anything moved.

`--record-interactions --trace DIRECTORY` enables recording in the archived
game runner. Each fast-play interval has a stream-start marker. A running
manifest precedes play. Dropped events, storage errors and interruption keep
the final manifest incomplete. The queue and raw-byte budgets bound memory
and disk use; overflow cannot delay a control to wait for storage.

This recording preserves pixels and port calls. It omits drawing commands,
sounds and internal decisions, and records whether a frame had a drawing
scene. It cannot establish native-app use, a body identity, a win or demo
readiness. Existing causal thresholds are unchanged.

Validation before the first recorded original-game probe: 41 focused tests
passed, including the recorder, campaign recovery and CPU model-worker
refusal. The recorder's seven tests also passed after adding stream markers.
Smoke: 165 passed, one skipped. Ruff, layering and dependency checks passed.
The writer publishes frames before their journal entries; the test reads both
while the recording context is still running, then mutates the source array
and verifies that the recorded pixels and clock receipt remain intact.

G09 remains owned by its existing process. This recorder adds no model,
browser or input source. The next actual probe will use the established
`--cpu-only` envelope and leave case 9 and the ten holdouts excluded.
