# Autonomous repair and three-attempt replay, 5 October 2026

The exact request that failed at 20:18 PDT completed on replay from Aura's
normal chat at 20:49:49 PDT. Aura repaired all five faults and played three
finished attempts: won 5–3, lost 1–5, won 5–2. Codex supplied zero repair
edits and zero game inputs. Codex submitted the ordinary request, observed
progress, checked the saved artifact after completion, and reset the fixture.

> Fix the broken game at /Users/bryan/aura-demos/pong/pong.html, then play it
> for three attempts to show the repair works.

The request is an example of ordinary use. Production changes contain no
Pong-specific policy, private game-state access, prompt-template edits or
weight changes. Existing repair code reads the program and checks candidate
edits; the general play controller learns what its controls affect from
observations. Prior game control memories were archived for fresh learning.
Prior conversation history remained in Aura's ordinary chat.

## Failure and mechanism

The first three-attempt run completed zero steps: a pre-cognition desktop
shortcut replaced the registered artifact operation with generic OS
automation, which correctly refused an absent acceptance contract. Repair
ranked highest in the live catalogue, but the shortcut bypassed selection.

`6ec1ff44a` makes a strongest declared artifact effect defer the shortcut to
capability selection. It neither picks a hardcoded tool nor relaxes the
dispatch confidence threshold. The live replay selected repair by semantic
meaning and invoked it through governed dispatch. The classifier uses only
an initialized catalogue, so classifying a request cannot initialize services.
The measured catalogue check took a median 5.3 ms across 30 offline calls.

`4b0234afa` adds precise visible geometry and text from supported canvas
paint operations, captured with the pixels. It observes renderer output,
without reading application variables or writing game inputs. Unsupported
paint falls back to pictures. Control attribution starts when browser input
is delivered and uses the key held at observation time. The browser uses the
Mac Metal renderer; a live GPU process sample contains Metal driver execution.
These mechanisms are shared with other environments.

## Recorded outcome

The delivery `aura-chat-00860dea-45ad-44de-9fd3-1014b53f026c`, turn
`db9df2dde5754b339ab749677f200287`, ended `completed` with HTTP 200. Durable
skill audit 14008 has `ok=true`, no remaining repair faults, five retained
changes, `played.completed=true` and `requested_attempts=3`.

| Attempt | Terminal result | Score | Observation samples per second |
| --- | --- | --- | --- |
| 1 | Won | 5–3 | 48.4 |
| 2 | Lost | 1–5 | 42.3 |
| 3 | Won | 5–2 | 41.8 |

All 19,271 gameplay observation samples used the generic canvas drawing
observer. These rates count observations, not distinct rendered frames.
The complete request took 912.536 seconds. Repair took 380.5 seconds.
Cold semantic catalogue embeddings contributed to the first-dispatch delay.
No controlled end-to-end latency comparison was performed.

The independent grader passed controls, opponent movement, collision, top
wall bounce, score credit, starting, serving again, ending at five, paddle
bounds and absence of errors. The repaired file has SHA-256
`af4c7949d8154013a342279ceab505067d5c03bd9af630570ca49fcc2caf89ea`.

The earlier final reply mentioned only the first win. `ab9a5b054` fixes that
reporting omission and lists all attempt outcomes. Its 32 focused checks and
smoke passed; the reporting module was reloaded through the supported live
API in 3.6 ms. No additional game run was performed for this reporting edit.

## Recording state and repetition

After the completed turn, Codex archived the repaired file, original backup,
receipts, screenshots and new game control memories. The demo file was reset
and independently graded: all five original faults are present, the five
preserved checks pass and there are no browser errors. Its SHA-256 is
`fecbf4563569972db4a4949c75ba04849a7c4c7ea86f19ea90be465c5c509a5d`.

A reusable copy remains at
`/Users/bryan/.aura/control-proof-2026-10-05/reusable-broken-pong.html` and a
spare at `/Users/bryan/aura-demos/pong-spare/pong.html`. To repeat after a
recording, wait for the task to finish, then run `tools/reset_pong_demo.py`.
The currently running Aura process is PID 9353, started at revision
`6ec1ff44a`, with the reporting module updated to `ab9a5b054`. One replacement
process owns port 8000; the preceding runtime saved state and exited cleanly.

## Validation and boundaries

The structured observation/control work passed 110 focused checks, 70 owner
checks, and smoke (165 passed, one skipped). Routing passed 95 initial focused
checks and 11 final regression checks. Reporting passed 32 focused checks
and smoke (165 passed, one skipped). Compilation, lint and layering passed.
Governance retains the same 80 pre-existing offending buckets; this work
introduced none. The full suite, aggregate quality gate and soak were not run.

Six synthetic worlds ran in both drawing and pixel modes. They measure
adapter/control coverage and fallback, not mastery or universal competence.
The transfer timing is not a controlled performance comparison. The live
receipt does not claim full cognitive certification; task effects are proven
separately by the durable skill audit and independent grader.

An independent review found a remaining general canvas-text limit: later
opaque paint can cover a previously recorded text item. The observer does
not yet remove every such hidden item. This game's drawing order does not use
that pattern. Visible-text correctness under overlapping repaint remains
open before broadening the claim to arbitrary canvas applications.

Compact receipts and a checksummed evidence manifest are in
`artifacts/closeout/general-control-2026-10-05/`. Full local evidence is in
`/Users/bryan/.aura/control-proof-2026-10-05/`; full private runtime history is
kept locally and is not included in the repository evidence bundle.
