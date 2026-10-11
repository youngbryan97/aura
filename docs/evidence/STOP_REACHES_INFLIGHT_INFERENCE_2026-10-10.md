# Stopping in-flight inference, 10 October 2026

Base revision: `64aefec64`. Presentation qualification remains **unproved**.

An ordinary request in the installed app opened the archived game on this
revision. Stop closed the owned browser within the observed second at 17:43:37;
no further browser actions appeared during the following minute. The generation
already running for the same request continued until 17:44:30. Later attempts
from that stopped background work received empty replies and kept retrying.
This is a bounded live observation of browser interruption and a live failure
of inference interruption, not a completed game.

The existing execution token now binds the complete public router request and
local client generation, including admission waits. The inference gate binds
its in-flight work while preserving its existing typed admission refusal.
Stopping the owner cancels the executing task, including a task behind a
shielded waiter. Cancellation reaches the client's existing request-specific
worker signal, acknowledgement handling and ownership cleanup. It propagates
as cancellation, so fallback and retry do not treat Stop as endpoint failure.
The loaded worker stays warm when it acknowledges cancellation; missing
acknowledgement continues to require recovery. The wrapper preserves the
public API's signature and source metadata.

Offline validation used the shared virtual environment and isolated logs:

- Router, inference, execution, admission, pressure authority, cancellation,
  acknowledgement and resilience regressions: **223 passed, 4 deselected**,
  59.33 seconds. Hardware cases were excluded from this run.
- The new public-client regression used the actual generation endpoint and
  inner request path with simulated IPC. It observed the request-specific stop
  signal, released request lock, zero active generations and preserved worker.
- Smoke: **165 passed, 1 skipped**, 66.84 seconds.
- Lint, compile and layering passed, with 36 grandfathered layering edges.
- Governance lint still reports 101 pre-existing regressions. Changed production
  modules have **zero effect bucket changes** against the base revision.
- The preceding checkpoint's additional gameplay/browser regression run passed
  **136 tests**, 132.29 seconds. These counts overlap other runs.

Local live records are retained in
`/Users/bryan/.aura/flash-proof-2026-10-10/live-64aefec64-complete.log`
and `live-64aefec64-final-health.json`. The final health read verified the source
revision and reported operational readiness. It does not certify gameplay.
Normal application quit saved state and completed graceful shutdown with exit
code zero at 17:55:46.

Fresh ordinary app proof of the inference repair, measured construction,
completed gameplay and full-suite validation remain outstanding.
