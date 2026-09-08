# R07: one health authority across transports

Date: 2026-09-08. Code: 43b32d714. Live checkout: 7480fec8d.

## Failure and repair

Before restart, the actual neural feed alternated between websocket
`health=healthy` and HTTP `runtime_revision_unverified` or
`health_snapshot_expired`. The websocket recomputed a weaker verdict rather
than using the versioned health snapshot. One captured snapshot was 175.072
seconds old, beyond its 30-second maximum age.

`read_runtime_health_snapshot` now applies snapshot expiry, runtime revision
and shutdown truth once. HTTP health, readiness and heartbeat consume it;
websocket and SSE heartbeat use the same projection. A busy conversation
does not erase an independent failure. Missing probe components cannot pass
because a summary flag says they passed. Operational health grants no
scientific certification.

## Live evidence

- Supported authenticated reboot replaced PID 70837 with PID 57421. Shutdown
  saved substrate and cognitive state, and reported `clean=True`.
- Exactly one `aura_main` successor remained. Expected and actual commit were
  `7480fec8dc611a3a97a07ddc07a468ac13f06f42`; workspace hashes matched.
- Initially the successor correctly refused readiness with
  `resident_app_not_running`. Opening the installed signed app in observer
  mode restored its connection without loading a second cortex. Verification
  then returned true, source_current true, and issues empty.
- Resident model: `Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15`.
- Live chat, 15:29:25: departure 14:35, journey 50 minutes. Answer at 15:29:50:
  `15:25.` Follow-up at 15:30:04: departure ten minutes later, same journey.
  Answer at 15:30:22: `15:35.` Each appeared once, complete.
- During generation, heartbeat reported `working`, conversation_busy true,
  no blockers, verified revision and an unexpired snapshot.
- Four rounds of health, readyz and heartbeat produced twelve HTTP 200
  responses. Health/heartbeat generations agreed within each round:
  23, 24, 26, 28. None expired. Request durations were 0.002-0.021 seconds.
  The accompanying JSON preserves those observations.
- The visible neural feed reported healthy websocket heartbeat after restart;
  the terminal reported required probes and conversation PASS.

## Tests and boundaries

155 focused tests passed before deployment; smoke 164 passed, one skipped.
Lint, compile and layering passed. Transport tests exercise expiry, revision
failure, shutdown, missing probes, busy-lane failures and certification
separation without probing live services from the heartbeat.

A collector wait warning occurred during the first live turn: its 2.5-second
HTTP wait elapsed while singleflight collection continued. This is not the
30-second snapshot expiry and did not become a false healthy assertion or
cancel the turn. Collector latency remains under R06/R11; warning semantics
and other neural-feed findings remain R08. The earlier cold-model admission
failure remains R03. This receipt does not establish a soak or broad reasoning
gain, and does not close those obligations.
