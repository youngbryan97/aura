# Black-box flight recorder — crash-survivable last moments (roadmap A5)

<!-- reader-navigation:start -->
> Logs and timing records show what Aura did, which parts ran, and where a check failed.
>
> You can use those records to investigate delays and errors that look the same in chat. For a late spoken reply, check how long it took to hear the request, prepare the answer, make audio, and finish playback.
>
> [Plain-language guide](PLAIN_ENGLISH_GUIDE.md#observability) · [Terms and examples](VARIABLES_EXPLAINED.md#observability) · [Evidence and related work](DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](DOCUMENTATION_INDEX.md) · [Glossary](GLOSSARY.md) · [Worked formulas](FORMULAS.md)
<!-- reader-navigation:end -->

If Aura crashes, the flight recorder gives you a short trail of what the
process was doing before it stopped. It writes a compact record for each
cognitive cycle into a fixed-size memory-mapped ring at
`data/error_logs/flight/flight_ring.bin`.

Each record contains wall and monotonic time, the tick ID, stage, mode,
memory use (RSS), tick duration, consecutive loop failures, typed health
conditions (K6), and the degradation count.

## Why it survives

The append path writes fixed-size 512-byte frames to a shared memory-mapped
file. This avoids a per-frame fsync on the event loop. The retained pages
can survive an abrupt process exit;
`tests/test_flight_recorder.py::test_frames_survive_sigkill` checks one real
SIGKILL recovery case. That test does not establish immunity to every OOM
scenario, filesystem failure, corruption, host crash or power loss. A
memory-mapped write is not itself a durable-storage guarantee.

## The death report

Graceful shutdown stamps a clean marker in the ring header
(`core/coordinators/lifecycle_coordinator.py`). On boot
(`core/orchestrator/boot.py`), the previous ring is inspected:

- **marker present** → a chosen exit; no report, ring archived to
  `flight_ring.prev`.
- **marker absent** → a hard death. The last recorded moments are extracted
  into a governed artifact `data/error_logs/flight/death_<ts>.json`
  (`aura.flight_recorder.death.v1`): time of death, uptime, final tick/stage/
  mode, final RSS and its last-minute trend, max consecutive tick failures,
  unhealthy conditions at the end, the last 24 frames, and a deterministic
  narrative.

## Who consumes it (the causal part)

- **Incident narrator** (`core/observability/incident_narrator.py`) collects
  death reports as `unclean_shutdown` evidence (severity critical) — "what
  happened?" on the conversation lane gets the actual final moments with a
  receipt, not confabulation.
- **Continuity waking sequence** (`core/continuity.py`): the continuity
  record is written *before* a death, so its shutdown reason can be stale
  optimism. When the black box recovered a death report, its note supersedes
  the record in `get_waking_context()` — Aura wakes knowing how she actually
  died.
- **FMEA**: FM-FORENSICS-001 tracks the failure mode this closes.

## Honesty guarantees

- Per-slot CRCs detect covered corruption/torn-write cases; a CRC is not a
  cryptographic integrity proof or a guarantee against every collision.
- An unreadable previous ring is reported as unreadable, not invented.
- A clean shutdown produces no death note at all.
- A death before the first frame is reported as "died during boot".
- A second runtime cannot rotate or map a live ring (`flight_ring.lock`,
  taken before the previous ring is even read — the duplicate-runtime
  cascade cannot destroy the record).

## Knobs

- `AURA_FLIGHT_RECORDER` (bool, default on) — kill switch.
- `AURA_FLIGHT_RECORDER_SLOTS` (int, default 4096) — ring capacity;
  4096 × 512 B ≈ 2 MB ≈ 68 min of history at 1 Hz.

Both are declared typed flags (C1), enumerable via `flag_report()`.
