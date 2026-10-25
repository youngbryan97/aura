# Black-box flight recorder — crash-survivable last moments (roadmap A5)

<!-- reader-navigation:start -->
> **In plain English:** Observability records what the system did, how long each stage took, and where an assumption failed.
>
> **Why it matters:** A visible error or delay needs a traceable cause; otherwise a repair can change the wrong thing while hiding the real failure.
>
> **In everyday use:** When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#observability) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#observability) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

Aerospace keeps a record that survives the crash. Aura now does too: an
always-on, fixed-size, mmap-backed ring (`data/error_logs/flight/flight_ring.bin`)
receives one compact **mind-moment** frame per cognitive tick — wall/monotonic
time, tick id, stage, mode, RSS, tick duration, consecutive loop failures,
unhealthy typed conditions (K6), degradation count.

## Why it survives

Appends are a single 512-byte memcpy into a `MAP_SHARED` mapping. The kernel
owns the pages the instant they are written — no fsync, no syscalls, nothing
on the event loop. SIGKILL, OOM-kill, and segfaults cannot erase them; only
whole-machine loss can. `tests/test_flight_recorder.py::test_frames_survive_sigkill`
proves this on a real killed subprocess, not a simulation.

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

- Per-slot CRCs: a torn write is skipped, never misread.
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
