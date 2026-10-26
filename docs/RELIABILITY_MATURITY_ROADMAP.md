# Reliability & Maturity Roadmap — Chrome / Kubernetes / Aerospace

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

The goal: Chrome/Kubernetes/aerospace-grade reliability *given* Aura's
complexity. This is a multi-session program. It is organized by the three
paradigms, each mapped to concrete Aura work, grounded in what already exists so
we deepen rather than duplicate.

Guiding principle from the field: reliability at this level is not "add more
try/except." It is **declarative desired-state + control loops that converge**,
**explicit resource contracts**, **bounded execution with watchdogs**, and
**systematic failure-mode enumeration** — with everything observable.

---

## Where Aura already is (do not rebuild — extend)

- Supervision: `core/supervisor/tree.py`, `core/resilience/{supervisor,stall_watchdog,memory_watchdog,sovereign_watchdog,cognitive_governor,resource_governor}.py`, `core/runtime/organ_supervisor.py`.
- Health/liveness: `core/runtime/{health_contract,liveness,boot_probes,flagship_readiness,concurrency_health}.py`, `core/health/*`, generated `docs/RUNTIME_CONTRACT.md`.
- Recovery: `core/runtime/self_repair_ladder.py`, the local fallback/degradation contracts (legacy cloud descriptors do not enable a remote provider), the watchdog dead-man clock.
- Backpressure/circuits: `core/runtime/backpressure.py`, `core/resilience/circuit_breaker*.py`, `core/utils/token_budget.py`.
- Forensics: `make triage` / `tools/crash_triage.py` (fingerprinted incident classes), the incident narrator, `data/error_logs/{crash,stalls,memory}/`, and the A5 black-box flight recorder (`core/runtime/flight_recorder.py`, `docs/FLIGHT_RECORDER.md`).
- SLOs: `docs/SLO.md`, `core/runtime/telemetry_sli.py`.
- The async-write-lane ratchet (`tests/test_async_write_lane_ratchet.py`) — the model for turning a whole class of bugs into a static gate.

The dominant historical reliability ceiling (from the improvement-pass history):
**model-serving memory over-commitment on 64 GB → stall → force-kill → cold
reload → doom loop**, plus **boot readiness conflated with liveness** ("booting
forever" over a live mind), **env-flag sprawl**, and **fail-closed escalation
storms**. The roadmap targets these roots.

---

## Source reconciliation, 25 October 2026

Several mechanisms proposed in the original sections below now exist. Their
source integration is distinct from complete live acceptance:

| Earlier proposal | Current source owner | What remains evidence-dependent |
|---|---|---|
| Desired/observed lane reconciliation (K1) | `core/runtime/lane_reconciler.py`, `control_plane.py` | Recovery and singularity on the installed revision |
| Startup/readiness/liveness distinctions (K2) | `core/runtime/boot_probes.py`, `health_contract.py` | Correct classification under real startup and failure conditions |
| Lane admission and budgets (K3) | `core/brain/lane_admission.py`, `core/runtime/model_lane_control.py` | Safe behavior for the measured concurrent workload |
| Typed conditions and finalizers (K6) | `core/runtime/service_conditions.py` | Consumer behavior and graceful completion under faults |
| Typed runtime settings (C1) | `core/runtime/runtime_settings.py`, `settings_control_plane.py` | Correct propagation for the setting and process being tested |
| Crash capture (A5) | [Flight recorder](FLIGHT_RECORDER.md) | Recoverable records under each tested failure, not universal durability |

The proposals below retain their earlier design context. “Build” and “today”
inside them are not assertions that the mechanism is still absent. Use the
[master obligations](AURA_1_0_MASTER_TODO.md) and revision-bound acceptance
receipts for closure; source presence alone closes no live obligation.

## Earlier adoption proposals

## Kubernetes — orchestration & control-plane maturity

**K1. Declarative reconciler for the model-serving lane (highest leverage).**
Recovery today is imperative and ad-hoc. Build a control loop: *desired state* =
{exactly one warm primary cortex, admitted adapters, within memory budget};
*observed state* = live; the reconciler converges with backoff. Kills the
doom-loop and the duplicate-runtime cascade at the root (no second 32B ever spawns
beside a wedged one). Build on `organ_supervisor` + `resource_governor`.

**K2. Distinct liveness / readiness / startup probes.** Aura conflates them —
the "booting forever while serving" bug is exactly a liveness-vs-readiness
confusion. Formalize three probe *types* in `health_contract` with independent
semantics: startup (long warmup deadline, gates the other two), liveness (cheap,
fast, restart-on-fail), readiness (gates traffic, may flap without a restart).

**K3. Resource requests/limits + admission control + QoS + eviction.** THE
over-commitment root. Generalize the fuse-admission fix into a real model:
every model lane declares a memory *request* and *limit*; a scheduler admits only
within the host budget; priority classes (primary cortex = guaranteed;
brainstem/reflex = burstable; background trainers/compounding = best-effort,
evicted first) drive graceful eviction instead of OOMKill.

**K4. CrashLoopBackoff.** Respawn storms thrash (kill → cold reload → stall →
kill). Add exponential backoff + a per-lane respawn circuit breaker so a
persistently-failing lane backs off and surfaces, instead of burning the host.

**K5. Graceful termination + disruption budget.** Formalize drain → soft-cancel →
grace → force-kill, and a disruption budget that *never* kills the last warm
lane. Extends the gate-holder soft-cancel work.

**K6. Typed conditions with reasons.** Expose `Ready/Progressing/Degraded`
conditions (reason + message + lastTransition) from every managed component,
consumed by the narrator and the reconciler.

## Chrome — product-grade robustness

**C1. Feature-flag / field-trial system (Finch).** Replace scattered
`os.environ.get("AURA_*")` reads with a typed config layer on `runtime_settings`:
declared flags, defaults, per-flag kill switches, staged rollout. Makes risky
behavior changes safe to ship and instantly revertible.

**C2. Crash pipeline: capture → fingerprint → dedup → trend (Crashpad).** Finish
what `crash_triage` started: automatic capture, dedup by fingerprint, trend over
time, and the narrator consuming triage *classes* (open remainder item).

**C3. Fault isolation (site isolation).** Audit which subsystem failures are
contained vs fatal; formalize fault-containment regions so one organ's crash
degrades a capability, never the organism.

**C4. Startup performance budgets + regression gating.** Extend the cold-open
probe (`readiness_coherence`) with explicit budgets that gate regressions.

## Aerospace — high-assurance & safety-critical

**A1. Bounded-execution gate (no unbounded await).** 12 recorded loop-wedge
crashes came from unbounded awaits / on-loop fsync. Turn "every await is bounded +
watchdog-covered" into a static ratchet like the async-write-lane test.

**A2. FMEA / failure-mode registry.** The registry now exists in
`core/runtime/fmea.py` and renders [FMEA.md](FMEA.md). Its coverage and mitigation
gaps remain separate questions; the earlier absence is no longer current. For each
subsystem: failure modes → detection → mitigation → blast radius. This *drives*
the reconciler (K1), probes (K2), and isolation (C3), rather than reacting to
soak findings one at a time.

**A3. Formal degradation ladder (N-version).** The local fallback contracts are
declared in `core/brain/degradation_ladder.py` and checked by
`tests/test_degradation_ladder.py`. Their presence is distinct from measured
end-to-end recovery under a current failure and resource budget.

**A4. Envelope protection.** Runtime guards that refuse to enter unsafe states:
never admit a model that would exceed the memory envelope (→ K3); never let SLO
error-budget exhaustion cascade into a fail-closed storm (cap escalation rate).

**A5. Black-box flight recorder.** LANDED 2026-07-10 — `core/runtime/flight_recorder.py`:
an always-on mmap ring of per-tick mind-moments + K6 conditions that survives
SIGKILL/OOM-kill; unclean shutdowns become governed death reports consumed by
the incident narrator and the continuity waking sequence (`docs/FLIGHT_RECORDER.md`,
FM-FORENSICS-001).

---

## Cross-cutting / foundational (open from prior sessions)

- **Memory leak** ~242 MB/h, linear (H1 real-leak vs H2 proof-defers-reclamation
  unresolved) — needs an app-down soak or a live RSS trend. Blocks A4/K3 tuning.
- **Longevity:** a green 24–72 h soak. Prior soaks die on *session teardown*, not
  runtime fault — needs a truly-detached harness (`caffeinate`, nohup+disown).
- **Clean-machine install verification.**
- **`core/` consolidation** (145 subsystems / 922 edges) — weeks-scale; reduces
  the surface all of the above must cover.
- **SLO error-budget review:** `error_events_per_hour` exhausts noisily under any
  storm (ties to A4).

---

## Sequenced first moves (highest leverage first)

1. **K3 memory-admission model** + **K2 three-probe split** — together they kill
   the over-commitment doom-loop and the booting-forever lie, the two dominant
   ceilings.
2. **K1 reconciler** built on the admission model, with **K4 backoff**.
3. **A2 FMEA registry** to make the rest systematic rather than soak-driven.
4. **C1 feature-flag system** so every subsequent change ships safely.
5. **A1 bounded-execution ratchet** to freeze the wedge class permanently.
6. Everything else, informed by the FMEA.

Each item is independently shippable, testable, and checkpointable. Land them the
way the Ghost substrate landed: one focused, tested, pushed increment at a time.
