# Engineering adoption: what Aura took from ten mature projects

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

Aura's runtime now carries disciplines borrowed, clean-room, from projects
that earned them the expensive way. This document says what was taken,
from where, why it applies here, and where the code lives — so that the
next person to touch one of these knows what contract they are inside.

Everything listed is **on by default and live at runtime**. The single
boot entry point is [`core/runtime/foundations.py`](../core/runtime/foundations.py),
called once from `aura_main._boot_runtime_orchestrator`. Each wave has an
activator; every activator is non-fatal and reports what came up.

Nothing here is a copy of anyone's code. These are adoptions of *ideas*,
re-derived against Aura's actual failure modes — most of which are
recorded in [KNOWN_FAILURE_MODES.md](../KNOWN_FAILURE_MODES.md) and the
crash forensics under `data/error_logs/`.

---

## Why these ideas and not others

The selection criterion throughout was: **does this address a failure Aura
has actually had, or an assumption it currently makes with nothing
enforcing it?** Several attractive ideas were left out because the answer
was no. What survived maps onto real incidents:

| Recorded incident | What was missing | Adopted from |
|---|---|---|
| 20-minute event-loop freeze on an on-loop fsync | Nothing checked whether a blocking call ran under a lock | lockdep (Linux) |
| Welfare deadlock | Lock order was a convention, not a checked invariant | lockdep (Linux) |
| Endurance run OOM-killed whole at 35GB | The host chose the victim, and it always chooses us | OOM badness (Linux) |
| ~242MB/h leak, root unresolved | RSS says the process grew; nothing said *which part* | memory-infra (Chromium) |
| Duplicate 32B spawned beside a wedged one | No cross-process exclusivity, and no way to detect a second runtime | Leases (Kubernetes) |
| `mind_tick` falsely declared dead under load | "Slow" and "wedged" were the same signal | Svc::Health (F Prime) |
| Green health over a degraded runtime | The verdict had no memory of what already broke | Taint flags (Linux) |
| Resident 32B degrades after ~15 turns | No declared notion of what may be dropped under load | 1202 response (Apollo) |
| Periodic loops slipping together under load | `sleep(i)` makes the interval a gap, not a period | Rate groups (F Prime) |

---

## Wave 1 — Linux kernel: runtime discipline

| Module | Adopted | The idea |
|---|---|---|
| [`core/runtime/taint.py`](../core/runtime/taint.py) | `/proc/sys/kernel/tainted` | Once an assumption breaks, **no later report may look clean**. One-way flags; the health verdict carries the caveat. |
| [`core/runtime/lockdep.py`](../core/runtime/lockdep.py) | `CONFIG_PROVE_LOCKING` | **You do not have to observe the deadlock.** Watch acquisition *order* across the process lifetime; report when two paths establish opposing edges, even if they never raced. |
| [`core/runtime/pressure_stall.py`](../core/runtime/pressure_stall.py) | PSI (`/proc/pressure`) | Utilization does not say whether anything is *waiting*. Measure lost work time, split `some` (latency) from `full` (throughput collapse). |
| [`core/runtime/oom_policy.py`](../core/runtime/oom_policy.py) | `oom_badness()` | Under exhaustion, dying is not the worst outcome — **dying arbitrarily is**. Score candidates in advance; log the table that justified the choice. |

Lockdep's four hazards: order inversion (observed *and* declared-rank),
sync lock held across an await, self-deadlock, and loop-blocking holds.
`assert_no_locks_held()` guards blocking operations; every `fsync` in the
runtime goes through it.

## Wave 2 — LLVM: verification and pass management

| Module | Adopted | The idea |
|---|---|---|
| [`core/verify/invariants.py`](../core/verify/invariants.py) | The `Verifier` pass, `-verify-each` | LLVM does not trust its own transforms. Structural invariants are re-checked after mutation, so **the pass that broke it names itself** instead of the failure surfacing twelve passes later. |
| [`core/verify/runtime_invariants.py`](../core/verify/runtime_invariants.py) | — | 30+ standing invariants that were true by convention and enforced nowhere. |
| [`core/pipeline/pass_manager.py`](../core/pipeline/pass_manager.py) | New PassManager, `-opt-bisect-limit` | Analyses cached with a precise invalidation contract; and **bisect**, which turns "which of ~30 phases ruined this answer" into about five runs. |
| [`core/runtime/sanitizers.py`](../core/runtime/sanitizers.py) | ASan / TSan / UBSan | Translated to this runtime's shapes: use-after-release (poisoned reuse), non-finite propagation, sequence affinity. |

A check that *raises* is itself a violation — a verifier that silently
skips reports clean, which is worse than reporting a breach.

The live kernel tick loop consults the pass instrumentation before each
phase and records it after, in a `finally`, so a timed-out phase is still
timed. That gives the running mind bisect and per-phase timing through a
seam rather than a rewrite of a load-bearing loop.

## Wave 3 — Kubernetes: orchestration

| Module | Adopted | The idea |
|---|---|---|
| [`core/runtime/reconcile.py`](../core/runtime/reconcile.py) | controller-runtime | **Level-triggered, not edge-triggered.** A reconciler never trusts the event; it reads current state and steps toward desired. Missing an event costs latency, never correctness. |
| [`core/runtime/admission.py`](../core/runtime/admission.py) | Admission webhooks | Mutation before validation, always. Validators may not mutate. Failure policy is declared per hook — silence must not mean consent. |
| [`core/runtime/quota.py`](../core/runtime/quota.py) | ResourceQuota, LimitRange | Requests vs limits, which is what produces QoS classes. Enforced at *admission*, because a budget checked at the point of use is checked where somebody remembered. |
| [`core/runtime/eviction.py`](../core/runtime/eviction.py) | kubelet eviction, PDB | Soft thresholds with grace periods and hard thresholds without; reclaim before eviction; disruption budgets refuse taking the last member. |
| [`core/runtime/lease.py`](../core/runtime/lease.py) | `coordination.k8s.io/Lease` | **The old leader gives up before the new one can take over.** The gap between renew deadline and lease duration is the safety margin. |

QoS class maps onto `oom_score_adj`, so eviction and the OOM killer shed
in the same order instead of contradicting each other.

`is_leader()` fails closed for mutative work; `should_act_as_singleton()`
fails open for protective work. Choosing the wrong one disables a safety
mechanism with the machinery meant to coordinate it.

## Wave 4 — ROS 2: middleware

| Module | Adopted | The idea |
|---|---|---|
| [`core/runtime/lifecycle.py`](../core/runtime/lifecycle.py) | Managed nodes | "Constructed" and "ready" are different states. Explicit transitions mean a component that failed to configure is *known* to be unconfigured. |
| [`core/bus/qos.py`](../core/bus/qos.py) | DDS QoS profiles | Reliability, durability (transient-local answers the late-joiner problem), deadline, liveliness, lifespan. Incompatible pub/sub QoS is reported at connect, not discovered as silence. |
| [`core/runtime/parameters.py`](../core/runtime/parameters.py) | Declared parameters | Every knob has a descriptor, a range, and a set-callback that can *reject*. Read-only after startup is enforceable. |
| [`core/observability/bus_recorder.py`](../core/observability/bus_recorder.py) | rosbag | Record and replay the bus. The single most useful debugging tool in robotics, and Aura had no equivalent. |
| [`core/health/diagnostics_aggregator.py`](../core/health/diagnostics_aggregator.py) | diagnostic_updater | Hierarchical rollup with declared analyzers, so a hundred signals become a handful of statuses without losing the detail. |

## Wave 5 — Chromium: observability and layering

| Module | Adopted | The idea |
|---|---|---|
| [`core/observability/histograms.py`](../core/observability/histograms.py) | UMA + `histograms.xml` | A mean says nothing about the tail. Bucketed histograms are O(1) memory and keep the shape. **A histogram without an owner is refused**; one past its expiry is reported. |
| [`core/observability/trace_events.py`](../core/observability/trace_events.py) | Trace Event format | Logs are bad at "what happened at the same time as what". Slices, async slices, flow events, counters — loads in any Perfetto UI. |
| [`core/runtime/memory_infra.py`](../core/runtime/memory_infra.py) | memory-infra | Components declare what they hold; the diff between two dumps names the culprit. |
| [`core/runtime/field_trials.py`](../core/runtime/field_trials.py) | Finch | Both arms concurrently over the same work, deterministic sticky assignment, declared hypothesis and metrics, and an **expiry** — a permanent experiment is a config flag keeping a dead arm alive. |
| [`core/security/rule_of_two.py`](../core/security/rule_of_two.py) | The Rule of Two | Untrusted input + ability to act + no sandbox is forbidden. Raises at *declaration*, with the three remedies in the message. |
| [`tools/check_layering.py`](../tools/check_layering.py) | `checkdeps` / DEPS | Architecture documents describe intended layering; only code enforces actual layering. `make layering` is the gate; the baseline only shrinks. |

## Wave 6 — Flight software: F Prime, Apollo, OpenMCT

| Module | Adopted | The idea |
|---|---|---|
| [`core/fsw/telemetry_dictionary.py`](../core/fsw/telemetry_dictionary.py) | F Prime channels + EVRs, OpenMCT limits | Every value has an id, a unit, and declared limits; every occurrence is an event with a severity that means what an operator should **do**. A limit crossing is a *transition*, not a repeated alarm. A silent channel reads STALE, not nominal. |
| [`core/fsw/restart_protection.py`](../core/fsw/restart_protection.py) | The Apollo 1202 | Notice before anything is missed, announce with a specific code, shed low-priority work, resume the rest **from declared restart points**, keep guiding. All five decided in advance. |
| [`core/fsw/rate_groups.py`](../core/fsw/rate_groups.py) | F Prime rate groups | The period is a period, not a gap after the work. Cycle slips are measured, and the member that ate the budget is named. |
| [`core/fsw/assertions.py`](../core/fsw/assertions.py) | `FW_ASSERT` | Always runs (`assert` is removed under `-O`), records site and argument values persistently, and triggers a *declared* response. |
| [`core/fsw/command_dispatch.py`](../core/fsw/command_dispatch.py) | Command dictionary + sequencer | A plan whose step 7 has a bad argument means steps 1–6 already happened. Validate the whole sequence first. |
| [`core/fsw/health_checker.py`](../core/fsw/health_checker.py) | `Svc::Health` | Active pings establish unresponsiveness as a fact rather than inferring it from silence — and separate **slow** (contended) from **unresponsive** (wedged). |

The telemetry report is OpenMCT-shaped (domain objects, composition,
limit metadata), so a client can render it without knowing about Aura.

## Wave 7 — Hyperon and OpenWorm: cognition

| Module | Adopted | The idea |
|---|---|---|
| [`core/knowledge/metta.py`](../core/knowledge/metta.py) | MeTTa evaluation | **Rewriting is knowledge too.** Rules are data — added at runtime, attributed, truth-valued, retractable. Evaluation is non-deterministic, because "what follows" usually has more than one answer. |
| [`core/organism/model_validation.py`](../core/organism/model_validation.py) | OpenWorm / sciunit | **Every claim carries a test; every test is scored against a recorded observation.** A claim without a test cannot be registered. Documents drift; suites do not. |

Rules live in their own space rather than among the facts — the AtomSpace
refuses pattern atoms and is right to. Hyperon's multiple-Spaces model is
exactly this separation.

---

## Operating these

```bash
make layering            # DEPS include-rule gate (ratchet baseline only shrinks)
make layering-baseline   # regenerate the baseline after fixing violations
```

Runtime surfaces, all under the health contract's `integrity` block:

- `taint_report()` / `taint_narrative()` — what broke, when, how often
- `lockdep_report()` — the acquisition-order graph and any splats
- `psi_report()` — per-resource `some`/`full` pressure over 10s/60s/300s
- `oom_report()` — the shed order and the scoring table
- `verifier_report()` / `verify(*scopes)` — the invariant set and last result
- `pass_manager_report()` — per-phase timing and the bisect limit
- `telemetry_report()` — channel violations and the recent event log
- `restart_report()` — core-set utilization, essential set, recent alarms
- `validation_report()` — claims, tests, and which claims are unsupported

Debugging entry points worth knowing:

- `AURA_PASS_BISECT_LIMIT=N` — run only the first N cognitive passes.
  Binary-search N to find which phase ruined an answer.
- `AURA_PASS_TRACE=1` — announce every pass as it runs.
- `bisect_pipeline(run, is_good, max_ordinal=N)` — automates that search.
- `get_bus_recorder().dump()` — write the event-bus ring for replay.
- `get_tracer().write()` — write a Perfetto-loadable trace.
- `get_memory_infra().diff(a, b).narrative()` — one sentence naming what grew.

## Adding to this

Each discipline is a registry with a declaration site, not a framework:

- A new **invariant** is a decorated function next to the thing it
  protects (`@invariant(name, scope=..., owner=...)`).
- A new **telemetry channel** is one `channel(id, name, ..., red_high=...)`
  call; ids are the contract and cannot be reused.
- A new **evictable organ** declares `requests`/`limits` and gets a QoS
  class, an eviction rank, and an `oom_score_adj` from that one act.
- A new **claim about Aura** must be registered with the test that
  validates it, and that test with an observation that has a source.

The pattern throughout: **declare it where it lives, enforce it centrally,
and make the absence of a declaration visible.**
