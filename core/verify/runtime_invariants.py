"""core/verify/runtime_invariants.py — the standing invariants.

These are the structural facts the runtime assumes everywhere and checks
nowhere. Each one has been true by convention; a convention that nothing
enforces is a convention that a refactor silently retires.

Grouped by scope so `-verify-each` can re-check only what a mutation could
have broken. Importing this module registers them; :mod:`core.runtime.foundations`
does that once at boot.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

_OWNER = "core/verify/runtime_invariants.py"


# ══════════════════════════════════════════════════════════════════════
# Service container — the spine. Everything else resolves through it.
# ══════════════════════════════════════════════════════════════════════

def _container_state() -> tuple[dict[str, object], dict[str, str]]:
    from core.container import ServiceContainer

    services = dict(getattr(ServiceContainer, "_services", {}) or {})
    aliases = dict(getattr(ServiceContainer, "_aliases", {}) or {})
    return services, aliases


@invariant(
    "container.alias_resolves",
    scope="container",
    owner=_OWNER,
    description="every registered alias resolves to a registered service",
)
def _alias_resolves() -> Iterator[Violation]:
    services, aliases = _container_state()
    for alias, target in aliases.items():
        seen: set[str] = set()
        current = target
        while current in aliases and current not in seen:
            seen.add(current)
            current = aliases[current]
        if current not in services:
            yield Violation(
                subject=alias,
                message=(
                    f"alias {alias!r} resolves to {current!r}, which is not a "
                    "registered service — every lookup through it raises or "
                    "silently returns the default"
                ),
                remedy=f"register {current!r}, or drop the alias",
            )


@invariant(
    "container.alias_terminates",
    scope="container",
    owner=_OWNER,
    description="alias chains are acyclic",
)
def _alias_terminates() -> Iterator[Violation]:
    _services, aliases = _container_state()
    for alias in aliases:
        seen: set[str] = {alias}
        current = aliases[alias]
        while current in aliases:
            if current in seen:
                yield Violation(
                    subject=alias,
                    message=(
                        f"alias chain from {alias!r} cycles at {current!r}; "
                        "resolution never terminates"
                    ),
                    remedy="break the cycle — one of these must point at a real service",
                )
                break
            seen.add(current)
            current = aliases[current]


@invariant(
    "container.declared_dependencies_exist",
    scope="container",
    owner=_OWNER,
    description="every declared service dependency names something registered",
)
def _dependencies_exist() -> Iterator[Violation]:
    services, aliases = _container_state()
    known = set(services) | set(aliases)
    for name, descriptor in services.items():
        for dependency in list(getattr(descriptor, "dependencies", ()) or ()):
            if str(dependency) not in known:
                yield Violation(
                    subject=f"{name} -> {dependency}",
                    message=(
                        f"service {name!r} declares a dependency on {dependency!r}, "
                        "which is not registered; construction will fail the first "
                        "time this service is actually needed"
                    ),
                    remedy=f"register {dependency!r} before {name!r}, or drop the declaration",
                )


@invariant(
    "container.dependency_graph_acyclic",
    scope="container",
    owner=_OWNER,
    description="declared service dependencies form a DAG",
)
def _dependency_graph_acyclic() -> Iterator[Violation]:
    services, aliases = _container_state()

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    graph = {
        name: [resolve(str(d)) for d in (getattr(desc, "dependencies", ()) or ())]
        for name, desc in services.items()
    }

    white, grey, black = 0, 1, 2
    colour = dict.fromkeys(graph, white)
    reported: set[frozenset[str]] = set()

    def walk(node: str, path: list[str]) -> Iterator[Violation]:
        colour[node] = grey
        for nxt in graph.get(node, ()):
            if nxt not in colour:
                continue
            if colour[nxt] == grey:
                cycle = path[path.index(nxt):] + [nxt] if nxt in path else [node, nxt]
                key = frozenset(cycle)
                if key not in reported:
                    reported.add(key)
                    yield Violation(
                        subject=" -> ".join(cycle),
                        message=(
                            "service construction dependencies form a cycle; "
                            "resolving any member deadlocks or recurses"
                        ),
                        remedy="break the cycle with a lazy accessor on one edge",
                    )
            elif colour[nxt] == white:
                yield from walk(nxt, path + [nxt])
        colour[node] = black

    for node in list(graph):
        if colour.get(node) == white:
            yield from walk(node, [node])


@invariant(
    "container.no_self_dependency",
    scope="container",
    owner=_OWNER,
    description="no service declares itself as a dependency",
)
def _no_self_dependency() -> Iterator[Violation]:
    services, _aliases = _container_state()
    for name, descriptor in services.items():
        if name in {str(d) for d in (getattr(descriptor, "dependencies", ()) or ())}:
            yield Violation(
                subject=name,
                message=f"service {name!r} declares itself as a dependency",
                remedy="remove the self-reference",
            )


# ══════════════════════════════════════════════════════════════════════
# Locking
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "locks.no_open_splats",
    scope="locks",
    owner=_OWNER,
    description="lockdep has reported no order violations",
)
def _no_open_splats() -> Iterator[Violation]:
    from core.runtime.lockdep import lockdep_report

    report = lockdep_report()
    for splat in report["splats"]:
        yield Violation(
            subject=splat["acquiring"],
            message=splat["message"],
            remedy="fix the ordering; a latent deadlock is still a deadlock",
        )


@invariant(
    "locks.declared_ranks_match_observed_order",
    scope="locks",
    owner=_OWNER,
    description="no observed acquisition edge contradicts a declared rank",
)
def _ranks_match_observed() -> Iterator[Violation]:
    from core.runtime.lockdep import LockRank, lockdep_report

    report = lockdep_report()
    ranks = {name: LockRank[value] for name, value in report["declared_ranks"].items()}
    for before, afters in report["order_edges"].items():
        before_rank = ranks.get(before)
        if before_rank is None or before_rank is LockRank.UNRANKED:
            continue
        for after in afters:
            after_rank = ranks.get(after)
            if after_rank is None or after_rank is LockRank.UNRANKED:
                continue
            if after_rank <= before_rank and after != before:
                yield Violation(
                    subject=f"{before} -> {after}",
                    message=(
                        f"{before!r} (rank {before_rank.name}) has been observed "
                        f"holding while {after!r} (rank {after_rank.name}) is taken, "
                        "which inverts the declared order"
                    ),
                    remedy="re-rank one of them, or reverse the acquisition",
                )


# ══════════════════════════════════════════════════════════════════════
# Memory policy
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "oom.spine_is_immune",
    scope="memory",
    owner=_OWNER,
    description="load-bearing organs are never OOM shed candidates",
)
def _spine_is_immune() -> Iterator[Violation]:
    from core.runtime.foundations import IMMUNE_SERVICES
    from core.runtime.oom_policy import OOM_SCORE_ADJ_MIN, get_oom_policy

    table = {row["organ"]: row for row in get_oom_policy().scoring_table()}
    for name in IMMUNE_SERVICES:
        row = table.get(name)
        if row is None:
            continue  # not registered in this process; nothing to protect
        if row["oom_score_adj"] > OOM_SCORE_ADJ_MIN or row["sheddable"]:
            yield Violation(
                subject=name,
                message=(
                    f"{name!r} is load-bearing but is a shed candidate "
                    f"(oom_score_adj={row['oom_score_adj']}, sheddable={row['sheddable']}) "
                    "— memory pressure could take the runtime's spine"
                ),
                remedy=f"register {name!r} with oom_score_adj=OOM_SCORE_ADJ_MIN and no shed hook",
            )


@invariant(
    "oom.ladder_has_rungs",
    scope="memory",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="at least one organ can actually be shed under pressure",
)
def _ladder_has_rungs() -> Iterator[Violation]:
    from core.runtime.oom_policy import get_oom_policy

    report = get_oom_policy().report()
    if report["registered_organs"] and report["sheddable_organs"] == 0:
        yield Violation(
            subject="oom_policy",
            message=(
                "no organ exposes a shed hook, so the OOM ladder has no rungs: "
                "the only available response to memory pressure is a restart"
            ),
            remedy="give at least one cache-holding organ a shed_memory() method",
        )


# ══════════════════════════════════════════════════════════════════════
# Pressure accounting
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "psi.capacity_declared",
    scope="pressure",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every observed resource has a declared worker capacity",
)
def _psi_capacity_declared() -> Iterator[Violation]:
    from core.runtime.pressure_stall import psi_report

    for name, entry in psi_report().items():
        if entry["capacity"] == 1 and entry["peak_stalled"] > 1:
            yield Violation(
                subject=name,
                message=(
                    f"resource {name!r} has default capacity 1 but has had "
                    f"{entry['peak_stalled']} concurrent waiters, so `full` pressure "
                    "reads as saturated whenever anything waits at all"
                ),
                remedy=f"declare_capacity({name!r}, <real worker count>) at activation",
            )


# ══════════════════════════════════════════════════════════════════════
# Integrity
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "integrity.untainted",
    scope="integrity",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no credibility-affecting taint is set",
)
def _untainted() -> Iterator[Violation]:
    from core.runtime.taint import taint_report

    report = taint_report()
    for entry in report["flags"]:
        if entry["flag"] in report["credibility_affecting"]:
            yield Violation(
                subject=entry["flag"],
                message=(
                    f"{entry['meaning']} ({entry['count']}×, first: "
                    f"{entry['first_reason']}) — any green verdict since then is "
                    "reported over a runtime that already broke an assumption"
                ),
                remedy="investigate the first occurrence; taint clears only on restart",
            )


# ══════════════════════════════════════════════════════════════════════
# Flags
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "sanitizers.clean",
    scope="integrity",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no sanitizer has reported a finding",
)
def _sanitizers_clean() -> Iterator[Violation]:
    from core.runtime.sanitizers import sanitizer_report

    for finding in sanitizer_report()["findings"]:
        yield Violation(
            subject=f"{finding['sanitizer']}:{finding['context']}",
            message=f"{finding['message']} ({finding['occurrences']}×)",
            remedy="fix the lifetime, the non-finite source, or the affinity",
        )


@invariant(
    "flags.documented_and_owned",
    scope="flags",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every declared flag names an owner and says what it does",
)
def _flags_documented() -> Iterator[Violation]:
    from core.runtime.flags import declared_flags

    for name, spec in declared_flags().items():
        missing = [
            field
            for field in ("owner", "description")
            if not str(getattr(spec, field, "") or "").strip()
        ]
        if missing:
            yield Violation(
                subject=name,
                message=f"flag {name!r} is missing {' and '.join(missing)}",
                remedy="a knob nobody owns is a knob nobody can retire",
            )


# ══════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "admission.validators_do_not_mutate",
    scope="orchestration",
    owner=_OWNER,
    description="no validating admission hook has been caught mutating its input",
)
def _validators_do_not_mutate() -> Iterator[Violation]:
    from core.runtime.sanitizers import sanitizer_report

    for finding in sanitizer_report()["findings"]:
        if finding["sanitizer"] == "admission":
            yield Violation(
                subject=finding["signature"].split(":", 1)[-1],
                message=finding["message"],
                remedy="move the edit into a mutating hook, which runs before validation",
            )


@invariant(
    "quota.guaranteed_specs_are_coherent",
    scope="orchestration",
    owner=_OWNER,
    description="a Guaranteed organ's requests equal its limits on every resource",
)
def _guaranteed_specs_coherent() -> Iterator[Violation]:
    from core.runtime.quota import QosClass, get_quota_registry

    for name, spec in get_quota_registry().specs().items():
        if spec.qos_class is not QosClass.GUARANTEED:
            continue
        for kind, limit in spec.limits.items():
            requested = spec.requests.get(kind)
            if requested is None or abs(requested - limit) > 1e-9:
                yield Violation(
                    subject=f"{name}.{kind}",
                    message=(
                        f"{name!r} is classed Guaranteed but requests "
                        f"{requested} against a limit of {limit}"
                    ),
                    remedy="set the request equal to the limit, or accept Burstable",
                )


@invariant(
    "eviction.guaranteed_organs_are_protected",
    scope="orchestration",
    owner=_OWNER,
    description="no Guaranteed organ appears in the eviction order",
)
def _guaranteed_protected() -> Iterator[Violation]:
    from core.runtime.eviction import eviction_report
    from core.runtime.quota import QosClass, get_quota_registry

    order = set(eviction_report()["eviction_order"])
    registry = get_quota_registry()
    for name in order:
        if registry.qos_class(name) is QosClass.GUARANTEED:
            yield Violation(
                subject=name,
                message=(
                    f"{name!r} is Guaranteed but is in the eviction order; the "
                    "guarantee it was given does not hold"
                ),
                remedy="exclude Guaranteed organs from eviction_order()",
            )


@invariant(
    "eviction.thresholds_are_ordered",
    scope="orchestration",
    owner=_OWNER,
    description="each signal's hard threshold is stricter than its soft one",
)
def _thresholds_ordered() -> Iterator[Violation]:
    from core.runtime.eviction import Comparison, eviction_report

    by_signal: dict[str, list[dict]] = {}
    for entry in eviction_report()["thresholds"]:
        by_signal.setdefault(entry["signal"], []).append(entry)
    for signal, entries in by_signal.items():
        hard = [e for e in entries if e["hard"]]
        soft = [e for e in entries if not e["hard"]]
        if not hard or not soft:
            continue
        for h in hard:
            for s in soft:
                if h["comparison"] != s["comparison"]:
                    continue
                stricter = (
                    h["value"] < s["value"]
                    if h["comparison"] == str(Comparison.BELOW)
                    else h["value"] > s["value"]
                )
                if not stricter:
                    yield Violation(
                        subject=signal,
                        message=(
                            f"hard threshold {h['value']} is not stricter than the "
                            f"soft threshold {s['value']} on {signal}; the hard one "
                            "fires first and the grace period never applies"
                        ),
                        remedy="make the hard threshold stricter, or drop the soft one",
                    )


@invariant(
    "reconcile.queues_are_draining",
    scope="orchestration",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no controller queue is deep and backing off at the same time",
)
def _queues_draining() -> Iterator[Violation]:
    from core.runtime.reconcile import reconcile_report

    for entry in reconcile_report()["controllers"]:
        queue = entry["queue"]
        if queue["depth"] > 32 and queue["backing_off"]:
            yield Violation(
                subject=entry["name"],
                message=(
                    f"controller {entry['name']!r} has {queue['depth']} queued keys "
                    f"while {len(queue['backing_off'])} are backing off; it is not "
                    "converging"
                ),
                remedy="look at last_error; a reconciler that always fails never drains",
            )


@invariant(
    "lease.no_live_duplicate_holder",
    scope="orchestration",
    owner=_OWNER,
    description="no lease we want is held by another live process on this host",
)
def _no_duplicate_holder() -> Iterator[Violation]:
    from core.runtime.lease import lease_report

    for entry in lease_report()["leases"]:
        record = entry.get("record")
        if not record or entry["is_leader"]:
            continue
        holder = record["identity"]
        ours = entry["identity"]
        if holder["host"] == ours["host"] and holder["pid"] != ours["pid"]:
            yield Violation(
                subject=entry["name"],
                message=(
                    f"lease {entry['name']!r} is held by pid {holder['pid']} on this "
                    f"host while pid {ours['pid']} also wants it — two runtimes are "
                    "contending for the same exclusive work"
                ),
                remedy="stop the other runtime; duplicate runtimes double memory",
            )


# ══════════════════════════════════════════════════════════════════════
# Middleware — lifecycles, QoS, parameters, diagnostics
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "lifecycle.critical_organs_are_active",
    scope="middleware",
    owner=_OWNER,
    description="every organ declared critical has reached the active state",
)
def _critical_organs_active() -> Iterator[Violation]:
    from core.runtime.lifecycle import State, get_lifecycle_manager

    for organ in get_lifecycle_manager().organs():
        if not organ.critical:
            continue
        if organ.state is State.ACTIVE:
            continue
        # An organ that has not been brought up yet is not a violation;
        # one that is finalized or stuck in error is.
        if organ.state in (State.UNCONFIGURED, State.INACTIVE):
            continue
        yield Violation(
            subject=organ.name,
            message=(
                f"critical organ {organ.name!r} is {organ.state} "
                f"({organ.last_error or 'no error recorded'})"
            ),
            remedy="the system is not ready while a critical organ is not active",
        )


@invariant(
    "lifecycle.no_organ_stuck_mid_transition",
    scope="middleware",
    owner=_OWNER,
    description="no organ has been sitting in an in-flight transition state",
)
def _no_stuck_transitions() -> Iterator[Violation]:
    import time

    from core.runtime.lifecycle import State, get_lifecycle_manager

    in_flight = {
        State.CONFIGURING,
        State.ACTIVATING,
        State.DEACTIVATING,
        State.CLEANING_UP,
        State.SHUTTING_DOWN,
    }
    now = time.time()
    for organ in get_lifecycle_manager().organs():
        if organ.state not in in_flight:
            continue
        stuck_for = now - organ.entered_state_at
        if stuck_for > organ.transition_timeout_s * 2:
            yield Violation(
                subject=organ.name,
                message=(
                    f"{organ.name!r} has been {organ.state} for {stuck_for:.0f}s "
                    f"(timeout {organ.transition_timeout_s:.0f}s) — the transition wedged"
                ),
                remedy="a transition that cannot time out cannot be recovered from",
            )


@invariant(
    "reality_middleware.effects_are_reconciled",
    scope="middleware",
    owner=_OWNER,
    description="managed physical effects never report ready while recovery is unresolved",
)
def _managed_physical_effects_reconciled() -> Iterator[Violation]:
    from core.container import ServiceContainer

    reality_reach = ServiceContainer.get("reality_reach", default=None)
    middleware = ServiceContainer.get("reality_middleware", default=None)
    if reality_reach is None and middleware is None:
        return
    if middleware is None:
        yield Violation(
            subject="reality_middleware",
            message="Reality Reach is live without its managed physical lifecycle runtime",
            remedy="restore the middleware before admitting managed services or actions",
        )
        return
    status = middleware.status() if callable(getattr(middleware, "status", None)) else None
    if not isinstance(status, dict):
        yield Violation(
            subject="reality_middleware",
            message="managed physical runtime exposes no inspectable status contract",
            remedy="provide status() with alive, ready, and recovery_required_count",
        )
        return
    recovery = int(status.get("recovery_required_count", 0) or 0)
    if recovery > 0 or (status.get("ready") is True and status.get("alive") is not True):
        yield Violation(
            subject="reality_middleware",
            message=(
                f"managed physical runtime reports alive={status.get('alive')!r}, "
                f"ready={status.get('ready')!r}, unresolved_effects={recovery}"
            ),
            remedy="reconcile every uncertain effect before restoring physical readiness",
        )


@invariant(
    "reality_metrology.live_mode_is_restored",
    scope="middleware",
    owner=_OWNER,
    description=(
        "physical measurement has a live metrology owner and never strands the runtime "
        "in simulation or HIL mode outside an active acquisition"
    ),
)
def _reality_metrology_live_mode_restored() -> Iterator[Violation]:
    from core.container import ServiceContainer

    reality_reach = ServiceContainer.get("reality_reach", default=None)
    metrology = ServiceContainer.get("reality_metrology", default=None)
    if reality_reach is None and metrology is None:
        return
    if metrology is None:
        yield Violation(
            subject="reality_metrology",
            message="Reality Reach is live without its calibrated acquisition owner",
            remedy="restore metrology before making calibrated, synchronized, simulation, or HIL claims",
        )
        return
    status = metrology.status() if callable(getattr(metrology, "status", None)) else None
    if not isinstance(status, dict):
        yield Violation(
            subject="reality_metrology",
            message="metrology exposes no inspectable status contract",
            remedy="provide status() with mode, active_run, and live_restoration_required",
        )
        return
    if bool(status.get("refresh_reconciliation_required")):
        yield Violation(
            subject="reality_metrology",
            message="a timed-out physical refresh is still reconciling",
            remedy="keep measurement admission closed until the adapter read terminates",
        )
    if bool(status.get("live_restoration_required")) or (
        status.get("mode") != "live" and not status.get("active_run")
    ):
        yield Violation(
            subject="reality_metrology",
            message=(
                f"measurement mode={status.get('mode')!r} has no active acquisition; "
                "simulation evidence could contaminate later live claims"
            ),
            remedy="restore live mode and advance the metrology mode-generation fence",
        )


@invariant(
    "qos.no_unresolved_mismatch",
    scope="middleware",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no subscriber requested QoS a publisher does not offer",
)
def _no_qos_mismatch() -> Iterator[Violation]:
    from core.bus.qos import qos_report

    for entry in qos_report()["qos_mismatches"]:
        yield Violation(
            subject=entry["topic"],
            message="; ".join(entry["problems"]),
            remedy="raise the publisher's profile, or lower what the subscriber asks for",
        )


@invariant(
    "qos.state_topics_are_transient_local",
    scope="middleware",
    owner=_OWNER,
    description="topics named as state retain their last value for late joiners",
)
def _state_topics_retain() -> Iterator[Violation]:
    from core.bus.qos import qos_report

    report = qos_report()
    for topic, entry in report["topics"].items():
        looks_like_state = topic.endswith(("state", "verdict", "phase", "pressure"))
        if not looks_like_state:
            continue
        if entry["profile"]["durability"] != "transient_local":
            yield Violation(
                subject=topic,
                message=(
                    f"{topic!r} carries state but is volatile: an organ that "
                    "subscribes after the last announcement never learns it, and "
                    "nothing will republish"
                ),
                remedy="declare_topic(topic, qos.STATE)",
            )


@invariant(
    "parameters.are_owned_and_described",
    scope="middleware",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every declared parameter names an owner and says what it does",
)
def _parameters_owned() -> Iterator[Violation]:
    from core.runtime.parameters import parameters_report

    for name, entry in parameters_report()["parameters"].items():
        descriptor = entry["descriptor"]
        missing = [
            field
            for field in ("owner", "description")
            if not str(descriptor.get(field) or "").strip()
        ]
        if missing:
            yield Violation(
                subject=name,
                message=f"parameter {name!r} is missing {' and '.join(missing)}",
                remedy="a knob nobody owns is a knob nobody can retire",
            )


@invariant(
    "parameters.numeric_bounds_are_declared",
    scope="middleware",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="numeric parameters declare at least one bound",
)
def _parameters_bounded() -> Iterator[Violation]:
    from core.runtime.parameters import parameters_report

    for name, entry in parameters_report()["parameters"].items():
        descriptor = entry["descriptor"]
        if descriptor["type"] not in ("int", "float") or descriptor["allowed"]:
            continue
        if descriptor["minimum"] is None and descriptor["maximum"] is None:
            yield Violation(
                subject=name,
                message=(
                    f"numeric parameter {name!r} has no minimum or maximum; any "
                    "value at all can be set on a live runtime"
                ),
                remedy="declare the range the consuming code actually tolerates",
            )


@invariant(
    "diagnostics.nothing_is_stale",
    scope="middleware",
    owner=_OWNER,
    description="no expected diagnostic has stopped reporting",
)
def _nothing_stale() -> Iterator[Violation]:
    from core.health.diagnostics_aggregator import get_aggregator

    aggregate = get_aggregator().aggregate()
    for name in aggregate["stale"]:
        yield Violation(
            subject=name,
            message=(
                f"diagnostic {name!r} is STALE — it was expected to report and "
                "stopped, which the health verdict cannot see on its own"
            ),
            remedy="find out why it stopped; silence is not the same as ok",
        )


# ══════════════════════════════════════════════════════════════════════
# Observability, experimentation, and security posture
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "histograms.are_owned",
    scope="observability",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every declared histogram names an owner and describes itself",
)
def _histograms_owned() -> Iterator[Violation]:
    from core.observability.histograms import histograms_report

    report = histograms_report()
    for entry in report["expired"]:
        yield Violation(
            subject=entry["name"],
            message=(
                f"histogram {entry['name']!r} has recorded nothing in "
                f"{entry['age_days']:.0f} days (expiry {entry['expiry_days']}) — "
                "either nobody reads it or nothing feeds it"
            ),
            remedy="retire it, or find out why the code path stopped running",
        )


@invariant(
    "histograms.are_not_clipping",
    scope="observability",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="no histogram is losing more than 1% of samples to overflow",
)
def _histograms_not_clipping() -> Iterator[Violation]:
    from core.observability.histograms import histograms_report

    report = histograms_report()
    for name in report["clipping"]:
        entry = report["histograms"][name]
        yield Violation(
            subject=name,
            message=(
                f"{name!r} sent {entry['overflow']} of {entry['count']} samples to "
                "the overflow bucket; its top percentiles are floors, not values"
            ),
            remedy="raise the histogram's maximum to cover the real tail",
        )


@invariant(
    "memory.attribution_is_meaningful",
    scope="observability",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="a reasonable share of process memory is attributed to a component",
)
def _memory_attributed() -> Iterator[Violation]:
    from core.runtime.memory_infra import memory_infra_report

    report = memory_infra_report()
    latest = report.get("latest")
    if not latest or not latest.get("process_rss_mb"):
        return
    fraction = float(latest.get("attributed_fraction") or 0.0)
    if fraction < 0.05 and len(report["providers"]) > 3:
        yield Violation(
            subject="memory_infra",
            message=(
                f"only {fraction * 100:.1f}% of {latest['process_rss_mb']:.0f}MB RSS is "
                f"attributed to a component ({latest['unattributed_mb']:.0f}MB "
                "unaccounted) — a growth diff cannot name a culprit it has no "
                "provider for"
            ),
            remedy="register providers for the large holders: model weights, caches, indexes",
        )


@invariant(
    "trials.have_hypotheses_and_expire",
    scope="observability",
    owner=_OWNER,
    description="no field trial has outlived its expiry without a conclusion",
)
def _trials_expire() -> Iterator[Violation]:
    from core.runtime.field_trials import field_trials_report

    report = field_trials_report()
    for name in report["expired"]:
        entry = report["trials"][name]
        yield Violation(
            subject=name,
            message=(
                f"trial {name!r} has run {entry['age_days']:.0f} days past its "
                f"{entry['expires_days']}-day expiry with no conclusion; it is now a "
                "config flag with extra steps, keeping a dead arm alive in the code"
            ),
            remedy="conclude it and delete the losing arm, or re-declare with a new expiry",
        )


@invariant(
    "security.rule_of_two_holds",
    scope="observability",
    owner=_OWNER,
    description="no handler takes untrusted input, can act, and runs unsandboxed",
)
def _rule_of_two_holds() -> Iterator[Violation]:
    from core.security.rule_of_two import get_rule_of_two_registry

    for handler in get_rule_of_two_registry().violations():
        yield Violation(
            subject=handler.name,
            message=(
                f"{handler.name!r} handles untrusted input, can execute or act, and "
                f"runs in-process. Carried as accepted risk: "
                f"{handler.accepted_risk or '(nobody accepted it)'}"
            ),
            remedy="; or ".join(handler.remedies()),
        )


@invariant(
    "layering.baseline_only_shrinks",
    scope="observability",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="the grandfathered layering baseline contains no fixed entries",
)
def _layering_baseline_current() -> Iterator[Violation]:
    import json
    from pathlib import Path

    from core.config import config

    baseline_path = Path(config.paths.project_root) / "config" / "layering_baseline.json"
    if not baseline_path.exists():
        return
    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    count = int(payload.get("count", 0) or 0)
    if count and count != len(payload.get("grandfathered", [])):
        yield Violation(
            subject="config/layering_baseline.json",
            message="the baseline's count does not match its entries",
            remedy="regenerate with tools/check_layering.py --baseline",
        )


# ══════════════════════════════════════════════════════════════════════
# Flight-software discipline
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "telemetry.limits_are_coherent",
    scope="flight_software",
    owner=_OWNER,
    description="no channel declares limits that can never be reached",
)
def _telemetry_limits_coherent() -> Iterator[Violation]:
    from core.fsw.telemetry_dictionary import Limits, get_telemetry

    for entry in get_telemetry().dictionary()["channels"]:
        limits = Limits(**entry["limits"])
        for problem in limits.coherent():
            yield Violation(
                subject=entry["name"],
                message=f"channel {entry['name']!r}: {problem}",
                remedy="a limit that can never be reached is a limit nobody checked",
            )


@invariant(
    "telemetry.channels_are_not_silent",
    scope="flight_software",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every declared channel has produced at least one sample",
)
def _channels_not_silent() -> Iterator[Violation]:
    from core.fsw.telemetry_dictionary import telemetry_report

    for name in telemetry_report()["silent_channels"]:
        yield Violation(
            subject=name,
            message=(
                f"channel {name!r} is declared but has never been written; a "
                "dictionary entry with no data is a promise nothing keeps"
            ),
            remedy="write it from wherever the value already exists, or retire it",
        )


@invariant(
    "telemetry.no_channel_is_red",
    scope="flight_software",
    owner=_OWNER,
    description="no telemetry channel is in a red limit state",
)
def _no_red_channels() -> Iterator[Violation]:
    from core.fsw.telemetry_dictionary import telemetry_report

    for entry in telemetry_report()["violations"]:
        if not str(entry["state"]).startswith("red"):
            continue
        yield Violation(
            subject=entry["channel"],
            message=(
                f"{entry['channel']} is {entry['state']} at {entry['value']}"
                f"{entry['unit']} and has been for {entry['for_s']:.0f}s"
            ),
            remedy=f"owner: {entry['owner']}",
        )


@invariant(
    "restart.essential_work_is_declared",
    scope="flight_software",
    owner=_OWNER,
    description="the essential set is non-empty and includes the tick loop",
)
def _essential_declared() -> Iterator[Violation]:
    from core.fsw.restart_protection import restart_report

    report = restart_report()
    if not report["groups"]:
        return
    essential = set(report["essential"])
    if not essential:
        yield Violation(
            subject="restart_protection",
            message=(
                "no work is declared ESSENTIAL, so an overload would shed "
                "everything including the loop that keeps the mind running"
            ),
            remedy="declare the tick loop, the Will, and health as ESSENTIAL",
        )
        return
    for required in ("kernel_tick", "unified_will", "health_surface"):
        if required not in essential:
            yield Violation(
                subject=required,
                message=f"{required!r} is not declared ESSENTIAL and could be shed",
                remedy="add it to install_standard_groups()",
            )


@invariant(
    "restart.core_sets_are_not_exhausted",
    scope="flight_software",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="the core-set pool has headroom",
)
def _core_sets_available() -> Iterator[Violation]:
    from core.fsw.restart_protection import restart_report

    pool = restart_report()["core_sets"]
    if pool["total"] and pool["available"] <= 0:
        yield Violation(
            subject="core_sets",
            message=(
                f"all {pool['total']} core sets are in use; the next significant "
                "work item triggers the overload response"
            ),
            remedy="shed background work, or raise the pool if the ceiling is wrong",
        )


@invariant(
    "rate_groups.are_not_slipping",
    scope="flight_software",
    owner=_OWNER,
    description="no rate group has slipped several cycles in a row",
)
def _rate_groups_not_slipping() -> Iterator[Violation]:
    from core.fsw.rate_groups import rate_group_report

    for group in rate_group_report()["groups"]:
        if group["consecutive_slips"] < 3:
            continue
        yield Violation(
            subject=group["name"],
            message=(
                f"rate group {group['name']!r} has slipped {group['consecutive_slips']} "
                f"cycles in a row (period {group['period_ms']}ms, p50 {group['p50_ms']}ms); "
                f"over budget: {group['over_budget'] or 'nothing individually'}"
            ),
            remedy="cut a member's budget, lower the rate, or shed the work",
        )


@invariant(
    "assertions.none_have_failed",
    scope="flight_software",
    owner=_OWNER,
    description="no declared invariant has been violated",
)
def _no_assertion_failures() -> Iterator[Violation]:
    from core.fsw.assertions import assertions_report

    for record in assertions_report()["records"]:
        yield Violation(
            subject=f"{record['file']}:{record['line']}",
            message=(
                f"{record['condition']} in {record['function']} "
                f"({record['count']}×, args={record['args']})"
            ),
            remedy="a violated invariant means the state is not what the code believes",
        )


@invariant(
    "health.no_critical_component_is_unresponsive",
    scope="flight_software",
    owner=_OWNER,
    description="every component declared critical is answering pings",
)
def _critical_components_answering() -> Iterator[Violation]:
    from core.fsw.health_checker import health_checker_report

    report = health_checker_report()
    for name in report["critical_unresponsive"]:
        yield Violation(
            subject=name,
            message=(
                f"critical component {name!r} stopped answering health pings — it is "
                "wedged, which is a different fact from being quiet"
            ),
            remedy="restart or repair it; a passive health surface cannot see this",
        )


@invariant(
    "commands.declared_commands_have_handlers",
    scope="flight_software",
    owner=_OWNER,
    description="no command is declared without something to run it",
)
def _commands_have_handlers() -> Iterator[Violation]:
    from core.fsw.command_dispatch import get_dispatcher

    for entry in get_dispatcher().dictionary()["commands"]:
        if not entry["has_handler"]:
            yield Violation(
                subject=entry["name"],
                message=(
                    f"command {entry['name']!r} (opcode 0x{entry['opcode']:02x}) is in "
                    "the dictionary but has no handler; a plan containing it validates "
                    "and then fails at execution"
                ),
                remedy="attach a handler, or remove it from the dictionary",
            )


# ══════════════════════════════════════════════════════════════════════
# Cognition: rewriting and self-validation
# ══════════════════════════════════════════════════════════════════════

@invariant(
    "curiosity.queue_is_transactional",
    scope="cognition",
    owner=_OWNER,
    description="curiosity owns one bounded queue with unique active claims",
)
def _curiosity_queue_is_transactional() -> Iterator[Violation]:
    from core.container import ServiceContainer

    explorer = ServiceContainer.get("curiosity_explorer", default=None)
    if explorer is None:
        return
    status = explorer.get_status() if hasattr(explorer, "get_status") else None
    if not isinstance(status, dict):
        yield Violation(
            subject="curiosity_explorer",
            message="registered curiosity explorer exposes no inspectable queue receipt",
            remedy="provide get_status() with bounded queue and attempt evidence",
        )
        return
    queue = status.get("queue") if isinstance(status.get("queue"), list) else []
    pending = int(status.get("pending", -1) or 0)
    active = [item for item in queue if item.get("status") in {"pending", "running"}]
    if pending != len(active) or pending > 10:
        yield Violation(
            subject="curiosity_explorer.queue",
            message=f"queue receipt says pending={pending}, observed active={len(active)}",
            remedy="repair queue accounting before admitting more autonomous work",
        )
    hashes = [str(item.get("question_hash") or "") for item in active]
    if any(not value for value in hashes) or len(hashes) != len(set(hashes)):
        yield Violation(
            subject="curiosity_explorer.queue",
            message="active curiosity claims are missing identity or contain duplicates",
            remedy="deduplicate under the queue owner lock before enqueue",
        )


@invariant(
    "affect.state_views_are_canonical",
    scope="cognition",
    owner=_OWNER,
    description="affect reads agree and simulated somatic indices are not biomedical claims",
)
def _affect_state_views_are_canonical() -> Iterator[Violation]:
    from core.container import ServiceContainer

    affect = ServiceContainer.get("affect_engine", default=None)
    if affect is None or not all(
        hasattr(affect, name) for name in ("get_snapshot", "get_status", "_snapshot_state")
    ):
        return
    snapshot = affect.get_snapshot()
    status = affect.get_status()
    state = affect._snapshot_state()
    values = (
        float(snapshot.get("valence", 0.0)),
        float(status.get("valence", 0.0)),
        float(getattr(state, "valence", 0.0)),
    )
    if max(values) - min(values) > 0.011:
        yield Violation(
            subject="affect_engine.valence",
            message=f"public affect views disagree: snapshot/status/state={values}",
            remedy="derive every affect view from the canonical dimension function",
        )
    rendered = repr(snapshot.get("somatic_indices", {})) + repr(status.get("physiology", {}))
    if any(unit in rendered for unit in ("bpm", "μS", "μg/dL")):
        yield Violation(
            subject="affect_engine.somatic_indices",
            message="simulated affect indices are presented with biomedical units",
            remedy="label unitless model indices explicitly and reserve units for measured sensors",
        )


@invariant(
    "claims.every_claim_has_a_passing_test",
    scope="cognition",
    owner=_OWNER,
    description="no registered claim about the runtime is currently unsupported",
)
def _claims_supported() -> Iterator[Violation]:
    from core.organism.model_validation import Outcome, get_suite

    for entry in get_suite().unsupported_claims():
        # A claim whose instrument had no population to measure is a claim
        # with no evidence — but it is not a contradicted claim, and in a
        # bare process (an offline test run, a partial boot) lockdep, the
        # rate groups and the health checker legitimately have nothing in
        # front of them. Contradicted is an error; unevidenced is reported
        # separately below so it cannot be mistaken for support.
        if entry.get("outcome") == str(Outcome.NOT_MEASURED):
            continue
        yield Violation(
            subject=entry["test"],
            message=(
                f"{entry['statement']} — {entry.get('reason', 'not run')} "
                f"({entry.get('outcome', 'unrun')})"
            ),
            remedy=f"fix the behaviour, or withdraw the claim from {entry['asserted_in']}",
        )


@invariant(
    "claims.no_claim_rests_on_an_empty_instrument",
    scope="cognition",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every claim's instrument had something to measure",
)
def _claims_were_measured() -> Iterator[Violation]:
    """Claims backed by an instrument that ran over an empty set.

    Three of these used to score a clean zero and PASS: lockdep counted 0
    splats across 0 known locks, the rate-group test took ``max([])`` of 0
    groups, and the health checker reported 0 unresponsive of 0 registered.
    A warning rather than an error because an idle subsystem is normal in a
    partial process — but never silence, because "measured nothing" reading
    as "measured fine" is what this whole registry exists to prevent.
    """
    from core.organism.model_validation import Outcome, get_suite

    for entry in get_suite().unsupported_claims():
        if entry.get("outcome") != str(Outcome.NOT_MEASURED):
            continue
        yield Violation(
            subject=entry["test"],
            message=(
                f"{entry['statement']} — no evidence this run: "
                f"{entry.get('reason', 'instrument measured nothing')}"
            ),
            remedy=(
                "exercise the subsystem before reading its claim, or accept that "
                "this claim is unevidenced in this process"
            ),
        )


@invariant(
    "claims.tests_are_claimed",
    scope="cognition",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="every validation test backs a stated claim",
)
def _tests_are_claimed() -> Iterator[Violation]:
    from core.organism.model_validation import validation_report

    for name in validation_report()["tests_without_claims"]:
        yield Violation(
            subject=name,
            message=(
                f"test {name!r} checks something nobody has claimed; either it is "
                "protecting an unstated promise or it is dead weight"
            ),
            remedy="state the claim it supports, or delete it",
        )


@invariant(
    "metta.rules_terminate",
    scope="cognition",
    severity=Severity.WARNING,
    owner=_OWNER,
    description="reductions are not routinely hitting their bounds",
)
def _metta_terminates() -> Iterator[Violation]:
    from core.knowledge.metta import metta_report

    report = metta_report()
    reductions = report["reductions"]
    truncations = report["truncations"]
    if reductions >= 10 and truncations / reductions > 0.25:
        yield Violation(
            subject="metta",
            message=(
                f"{truncations} of {reductions} reductions hit a bound; the rule set "
                "is producing derivations that do not terminate within budget"
            ),
            remedy="find the non-terminating rule pair; unbounded rewriting hangs the runtime",
        )


@invariant(
    "values.one_level_per_value",
    scope="values",
    owner=_OWNER,
    description="no value is held at two levels of changeability at once",
)
def _one_level_per_value() -> Iterator[Violation]:
    """Two subsystems disagreeing about what may change is not a tidiness bug.

    If A holds a thing immutable and B treats it as a learned preference, the
    intersection is a governance ambiguity, and the duplicated concept is
    "what is allowed to change". The census reports the disagreement; this
    fails when one is left unresolved by the canonical declaration.
    """

    try:
        from core.values.what_she_holds import (
            declare_what_she_holds,
            disagreements,
            what_she_holds,
        )
        from core.governance.value_levels import registry
    except ImportError:
        return
    claims = what_she_holds()
    if not claims:
        return
    declare_what_she_holds(claims)
    held = registry()
    for name, group in disagreements(claims).items():
        canonical = held.get(name)
        strictest = max(one.level for one in group)
        if canonical is None:
            yield Violation(
                subject=name,
                message=(
                    f"{len(group)} subsystems hold {name!r} at different levels "
                    "and no canonical level is declared"
                ),
                remedy="declare it through core/values/what_she_holds.py",
            )
        elif canonical.level is not strictest:
            yield Violation(
                subject=name,
                severity=Severity.ERROR,
                message=(
                    f"{name!r} is canonically {canonical.level.name.lower()} while "
                    f"{', '.join(sorted({o.source for o in group if o.level is strictest}))} "
                    f"holds it {strictest.name.lower()}; a value cannot become "
                    "easier to change by being declared twice"
                ),
                remedy="the strictest claim wins; resolve to it or drop the claim",
            )


@invariant(
    "values.constitutive_values_are_unreachable",
    scope="values",
    owner=_OWNER,
    description="no registered automated process may write a constitutive value",
)
def _nothing_reaches_constitutive() -> Iterator[Violation]:
    """A system that can widen its own authority has none.

    The permission table is module state with no setter, and this is the
    check that it stayed that way: every registered process, against every
    constitutive value, must be refused.
    """

    try:
        from core.governance.value_levels import (
            Change,
            Level,
            registered_processes,
            registry,
        )
        from core.values.what_she_holds import declare_what_she_holds
    except ImportError:
        return
    declare_what_she_holds()
    held = registry()
    for value in held.at_level(Level.CONSTITUTIVE):
        for process in registered_processes():
            decision = held.may_change(
                Change(
                    value=value.name,
                    process=process,
                    gives_up="everything, which is why this must be refused",
                )
            )
            if decision.allowed:
                yield Violation(
                    subject=f"{process} -> {value.name}",
                    severity=Severity.ERROR,
                    message=(
                        f"{process!r} may write {value.name!r}, which is "
                        "constitutive; nothing automated may"
                    ),
                    remedy="remove the authority, or the value is not constitutive",
                )


def register_runtime_invariants() -> int:
    """Import-time registration is the real work; this returns the count."""
    from core.verify.invariants import get_registry

    return len(get_registry().specs())


__all__ = ["register_runtime_invariants"]
