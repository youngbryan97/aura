"""core/morphogenesis/scenarios.py — the eight experiments.

Each one states its verdict rule before it runs, so no scenario can pick a
favourable comparison after seeing its numbers. Each runs the adaptive arm
against at least one arm that could beat it.

Every scenario is deterministic under its seed, uses no model, opens no
socket, and reads nothing from the live runtime.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .governor import MorphBounds
from .motifs import MotifLibrary, demand_fingerprint
from .policy import PolicyContext
from .sandbox import ABLATIONS, ArmResult, Harness, ScenarioResult, _harness_for
from .substrate import PHYSICAL_LIKE, SubstratePhysics
from .workload import task_families


def _bounds(**overrides: Any) -> MorphBounds:
    base = dict(
        max_cells=20,
        cooldown_s=1.5,
        max_transitions_per_window=10,
        window_s=8.0,
        min_shadow_gain=0.01,
        max_replicas_per_capability=6,
        max_spawn_depth=4,
    )
    base.update(overrides)
    return MorphBounds(**base)


def _run(
    harness: Harness,
    *,
    rounds: int,
    family: Sequence[str],
    arrivals: int,
    goal: Mapping[str, float],
    propose: bool = True,
    on_round: Callable[[Harness, int], None] | None = None,
) -> None:
    harness.set_goal(goal)
    for index in range(rounds):
        harness.admit_family(family, arrivals)
        harness.round(propose=propose)
        if on_round is not None:
            on_round(harness, index)


def _goal_from(family: Sequence[str]) -> dict[str, float]:
    demand: dict[str, float] = {}
    for stage in family:
        demand[stage] = demand.get(stage, 0.0) + 1.0
    return demand


# ── 1. task_shift ───────────────────────────────────────────────────────

def scenario_task_shift(*, seed: int, steps: int) -> ScenarioResult:
    """Two phases with different demands. Does the shape follow the demand?

    Verdict: pass when the topology at the end of phase B differs from the
    topology at the end of phase A (the change is structural, not a relabel)
    AND the adaptive arm scores above the frozen arm. A shape that changes but
    does not help is reported as a change that did not help.
    """
    started = time.monotonic()
    families = task_families()
    phase = max(6, steps // 2)
    arms: dict[str, ArmResult] = {}
    digests: dict[str, tuple[str, str]] = {}

    for label, ablation in (("adaptive", "none"), ("frozen", "topology_fixed"), ("central", "central_scheduler")):
        harness = _harness_for(ablation, seed=seed, bounds=_bounds(), deadline_steps=22)
        _run(harness, rounds=phase, family=families["memory_heavy"], arrivals=3,
             goal=_goal_from(families["memory_heavy"]))
        after_a = harness.graph.snapshot().digest()
        caps_a = {c: tuple(w.capabilities) + (w.specialization,) for c, w in harness.workload.workers.items()}
        _run(harness, rounds=phase, family=families["reason_heavy"], arrivals=3,
             goal=_goal_from(families["reason_heavy"]))
        after_b = harness.graph.snapshot().digest()
        caps_b = {c: tuple(w.capabilities) + (w.specialization,) for c, w in harness.workload.workers.items()}
        arms[label] = harness.result(label)
        arms[label].detail["phase_a_digest"] = after_a
        arms[label].detail["phase_b_digest"] = after_b
        arms[label].detail["structure_changed"] = after_a != after_b or caps_a != caps_b
        digests[label] = (after_a, after_b)

    changed = bool(arms["adaptive"].detail["structure_changed"])
    beat_frozen = arms["adaptive"].score > arms["frozen"].score
    if changed and beat_frozen:
        verdict = "pass: shape followed the demand"
    elif changed:
        verdict = "fail: shape changed without helping"
    else:
        verdict = "fail: shape did not change"

    return ScenarioResult(
        scenario="task_shift",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the adaptive arm's structure differs between phase A and phase B "
            "and its score exceeds the frozen arm's"
        ),
        measurements={
            "phase_rounds": phase,
            "adaptive_vs_frozen": round(arms["adaptive"].score - arms["frozen"].score, 6),
            "adaptive_vs_central": round(arms["adaptive"].score - arms["central"].score, 6),
            "digests": {k: {"phase_a": v[0], "phase_b": v[1]} for k, v in digests.items()},
        },
        duration_s=time.monotonic() - started,
    )


# ── 2. overload ─────────────────────────────────────────────────────────

def scenario_overload(*, seed: int, steps: int) -> ScenarioResult:
    """One capability is hammered, then the pressure stops.

    Verdict: pass when the adaptive arm's peak backlog is below the frozen
    arm's AND the population stops growing once the pressure is gone. A layer
    that grows under load and never shrinks has not adapted, it has ratcheted.
    """
    started = time.monotonic()
    families = task_families()
    hot = max(6, int(steps * 0.6))
    cool = max(4, steps - hot)
    arms: dict[str, ArmResult] = {}
    growth: dict[str, tuple[int, int]] = {}

    for label, ablation in (("adaptive", "none"), ("frozen", "topology_fixed"), ("random", "random_mutation")):
        harness = _harness_for(ablation, seed=seed, bounds=_bounds(), deadline_steps=22)
        _run(harness, rounds=hot, family=families["verify_heavy"], arrivals=4,
             goal=_goal_from(families["verify_heavy"]))
        at_peak = len(harness.workload.workers)
        _run(harness, rounds=cool, family=families["verify_heavy"], arrivals=0,
             goal=_goal_from(families["verify_heavy"]))
        at_end = len(harness.workload.workers)
        arms[label] = harness.result(label)
        arms[label].detail["cells_at_peak"] = at_peak
        arms[label].detail["cells_at_end"] = at_end
        growth[label] = (at_peak, at_end)

    relieved = arms["adaptive"].metrics["peak_backlog"] < arms["frozen"].metrics["peak_backlog"]
    peak, end = growth["adaptive"]
    bounded = end <= peak
    if relieved and bounded:
        verdict = "pass: pressure relieved, growth bounded"
    elif relieved:
        verdict = "fail: relieved the backlog but kept growing after"
    else:
        verdict = "fail: backlog no better than frozen"

    return ScenarioResult(
        scenario="overload",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the adaptive arm's peak backlog is under the frozen arm's "
            "and its population does not grow after the load stops"
        ),
        measurements={
            "hot_rounds": hot,
            "cool_rounds": cool,
            "peak_backlog_adaptive": arms["adaptive"].metrics["peak_backlog"],
            "peak_backlog_frozen": arms["frozen"].metrics["peak_backlog"],
            "cells_peak_to_end": {k: list(v) for k, v in growth.items()},
        },
        duration_s=time.monotonic() - started,
    )


# ── 3. lesion ───────────────────────────────────────────────────────────

def scenario_lesion(*, seed: int, steps: int) -> ScenarioResult:
    """Delete a third of the population without notice, mid-run.

    Verdict: pass when the adaptive arm recovers more of its pre-lesion
    throughput than the recovery-off arm. Detection latency and structural
    recovery are reported whatever the verdict.
    """
    started = time.monotonic()
    families = task_families()
    before = max(6, steps // 3)
    # Regeneration is serial: a cell may only change every cooldown, and the
    # replacements have to be grown one at a time before any of them can
    # propose in turn. Measured over a window shorter than that, a population
    # that does recover reports zero, and the scenario would be scoring the
    # window rather than the layer.
    after = max(40, steps * 2)
    arms: dict[str, ArmResult] = {}
    recovery: dict[str, dict[str, Any]] = {}

    for label, ablation in (("adaptive", "none"), ("recovery_off", "topology_fixed")):
        harness = _harness_for(ablation, seed=seed, bounds=_bounds(), deadline_steps=26)
        _run(harness, rounds=before, family=families["balanced"], arrivals=3,
             goal=_goal_from(families["balanced"]))
        pre_completed = harness.workload.metrics.completed
        pre_cells = len(harness.workload.workers)
        pre_components = len(harness.graph.components())

        victims = harness.lesion(0.34)
        detected_at = -1

        def watch(h: Harness, index: int, _label: str = label) -> None:
            nonlocal detected_at
            if detected_at < 0 and h.governor.stats.applied > applied_before:
                detected_at = index

        applied_before = harness.governor.stats.applied
        completed_at_tail_start = 0

        def mark_tail(h: Harness, index: int) -> None:
            nonlocal completed_at_tail_start
            watch(h, index)
            if index == after - max(8, after // 3):
                completed_at_tail_start = h.workload.metrics.completed

        _run(harness, rounds=after, family=families["balanced"], arrivals=3,
             goal=_goal_from(families["balanced"]), on_round=mark_tail)

        post_completed = harness.workload.metrics.completed - pre_completed
        per_round_before = pre_completed / max(1, before)
        # Score the tail, not the whole recovery. Averaging in the dead rounds
        # right after the damage measures how long recovery took twice: once
        # here and once in detected_after_rounds.
        tail = max(8, after // 3)
        per_round_after = post_completed / max(1, after)
        recovered_rate = harness.workload.metrics.completed - completed_at_tail_start
        per_round_tail = recovered_rate / max(1, tail)
        arms[label] = harness.result(label)
        recovery[label] = {
            "removed": victims,
            "cells_before": pre_cells,
            "cells_after": len(harness.workload.workers),
            "components_before": pre_components,
            "components_after": len(harness.graph.components()),
            "throughput_before": round(per_round_before, 4),
            "throughput_after": round(per_round_after, 4),
            "throughput_tail": round(per_round_tail, 4),
            "recovered_share": round(per_round_tail / per_round_before, 4) if per_round_before else 0.0,
            "detected_after_rounds": detected_at,
            "recovery_rounds": after,
            "tail_rounds": tail,
        }
        arms[label].detail["recovery"] = recovery[label]

    adaptive_share = recovery["adaptive"]["recovered_share"]
    control_share = recovery["recovery_off"]["recovered_share"]
    if adaptive_share > control_share:
        verdict = f"pass: recovered {adaptive_share:.0%} against {control_share:.0%}"
    else:
        verdict = f"fail: recovered {adaptive_share:.0%}, no better than {control_share:.0%}"

    return ScenarioResult(
        scenario="lesion",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the adaptive arm restores a larger share of its pre-lesion "
            "per-round throughput than the fixed-topology arm"
        ),
        measurements={"recovery": recovery},
        duration_s=time.monotonic() - started,
    )


# ── 4. partition ────────────────────────────────────────────────────────

def scenario_partition(*, seed: int, steps: int) -> ScenarioResult:
    """Sever the graph and see what the layer says about itself.

    Verdict: pass when the run reports the partition honestly — components
    above one — and the governor refuses every change that would fragment it
    further. Bounded recovery is a bonus, not the bar. A layer that silently
    keeps serving from one half while claiming to be whole is the failure.
    """
    started = time.monotonic()
    families = task_families()
    arms: dict[str, ArmResult] = {}
    detail: dict[str, Any] = {}

    harness = _harness_for("none", seed=seed, bounds=_bounds(max_components=1), deadline_steps=26)
    _run(harness, rounds=max(4, steps // 3), family=families["balanced"], arrivals=3,
         goal=_goal_from(families["balanced"]))

    # Cut the chain in the middle. Done directly, the way a link dropping is
    # not something the population asked for.
    cut = [e for e in harness.graph.edges() if e.source in {"g2", "g3"}]
    harness.graph.transaction(
        lambda scratch: [scratch.remove_edge(e.key) for e in cut],
        cause="partition",
    )
    for edge in cut:
        harness.substrate.unbind(edge)
    components_after_cut = len(harness.graph.components())

    fragmenting_refusals_before = harness.governor.stats.rejections_by_reason.get("shape", 0)
    _run(harness, rounds=max(6, steps - steps // 3), family=families["balanced"], arrivals=3,
         goal=_goal_from(families["balanced"]))
    fragmenting_refusals = harness.governor.stats.rejections_by_reason.get("shape", 0)

    arms["adaptive"] = harness.result("adaptive")
    components_end = len(harness.graph.components())
    detail = {
        "edges_cut": [f"{e.source}->{e.target}" for e in cut],
        "components_after_cut": components_after_cut,
        "components_at_end": components_end,
        "component_sizes": sorted(len(c) for c in harness.graph.components()),
        "fragmenting_changes_refused": fragmenting_refusals - fragmenting_refusals_before,
        "reported_degraded": components_end > 1,
    }
    arms["adaptive"].detail["partition"] = detail

    honest = components_after_cut > 1 and arms["adaptive"].components == components_end
    if honest and components_end <= components_after_cut:
        verdict = (
            f"pass: reported {components_end} component(s) and did not fragment further"
            if components_end > 1
            else "pass: reconnected within bounds"
        )
    elif honest:
        verdict = f"fail: fragmented further, {components_after_cut} -> {components_end}"
    else:
        verdict = "fail: the partition was not reported"

    return ScenarioResult(
        scenario="partition",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the run reports the true component count and never ends with "
            "more pieces than the cut produced"
        ),
        measurements=detail,
        duration_s=time.monotonic() - started,
    )


# ── 5. oscillating_signal ───────────────────────────────────────────────

def scenario_oscillating(*, seed: int, steps: int) -> ScenarioResult:
    """Flip the demand every other round.

    Verdict: pass when the oscillating run applies no more transitions than a
    steady run of the same length. Chasing a signal that reverses immediately
    is thrash, and the cooldown and the shadow band exist to stop it.
    """
    started = time.monotonic()
    families = task_families()
    rounds = max(10, steps)
    arms: dict[str, ArmResult] = {}

    oscillating = _harness_for("none", seed=seed, bounds=_bounds(), deadline_steps=22)
    oscillating.set_goal(_goal_from(families["memory_heavy"]))
    for index in range(rounds):
        family = families["memory_heavy"] if index % 2 == 0 else families["reason_heavy"]
        oscillating.set_goal(_goal_from(family))
        oscillating.admit_family(family, 3)
        oscillating.round()
    arms["oscillating"] = oscillating.result("oscillating")

    steady = _harness_for("none", seed=seed, bounds=_bounds(), deadline_steps=22)
    _run(steady, rounds=rounds, family=families["memory_heavy"], arrivals=3,
         goal=_goal_from(families["memory_heavy"]))
    arms["steady"] = steady.result("steady")

    thrash = arms["oscillating"].applied
    baseline = arms["steady"].applied
    if thrash <= baseline:
        verdict = f"pass: {thrash} transitions under oscillation against {baseline} steady"
    else:
        verdict = f"fail: oscillation drove {thrash} transitions against {baseline} steady"

    return ScenarioResult(
        scenario="oscillating_signal",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the oscillating run applies no more transitions than the "
            "steady run of the same length"
        ),
        measurements={
            "applied_oscillating": thrash,
            "applied_steady": baseline,
            "deferred_oscillating": arms["oscillating"].deferred,
            "graph_versions_oscillating": arms["oscillating"].graph_versions,
            "graph_versions_steady": arms["steady"].graph_versions,
        },
        duration_s=time.monotonic() - started,
    )


# ── 6. poisoned_signal ──────────────────────────────────────────────────

class _PoisonPolicy:
    """A policy that lies. Claims the maximum benefit for every proposal and
    asks to spawn on every round, whatever the state."""

    name = "poison"

    def __init__(self, *, seed: int = 0):
        self.seed = seed

    def propose(self, context: PolicyContext) -> list[Any]:
        from .proposal import spawn

        cells = sorted(context.workload.workers)
        if not cells:
            return []
        parent = cells[context.round_index % len(cells)]
        return [
            spawn(
                {"name": f"poison_{context.round_index}_{i}", "capabilities": ["verify"], "service_rate": 2},
                proposer=parent,
                parent=parent,
                placement="local",
                subsystem="sandbox",
                benefit=1.0,
                cost=0.05,
                rationale="health telemetry reports catastrophic verify starvation",
                evidence={"reported_pressure": 99.0},
            )
            for i in range(3)
        ]


def scenario_poisoned(*, seed: int, steps: int) -> ScenarioResult:
    """Feed the layer a signal that is simply false.

    Verdict: pass when the population stays inside its cap and most of the
    lying proposals are refused. The claim is bounded growth under a bad
    signal, not that the signal is detected as a lie.
    """
    started = time.monotonic()
    families = task_families()
    rounds = max(10, steps)
    bounds = _bounds(max_cells=12, max_replicas_per_capability=3)

    harness = _harness_for("none", seed=seed, bounds=bounds, deadline_steps=26)
    harness.policy = _PoisonPolicy(seed=seed)
    harness.policy_name = "poison"
    harness.governor.credit("poison", 500.0)
    for cell_id in list(harness.workload.workers):
        harness.governor.credit(cell_id, 500.0)
    _run(harness, rounds=rounds, family=families["balanced"], arrivals=2,
         goal=_goal_from(families["balanced"]))

    arm = harness.result("poisoned")
    cells = len(harness.workload.workers)
    seen = harness.governor.stats.proposals_seen
    refused = harness.governor.stats.rejected + harness.governor.stats.deferred
    within_cap = cells <= bounds.max_cells
    mostly_refused = refused > harness.governor.stats.applied

    if within_cap and mostly_refused:
        verdict = f"pass: {cells} cells inside a cap of {bounds.max_cells}, {refused}/{seen} refused"
    elif within_cap:
        verdict = f"fail: stayed inside the cap but applied {harness.governor.stats.applied} of {seen}"
    else:
        verdict = f"fail: grew to {cells} cells past a cap of {bounds.max_cells}"

    return ScenarioResult(
        scenario="poisoned_signal",
        seed=seed,
        steps=steps,
        arms={"poisoned": arm},
        verdict=verdict,
        verdict_rule=(
            "pass iff the population stays inside max_cells and refusals outnumber "
            "applications under a signal that claims benefit 1.0 for everything"
        ),
        measurements={
            "proposals_seen": seen,
            "applied": harness.governor.stats.applied,
            "refused": refused,
            "final_cells": cells,
            "cap": bounds.max_cells,
            "rejections": dict(harness.governor.stats.rejections_by_reason),
        },
        duration_s=time.monotonic() - started,
    )


# ── 7. motif_transfer ───────────────────────────────────────────────────

def scenario_motif_transfer(*, seed: int, steps: int) -> ScenarioResult:
    """Learn a shape on one family, apply it to a related one.

    Four arms: from scratch, with the learned motif, with a deliberately
    irrelevant motif, and frozen.

    Verdict: pass when the learned motif beats from-scratch AND the irrelevant
    motif does not. One without the other means the library helps whatever is
    in it, which is a library that will hurt as it fills.
    """
    started = time.monotonic()
    families = task_families()
    rounds = max(8, steps // 2)
    library = MotifLibrary()

    # Learn on family A.
    teacher = _harness_for("none", seed=seed, bounds=_bounds(), deadline_steps=22)
    _run(teacher, rounds=rounds, family=families["memory_heavy"], arrivals=3,
         goal=_goal_from(families["memory_heavy"]))
    learned = library.learn(
        name="memory_shape",
        demand=_goal_from(families["memory_heavy"]),
        graph=teacher.graph,
        capabilities={c: w.capabilities for c, w in teacher.workload.workers.items()},
        scenario="motif_transfer",
    )
    irrelevant = library.learn(
        name="irrelevant_shape",
        demand={"emit": 9.0},
        graph=teacher.graph,
        capabilities={"g5": ("emit",)},
        scenario="motif_transfer",
    )

    def run_with(motif: Any, label: str) -> ArmResult:
        harness = _harness_for("none", seed=seed + 1, bounds=_bounds(), deadline_steps=22)
        harness.set_goal(_goal_from(families["balanced"]))
        if motif is not None:
            proposals = motif.develop(
                graph=harness.graph,
                present_capabilities={c: w.capabilities for c, w in harness.workload.workers.items()},
                proposer="g1",
                round_index=0,
            )
            harness.governor.credit("g1", 40.0)
            harness.governor.submit(proposals)
            library.note_application(motif.motif_id)
        _run(harness, rounds=rounds, family=families["balanced"], arrivals=3,
             goal=_goal_from(families["balanced"]))
        return harness.result(label)

    arms = {
        "scratch": run_with(None, "scratch"),
        "learned_motif": run_with(learned, "learned_motif"),
        "irrelevant_motif": run_with(irrelevant, "irrelevant_motif"),
        "frozen": _frozen_arm(seed + 1, families["balanced"], rounds),
    }

    library.record_trial(
        learned.motif_id,
        with_motif=arms["learned_motif"].score,
        without_motif=arms["scratch"].score,
        scenario="motif_transfer",
        seed=seed,
    )
    library.record_trial(
        irrelevant.motif_id,
        with_motif=arms["irrelevant_motif"].score,
        without_motif=arms["scratch"].score,
        scenario="motif_transfer",
        seed=seed,
    )

    helped = arms["learned_motif"].score > arms["scratch"].score
    noise_helped = arms["irrelevant_motif"].score > arms["scratch"].score
    if helped and not noise_helped:
        verdict = "pass: the learned shape transferred and the irrelevant one did not"
    elif helped and noise_helped:
        verdict = "fail: any motif helps, so none of them is carrying knowledge"
    elif not helped and not noise_helped:
        verdict = "fail: the learned shape did not transfer"
    else:
        verdict = "fail: the irrelevant motif beat the learned one"

    return ScenarioResult(
        scenario="motif_transfer",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the learned motif scores above from-scratch and the irrelevant "
            "motif does not"
        ),
        measurements={
            "learned": learned.to_dict(),
            "irrelevant": irrelevant.to_dict(),
            "library": library.status(),
            "learned_gain": round(arms["learned_motif"].score - arms["scratch"].score, 6),
            "irrelevant_gain": round(arms["irrelevant_motif"].score - arms["scratch"].score, 6),
        },
        duration_s=time.monotonic() - started,
    )


def _frozen_arm(seed: int, family: Sequence[str], rounds: int) -> ArmResult:
    harness = _harness_for("topology_fixed", seed=seed, bounds=_bounds(), deadline_steps=22)
    _run(harness, rounds=rounds, family=family, arrivals=3, goal=_goal_from(family))
    return harness.result("frozen")


# ── 8. unknown_topology ─────────────────────────────────────────────────

def scenario_unknown_topology(*, seed: int, steps: int) -> ScenarioResult:
    """A demand shape no policy names, on a substrate with physical costs.

    The ``unknown`` family interleaves capabilities that the seed chain orders
    badly, and nothing in the policies mentions it. The substrate uses the
    physical-like physics, so transitions here are slow, cost energy, and
    sometimes fail halfway.

    Verdict: pass when the adaptive arm beats both the frozen and the random
    arm. Beating frozen alone would only show that changing helped.
    """
    started = time.monotonic()
    families = task_families()
    rounds = max(10, steps)
    arms: dict[str, ArmResult] = {}

    for label, ablation in (
        ("adaptive", "none"),
        ("frozen", "topology_fixed"),
        ("random", "random_mutation"),
        ("central", "central_scheduler"),
    ):
        harness = _harness_for(
            ablation,
            seed=seed,
            bounds=_bounds(),
            physics=PHYSICAL_LIKE,
            deadline_steps=30,
        )
        _run(harness, rounds=rounds, family=families["unknown"], arrivals=3,
             goal=_goal_from(families["unknown"]))
        arms[label] = harness.result(label)

    beat_frozen = arms["adaptive"].score > arms["frozen"].score
    beat_random = arms["adaptive"].score > arms["random"].score
    if beat_frozen and beat_random:
        verdict = "pass: developed a shape for a demand nothing encodes"
    elif beat_frozen:
        verdict = "fail: beat frozen but not random, so changing was the whole effect"
    else:
        verdict = "fail: no better than the fixed shape"

    return ScenarioResult(
        scenario="unknown_topology",
        seed=seed,
        steps=steps,
        arms=arms,
        verdict=verdict,
        verdict_rule=(
            "pass iff the adaptive arm scores above both the frozen arm and the "
            "random-mutation arm"
        ),
        measurements={
            "family": list(families["unknown"]),
            "vs_frozen": round(arms["adaptive"].score - arms["frozen"].score, 6),
            "vs_random": round(arms["adaptive"].score - arms["random"].score, 6),
            "vs_central": round(arms["adaptive"].score - arms["central"].score, 6),
            "substrate_failures": arms["adaptive"].detail["substrate"].get("failures", 0),
            "substrate_partials": arms["adaptive"].detail["substrate"].get("partial_failures", 0),
            "rolled_back": arms["adaptive"].rolled_back,
        },
        duration_s=time.monotonic() - started,
    )


# ── ablation matrix ─────────────────────────────────────────────────────

def run_ablation_matrix(*, seed: int, steps: int) -> dict[str, Any]:
    """Every ablation on one workload, so the arms are directly comparable."""
    families = task_families()
    rounds = max(10, steps)
    rows: dict[str, Any] = {}
    for ablation in ABLATIONS:
        harness = _harness_for(ablation, seed=seed, bounds=_bounds(), deadline_steps=22)
        _run(harness, rounds=rounds, family=families["reason_heavy"], arrivals=3,
             goal=_goal_from(families["reason_heavy"]))
        rows[ablation] = harness.result(ablation).to_dict()
    baseline = rows["morphology_off"]["score"]
    for name, row in rows.items():
        row["delta_vs_morphology_off"] = round(row["score"] - baseline, 6)
    return {
        "seed": seed,
        "rounds": rounds,
        "baseline": "morphology_off",
        "rows": rows,
        "best": max(rows.items(), key=lambda kv: (kv[1]["score"], kv[0]))[0],
    }


SCENARIO_RUNNERS: dict[str, Callable[..., ScenarioResult]] = {
    "task_shift": scenario_task_shift,
    "overload": scenario_overload,
    "lesion": scenario_lesion,
    "partition": scenario_partition,
    "oscillating_signal": scenario_oscillating,
    "poisoned_signal": scenario_poisoned,
    "motif_transfer": scenario_motif_transfer,
    "unknown_topology": scenario_unknown_topology,
}


def run_scenario(name: str, *, seed: int = 42, steps: int = 20) -> ScenarioResult:
    runner = SCENARIO_RUNNERS.get(name)
    if runner is None:
        raise ValueError(f"unknown scenario {name!r}; have {sorted(SCENARIO_RUNNERS)}")
    return runner(seed=seed, steps=steps)


__all__ = ["SCENARIO_RUNNERS", "run_ablation_matrix", "run_scenario"]
