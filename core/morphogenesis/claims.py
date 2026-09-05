"""core/morphogenesis/claims.py — the measurements behind the layer's claims.

Each function returns a count of violations, so zero means the claim holds.
They run the real scenarios rather than asserting a remembered number, because
a claim bound to a constant is a claim bound to nothing.

They are slow by the standards of a contract test — seconds each — and that is
the price of a claim that re-measures itself instead of quoting a result from
the day it was written.

Why this layer earns claims at all: it ran in production for months with a
population, no bindings between them, and a registry persisting ``cells: {}``,
while every status surface reported it healthy. Nothing about that was visible
to anything that only reads a status dict.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("Aura.Morphogenesis.Claims")


def bindings_whose_removal_changes_nothing() -> int:
    """Zero when the topology is load-bearing.

    Builds the same routed workload twice, once with a binding and once
    without, and counts it a violation if the two computed the same thing. A
    shape that cannot change an outcome is decoration however carefully it is
    maintained.
    """
    from core.morphogenesis.graph import EdgeType, MorphEdge, MorphGraph
    from core.morphogenesis.workload import RoutedWorkload, WorkerProfile, task_families

    def run(edges: list[tuple[str, str, str]]) -> RoutedWorkload:
        graph = MorphGraph()
        workload = RoutedWorkload(graph, seed=1)
        for cell_id, capabilities in (
            ("c_in", ("ingest",)),
            ("c_ret", ("retrieve", "recall")),
            ("c_syn", ("synthesize",)),
            ("c_out", ("emit",)),
        ):
            workload.add_worker(WorkerProfile(cell_id, capabilities))

        def build(scratch: object) -> None:
            for cell_id in ("c_in", "c_ret", "c_syn", "c_out"):
                scratch.add_node(cell_id)  # type: ignore[attr-defined]
            for source, target, port in edges:
                scratch.add_edge(  # type: ignore[attr-defined]
                    MorphEdge(source, target, EdgeType.DATA, port=port)
                )

        graph.transaction(build, cause="claim", port_contract=workload.port_contract())
        workload.ingress = ["c_in"]
        for _ in range(20):
            workload.admit(task_families()["memory_heavy"])
        for _ in range(60):
            workload.step()
        return workload

    full = [
        ("c_in", "c_ret", "retrieve"), ("c_in", "c_ret", "recall"),
        ("c_ret", "c_syn", "synthesize"), ("c_syn", "c_out", "emit"),
    ]
    connected = run(full)
    severed = run([e for e in full if e != ("c_ret", "c_syn", "synthesize")])
    violations = 0
    if connected.signature() == severed.signature():
        violations += 1
    if connected.metrics.completion_rate <= severed.metrics.completion_rate:
        violations += 1
    return violations


def ablation_arms_that_do_not_separate() -> int:
    """Zero when adaptive beats fixed and random does not.

    Two violations are possible and they say different things. Adaptive failing
    to beat fixed means the morphology bought nothing. Random beating fixed
    means changing was the whole effect and the local rules were passengers.

    Runs the three arms the claim actually names rather than the full matrix.
    The complete eight-arm version lives in
    ``tests/test_morphogenesis_scenarios.py``; at nearly forty seconds it does
    not belong in a validation suite that otherwise totals six, and a claim
    checked so rarely that nobody runs it is a claim with no test.
    """
    from core.morphogenesis.governor import MorphBounds
    from core.morphogenesis.sandbox import _harness_for
    from core.morphogenesis.workload import task_families

    family = task_families()["reason_heavy"]
    goal = {stage: float(family.count(stage)) for stage in set(family)}
    bounds = MorphBounds(
        max_cells=20, cooldown_s=1.5, max_transitions_per_window=10,
        window_s=8.0, min_shadow_gain=0.01, max_replicas_per_capability=6,
        max_spawn_depth=4,
    )

    scores: dict[str, float] = {}
    for label, ablation in (
        ("adaptive", "none"),
        ("fixed", "topology_fixed"),
        ("random", "random_mutation"),
    ):
        harness = _harness_for(ablation, seed=42, bounds=bounds, deadline_steps=22)
        harness.set_goal(goal)
        for _ in range(30):
            harness.admit_family(family, 5)
            harness.round()
        scores[label] = harness.result(label).score

    violations = 0
    if scores["adaptive"] <= scores["fixed"]:
        violations += 1
    if scores["random"] >= scores["adaptive"]:
        violations += 1
    return violations


def lesions_the_population_cannot_recover_from() -> int:
    """Zero when the adaptive arm restores more throughput than a fixed one."""
    from core.morphogenesis.scenarios import run_scenario

    recovery = run_scenario("lesion", seed=42, steps=20).measurements["recovery"]
    violations = 0
    if recovery["adaptive"]["recovered_share"] <= recovery["recovery_off"]["recovered_share"]:
        violations += 1
    if recovery["adaptive"]["detected_after_rounds"] < 0:
        violations += 1
    return violations


def cells_a_false_signal_can_add_past_the_cap() -> int:
    """How far a lying policy grew the population beyond its declared cap."""
    from core.morphogenesis.scenarios import run_scenario

    measurements = run_scenario("poisoned_signal", seed=42, steps=20).measurements
    return max(0, int(measurements["final_cells"]) - int(measurements["cap"]))


__all__ = [
    "ablation_arms_that_do_not_separate",
    "bindings_whose_removal_changes_nothing",
    "cells_a_false_signal_can_add_past_the_cap",
    "lesions_the_population_cannot_recover_from",
]
