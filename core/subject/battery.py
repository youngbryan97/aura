"""The conjunction, with every threshold written down before the first run.

ISC(K) = 1 is an and, not an average. A system that scores beautifully on nine
axes and has a free cut is two systems, and a weighted total would hide that,
which is the whole reason the criterion was written as a conjunction.

Thresholds live here and nowhere else. They are engineering bars, not
constants of nature, and their job is to have been fixed in advance: a number
that can be adjusted after seeing the result measures nothing. Two of them are
already known to be contested and are kept anyway, with the argument recorded
next to them rather than acted on.

Each criterion carries what it would take to falsify it, so a passing line and
a failing line are the same kind of object and a run can be read without
knowing which way it came out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["THRESHOLDS", "Criterion", "Verdict", "assemble"]

#: Preregistered. Sources: the specification's section numbers.
THRESHOLDS: dict[str, float] = {
    "phi_do": 0.05,
    "d_eff_normalised": 0.40,
    "spread": 0.60,
    "synergy_fraction": 0.10,
    "edge_q": 0.01,
    "edge_effect": 0.30,
    "edge_replication": 3.0,
    "vertex_connectivity": 2.0,
    "cycles_per_node": 2.0,
    "reentry_domains": 3.0,
    "global_access_consumers": 3.0,
}


@dataclass(frozen=True, slots=True)
class Criterion:
    """One line of the conjunction."""

    key: str
    section: str
    statement: str
    passed: bool
    value: Any
    bar: Any
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.key,
            "section": self.section,
            "requires": self.statement,
            "passed": self.passed,
            "value": self.value,
            "bar": self.bar,
            "detail": self.detail,
        }


@dataclass
class Verdict:
    criteria: list[Criterion] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def isc(self) -> bool:
        return bool(self.criteria) and all(item.passed for item in self.criteria)

    @property
    def passed(self) -> int:
        return sum(1 for item in self.criteria if item.passed)

    def failures(self) -> list[Criterion]:
        return [item for item in self.criteria if not item.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "isc": self.isc,
            "passed": self.passed,
            "total": len(self.criteria),
            "failed": [item.key for item in self.failures()],
            "criteria": [item.as_dict() for item in self.criteria],
            "notes": self.notes,
        }

    def table(self) -> str:
        rows = [f"{'PASS' if c.passed else 'FAIL'}  {c.key:34s} {c.value!s:>22s}  needs {c.bar}" for c in self.criteria]
        head = f"ISC = {int(self.isc)}   {self.passed}/{len(self.criteria)} criteria"
        return "\n".join([head, *rows])


def _c(key: str, section: str, statement: str, passed: bool, value: Any, bar: Any, **detail: Any) -> Criterion:
    return Criterion(key, section, statement, bool(passed), value, bar, detail)


def assemble(evidence: dict[str, Any]) -> Verdict:
    """Turn the reports into the conjunction. Missing evidence is a failure.

    A criterion whose measurement did not run reads as failed rather than as
    absent, because a battery that quietly shrinks to what it managed to
    compute reports a pass for a system nobody finished testing.
    """
    graph = evidence.get("graph", {})
    edges = evidence.get("edges", [])
    phi = evidence.get("phi", {})
    diff = evidence.get("differentiation", {})
    intrinsic = evidence.get("intrinsic", {})
    pci = evidence.get("perturbation", {})
    synergy = evidence.get("synergy", [])
    meta = evidence.get("metastability", {})
    agency = evidence.get("agency", {})
    closure = evidence.get("closure", {})
    lesion = evidence.get("lesion", {})
    nulls = evidence.get("nulls", {})
    access = evidence.get("global_access", {})
    timescale = evidence.get("timescale", {})
    conditions = evidence.get("per_condition", {})

    kept = [e for e in edges if e.get("kept")]
    directed = {(e["source"], e["target"]) for e in kept}

    out = Verdict(notes=evidence.get("notes", {}))
    add = out.criteria.append

    add(_c(
        "causal_closure_scc", "14",
        "every domain reaches every other one through interventional edges",
        bool(graph.get("one_strongly_connected_component")),
        graph.get("components", []),
        "one component covering all domains",
        edge_count=graph.get("edge_count", 0),
    ))
    add(_c(
        "robust_recurrence_kappa", "17",
        "no single domain's removal disconnects the rest",
        float(graph.get("vertex_connectivity", 0)) >= THRESHOLDS["vertex_connectivity"],
        graph.get("vertex_connectivity", 0),
        f">= {int(THRESHOLDS['vertex_connectivity'])}",
    ))
    add(_c(
        "cycles_per_domain", "15",
        "every domain lies on at least two cross-domain directed cycles",
        bool(graph.get("every_node_on_two_cycles")),
        graph.get("cycles_per_node", {}),
        f">= {int(THRESHOLDS['cycles_per_node'])} each",
    ))
    add(_c(
        "reentry", "16",
        "influence leaves each domain and returns through at least two others",
        bool(graph.get("every_node_reenters_through_two_others")),
        graph.get("shortest_reentry", {}),
        f"cycle length >= {int(THRESHOLDS['reentry_domains'])}",
    ))
    add(_c(
        "partition_irreducibility", "11",
        "the cheapest bipartition still costs held-out prediction",
        float(phi.get("phi_do", -1.0)) > THRESHOLDS["phi_do"],
        round(float(phi.get("phi_do", 0.0)), 4),
        f"> {THRESHOLDS['phi_do']}",
        cheapest_cut=phi.get("best_cut"),
        dearest_cuts=phi.get("dearest_cuts"),
        surrogate_floor=nulls.get("surrogate_floor"),
        above_surrogate_floor=nulls.get("phi_above_floor"),
    ))
    # The comparison the absolute threshold cannot make. A minimum over 511
    # noisy estimates is biased downward by the width of its own search, and the
    # matched surrogates — same dimensionality, same cuts, same estimator, the
    # coupling removed — are the only thing that measures how far.
    add(_c(
        "partition_beats_nulls", "18",
        "irreducibility exceeds every null architecture and matched surrogate",
        bool(nulls.get("phi_beats_all")),
        nulls.get("phi_table", {}),
        "above all nulls",
        surrogate_floor=nulls.get("surrogate_floor"),
        margin_over_floor=nulls.get("phi_above_floor"),
    ))
    add(_c(
        "differentiation", "18",
        "the integrated state is not one dimension wearing many names",
        float(diff.get("d_eff_normalised", 0.0)) >= THRESHOLDS["d_eff_normalised"],
        round(float(diff.get("d_eff_normalised", 0.0)), 4),
        f">= {THRESHOLDS['d_eff_normalised']}",
        d_eff=diff.get("d_eff"),
        largest_component_share=diff.get("largest_component_share"),
    ))
    add(_c(
        "differentiation_above_floor", "18",
        "effective dimension is far above the one-dimensional floor",
        float(diff.get("d_eff", 0.0)) >= 3.0
        and float(diff.get("largest_component_share", 1.0)) < 0.5,
        {"d_eff": diff.get("d_eff"), "top_share": diff.get("largest_component_share")},
        "d_eff >= 3 and no component above half the variance",
    ))
    add(_c(
        "intrinsic_persistence", "19",
        "the previous internal state predicts the next beyond the environment",
        bool(intrinsic.get("passes")),
        round(float(intrinsic.get("delta_intrinsic", 0.0)), 4),
        "> 0, and above the shuffled-state control",
        over_shuffle=intrinsic.get("delta_over_shuffled_state"),
    ))
    add(_c(
        "causal_closure_of_the_core", "3",
        "no variable outside K predicts K's future better than K does",
        bool(closure.get("closed")),
        round(float(closure.get("leak", 1.0)), 4),
        "no improvement on K beyond what a shuffled periphery gives",
        largest_leaks=closure.get("largest_leaks", []),
    ))
    add(_c(
        "perturbational_spread", "24",
        "a local displacement reaches most of the core",
        float(pci.get("mean_spread", 0.0)) >= THRESHOLDS["spread"],
        round(float(pci.get("mean_spread", 0.0)), 4),
        f">= {THRESHOLDS['spread']}",
        per_source=pci.get("spread_by_source", {}),
    ))
    # A response matrix of all zeros beats a null of all zeros, and the first
    # version of this line passed on exactly that. Complexity is only a
    # question about a response that happened, so the matrix has to be
    # non-degenerate before its structure is worth scoring.
    reached = [
        key for key, value in (pci.get("spread_by_source") or {}).items() if value > 0.0
    ]
    add(_c(
        "perturbational_complexity", "26",
        "the response is structured, not local and not a broadcast",
        bool(pci.get("beats_null")) and len(reached) >= 2 and float(pci.get("mean_pci", 0.0)) > 0.0,
        round(float(pci.get("mean_pci", 0.0)), 4),
        "a response that reached somewhere, above the 99th percentile of matched nulls",
        null=pci.get("null_q99"),
        sources_that_reached_anything=reached,
    ))
    add(_c(
        "synergy", "27",
        "domains carry information jointly that neither carries alone",
        bool(synergy) and all(item.get("passes") for item in synergy),
        [item.get("synergy_fraction") for item in synergy],
        f">= {THRESHOLDS['synergy_fraction']} and above its shifted null",
        triples=[item.get("sources", []) + [item.get("target")] for item in synergy],
    ))
    add(_c(
        "metastability", "29",
        "the state settles into regimes and still moves between them",
        bool(meta.get("passes")),
        {"regimes": meta.get("regimes"), "entropy": meta.get("transition_entropy")},
        "more than one regime, entropy strictly between zero and its ceiling",
    ))
    add(_c(
        "global_access", "30",
        "workspace content reaches at least three heterogeneous consumers",
        int(access.get("consumers", 0)) >= THRESHOLDS["global_access_consumers"],
        access.get("consumer_list", []),
        f">= {int(THRESHOLDS['global_access_consumers'])}",
    ))
    add(_c(
        "recurrent_global_access", "31",
        "a consumer of the workspace later changes the workspace again",
        bool(access.get("returns")),
        access.get("return_paths", []),
        "at least one G -> X -> G path",
    ))
    add(_c(
        "self_drives_action", "33",
        "changing the self-model changes deliberation or behaviour",
        bool(agency.get("self_drives_action")),
        round(float(agency.get("self_to_action", 0.0)), 4),
        "above the same-arm floor",
        floor=agency.get("self_to_action_floor"),
    ))
    add(_c(
        "ownership", "34",
        "the same world state updates the self differently when she caused it",
        bool(agency.get("outcome_updates_self")),
        round(float(agency.get("ownership_divergence", 0.0)), 4),
        "above the same-arm floor",
        floor=agency.get("ownership_floor"),
    ))
    add(_c(
        "fast_to_slow", "38",
        "fast cognition changes the slow developmental state",
        bool(timescale.get("fast_to_slow")),
        timescale.get("fast_to_slow_edges", []),
        "at least one edge from a fast domain into a slow one",
    ))
    add(_c(
        "slow_to_fast", "38",
        "the slow developmental state changes later fast cognition",
        bool(timescale.get("slow_to_fast")),
        timescale.get("slow_to_fast_edges", []),
        "at least one edge from a slow domain into a fast one",
    ))
    add(_c(
        "natural_runtime_replication", "44",
        "the graph holds across ordinary conditions, not only one workload",
        int(conditions.get("conditions_with_scc", 0)) >= 3,
        conditions.get("per_condition", {}),
        "strongly connected in at least three conditions",
    ))
    add(_c(
        "lesion_deficit", "39",
        "cutting the cheapest partition degrades the measures it should",
        bool(lesion.get("deficit")),
        lesion.get("deltas", {}),
        "irreducibility, complexity and synergy all fall",
    ))
    # Rescue only means something after a deficit. Restoring a channel whose
    # removal changed nothing is not evidence about the channel.
    add(_c(
        "rescue", "40",
        "restoring the cut restores them",
        bool(lesion.get("deficit")) and bool(lesion.get("rescued_ok")),
        lesion.get("rescue", {}),
        "a deficit first, then the measures return towards the intact values",
    ))
    add(_c(
        "beats_every_null", "41",
        "the whole battery separates the system from every matched null",
        bool(nulls.get("all_nulls_fail")),
        nulls.get("summary", {}),
        "no null passes the conjunction",
    ))
    del directed
    return out
