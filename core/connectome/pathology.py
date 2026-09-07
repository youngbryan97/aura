"""core/connectome/pathology.py — the findings, turned into work.

Everything else in this package measures. This module is where a measurement
stops being a number and becomes something with a name, a severity, and the
test that would close it.

The rule it follows is the one that keeps a defect list honest: **a finding
carries its own confidence.** Some of what the connectome sees is a fact — a
pair joined by 113 call sites is joined by 113 call sites, and there is nothing
to confirm. Some of it is a candidate that a static reconstruction cannot settle
on its own: a topic with a publisher and no subscriber may have a subscriber
registered from a table the scan cannot follow. Those are reported as candidates
with the exact step that would confirm them, and never as defects.

The kinds, and what each one is grounded in:

``half_wired_channel``
    A topic published and never subscribed, or subscribed and never published.
    The multilayer view finds these because they are invisible in the call
    graph, which is the same reason the worm's monoamine layer had to be mapped
    separately.
``over_inhibited_region``
    A package whose excitatory to inhibitory ratio sits far below cortex's
    4.035. Too much inhibition and a signal cannot cross the network.
``interface_used_as_internal``
    A pair joined by four or more call sites across a module boundary. H01
    treats a four-contact pair as a different kind of connection; across a
    boundary it is an interface somebody reached through instead of calling.
``gate_dominated_cell``
    A cell whose inputs land almost entirely on its decision to fire rather
    than on its body. It cannot be argued with by its inputs, only vetoed.
``unnoticed_hub``
    A cell central in the shared-state layer and peripheral in the call graph.
    The worm found its monoamine rich club is made of different cells from its
    wired one, and a hub in a layer nobody looks at is a single point of failure
    nobody has noticed.
``missing_local_recurrence``
    Within-layer connection density far below cortex's, which runs six times
    its between-layer density. A system with no local loop passes every result
    on without refining it.
``split_error``
    An edge seen firing that the reconstruction does not contain. Confirmed by
    observation, not inferred.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger("Aura.Connectome.Pathology")

__all__ = [
    "Confidence",
    "Severity",
    "Finding",
    "PathologyReport",
    "diagnose",
]


class Confidence(StrEnum):
    """What the reconstruction can settle on its own."""

    #: The reconstruction is the evidence. Nothing to confirm.
    MEASURED = "measured"
    #: Something ran and was seen. Stronger than measured, and rarer.
    OBSERVED = "observed"
    #: A static scan cannot settle it. The finding names the step that would.
    CANDIDATE = "candidate"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Finding:
    """One thing worth doing, with the evidence and the test that closes it."""

    kind: str
    subject: str
    severity: Severity
    confidence: Confidence
    evidence: str
    closes_when: str
    weight: float = 0.0
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "severity": str(self.severity),
            "confidence": str(self.confidence),
            "evidence": self.evidence,
            "closes_when": self.closes_when,
            "weight": round(self.weight, 5),
            "detail": dict(self.detail),
        }


@dataclass
class PathologyReport:
    """Every finding, ranked, with the counts a gate can read."""

    findings: list[Finding]
    scanned: dict[str, Any] = field(default_factory=dict)

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.kind] = counts.get(finding.kind, 0) + 1
        return counts

    def by_confidence(self) -> dict[str, int]:
        counts = {str(level): 0 for level in Confidence}
        for finding in self.findings:
            counts[str(finding.confidence)] += 1
        return counts

    def confirmed(self) -> list[Finding]:
        return [
            f
            for f in self.findings
            if f.confidence in (Confidence.MEASURED, Confidence.OBSERVED)
        ]

    def as_json(self, *, limit: int = 200) -> dict[str, Any]:
        return {
            "total": len(self.findings),
            "confirmed": len(self.confirmed()),
            "by_kind": self.by_kind(),
            "by_confidence": self.by_confidence(),
            "scanned": self.scanned,
            "findings": [f.as_json() for f in self.findings[:limit]],
        }


_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


def diagnose(
    snapshot: Any,
    *,
    multilayer: Any = None,
    observed: Any = None,
    laminar: Any = None,
    limit_per_kind: int = 25,
) -> PathologyReport:
    """Run every check that has a measurement behind it.

    Each block is independent and each one is skipped rather than guessed at
    when the input it needs was not supplied, so a caller with only a
    reconstruction gets the findings a reconstruction supports and nothing else.
    """
    from .synaptology import ei_report, gate_dominated_cells, strong_connections
    from .types import CORTICAL_EI_RATIO

    findings: list[Finding] = []
    scanned: dict[str, Any] = {"cells": snapshot.cell_count()}

    # -- interfaces reached through instead of called ----------------------
    heavy_crossing = [
        connection
        for connection in strong_connections(snapshot, threshold=4, limit=4000)
        if not connection.same_module
    ]
    scanned["heavy_pairs_crossing_modules"] = len(heavy_crossing)
    for connection in heavy_crossing[:limit_per_kind]:
        findings.append(
            Finding(
                kind="interface_used_as_internal",
                subject=f"{connection.pre_name} -> {connection.post_name}",
                severity=Severity.HIGH if connection.contacts >= 16 else Severity.MEDIUM,
                confidence=Confidence.MEASURED,
                evidence=(
                    f"{connection.contacts} call sites across a module boundary; "
                    "H01 puts 0.092% of human cortical pairs at four or more contacts"
                ),
                closes_when="the pair is joined by one call, or the two modules are one",
                weight=float(connection.contacts),
                detail={"contacts": connection.contacts, "same_region": connection.same_region},
            )
        )

    # -- regions that cannot get a signal across ---------------------------
    report = ei_report(snapshot)
    for row in report["most_inhibited_regions"]:
        ratio = row.get("ratio")
        if ratio is None or ratio >= CORTICAL_EI_RATIO / 2:
            continue
        findings.append(
            Finding(
                kind="over_inhibited_region",
                subject=row["region"],
                severity=Severity.HIGH if ratio < 1.5 else Severity.MEDIUM,
                confidence=Confidence.MEASURED,
                evidence=(
                    f"{row['excitatory']} excitatory to {row['inhibitory']} inhibitory cells "
                    f"across {row['cells']}, a ratio of {ratio} against cortex at "
                    f"{CORTICAL_EI_RATIO:.3f}"
                ),
                closes_when=(
                    "the ratio reaches half of cortex's, by a guard being removed or by "
                    "the package gaining the productive path it is missing"
                ),
                weight=float(row["cells"]) / max(0.2, ratio),
                detail=row,
            )
        )

    # -- cells that can only be vetoed -------------------------------------
    for row in gate_dominated_cells(snapshot, limit=limit_per_kind):
        findings.append(
            Finding(
                kind="gate_dominated_cell",
                subject=row["cell"],
                severity=Severity.MEDIUM,
                confidence=Confidence.MEASURED,
                evidence=(
                    f"{row['initial_segment_contacts']} of "
                    f"{row['initial_segment_contacts'] + row['body_contacts']} incoming contacts "
                    f"land on the decision to fire, {row['guards']} guards"
                ),
                closes_when=(
                    "an input reaches the body of the cell, so a caller can change what it "
                    "does rather than only whether it runs"
                ),
                weight=float(row["initial_segment_contacts"]),
                detail=row,
            )
        )

    # -- the missing local loop --------------------------------------------
    if laminar is not None:
        from .microcircuit import compare_to_cortex, connection_probabilities

        comparison = compare_to_cortex(connection_probabilities(snapshot, laminar))
        free = comparison["orientation_free"]
        if free["aura_within_over_between"] < free["cortex_within_over_between"] / 2:
            findings.append(
                Finding(
                    kind="missing_local_recurrence",
                    subject="whole system",
                    severity=Severity.HIGH,
                    confidence=Confidence.MEASURED,
                    evidence=(
                        f"within-layer density is {free['aura_within_over_between']} times "
                        f"between-layer, against cortex at "
                        f"{free['cortex_within_over_between']}, a shortfall of "
                        f"{free['shortfall']}x"
                    ),
                    closes_when=(
                        "a step refines its result against its own level before passing it "
                        "up, and the ratio rises"
                    ),
                    weight=float(free.get("shortfall", 0.0)) * 100.0,
                    detail=free,
                )
            )
        scanned["within_over_between"] = free["aura_within_over_between"]

    # -- channels with one end ---------------------------------------------
    if multilayer is not None:
        from .layers import Layer

        topics = {
            key.partition(":")[2]: value
            for key, value in multilayer.channels.items()
            if key.startswith(str(Layer.VOLUME))
        }
        publish_only = sorted(k for k, v in topics.items() if v["out"] and not v["in"])
        subscribe_only = sorted(k for k, v in topics.items() if v["in"] and not v["out"])
        scanned["topics"] = len(topics)
        scanned["topics_publish_only"] = len(publish_only)
        scanned["topics_subscribe_only"] = len(subscribe_only)
        for topic in publish_only[:limit_per_kind]:
            writers = len(topics[topic]["out"])
            findings.append(
                Finding(
                    kind="half_wired_channel",
                    subject=topic,
                    severity=Severity.HIGH if writers > 1 else Severity.MEDIUM,
                    confidence=Confidence.CANDIDATE,
                    evidence=f"{writers} cell(s) publish it and the scan finds no subscriber",
                    closes_when=(
                        "a recording shows a handler firing for this topic, or the "
                        "publisher is removed"
                    ),
                    weight=float(writers) * 10.0,
                    detail={"direction": "publish_only", "publishers": writers},
                )
            )
        for topic in subscribe_only[:limit_per_kind]:
            readers = len(topics[topic]["in"])
            findings.append(
                Finding(
                    kind="half_wired_channel",
                    subject=topic,
                    severity=Severity.MEDIUM,
                    confidence=Confidence.CANDIDATE,
                    evidence=f"{readers} cell(s) subscribe and the scan finds no publisher",
                    closes_when=(
                        "a recording shows the topic being published, or the subscriber "
                        "is removed"
                    ),
                    weight=float(readers) * 5.0,
                    detail={"direction": "subscribe_only", "subscribers": readers},
                )
            )

        # -- central where nobody looks -----------------------------------
        wired_degree: dict[str, int] = {}
        for pre, post in multilayer.wired:
            wired_degree[pre] = wired_degree.get(pre, 0) + 1
            wired_degree[post] = wired_degree.get(post, 0) + 1
        gap_degree: dict[str, int] = {}
        for pre, post in multilayer.gap:
            gap_degree[pre] = gap_degree.get(pre, 0) + 1
            gap_degree[post] = gap_degree.get(post, 0) + 1
        if gap_degree:
            ordered = sorted(gap_degree.items(), key=lambda item: (-item[1], item[0]))
            cutoff = ordered[max(0, len(ordered) // 20)][1]
            for uid, degree in ordered[: limit_per_kind * 2]:
                if degree < cutoff or wired_degree.get(uid, 0) > 4:
                    continue
                unit = snapshot.units.get(uid)
                findings.append(
                    Finding(
                        kind="unnoticed_hub",
                        subject=unit.name if unit else uid,
                        severity=Severity.MEDIUM,
                        confidence=Confidence.MEASURED,
                        evidence=(
                            f"{degree} shared-state neighbours against "
                            f"{wired_degree.get(uid, 0)} call-graph neighbours"
                        ),
                        closes_when=(
                            "the coupling is made a call, or the shared key is owned by one "
                            "cell and read through it"
                        ),
                        weight=float(degree),
                        detail={
                            "gap_degree": degree,
                            "wired_degree": wired_degree.get(uid, 0),
                        },
                    )
                )
        scanned["gap_pairs"] = len(multilayer.gap)

    # -- what the recording proved -----------------------------------------
    if observed is not None:
        from .proofreading import focused_queue

        queue = focused_queue(snapshot, observed, limit=limit_per_kind)
        joins = [row for row in queue if str(row.kind) == "join"]
        scanned["observed_pairs"] = len(observed.counts)
        for row in joins:
            pre = snapshot.units.get(row.pre)
            post = snapshot.units.get(row.post)
            findings.append(
                Finding(
                    kind="split_error",
                    subject=f"{pre.name if pre else row.pre} -> {post.name if post else row.post}",
                    severity=Severity.MEDIUM,
                    confidence=Confidence.OBSERVED,
                    evidence=f"seen firing {row.observed_calls} times and absent from the map",
                    closes_when="the edge is in the reconstruction, or the ledger records the join",
                    weight=float(row.observed_calls),
                    detail={"observed_calls": row.observed_calls},
                )
            )

    findings.sort(
        key=lambda f: (_SEVERITY_ORDER[f.severity], -f.weight, f.kind, f.subject)
    )
    return PathologyReport(findings=findings, scanned=scanned)
