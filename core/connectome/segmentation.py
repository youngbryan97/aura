"""core/connectome/segmentation.py — attaching the uncertain sites, and paying for it.

Every automated reconstruction reaches the same fork. A process runs into a
place where the evidence stops being decisive, and the choice is to attach it
to the most likely neighbour or to leave it hanging. Attaching buys coverage
and pays in merge errors, and a merge error is the expensive kind, because a
wrong join looks exactly like a circuit and gets published as one.

Connectomics answers this with a measured threshold rather than a preference.
The base segmentation is deliberately over-split; agglomeration then merges
segments whose affinity clears a threshold; and the threshold is chosen from a
curve, not from taste. The metric on that curve is expected run length: how far
a process can be followed before the reconstruction gets it wrong, with any
piece carrying a merge error scoring zero.

Aura gets something the fish and the fly never had. Running her produces ground
truth — an edge that fired happened — so the curve here is measured against
observation instead of against a few hours of human tracing.

Two asymmetries in what observation can prove, both of which this module keeps:

* An observed edge missing from the reconstruction **is** a split error. There
  is no other reading.
* A reconstructed edge that was never observed is only weak evidence of a merge
  error, because the branch may never have run. It is reported as a suspicion
  with the caller's own firing count attached, and never as a defect.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .activity import ObservedEdges
from .types import Compartment, Connection, ConnectomeSnapshot, ContactSite, EdgeKind, Unit
from .volume import AmbiguousSite

logger = logging.getLogger("Aura.Connectome.Segmentation")

__all__ = [
    "AffinityWeights",
    "Attachment",
    "score_site",
    "agglomerate",
    "AgglomerationResult",
    "expected_run_length",
    "sweep_threshold",
    "ReconstructionScore",
    "score_against_observation",
]


@dataclass(frozen=True)
class AffinityWeights:
    """What each kind of evidence is worth when scoring a candidate.

    These are ordered by how much they constrain, not by how often they fire.
    Sharing a module is the strongest evidence short of an exact resolution,
    because Python name lookup happens there first; sharing a package is much
    weaker, and being reachable through an import sits between them.
    """

    same_module: float = 1.0
    imported_module: float = 0.8
    same_region: float = 0.45
    name_is_unique_in_region: float = 0.2
    private_name: float = 0.15
    penalty_per_extra_candidate: float = 0.03

    def as_json(self) -> dict[str, float]:
        return {
            "same_module": self.same_module,
            "imported_module": self.imported_module,
            "same_region": self.same_region,
            "name_is_unique_in_region": self.name_is_unique_in_region,
            "private_name": self.private_name,
            "penalty_per_extra_candidate": self.penalty_per_extra_candidate,
        }


@dataclass(frozen=True)
class Attachment:
    """One candidate scored against one site."""

    site_index: int
    uid: str
    score: float
    margin: float
    evidence: tuple[str, ...]


def score_site(
    site: AmbiguousSite,
    units: Mapping[str, Unit],
    weights: AffinityWeights,
) -> list[tuple[str, float, tuple[str, ...]]]:
    """Score every candidate for one site. Highest first, ties broken by uid."""
    imported = set(site.imported)
    scored: list[tuple[str, float, tuple[str, ...]]] = []
    region_counts: dict[str, int] = {}
    for uid in site.candidates:
        unit = units.get(uid)
        if unit is not None:
            region_counts[unit.region] = region_counts.get(unit.region, 0) + 1
    for uid in site.candidates:
        unit = units.get(uid)
        if unit is None:
            continue
        score = 0.0
        evidence: list[str] = []
        if unit.neuropil == site.caller_module:
            score += weights.same_module
            evidence.append("same_module")
        if unit.neuropil in imported:
            score += weights.imported_module
            evidence.append("imported")
        if unit.region == site.caller_region:
            score += weights.same_region
            evidence.append("same_region")
            if region_counts.get(unit.region, 0) == 1:
                score += weights.name_is_unique_in_region
                evidence.append("unique_in_region")
        if site.target_name.startswith("_") and unit.neuropil == site.caller_module:
            score += weights.private_name
            evidence.append("private_local")
        score -= weights.penalty_per_extra_candidate * max(0, len(site.candidates) - 2)
        scored.append((uid, score, tuple(evidence)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


@dataclass
class AgglomerationResult:
    """A snapshot with attachments applied, and what applying them cost."""

    snapshot: ConnectomeSnapshot
    threshold: float
    margin: float
    attached: int
    left_split: int
    attachments: list[Attachment] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "margin": self.margin,
            "attached": self.attached,
            "left_split": self.left_split,
            "drive_edges": len(self.snapshot.edges(EdgeKind.DRIVE)),
        }


def agglomerate(
    snapshot: ConnectomeSnapshot,
    sites: Sequence[AmbiguousSite],
    *,
    threshold: float = 1.0,
    margin: float = 0.2,
    weights: AffinityWeights | None = None,
) -> AgglomerationResult:
    """Attach the sites that clear both bars, and leave the rest split.

    The margin is the part that matters. A site whose best two candidates score
    the same has no winner however high the score is, and attaching to the first
    of them is a coin flip recorded as a fact.
    """
    weights = weights or AffinityWeights()
    result_units = snapshot.units
    attachments: list[Attachment] = []
    new_contacts: list[ContactSite] = []
    left = 0
    for index, site in enumerate(sites):
        scored = score_site(site, result_units, weights)
        if not scored:
            left += 1
            continue
        best_uid, best_score, evidence = scored[0]
        runner_up = scored[1][1] if len(scored) > 1 else float("-inf")
        gap = best_score - runner_up if len(scored) > 1 else best_score
        if best_score < threshold or gap < margin:
            left += 1
            continue
        attachments.append(
            Attachment(
                site_index=index, uid=best_uid, score=best_score, margin=gap, evidence=evidence
            )
        )
        new_contacts.append(
            ContactSite(
                pre=site.caller,
                post=best_uid,
                locus=site.locus,
                compartment=Compartment.SOMA,
                sign=1,
                kind=EdgeKind.DRIVE,
            )
        )
        if site.value_used:
            callee = result_units[best_uid]
            sign = -1 if str(callee.cell_class) == "inhibitory" else 1
            new_contacts.append(
                ContactSite(
                    pre=best_uid,
                    post=site.caller,
                    locus=f"{site.locus}:r",
                    compartment=(
                        Compartment.AXON_INITIAL_SEGMENT if site.in_guard else Compartment.DENDRITE
                    ),
                    sign=sign,
                    kind=EdgeKind.RETURN,
                )
            )

    merged = dict(snapshot.connections)
    for contact in new_contacts:
        key = (contact.pre, contact.post, str(contact.kind))
        existing = merged.get(key)
        if existing is None:
            merged[key] = Connection(
                pre=contact.pre,
                post=contact.post,
                contacts=1,
                sign=contact.sign,
                compartments=(contact.compartment,),
                kind=contact.kind,
            )
        else:
            merged[key] = Connection(
                pre=existing.pre,
                post=existing.post,
                contacts=existing.contacts + 1,
                sign=min(existing.sign, contact.sign),
                compartments=tuple(sorted(set(existing.compartments) | {contact.compartment}, key=str)),
                kind=existing.kind,
            )

    grown = ConnectomeSnapshot(
        version=snapshot.version + 1,
        units=snapshot.units,
        connections=merged,
        neuropils=snapshot.neuropils,
        built_at=snapshot.built_at,
        source=snapshot.source,
        attrs=dict(snapshot.attrs),
    )
    grown.attrs.update(
        {
            "agglomeration_threshold": threshold,
            "agglomeration_margin": margin,
            "agglomerated_sites": len(attachments),
            "weights": weights.as_json(),
        }
    )
    return AgglomerationResult(
        snapshot=grown,
        threshold=threshold,
        margin=margin,
        attached=len(attachments),
        left_split=left,
        attachments=attachments,
    )


# ---------------------------------------------------------------------------
# Scoring a reconstruction against what the runtime actually did
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructionScore:
    """What observation can and cannot prove about a reconstruction."""

    observed_pairs: int
    recovered: int
    split_errors: int
    recall: float
    suspected_merges: int
    suspicion_denominator: int
    expected_run_length: float
    mean_run_length: float
    runs: int

    def as_json(self) -> dict[str, Any]:
        return {
            "observed_pairs": self.observed_pairs,
            "recovered": self.recovered,
            "split_errors": self.split_errors,
            "recall": round(self.recall, 4),
            "suspected_merges": self.suspected_merges,
            "suspicion_denominator": self.suspicion_denominator,
            "expected_run_length": round(self.expected_run_length, 4),
            "mean_run_length": round(self.mean_run_length, 4),
            "runs": self.runs,
        }


def _observed_chains(observed: ObservedEdges, limit: int = 4096) -> list[list[tuple[str, str]]]:
    """Follow the busiest successor from each source to build traceable chains.

    A chain stands in for the skeleton a human tracer follows. Picking the
    heaviest successor is deterministic and biases towards the paths that carry
    the most traffic, which are the ones a reconstruction error hurts most.
    """
    successors: dict[str, list[tuple[int, str]]] = {}
    for (pre, post), count in observed.counts.items():
        successors.setdefault(pre, []).append((count, post))
    for entries in successors.values():
        entries.sort(key=lambda item: (-item[0], item[1]))
    posts = {post for _, post in observed.counts}
    sources = sorted(set(successors) - posts) or sorted(successors)
    chains: list[list[tuple[str, str]]] = []
    for source in sources[:limit]:
        seen = {source}
        node = source
        chain: list[tuple[str, str]] = []
        while True:
            entries = successors.get(node)
            if not entries:
                break
            nxt = entries[0][1]
            if nxt in seen:
                break
            chain.append((node, nxt))
            seen.add(nxt)
            node = nxt
        if chain:
            chains.append(chain)
    return chains


def expected_run_length(
    chains: Sequence[Sequence[tuple[str, str]]],
    present: set[tuple[str, str]],
) -> tuple[float, float, int]:
    """Expected run length over traced chains, weighted by run length.

    A chain of ``n`` edges with a break in the middle gives two runs. Weighting
    each run by its own length is what makes the metric care about long correct
    stretches rather than about counting edges, which is the whole reason
    connectomics uses it instead of accuracy.
    """
    runs: list[int] = []
    for chain in chains:
        current = 0
        for edge in chain:
            if edge in present:
                current += 1
            else:
                if current:
                    runs.append(current)
                current = 0
        if current:
            runs.append(current)
    if not runs:
        return 0.0, 0.0, 0
    total = sum(runs)
    erl = sum(length * length for length in runs) / total
    return erl, total / len(runs), len(runs)


def score_against_observation(
    snapshot: ConnectomeSnapshot,
    observed: ObservedEdges,
    *,
    min_caller_calls: int = 8,
) -> ReconstructionScore:
    """Measure a reconstruction against edges that were seen to fire."""
    present = {(c.pre, c.post) for c in snapshot.edges(EdgeKind.DRIVE)}
    observed_pairs = observed.pairs()
    recovered = len(observed_pairs & present)
    split_errors = len(observed_pairs - present)
    recall = recovered / len(observed_pairs) if observed_pairs else 0.0

    fired: dict[str, int] = {}
    for (pre, _), count in observed.counts.items():
        fired[pre] = fired.get(pre, 0) + count
    busy = {uid for uid, count in fired.items() if count >= min_caller_calls}
    denominator = sum(1 for pre, _ in present if pre in busy)
    suspected = sum(1 for pair in present if pair[0] in busy and pair not in observed_pairs)

    chains = _observed_chains(observed)
    erl, mean_run, run_count = expected_run_length(chains, present)
    return ReconstructionScore(
        observed_pairs=len(observed_pairs),
        recovered=recovered,
        split_errors=split_errors,
        recall=recall,
        suspected_merges=suspected,
        suspicion_denominator=denominator,
        expected_run_length=erl,
        mean_run_length=mean_run,
        runs=run_count,
    )


def sweep_threshold(
    snapshot: ConnectomeSnapshot,
    sites: Sequence[AmbiguousSite],
    observed: ObservedEdges,
    *,
    thresholds: Sequence[float] = (0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.45),
    margin: float = 0.2,
    weights: AffinityWeights | None = None,
) -> list[dict[str, Any]]:
    """The curve the operating point is chosen from.

    Each row is one threshold with what it bought and what it risked. Nothing
    here picks a winner: the choice depends on whether the reconstruction is
    about to be used for something a merge error would ruin.
    """
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        result = agglomerate(
            snapshot, sites, threshold=threshold, margin=margin, weights=weights
        )
        score = score_against_observation(result.snapshot, observed)
        row = {"threshold": threshold, **result.summary(), **score.as_json()}
        rows.append(row)
    return rows
