"""Apply an existing pairwise evidence selector without dropping alternatives."""

from collections.abc import Mapping
from dataclasses import dataclass

from core.evidence.necessary_condition_selector import (
    CandidateSelectionDecision,
    PairwiseSelectionEvidence,
)
from core.evidence.packet import EvidencePacket, fuse


@dataclass(frozen=True)
class CandidatePortfolioDecision:
    selected: str
    candidate_order: tuple[str, ...]
    comparisons: tuple[CandidateSelectionDecision, ...]


def select_candidate_portfolio(selector, *, incumbent: str,
                               measurements: Mapping[str, Mapping[str, float]],
                               provenance: Mapping[str, EvidencePacket]) -> CandidatePortfolioDecision:
    """Replay a declared candidate order using measured evidence only.

    All candidates remain available to the caller. The caller fixes order
    before scoring; neither an answer key nor a benchmark identity is an
    input. Different valid answers can still disagree after this selection.
    """
    if (not measurements or incumbent not in measurements or set(measurements) != set(provenance)
            or any(not isinstance(name, str) or not name for name in measurements)):
        raise ValueError("portfolio needs identified candidates and complete provenance")
    if len({packet.subject for packet in provenance.values()}) != 1:
        raise ValueError("portfolio candidates describe different observations")
    # Validate every row, even a sole incumbent or an alternative never chosen.
    for name, values in measurements.items():
        PairwiseSelectionEvidence.from_mappings(
            incumbent=measurements[incumbent], challenger=values, packet=provenance[name])
    order = (incumbent, *(name for name in measurements if name != incumbent))
    selected, decisions = incumbent, []
    for name in order[1:]:
        evidence = PairwiseSelectionEvidence.from_mappings(
            incumbent=measurements[selected], challenger=measurements[name],
            packet=fuse((provenance[selected], provenance[name])),
        )
        decision = selector.select(incumbent=selected, challenger=name, evidence=evidence)
        if decision.selected not in {selected, name}:
            raise ValueError("pairwise selector chose an absent candidate")
        decisions.append(decision)
        selected = decision.selected
    return CandidatePortfolioDecision(selected, order, tuple(decisions))
