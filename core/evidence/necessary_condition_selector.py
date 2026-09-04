"""Select a challenger only when it repairs a necessary-condition failure.

This selector does not estimate correctness. A consumer declares evidence that
is mechanically necessary for its own notion of success. The challenger can
replace the incumbent only when the incumbent lacks that evidence and the
challenger has all of it. If both satisfy the conditions, the incumbent keeps
the decision.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.evidence.packet import EvidencePacket, derive

NECESSARY_CONDITION_SELECTOR_SCHEMA: Final = (
    "aura.evidence.necessary_condition_selector.v1"
)
NECESSARY_CONDITION_SELECTOR_RECEIPT_SCHEMA: Final = (
    "aura.evidence.necessary_condition_selector_receipt.v1"
)
CANDIDATE_SELECTION_DECISION_SCHEMA: Final = (
    "aura.evidence.candidate_selection_decision.v1"
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _finite(value: Any, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class NecessaryEvidenceCondition:
    """One measured condition that success requires."""

    name: str
    minimum: float
    necessity_contract: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.necessity_contract, str)
            or not self.necessity_contract
        ):
            raise ValueError("necessary evidence condition is invalid")
        object.__setattr__(
            self,
            "minimum",
            _finite(self.minimum, field=f"minimum for {self.name!r}"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "necessity_contract": self.necessity_contract,
        }


@dataclass(frozen=True, slots=True)
class PairwiseSelectionEvidence:
    """Comparable incumbent and challenger measurements with provenance."""

    incumbent: tuple[tuple[str, float], ...]
    challenger: tuple[tuple[str, float], ...]
    packet: EvidencePacket

    def __post_init__(self) -> None:
        incumbent_names = tuple(name for name, _value in self.incumbent)
        challenger_names = tuple(name for name, _value in self.challenger)
        if (
            not self.incumbent
            or incumbent_names != tuple(sorted(incumbent_names))
            or challenger_names != incumbent_names
            or len(set(incumbent_names)) != len(incumbent_names)
            or any(not isinstance(name, str) or not name for name in incumbent_names)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for _name, value in (*self.incumbent, *self.challenger)
            )
            or not self.packet.sources
            or self.packet.mass <= 0.0
        ):
            raise ValueError("pairwise selection evidence is invalid")

    @classmethod
    def from_mappings(
        cls,
        *,
        incumbent: Mapping[str, float],
        challenger: Mapping[str, float],
        packet: EvidencePacket,
    ) -> PairwiseSelectionEvidence:
        if not isinstance(incumbent, Mapping) or not isinstance(challenger, Mapping):
            raise ValueError("pairwise selection values must be mappings")

        def normalize(
            values: Mapping[str, float], *, role: str
        ) -> tuple[tuple[str, float], ...]:
            rows = []
            for name, value in values.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{role} evidence name is invalid")
                rows.append((name, _finite(value, field=f"{role} evidence {name!r}")))
            return tuple(sorted(rows))

        return cls(
            incumbent=normalize(incumbent, role="incumbent"),
            challenger=normalize(challenger, role="challenger"),
            packet=packet,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _value in self.incumbent)

    @property
    def evidence_sha256(self) -> str:
        return _sha(
            {
                "incumbent": [list(row) for row in self.incumbent],
                "challenger": [list(row) for row in self.challenger],
            }
        )

    def values(self, role: str) -> dict[str, float]:
        if role == "incumbent":
            return dict(self.incumbent)
        if role == "challenger":
            return dict(self.challenger)
        raise ValueError("selection role is invalid")


@dataclass(frozen=True, slots=True)
class CandidateSelectionDecision:
    """One replayable incumbent/challenger decision."""

    selected: str
    evidence: EvidencePacket
    receipt: dict[str, Any]

    def __post_init__(self) -> None:
        body = {
            key: value
            for key, value in self.receipt.items()
            if key != "receipt_sha256"
        }
        if (
            not self.selected
            or self.receipt.get("schema") != CANDIDATE_SELECTION_DECISION_SCHEMA
            or self.receipt.get("selected") != self.selected
            or self.receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("candidate selection decision is invalid")


@dataclass(frozen=True, slots=True)
class NecessaryConditionSelector:
    """A pairwise selector whose switch is justified by necessary evidence."""

    conditions: tuple[NecessaryEvidenceCondition, ...]
    construction_receipt: dict[str, Any]
    schema: str = NECESSARY_CONDITION_SELECTOR_SCHEMA

    def __post_init__(self) -> None:
        names = tuple(condition.name for condition in self.conditions)
        body = {
            key: value
            for key, value in self.construction_receipt.items()
            if key != "receipt_sha256"
        }
        if (
            self.schema != NECESSARY_CONDITION_SELECTOR_SCHEMA
            or not self.conditions
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
            or self.construction_receipt.get("schema")
            != NECESSARY_CONDITION_SELECTOR_RECEIPT_SCHEMA
            or self.construction_receipt.get("conditions")
            != [condition.to_dict() for condition in self.conditions]
            or self.construction_receipt.get("labels_available_to_runtime") is not False
            or self.construction_receipt.get("text_available_to_selector") is not False
            or self.construction_receipt.get("domain_identity_available_to_selector")
            is not False
            or self.construction_receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("necessary condition selector envelope is invalid")
        object.__setattr__(
            self,
            "construction_receipt",
            json.loads(json.dumps(self.construction_receipt)),
        )

    @property
    def receipt_sha256(self) -> str:
        return str(self.construction_receipt["receipt_sha256"])

    def select(
        self,
        *,
        incumbent: str,
        challenger: str,
        evidence: PairwiseSelectionEvidence,
    ) -> CandidateSelectionDecision:
        if not incumbent or not challenger or incumbent == challenger:
            raise ValueError("selection candidates are invalid")
        required = {condition.name for condition in self.conditions}
        if not required.issubset(evidence.names):
            raise ValueError("selection evidence omits a necessary condition")
        incumbent_values = evidence.values("incumbent")
        challenger_values = evidence.values("challenger")
        incumbent_missing = tuple(
            condition.name
            for condition in self.conditions
            if incumbent_values[condition.name] < condition.minimum
        )
        challenger_missing = tuple(
            condition.name
            for condition in self.conditions
            if challenger_values[condition.name] < condition.minimum
        )
        selected = (
            challenger if incumbent_missing and not challenger_missing else incumbent
        )
        result_evidence = derive(
            1.0,
            (evidence.packet,),
            subject=evidence.packet.subject,
            produced_by="core.evidence.necessary_condition_selector",
        )
        body = {
            "schema": CANDIDATE_SELECTION_DECISION_SCHEMA,
            "selector_receipt_sha256": self.receipt_sha256,
            "evidence_sha256": evidence.evidence_sha256,
            "evidence_sources": sorted(evidence.packet.sources),
            "incumbent": incumbent,
            "challenger": challenger,
            "incumbent_missing": list(incumbent_missing),
            "challenger_missing": list(challenger_missing),
            "selected": selected,
            "reason": (
                "challenger_repairs_necessary_condition_failure"
                if selected == challenger
                else "incumbent_retained_without_proven_necessary_condition_repair"
            ),
        }
        return CandidateSelectionDecision(
            selected=selected,
            evidence=result_evidence,
            receipt={**body, "receipt_sha256": _sha(body)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "construction_receipt": self.construction_receipt,
        }


def build_necessary_condition_selector(
    conditions: Sequence[NecessaryEvidenceCondition],
) -> NecessaryConditionSelector:
    normalized = tuple(sorted(conditions, key=lambda condition: condition.name))
    if not normalized:
        raise ValueError("necessary condition selector requires conditions")
    receipt_body = {
        "schema": NECESSARY_CONDITION_SELECTOR_RECEIPT_SCHEMA,
        "algorithm": "necessary_condition_repair_v1",
        "conditions": [condition.to_dict() for condition in normalized],
        "labels_available_to_runtime": False,
        "text_available_to_selector": False,
        "domain_identity_available_to_selector": False,
        "tie_policy": "retain_incumbent",
        "unmeasured_policy": "refuse_selection",
    }
    return NecessaryConditionSelector(
        conditions=normalized,
        construction_receipt={
            **receipt_body,
            "receipt_sha256": _sha(receipt_body),
        },
    )


def necessary_condition_selector_from_dict(value: Any) -> NecessaryConditionSelector:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "conditions",
        "construction_receipt",
    }:
        raise ValueError("necessary condition selector payload is invalid")
    raw_conditions = value["conditions"]
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise ValueError("necessary condition selector conditions are invalid")
    conditions = []
    for raw in raw_conditions:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "minimum",
            "necessity_contract",
        }:
            raise ValueError("necessary evidence condition payload is invalid")
        conditions.append(
            NecessaryEvidenceCondition(
                name=str(raw["name"]),
                minimum=raw["minimum"],
                necessity_contract=str(raw["necessity_contract"]),
            )
        )
    return NecessaryConditionSelector(
        schema=str(value["schema"]),
        conditions=tuple(conditions),
        construction_receipt=dict(value["construction_receipt"]),
    )


__all__ = [
    "CANDIDATE_SELECTION_DECISION_SCHEMA",
    "NECESSARY_CONDITION_SELECTOR_RECEIPT_SCHEMA",
    "NECESSARY_CONDITION_SELECTOR_SCHEMA",
    "CandidateSelectionDecision",
    "NecessaryConditionSelector",
    "NecessaryEvidenceCondition",
    "PairwiseSelectionEvidence",
    "build_necessary_condition_selector",
    "necessary_condition_selector_from_dict",
]
