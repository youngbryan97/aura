"""Proof-preserving composition of complete semantic-program pathways."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from core.evidence.necessary_condition_selector import (
    CandidateSelectionDecision,
    NecessaryConditionSelector,
    NecessaryEvidenceCondition,
    PairwiseSelectionEvidence,
    build_necessary_condition_selector,
    necessary_condition_selector_from_dict,
)
from core.evidence.packet import fuse, observe
from core.learning.semantic_program_compositional_transducer import (
    CompositionalSemanticProgramTransducer,
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_transducer import SemanticTransductionOutcome

SEMANTIC_PATH_ENSEMBLE_SCHEMA: Final = "aura.semantic_program_path_ensemble.v1"
SEMANTIC_PATH_ENSEMBLE_RECEIPT_SCHEMA: Final = (
    "aura.semantic_program_path_ensemble_receipt.v1"
)
EXECUTABLE_PROGRAM_CONDITION: Final = "executable_program"
EXECUTABLE_PROGRAM_NECESSITY_CONTRACT: Final = (
    "semantic_exact_execution_requires_executable_ir"
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


def semantic_path_selection_values(
    outcome: SemanticTransductionOutcome,
) -> dict[str, float]:
    """Return evidence that is logically necessary for exact execution."""

    return {EXECUTABLE_PROGRAM_CONDITION: float(outcome.ir is not None)}


@dataclass(frozen=True, slots=True)
class ArbitratedSemanticTransduction:
    outcome: SemanticTransductionOutcome
    decision: CandidateSelectionDecision
    incumbent: SemanticTransductionOutcome
    challenger: SemanticTransductionOutcome


@dataclass(frozen=True, slots=True)
class SemanticProgramPathEnsemble:
    """Two immutable pathways composed by a necessary-evidence selector."""

    incumbent: CompositionalSemanticProgramTransducer
    challenger: CompositionalSemanticProgramTransducer
    selector: NecessaryConditionSelector
    composition_receipt: dict[str, Any]
    schema: str = SEMANTIC_PATH_ENSEMBLE_SCHEMA

    def __post_init__(self) -> None:
        body = {
            key: value
            for key, value in self.composition_receipt.items()
            if key != "receipt_sha256"
        }
        if (
            self.schema != SEMANTIC_PATH_ENSEMBLE_SCHEMA
            or self.incumbent.model_basis_sha256 != self.challenger.model_basis_sha256
            or self.incumbent.hidden_channels != self.challenger.hidden_channels
            or self.incumbent.hidden_channel_widths
            != self.challenger.hidden_channel_widths
            or self.incumbent.receipt_sha256 == self.challenger.receipt_sha256
            or self.composition_receipt.get("schema")
            != SEMANTIC_PATH_ENSEMBLE_RECEIPT_SCHEMA
            or self.composition_receipt.get("incumbent_receipt_sha256")
            != self.incumbent.receipt_sha256
            or self.composition_receipt.get("challenger_receipt_sha256")
            != self.challenger.receipt_sha256
            or self.composition_receipt.get("selector_receipt_sha256")
            != self.selector.receipt_sha256
            or self.composition_receipt.get("expected_answers_available_to_runtime")
            is not False
            or self.composition_receipt.get("expected_answers_available_to_build")
            is not False
            or self.composition_receipt.get("text_available_to_selector") is not False
            or self.composition_receipt.get("domain_identity_available_to_selector")
            is not False
            or self.composition_receipt.get("receipt_sha256") != _sha(body)
        ):
            raise ValueError("semantic path ensemble envelope is invalid")
        object.__setattr__(
            self,
            "composition_receipt",
            json.loads(json.dumps(self.composition_receipt)),
        )

    @property
    def model_basis_sha256(self) -> str:
        return self.incumbent.model_basis_sha256

    @property
    def receipt_sha256(self) -> str:
        return str(self.composition_receipt["receipt_sha256"])

    def decode_with_receipt(
        self,
        *,
        source_token_ids: Sequence[int],
        hidden_states: Any,
        public_inputs: Sequence[Any],
        source_text_sha256: str,
        model_basis_sha256: str,
    ) -> ArbitratedSemanticTransduction:
        arguments = {
            "source_token_ids": source_token_ids,
            "hidden_states": hidden_states,
            "public_inputs": public_inputs,
            "source_text_sha256": source_text_sha256,
            "model_basis_sha256": model_basis_sha256,
        }
        incumbent = self.incumbent.decode(**arguments)
        challenger = self.challenger.decode(**arguments)
        packets = (
            observe(
                1.0,
                origin="semantic_path_incumbent",
                ref=_sha(
                    {
                        "receipt": self.incumbent.receipt_sha256,
                        "source": source_text_sha256,
                        "selection_values": semantic_path_selection_values(incumbent),
                    }
                ),
                subject=f"semantic_path_selection:{source_text_sha256}",
            ),
            observe(
                1.0,
                origin="semantic_path_challenger",
                ref=_sha(
                    {
                        "receipt": self.challenger.receipt_sha256,
                        "source": source_text_sha256,
                        "selection_values": semantic_path_selection_values(challenger),
                    }
                ),
                subject=f"semantic_path_selection:{source_text_sha256}",
            ),
        )
        decision = self.selector.select(
            incumbent="incumbent",
            challenger="challenger",
            evidence=PairwiseSelectionEvidence.from_mappings(
                incumbent=semantic_path_selection_values(incumbent),
                challenger=semantic_path_selection_values(challenger),
                packet=fuse(packets),
            ),
        )
        outcome = challenger if decision.selected == "challenger" else incumbent
        return ArbitratedSemanticTransduction(
            outcome=outcome,
            decision=decision,
            incumbent=incumbent,
            challenger=challenger,
        )

    def decode(self, **kwargs: Any) -> SemanticTransductionOutcome:
        return self.decode_with_receipt(**kwargs).outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "incumbent": self.incumbent.to_dict(),
            "challenger": self.challenger.to_dict(),
            "selector": self.selector.to_dict(),
            "composition_receipt": self.composition_receipt,
        }


def build_semantic_program_path_ensemble(
    incumbent: CompositionalSemanticProgramTransducer,
    challenger: CompositionalSemanticProgramTransducer,
) -> SemanticProgramPathEnsemble:
    """Compose two intact paths without labels, fitting, or changed weights."""

    selector = build_necessary_condition_selector(
        (
            NecessaryEvidenceCondition(
                name=EXECUTABLE_PROGRAM_CONDITION,
                minimum=1.0,
                necessity_contract=EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
            ),
        )
    )
    receipt_body = {
        "schema": SEMANTIC_PATH_ENSEMBLE_RECEIPT_SCHEMA,
        "incumbent_receipt_sha256": incumbent.receipt_sha256,
        "challenger_receipt_sha256": challenger.receipt_sha256,
        "selector_receipt_sha256": selector.receipt_sha256,
        "path_coefficients_changed": False,
        "expected_answers_available_to_build": False,
        "expected_answers_available_to_runtime": False,
        "text_available_to_selector": False,
        "domain_identity_available_to_selector": False,
        "selection_contract": EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
        "selection_policy": "necessary_condition_repair",
        "tie_policy": "retain_incumbent",
        "serving_authority": False,
    }
    return SemanticProgramPathEnsemble(
        incumbent=incumbent,
        challenger=challenger,
        selector=selector,
        composition_receipt={
            **receipt_body,
            "receipt_sha256": _sha(receipt_body),
        },
    )


def semantic_program_path_ensemble_from_dict(value: Any) -> SemanticProgramPathEnsemble:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "incumbent",
        "challenger",
        "selector",
        "composition_receipt",
    }:
        raise ValueError("semantic program path ensemble payload is invalid")
    return SemanticProgramPathEnsemble(
        schema=str(value["schema"]),
        incumbent=compositional_semantic_program_transducer_from_dict(value["incumbent"]),
        challenger=compositional_semantic_program_transducer_from_dict(value["challenger"]),
        selector=necessary_condition_selector_from_dict(value["selector"]),
        composition_receipt=dict(value["composition_receipt"]),
    )


__all__ = [
    "EXECUTABLE_PROGRAM_CONDITION",
    "EXECUTABLE_PROGRAM_NECESSITY_CONTRACT",
    "SEMANTIC_PATH_ENSEMBLE_RECEIPT_SCHEMA",
    "SEMANTIC_PATH_ENSEMBLE_SCHEMA",
    "ArbitratedSemanticTransduction",
    "SemanticProgramPathEnsemble",
    "build_semantic_program_path_ensemble",
    "semantic_path_selection_values",
    "semantic_program_path_ensemble_from_dict",
]
