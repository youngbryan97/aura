from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path

import pytest

from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    PairwiseSelectionEvidence,
    build_necessary_condition_selector,
    necessary_condition_selector_from_dict,
)
from core.evidence.packet import observe
from core.learning.semantic_program_compositional_transducer import (
    compositional_semantic_program_transducer_from_dict,
)
from core.learning.semantic_program_path_ensemble import (
    EXECUTABLE_PROGRAM_CONDITION,
    EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
    build_semantic_program_path_ensemble,
    semantic_path_selection_values,
    semantic_program_path_ensemble_from_dict,
)
from core.learning.semantic_program_transducer import SemanticTransductionOutcome


def _selector():
    return build_necessary_condition_selector(
        (
            NecessaryEvidenceCondition(
                name=EXECUTABLE_PROGRAM_CONDITION,
                minimum=1.0,
                necessity_contract=EXECUTABLE_PROGRAM_NECESSITY_CONTRACT,
            ),
        )
    )


def _evidence(incumbent: float, challenger: float):
    return PairwiseSelectionEvidence.from_mappings(
        incumbent={EXECUTABLE_PROGRAM_CONDITION: incumbent},
        challenger={EXECUTABLE_PROGRAM_CONDITION: challenger},
        packet=observe(1.0, origin="runtime_test", ref=f"{incumbent}:{challenger}"),
    )


@pytest.mark.parametrize(
    ("incumbent", "challenger", "selected"),
    (
        (0.0, 0.0, "old"),
        (0.0, 1.0, "new"),
        (1.0, 0.0, "old"),
        (1.0, 1.0, "old"),
    ),
)
def test_selector_switches_only_when_challenger_repairs_a_necessary_condition(
    incumbent: float,
    challenger: float,
    selected: str,
) -> None:
    selector = _selector()
    replay = necessary_condition_selector_from_dict(selector.to_dict())
    evidence = _evidence(incumbent, challenger)

    decision = replay.select(
        incumbent="old",
        challenger="new",
        evidence=evidence,
    )

    assert decision.selected == selected
    assert decision.evidence.sources == evidence.packet.sources
    assert replay.to_dict() == selector.to_dict()


def test_exact_execution_contract_cannot_regress_a_correct_incumbent() -> None:
    selector = _selector()
    for incumbent_available, challenger_available, incumbent_correct in product(
        (False, True), repeat=3
    ):
        if incumbent_correct and not incumbent_available:
            continue
        decision = selector.select(
            incumbent="old",
            challenger="new",
            evidence=_evidence(
                float(incumbent_available),
                float(challenger_available),
            ),
        )
        assert not (incumbent_correct and decision.selected == "new")


def test_selector_rejects_tampering_missing_evidence_and_boolean_scores() -> None:
    selector = _selector()
    payload = copy.deepcopy(selector.to_dict())
    payload["conditions"][0]["minimum"] = 0.0

    with pytest.raises(ValueError, match="envelope"):
        necessary_condition_selector_from_dict(payload)
    with pytest.raises(ValueError, match="omits"):
        selector.select(
            incumbent="old",
            challenger="new",
            evidence=PairwiseSelectionEvidence.from_mappings(
                incumbent={"other": 0.0},
                challenger={"other": 1.0},
                packet=observe(1.0, origin="runtime_test", ref="missing"),
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        PairwiseSelectionEvidence.from_mappings(
            incumbent={EXECUTABLE_PROGRAM_CONDITION: False},
            challenger={EXECUTABLE_PROGRAM_CONDITION: 1.0},
            packet=observe(1.0, origin="runtime_test", ref="boolean"),
        )


def test_semantic_path_selection_preserves_quality_evidence_without_text() -> None:
    refused = SemanticTransductionOutcome(
        None,
        "arbitrary refusal text",
        {"input:0": 100.0},
        {"operation:0": 0.99},
    )

    values = semantic_path_selection_values(refused)

    assert values[EXECUTABLE_PROGRAM_CONDITION] == 0.0
    assert set(values) == {
        EXECUTABLE_PROGRAM_CONDITION,
        "argument_graph_mean",
        "argument_graph_margin",
        "argument_graph_runner_up_available",
        "input_pointer_mean",
        "input_pointer_min",
        "operation_pointer_mean",
        "operation_pointer_min",
        "operation_confidence_mean",
        "operation_confidence_min",
        "input_count",
        "instruction_count",
    }
    assert values["input_pointer_mean"] == 100.0
    assert values["input_pointer_min"] == 100.0
    assert values["input_count"] == 0.0
    assert values["instruction_count"] == 0.0


def test_real_semantic_paths_form_a_replayable_label_free_ensemble() -> None:
    root = Path(__file__).resolve().parent.parent

    def load(name: str):
        payload = json.loads(
            (
                root
                / "artifacts"
                / "rlc"
                / name
                / "transducer.json"
            ).read_text(encoding="ascii")
        )
        return compositional_semantic_program_transducer_from_dict(payload)

    incumbent = load("semantic_program_27b_compositional_v17_alias_local_dev")
    challenger = load("semantic_program_27b_compositional_v19_clause_local_identity_dev")

    ensemble = build_semantic_program_path_ensemble(incumbent, challenger)
    replay = semantic_program_path_ensemble_from_dict(ensemble.to_dict())

    assert replay.to_dict() == ensemble.to_dict()
    assert ensemble.composition_receipt["expected_answers_available_to_build"] is False
    assert ensemble.composition_receipt["path_coefficients_changed"] is False
    assert ensemble.composition_receipt["selection_contract"] == (
        EXECUTABLE_PROGRAM_NECESSITY_CONTRACT
    )

    with pytest.raises(ValueError, match="envelope"):
        build_semantic_program_path_ensemble(incumbent, incumbent)
