from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.learning.semantic_program_calibrated_path_replication import (
    adjudicate_calibrated_path_replication,
)
from core.learning.semantic_program_corpus import (
    build_semantic_program_natural_weave_replication_corpus,
)
from core.learning.semantic_program_natural_transfer import procedure_schema_signature
from tools.run_semantic_program_calibrated_path_replication import (
    _validate_calibration_report,
    _validated_output_path,
)


def _row(
    identity: str,
    *,
    answer: bool,
    selected: str = "incumbent",
    incumbent: bool | None = None,
    challenger: bool | None = None,
) -> dict:
    return {
        "source_text_sha256": identity,
        "answer_exact": answer,
        "program_exact": answer,
        "selected_path": selected,
        "incumbent_answer_exact": answer if incumbent is None else incumbent,
        "challenger_answer_exact": answer if challenger is None else challenger,
    }


def _passing_arms() -> dict[str, list[dict]]:
    identities = [f"task-{index}" for index in range(8)]
    incumbent = [False] * 5 + [True] * 3
    treatment = [
        _row(
            identity,
            answer=True,
            selected="challenger" if index < 5 else "incumbent",
            incumbent=incumbent[index],
            challenger=True,
        )
        for index, identity in enumerate(identities)
    ]
    incumbent_rows = [
        _row(identity, answer=value, incumbent=value, challenger=True)
        for identity, value in zip(identities, incumbent, strict=True)
    ]
    challenger_rows = [
        _row(identity, answer=True, selected="challenger", incumbent=value, challenger=True)
        for identity, value in zip(identities, incumbent, strict=True)
    ]
    return {
        "calibrated_path_ensemble": treatment,
        "frozen_incumbent": incumbent_rows,
        "frozen_challenger": challenger_rows,
        "necessary_condition_selector_lesion": [dict(row) for row in incumbent_rows],
        "forced_incumbent_selector_lesion": [dict(row) for row in incumbent_rows],
        "source_only_calibration_control": [dict(row) for row in incumbent_rows],
    }


def test_adjudication_requires_five_causal_repairs_and_beats_both_lesions() -> None:
    result = adjudicate_calibrated_path_replication(_passing_arms())

    assert result["mechanism_pass"] is True
    assert result["improvements_over_incumbent"] == 5
    assert result["regressions_from_incumbent"] == 0
    assert result["causal_improvement_contract_satisfied"] is True
    assert {
        name: test["one_sided_exact_p"] for name, test in result["paired_exact_tests"].items()
    } == {
        "forced_incumbent_selector_lesion": 0.03125,
        "necessary_condition_selector_lesion": 0.03125,
    }


def test_adjudication_rejects_a_regression_or_noncausal_repair() -> None:
    regression = _passing_arms()
    regression["calibrated_path_ensemble"][-1]["answer_exact"] = False
    assert adjudicate_calibrated_path_replication(regression)["mechanism_pass"] is False

    noncausal = _passing_arms()
    noncausal["calibrated_path_ensemble"][0]["challenger_answer_exact"] = False
    assert adjudicate_calibrated_path_replication(noncausal)["mechanism_pass"] is False


def test_adjudication_refuses_missing_or_task_mismatched_arms() -> None:
    incomplete = _passing_arms()
    incomplete.pop("source_only_calibration_control")
    with pytest.raises(ValueError, match="incomplete"):
        adjudicate_calibrated_path_replication(incomplete)

    mismatched = _passing_arms()
    mismatched["source_only_calibration_control"][0]["source_text_sha256"] = "different"
    with pytest.raises(ValueError, match="different tasks"):
        adjudicate_calibrated_path_replication(mismatched)


def test_weave_corpus_matches_the_committed_calibrated_preregistration() -> None:
    root = Path(__file__).resolve().parents[1]
    preregistration = json.loads(
        (
            root
            / "artifacts/rlc/semantic_program_27b_calibrated_path_replication_v1"
            / "preregistration.json"
        ).read_text("ascii")
    )
    corpus = build_semantic_program_natural_weave_replication_corpus()
    contract = preregistration["corpus"]

    assert preregistration["preregistered_before_generator_implementation"] is True
    assert preregistration["preregistered_before_target_generation"] is True
    assert len(corpus) == contract["max_examples"] == 48
    assert {item.topology_id for item in corpus} == set(contract["schemas"])
    assert {tuple(step.instruction.args for step in item.instructions) for item in corpus} == {
        tuple(tuple(pair) for pair in contract["operation_graph"]["arguments"])
    }
    assert {tuple(step.depends_on for step in item.instructions) for item in corpus} == {
        tuple(tuple(row) for row in contract["operation_graph"]["dependencies"])
    }
    assert len({procedure_schema_signature(item) for item in corpus}) == 24


def test_cli_validates_calibration_envelope_and_protects_the_feature_bundle(
    tmp_path: Path,
) -> None:
    body = {"schema": "test", "admitted": True}
    report_sha256 = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    report = {**body, "report_sha256": report_sha256}
    _validate_calibration_report(report, expected_report_sha256=report_sha256)
    with pytest.raises(ValueError, match="identity or admission"):
        _validate_calibration_report(report, expected_report_sha256="0" * 64)

    bundle = tmp_path / "features"
    bundle.mkdir()
    with pytest.raises(ValueError, match="immutable feature bundle"):
        _validated_output_path(bundle / "result.json", bundle=bundle)
