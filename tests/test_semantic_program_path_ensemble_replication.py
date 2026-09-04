from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.semantic_program_corpus import (
    build_semantic_program_natural_branch_replication_corpus,
)
from core.learning.semantic_program_natural_transfer import procedure_schema_signature
from core.learning.semantic_program_path_ensemble_replication import paired_exact_test
from tools.run_semantic_program_path_ensemble_replication import _validated_output_path


def _row(identity: str, *, answer: bool, program: bool | None = None) -> dict:
    return {
        "source_text_sha256": identity,
        "answer_exact": answer,
        "program_exact": answer if program is None else program,
    }


def test_paired_exact_test_recounts_the_preregistered_five_case_bar() -> None:
    treatment = [_row(str(index), answer=True) for index in range(8)]
    control = [
        _row(str(index), answer=index >= 5)
        for index in range(8)
    ]

    result = paired_exact_test(treatment, control)

    assert result == {
        "metric": "answer_exact",
        "treatment_only": 5,
        "control_only": 0,
        "discordant": 5,
        "one_sided_exact_p": 0.03125,
    }


def test_paired_exact_test_refuses_task_or_metric_drift() -> None:
    with pytest.raises(ValueError, match="different tasks"):
        paired_exact_test([_row("a", answer=True)], [_row("b", answer=False)])
    with pytest.raises(ValueError, match="unsupported"):
        paired_exact_test(
            [_row("a", answer=True)],
            [_row("a", answer=False)],
            metric="accepted",
        )


def test_generated_branch_corpus_matches_the_committed_preregistration() -> None:
    root = Path(__file__).resolve().parents[1]
    preregistration = json.loads(
        (
            root
            / "artifacts/rlc/semantic_program_27b_path_ensemble_replication_v1"
            / "preregistration.json"
        ).read_text("ascii")
    )
    corpus = build_semantic_program_natural_branch_replication_corpus()
    contract = preregistration["corpus"]

    assert preregistration["preregistered_before_generator_implementation"] is True
    assert contract["seed"] == 2718281828
    assert len(corpus) == contract["max_examples"]
    assert {item.topology_id for item in corpus} == set(contract["schemas"])
    assert {
        tuple(step.instruction.args for step in item.instructions) for item in corpus
    } == {tuple(tuple(pair) for pair in contract["operation_graph"]["arguments"])}
    assert len({procedure_schema_signature(item) for item in corpus}) == 24


def test_replication_output_cannot_mutate_the_feature_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "features"
    bundle.mkdir()

    with pytest.raises(ValueError, match="immutable feature bundle"):
        _validated_output_path(bundle / "result.json", bundle=bundle)

    output = _validated_output_path(tmp_path / "results" / "result.json", bundle=bundle)

    assert output == tmp_path / "results" / "result.json"
