from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.learning.semantic_program_corpus import build_semantic_program_natural_request_corpus
from core.learning.semantic_program_natural_transfer import (
    build_natural_request_transfer_preflight,
    procedure_schema_signature,
)

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "artifacts/rlc/semantic_program_27b_compositional_v14"


def _frozen(name: str) -> dict:
    return json.loads((_ARTIFACT / name).read_text("ascii"))


def test_frozen_tissue_never_saw_the_natural_request_schemas() -> None:
    examples = build_semantic_program_natural_request_corpus()

    report = build_natural_request_transfer_preflight(
        examples=examples,
        transducer=_frozen("transducer.json"),
        source_campaign=_frozen("source_campaign.json"),
        frozen_verification=_frozen("verification.json"),
    )

    assert report["example_count"] == 24
    assert report["target_schema_count"] == 24
    assert report["target_source_schema_overlap"] == 0
    assert report["fit_or_refit_calls"] == 0
    assert report["expected_answers_available_to_decode"] is False
    assert report["unsupported_operations"] == []


def test_schema_identity_ignores_values_and_words_but_not_computation() -> None:
    first = build_semantic_program_natural_request_corpus(seed=1)
    second = build_semantic_program_natural_request_corpus(seed=2)

    assert procedure_schema_signature(first[0]) == procedure_schema_signature(second[0])
    assert procedure_schema_signature(first[0]) != procedure_schema_signature(first[1])


def test_preflight_refuses_a_schema_that_was_in_fitting() -> None:
    from core.learning.semantic_program_corpus import build_semantic_program_corpus

    source_example = next(item for item in build_semantic_program_corpus() if item.split != "train")
    with pytest.raises(ValueError, match="present in fitting"):
        build_natural_request_transfer_preflight(
            examples=(source_example,),
            transducer=_frozen("transducer.json"),
            source_campaign=_frozen("source_campaign.json"),
            frozen_verification=_frozen("verification.json"),
        )
