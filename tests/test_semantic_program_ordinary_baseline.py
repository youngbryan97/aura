from __future__ import annotations

import hashlib

import pytest

from core.learning.semantic_program_ordinary_baseline import (
    adjudicate_ordinary_product_bar,
    best_possible_product_test,
    parse_integral_numeric_claim,
    product_bar_is_reachable,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The result is 2,284,654.", 2_284_654),
        ("Work: 3 then 4\nAnswer: -12", -12),
        ("Therefore 12.0", 12),
        ("The exact value is 1.2e3", 1200),
        ("The result is 24/2", 12),
        ("The result is 7/2", None),
        ("I cannot determine it", None),
    ],
)
def test_integral_parser_is_form_tolerant_and_value_strict(text: str, expected: int | None) -> None:
    assert parse_integral_numeric_claim(text) == expected


def _row(index: int, *, exact: bool) -> dict[str, object]:
    return {
        "source_text_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        "answer_exact": exact,
        "response_text": str(index),
        "response_sha256": "a" * 64,
    }


def test_best_possible_test_assumes_every_remaining_treatment_win_survives() -> None:
    treatment = [_row(index, exact=index < 21) for index in range(48)]
    ordinary = [_row(index, exact=False) for index in range(4)]
    best = best_possible_product_test(treatment, ordinary)
    assert best["treatment_only"] == 21
    assert best["control_only"] == 0
    assert product_bar_is_reachable(treatment, ordinary) is True


def test_futility_detects_when_ordinary_has_too_many_unique_wins() -> None:
    treatment = [_row(index, exact=index < 21) for index in range(48)]
    ordinary = [_row(index, exact=index >= 21) for index in range(32)]
    assert product_bar_is_reachable(treatment, ordinary) is False


def test_product_bar_passes_and_retains_each_ordinary_incumbent() -> None:
    treatment = [_row(index, exact=index < 21) for index in range(48)]
    ordinary = [_row(index, exact=False) for index in range(48)]
    result = adjudicate_ordinary_product_bar(treatment, ordinary)
    assert result["product_bar_pass"] is True
    assert result["ordinary_exact_incumbents_retained"] is True
    assert result["composition_policy"]["semantic_replacement_authorized_by_this_evaluator"] is False


def test_product_bar_rejects_an_ordinary_control_that_matches_treatment() -> None:
    treatment = [_row(index, exact=index < 21) for index in range(48)]
    ordinary = [_row(index, exact=index < 21) for index in range(48)]
    result = adjudicate_ordinary_product_bar(treatment, ordinary)
    assert result["product_bar_pass"] is False


def test_product_bar_rejects_mismatched_task_order() -> None:
    treatment = [_row(index, exact=False) for index in range(48)]
    ordinary = [_row(index, exact=False) for index in range(48)]
    ordinary[0]["source_text_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identities differ"):
        adjudicate_ordinary_product_bar(treatment, ordinary)
