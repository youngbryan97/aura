from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.learning.semantic_program_frozen_path_replication import (
    adjudicate_frozen_path_replication,
)


def _row(index: int, *, exact: bool, receipt: str | None) -> dict[str, object]:
    return {
        "source_text_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
        "answer_exact": exact,
        "transducer_receipt_sha256": receipt,
    }


def test_preregistration_freezes_failed_selector_and_next_causal_bar() -> None:
    path = (
        Path(__file__).parents[1]
        / "artifacts/rlc/semantic_program_27b_frozen_path_replication_v1/preregistration.json"
    )
    value = json.loads(path.read_text(encoding="ascii"))
    assert value["discovery_evidence"]["calibrated_selector_verdict"] == (
        "FAIL_PREREGISTERED_MECHANISM_BAR"
    )
    assert value["frozen_transducer"]["coefficients_or_hyperparameters_may_change"] is False
    assert value["replication_corpus"]["generated_only_after_this_preregistration_is_committed"]
    assert value["ordinary_decode_deferred_until_mechanism_pass"] is True
    assert value["serving_authority"] is False


def test_adjudicator_accepts_powered_receipt_bound_causal_transfer() -> None:
    receipt = "a" * 64
    rows = {
        "frozen_transducer": [_row(i, exact=i < 20, receipt=receipt) for i in range(48)],
        "coefficient_lesion": [_row(i, exact=False, receipt="b" * 64) for i in range(48)],
        "hidden_token_shuffle": [_row(i, exact=False, receipt=None) for i in range(48)],
    }
    result = adjudicate_frozen_path_replication(
        rows,
        treatment_receipt_sha256=receipt,
    )
    assert result["mechanism_pass"] is True
    assert result["causal_treatment_receipt_contract_satisfied"] is True
    assert result["paired_exact_tests"]["coefficient_lesion"]["one_sided_exact_p"] < 0.05


@pytest.mark.parametrize("failure", ["underpowered", "wrong_receipt", "control_survives"])
def test_adjudicator_rejects_each_causal_failure(failure: str) -> None:
    receipt = "a" * 64
    exact = 17 if failure == "underpowered" else 20
    treatment = [_row(i, exact=i < exact, receipt=receipt) for i in range(48)]
    if failure == "wrong_receipt":
        treatment[0]["transducer_receipt_sha256"] = "c" * 64
    control_exact = 20 if failure == "control_survives" else 0
    rows = {
        "frozen_transducer": treatment,
        "coefficient_lesion": [
            _row(i, exact=i < control_exact, receipt="b" * 64) for i in range(48)
        ],
        "hidden_token_shuffle": [_row(i, exact=False, receipt=None) for i in range(48)],
    }
    result = adjudicate_frozen_path_replication(
        rows,
        treatment_receipt_sha256=receipt,
    )
    assert result["mechanism_pass"] is False


def test_adjudicator_rejects_mismatched_task_identities() -> None:
    receipt = "a" * 64
    rows = {
        arm: [_row(i, exact=False, receipt=receipt) for i in range(48)]
        for arm in ("frozen_transducer", "coefficient_lesion", "hidden_token_shuffle")
    }
    rows["hidden_token_shuffle"][0]["source_text_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different tasks"):
        adjudicate_frozen_path_replication(rows, treatment_receipt_sha256=receipt)
