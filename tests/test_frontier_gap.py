"""Battery-level tests; v5 trust-chain tests live in test_frontier_evidence_v5."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path

import pytest

from core.brain.frontier_gap import (
    BATTERY_VERSION,
    SCHEMA_VERSION,
    ClassResult,
    build_battery,
    run_battery,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts" / "frontier_gap" / "latest.json"


def _answer_for(item) -> str:
    if item.task_class == "math":
        left, right = re.search(r"Compute (\d+) \* (\d+)", item.prompt).groups()
        return str(int(left) * int(right))
    if item.task_class == "reasoning":
        return re.findall(r"([A-Z][a-z]+) is older than", item.prompt)[0]
    if item.task_class == "coding":
        function_name = re.search(r"`([a-z0-9_]+)\(xs\)`", item.prompt).group(1)
        operation = function_name.partition("_")[0]
        return f"def {function_name}(xs):\n    return {operation}(xs)"
    candidates = (
        "Au", "Mars", "6", "Tokyo", "carbon dioxide", "Na", "Pacific Ocean",
        "90", "Nairobi", "oxygen", "barometer", "2", "South America", "ampere",
        "evaporation", "Wellington",
    )
    return next(candidate for candidate in candidates if item.grade(candidate))


def test_battery_is_challenge_derived_unique_and_seed_sensitive() -> None:
    first = build_battery(seed=1, per_class=5, challenge_nonce=b"a" * 32)
    second = build_battery(seed=2, per_class=5, challenge_nonce=b"b" * 32)
    assert len(first) == len(second) == 20
    assert len({item.item_id for item in first}) == 20
    assert len({hashlib.sha256(item.prompt.encode()).hexdigest() for item in first}) == 20
    assert [item.prompt for item in first] != [item.prompt for item in second]


def test_graders_discriminate_exact_answers_and_restricted_code() -> None:
    items = build_battery(
        seed=5, per_class=3, challenge_nonce=hashlib.sha256(b"gap-test-nonce").digest()
    )
    for item in items:
        assert item.grade(_answer_for(item)) is True
        assert item.grade("definitely incorrect") is False
        assert re.fullmatch(r"[0-9a-f]{64}", item.grader_implementation_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", item.expected_answer_commitment_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", item.hidden_case_commitment_sha256)
    coding = next(item for item in items if item.task_class == "coding")
    assert coding.grade("```python\nimport os\ndef bad(xs): return 999\n```") is False


def test_gap_is_zero_at_parity_and_grows_below_it() -> None:
    assert ClassResult("x", 10, 10, 0.9).gap == 0.0
    assert ClassResult("x", 10, 9, 0.9).gap == 0.0
    assert ClassResult("x", 10, 5, 1.0).gap == pytest.approx(0.5)


def test_unreferenced_run_is_diagnostic_only_even_when_perfect() -> None:
    items = build_battery(seed=7, per_class=2)
    cursor = 0

    async def perfect(prompt, task_type):
        nonlocal cursor
        item = items[cursor]
        cursor += 1
        assert (prompt, task_type) == (item.prompt, item.task_type)
        return _answer_for(item)

    report = asyncio.run(
        run_battery(perfect, seed=7, per_class=2, grade_to_foundry=False)
    )
    assert report["overall_candidate_score"] == 1.0
    assert report["overall_gap"] is None
    assert report["reference_basis"] == "unavailable"
    assert report["general_frontier_claim_eligible"] is False
    assert report["schema_version"] == SCHEMA_VERSION == 5
    assert report["battery_version"] == BATTERY_VERSION


def test_checked_in_v5_artifact_is_a_control_not_a_capability_claim() -> None:
    envelope = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    artifact = envelope.get("payload", envelope)
    assert artifact["schema"] == "aura.frontier_gap_report.v5"
    assert artifact["general_frontier_claim_eligible"] is False
    assert artifact["capability_claim_eligible"] is False
    assert artifact["evidence_class"] == "synthetic_pipeline_control"
    assert artifact["capability_ledger"]["runs"] == []
    assert artifact["rejected_ledger"]["runs"] == []
    assert artifact["control_ledger"]["runs"]
    assert artifact["legacy_artifact_sha256"]

    latest = artifact["latest_evidence"]
    digest = latest["evidence_sha256"]
    evidence_path = ARTIFACT.parent / "evidence-v5" / f"{digest}.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["evidence_sha256"] == digest
    assert evidence["payload"]["evidence_class"] == "synthetic_pipeline_control"
    assert evidence["payload"]["capability_claim_eligible"] is False
    assert evidence["payload"]["general_frontier_claim_eligible"] is False


def test_a_grader_reads_the_answer_out_of_the_contract_envelope():
    """A sealed measurement asks for the strict answer contract, which returns
    ``<answer>Tokyo</answer>``.

    These graders read the raw output. So the text grader normalised that to
    "answerTokyoanswer", the integer grader's full match failed on the tags,
    and the code grader found no fenced block — every class failing on the
    wrapper with the model answering correctly. Measured 2026-08-30 against the
    real model: candidate 0.0 before, 0.65 after, on the same battery.
    """
    from core.brain.frontier_gap import (
        _exact_integer_grader,
        _exact_text_grader,
        _extract_python,
    )

    says_tokyo = _exact_text_grader("Tokyo")
    assert says_tokyo("<answer>Tokyo</answer>")
    assert says_tokyo("Tokyo")
    assert not says_tokyo("<answer>Osaka</answer>")

    says_the_number = _exact_integer_grader(258662)
    assert says_the_number("<answer>258662</answer>")
    assert says_the_number("258662")
    assert not says_the_number("<answer>258663</answer>")
    assert not says_the_number("<answer>about 258662</answer>")

    assert _extract_python(
        "<answer>```python\ndef f(xs): return sum(xs)\n```</answer>"
    ) == "def f(xs): return sum(xs)"


def test_an_answer_with_no_envelope_grades_exactly_as_it_did():
    """Nothing that graded before grades differently now."""
    from core.brain.frontier_gap import _exact_integer_grader, _exact_text_grader

    assert _exact_text_grader("Mars")("Mars")
    assert not _exact_text_grader("Mars")("Venus")
    assert _exact_integer_grader(42)("42")
    assert not _exact_integer_grader(42)("42 apples")
