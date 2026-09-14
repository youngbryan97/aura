"""Formatting tolerance cannot manufacture an answer or erase ambiguity."""

import pytest
from pathlib import Path

from tools.audit_composition_decode_semantics import audit, public_values


@pytest.mark.parametrize("text", [
    '{"r0":2,"s0":3}',
    'FINAL_ANSWER:{"r0":2,"s0":3}',
    '```json\n{"r0":2,"s0":3}\n```',
    '```\n{"s0":3,"r0":2}\n```',
    'FINAL_ANSWER:\n```json\n{"s0":3,"r0":2}\n```',
])
def test_only_envelope_and_order_are_normalized(text):
    assert public_values(text, ("r0", "s0")) == {"r0": 2, "s0": 3}


@pytest.mark.parametrize("text", [
    '{"r0":2,"s0":3,"r0":4}',
    '{"r0":2,"s0":true}',
    '{"r0":2,"s0":3.0}',
    '{"r0":2,"s0":"3"}',
    '{"r0":2,"s0":3,"extra":0}',
    '{"r0":2}',
    '{"r0":2,"s0":3} {"r0":4,"s0":3}',
    'Here is my answer: {"r0":2,"s0":3}',
    '```json\n{"r0":2,"s0":3}\n```\nActually no.',
    'FINAL_ANSWER:FINAL_ANSWER:{"r0":2,"s0":3}',
    r'FINAL_ANSWER:{\"r0\":2,\"s0\":3}',
    '{"r0":NaN,"s0":3}',
    ' ' * 16385,
])
def test_ambiguous_malformed_or_wrongly_typed_answers_are_not_repaired(text):
    assert public_values(text, ("r0", "s0")) is None


def test_incorrect_values_remain_incorrect():
    assert public_values('{"r0":999,"s0":0}', ("r0", "s0")) == {"r0": 999, "s0": 0}


@pytest.mark.live
def test_retained_campaign_is_regraded_without_mutating_its_evidence():
    root = Path(__file__).resolve().parents[1] / "artifacts/closeout/latent_cortex/public_composition_diagnostic_20260914"
    report, journal = root / "result.json", root / "journal.jsonl"
    before = report.read_bytes(), journal.read_bytes()
    result = audit(report, journal)
    assert len(result["rows"]) == 48
    assert result["arms"]["treatment"]["semantic_correct"] == 8
    assert result["arms"]["ordinary_base"]["semantic_parsed"] == 6
    assert result["arms"]["ordinary_base"]["semantic_correct"] == 0
    assert result["serving_authority"] is False
    assert before == (report.read_bytes(), journal.read_bytes())
