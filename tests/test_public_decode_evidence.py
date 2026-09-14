import hashlib

import pytest

from core.brain.llm.public_channel_decode import PublicChannelDecode
from core.learning.public_decode_evidence import (
    public_decode_coverage,
    validate_public_decode_receipt,
)


def receipt(text="answer", reason="eos", count=4, closed=True):
    return PublicChannelDecode(
        text, count, 0, 0, 10, reason, True, closed, 0,
        hashlib.sha256(b"").hexdigest(), "a" * 64,
    ).receipt()


def test_uncensored_wrong_public_answer_is_measured_not_excluded():
    assert validate_public_decode_receipt(receipt("wrong"), "wrong", max_tokens=32)


def test_private_cap_is_censored_not_a_completed_incorrect_answer():
    assert not validate_public_decode_receipt(
        receipt("", "token_limit", 32, False), "", max_tokens=32,
    )


@pytest.mark.parametrize(("field", "value"), [
    ("policy", "legacy"), ("generated_tokens", True), ("generated_tokens", 33),
    ("stop_reason", "unknown"), ("boundary_closed", False), ("boundary_tokens", 1),
    ("public_text_sha256", "a" * 64), ("reasoning_sha256", "b" * 64),
])
def test_inconsistent_decode_evidence_fails(field, value):
    data = receipt()
    data[field] = value
    with pytest.raises(ValueError):
        validate_public_decode_receipt(data, "answer", max_tokens=32)


def test_retry_censoring_cannot_disappear_behind_successful_last_attempt():
    result = public_decode_coverage([
        {"arm": "treatment", "attempts": [
            {"decode": receipt("", "token_limit", 32, False), "raw_response": ""},
            {"decode": receipt("answer"), "raw_response": "answer"},
        ]},
        {"arm": "ordinary_base", "attempts": [{"decode": receipt("wrong"), "raw_response": "wrong"}]},
    ], max_tokens=32)
    assert result["attempts_by_arm"] == {"ordinary_base": 1, "treatment": 2}
    assert result["censored_attempts_by_arm"] == {"ordinary_base": 0, "treatment": 1}
    assert not result["uncensored"]
