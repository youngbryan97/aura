"""Validate channel and resource receipts without loading a model or scoring it."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from core.brain.llm.public_channel_decode import PUBLIC_CHANNEL_DECODE_POLICY
from core.verify.invariants import invariant


def validate_public_decode_receipt(
    receipt: dict[str, Any], public_text: str, *, max_tokens: int,
) -> bool:
    """Return whether decoding terminated uncensored; malformed evidence raises."""
    if not isinstance(receipt, dict) or receipt.get("policy") != PUBLIC_CHANNEL_DECODE_POLICY:
        raise ValueError("public decode policy is missing or unknown")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("public decode budget is invalid")
    for field in ("generated_tokens", "prefill_tokens", "boundary_tokens", "latency_ms", "reasoning_chars"):
        if type(receipt.get(field)) is not int or receipt[field] < 0:
            raise ValueError(f"public decode count is invalid: {field}")
    for field in ("native_thinking", "boundary_closed"):
        if type(receipt.get(field)) is not bool:
            raise ValueError(f"public decode channel flag is invalid: {field}")
    for field in ("reasoning_sha256", "token_ids_sha256", "public_text_sha256"):
        if not isinstance(receipt.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", receipt[field]):
            raise ValueError(f"public decode digest is invalid: {field}")
    if not isinstance(public_text, str) or receipt["public_text_sha256"] != hashlib.sha256(public_text.encode()).hexdigest():
        raise ValueError("public decode text differs from its receipt")
    if not receipt["boundary_closed"] and public_text:
        raise ValueError("public decode exposed an unclosed private channel")
    if receipt["reasoning_chars"] == 0 and receipt["reasoning_sha256"] != hashlib.sha256(b"").hexdigest():
        raise ValueError("public decode empty reasoning digest differs")
    reason, count = receipt.get("stop_reason"), receipt["generated_tokens"]
    if reason not in {"eos", "public_contract", "token_limit", "generator_exhausted"}:
        raise ValueError("public decode stop reason is unknown")
    if count > max_tokens or (reason == "token_limit" and count != max_tokens):
        raise ValueError("public decode token accounting differs from budget")
    if reason == "generator_exhausted" and count >= max_tokens:
        raise ValueError("public decode exhaustion reason differs from token count")
    if reason == "public_contract" and not receipt["boundary_closed"]:
        raise ValueError("public decode predicate ran on a private channel")
    if reason in {"eos", "public_contract"} and count < 1:
        raise ValueError("public decode termination has no sampled token")
    if receipt["boundary_tokens"] and not (receipt["native_thinking"] and receipt["prefill_tokens"]):
        raise ValueError("public decode boundary has no native public prefix")
    return receipt["boundary_closed"] and reason in {"eos", "public_contract"}


def public_decode_coverage(outputs: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    """Count all attempts, including failed/censored retries, for every arm."""
    counts: Counter[str] = Counter()
    censored: Counter[str] = Counter()
    for output in outputs:
        attempts = output.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError("public decode attempts are missing")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ValueError("public decode attempt is invalid")
            complete = validate_public_decode_receipt(
                attempt.get("decode"), attempt.get("raw_response"), max_tokens=max_tokens,
            )
            counts[output["arm"]] += 1
            censored[output["arm"]] += not complete
    return {
        "attempts_by_arm": dict(sorted(counts.items())),
        "censored_attempts_by_arm": dict(sorted(censored.items())),
        "uncensored": bool(counts) and not any(censored.values()),
    }


@invariant(
    "decode.private_budget_is_not_public_completion", scope="learning",
    owner="core/learning/public_decode_evidence.py", observational=False,
)
def _private_cap_is_censored() -> tuple:
    from core.brain.llm.public_channel_decode import PublicChannelDecode

    receipt = PublicChannelDecode(
        "", 32, 0, 0, 0, "token_limit", True, False, 4,
        hashlib.sha256(b"test").hexdigest(), "a" * 64,
    ).receipt()
    assert not validate_public_decode_receipt(receipt, "", max_tokens=32)
    return ()
