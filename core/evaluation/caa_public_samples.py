"""Validate the sampled public surface before a steering qualification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from core.brain.llm.public_channel_decode import PUBLIC_CHANNEL_SAMPLE_POLICY
from core.verify.invariants import invariant


def public_sample_complete(text: str, receipt: Mapping) -> bool:
    return (receipt.get("boundary_closed") is True
            and receipt.get("stop_reason") == "eos" and bool(text.strip()))


def validate_public_samples(result: Mapping, outputs: Mapping[str, list[str]]) -> dict:
    policy = result.get("generation_policy")
    if policy is None:
        return {"verified": False, "complete": False, "reason": "legacy_generation_channel_unmeasured",
                "completed_per_condition": {}}
    if policy != PUBLIC_CHANNEL_SAMPLE_POLICY:
        raise ValueError("caa_generation_policy_invalid")
    all_receipts = result.get("generation_receipts")
    if not isinstance(all_receipts, Mapping) or set(all_receipts) != set(outputs):
        raise ValueError("caa_generation_receipts_missing")
    completed = {}
    for condition, values in outputs.items():
        receipts = all_receipts[condition]
        if not isinstance(receipts, list) or len(receipts) != len(values):
            raise ValueError("caa_generation_receipts_shape")
        count = 0
        for text, receipt in zip(values, receipts, strict=True):
            if not isinstance(receipt, Mapping) or receipt.get("policy") != policy:
                raise ValueError("caa_generation_receipt_invalid")
            if receipt.get("public_text_sha256") != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("caa_generation_public_text_drift")
            for key in ("reasoning_sha256", "token_ids_sha256", "prompt_sha256"):
                digest = receipt.get(key)
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(char not in "0123456789abcdef" for char in digest)):
                    raise ValueError("caa_generation_digest_invalid")
            for key in ("generated_tokens", "max_tokens", "reasoning_chars"):
                if type(receipt.get(key)) is not int or receipt[key] < 0:
                    raise ValueError("caa_generation_counts_invalid")
            if (receipt["max_tokens"] < 1 or receipt["generated_tokens"] > receipt["max_tokens"]
                    or receipt["max_tokens"] != result.get("max_tokens")):
                raise ValueError("caa_generation_allocation_mismatch")
            if (type(receipt.get("native_thinking")) is not bool
                    or type(receipt.get("boundary_closed")) is not bool
                    or (not receipt["boundary_closed"] and text)):
                raise ValueError("caa_generation_boundary_invalid")
            reason = receipt.get("stop_reason")
            if reason not in {"eos", "token_limit", "generator_exhausted"}:
                raise ValueError("caa_generation_stop_invalid")
            if reason == "token_limit" and receipt["generated_tokens"] != receipt["max_tokens"]:
                raise ValueError("caa_generation_stop_invalid")
            count += public_sample_complete(text, receipt)
        completed[condition] = count
    complete = all(completed[name] == len(values) for name, values in outputs.items())
    return {"verified": True, "complete": complete,
            "reason": "" if complete else "public_generation_incomplete",
            "completed_per_condition": completed}


@invariant("evaluation.steering_public_completion", scope="evaluation",
           owner="core/evaluation/caa_public_samples.py", observational=False)
def _public_completion() -> tuple:
    assert not public_sample_complete("happy", {"boundary_closed": False, "stop_reason": "eos"})
    assert not public_sample_complete("happy", {"boundary_closed": True, "stop_reason": "token_limit"})
    assert public_sample_complete("happy", {"boundary_closed": True, "stop_reason": "eos"})
    return ()
