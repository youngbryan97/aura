"""Replay archived desktop delivery evidence without starting a model."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sealed(value: dict[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return value.get("receipt_sha256") == _sha(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ))


def delivery_errors(row: dict[str, Any]) -> list[str]:
    """Check independently retained task/visible text against the sealed journal."""
    errors: list[str] = []
    raw = row["response_json"]
    if _sha(raw) != row["response_hash"]:
        errors.append("journal_hash_mismatch")
    response = json.loads(raw)
    contract = response.get("live_turn_contract", {})
    answer = response.get("response", "")
    if row["state"] != "completed" or response.get("delivery_state") != "completed":
        errors.append("delivery_not_completed")
    if row["turn_id"] != response.get("turn_id"):
        errors.append("turn_identity_mismatch")
    if answer != row["observed_desktop_text"]:
        errors.append("desktop_answer_mismatch")
    if response.get("delivery_replayed") is not False:
        errors.append("not_a_fresh_delivery")
    if contract.get("request_surface") != "desktop-ui":
        errors.append("not_desktop_ingress")
    if contract.get("response_path") != "cognitive_engine_qualified_recurrent":
        errors.append("wrong_response_path")
    for field in (
        "answer_delivery_proven", "qualified_recurrent_path_proven",
        "qualified_recurrent_terminal_bytes_preserved",
        "qualified_recurrent_public_output_quality_proven",
        "final_requested_output_contract_satisfied", "semantic_completion_satisfied",
    ):
        if contract.get(field) is not True:
            errors.append(f"unproven:{field}")
    if contract.get("qualified_recurrent_delivery_errors") != []:
        errors.append("qualified_delivery_errors")
    if contract.get("foreground_model_generation_consumed") is not False:
        errors.append("unexpected_decoder_use")
    if contract.get("qualified_recurrent_family") != row["family"]:
        errors.append("family_mismatch")
    if response.get("conversation_lane", {}).get("model_path") != row["model_path"]:
        errors.append("model_mismatch")
    quality = contract.get("latent_cortex_public_output_quality", {})
    # This projection references the result receipt; it is not itself that receipt.
    reference = quality.get("receipt_sha256", "")
    if not (
        quality.get("passed") is True
        and quality.get("reasons") == []
        and quality.get("policy") == "qualified_recurrent_state_serialization_quality_v1"
        and isinstance(reference, str) and len(reference) == 64
        and all(char in "0123456789abcdef" for char in reference)
    ):
        errors.append("quality_receipt_invalid")
    if quality.get("text_sha256") != _sha(answer):
        errors.append("answer_binding_mismatch")
    if quality.get("objective_sha256") != _sha(row["prompt"]):
        errors.append("objective_binding_mismatch")
    admission = contract.get("preflight_evidence_owner", {})
    if not _sealed(admission):
        errors.append("admission_receipt_invalid")
    if admission.get("public_source_sha256") != _sha(row["prompt"]):
        errors.append("admission_binding_mismatch")
    prefix = "FINAL_ANSWER: "
    try:
        actual = json.loads(answer.removeprefix(prefix))
    except (TypeError, ValueError):
        errors.append("invalid_answer_json")
    else:
        if not answer.startswith(prefix) or actual != row["expected_answer"]:
            errors.append("incorrect_answer")
    return errors


def verify_archive(archive: dict[str, Any]) -> dict[str, Any]:
    revision = archive["boot_observation"]["runtime_revision"]
    errors = []
    if not (
        revision.get("source_verified") is True
        and revision.get("source_current") is True
        and revision.get("capture_stable") is True
        and revision.get("issues") == []
        and revision.get("actual_commit_sha") == archive["source_commit"]
        and revision.get("expected_commit_sha") == archive["source_commit"]
    ):
        errors.append("runtime_source_mismatch")
    rows = archive["deliveries"]
    expected = {
        "frontier_coding", "frontier_calibration", "frontier_misleading_premise",
        "frontier_scientific_inference",
    }
    if {row["family"] for row in rows} != expected:
        errors.append("incomplete_family_coverage")
    if len({row["turn_id"] for row in rows}) != len(rows):
        errors.append("duplicate_delivery")
    results = [{"turn_id": row["turn_id"], "errors": delivery_errors(row)} for row in rows]
    return {
        "passed": not errors and all(not row["errors"] for row in results),
        "errors": errors,
        "deliveries": results,
        "claim": "qualified_mechanism_serves_eligible_desktop_requests",
        "broad_transfer_proven": False,
        "runtime_health_proven": False,
    }

