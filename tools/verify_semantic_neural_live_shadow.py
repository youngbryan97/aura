#!/usr/bin/env python3
"""Independently reopen a private CP568 desktop shadow transcript."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from tools.run_semantic_neural_live_shadow import (  # noqa: E402
    DOMAINS,
    PACKAGE_ID,
    PROMOTION_MODE,
    SCHEMA,
    _canonical_sha,
    _contract_issues,
    _sanitized,
    _tasks,
    _validate_shadow_row,
)

VERIFICATION_SCHEMA: Final = "aura.semantic_neural_live_shadow_verification.v1"


class SemanticNeuralLiveShadowVerificationError(RuntimeError):
    """The private producer artifact failed independent reopening."""


def _read_private(path: Path) -> dict[str, Any]:
    source = path.expanduser().absolute()
    metadata = source.stat()
    if (
        source.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SemanticNeuralLiveShadowVerificationError(
            "private producer artifact custody is invalid"
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SemanticNeuralLiveShadowVerificationError(
            "private producer artifact is invalid"
        ) from exc
    if not isinstance(document, dict):
        raise SemanticNeuralLiveShadowVerificationError(
            "private producer artifact is not an object"
        )
    return document


def verify_document(document: dict[str, Any]) -> dict[str, Any]:
    result_sha256 = str(document.get("result_sha256") or "")
    body = {key: value for key, value in document.items() if key != "result_sha256"}
    if document.get("schema") != SCHEMA or result_sha256 != _canonical_sha(body):
        raise SemanticNeuralLiveShadowVerificationError(
            "producer schema or result receipt differs"
        )
    if (
        document.get("package_id") != PACKAGE_ID
        or document.get("promotion_mode") != PROMOTION_MODE
        or tuple(document.get("domains") or ()) != DOMAINS
    ):
        raise SemanticNeuralLiveShadowVerificationError(
            "producer package or domain identity differs"
        )
    seed = document.get("seed")
    tasks_per_difficulty = document.get("tasks_per_difficulty")
    if type(seed) is not int or type(tasks_per_difficulty) is not int:
        raise SemanticNeuralLiveShadowVerificationError(
            "producer task identity is invalid"
        )
    tasks = _tasks(seed=seed, tasks_per_difficulty=tasks_per_difficulty)
    transcript = document.get("transcript")
    if not isinstance(transcript, list) or len(transcript) != len(tasks):
        raise SemanticNeuralLiveShadowVerificationError(
            "producer transcript cardinality differs"
        )

    verified_rows = []
    ordinary_correct = 0
    matches = 0
    gains = 0
    for task, row in zip(tasks, transcript, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("task_id") != task.task_id
            or row.get("family") != task.family
            or row.get("depth") != task.depth
            or row.get("prompt") != task.prompt
        ):
            raise SemanticNeuralLiveShadowVerificationError(
                "producer task reconstruction differs"
            )
        response = row.get("response")
        shadow_row = row.get("shadow_row")
        if not isinstance(response, str) or not isinstance(shadow_row, dict):
            raise SemanticNeuralLiveShadowVerificationError(
                "producer response or shadow row is unavailable"
            )
        contract_issues = _contract_issues(
            {
                "live_turn_contract": row.get("live_turn_contract"),
                "response_confidence": row.get("response_confidence"),
                "status": row.get("status"),
            }
        )
        shadow_issues = _validate_shadow_row(
            shadow_row,
            objective=task.prompt,
            response=response,
            family=task.family,
            activation_sha256=str(document.get("activation_sha256") or ""),
        )
        grade = task.grade(response)
        correct = grade.get("correct") is True
        if (
            row.get("http_status") != 200
            or row.get("ordinary_correct") is not correct
            or row.get("ordinary_parsed") != grade.get("parsed")
            or row.get("expected") != grade.get("expected")
            or row.get("contract_issues") != contract_issues
            or row.get("shadow_issues") != shadow_issues
            or contract_issues
            or shadow_issues
        ):
            raise SemanticNeuralLiveShadowVerificationError(
                f"producer row failed independent reopening: {task.task_id}"
            )
        answer_match = shadow_row.get("answer_match") is True
        gain = shadow_row.get("qualified_gain_candidate") is True
        if correct is not answer_match or gain is answer_match:
            raise SemanticNeuralLiveShadowVerificationError(
                f"ordinary grade and shadow comparison differ: {task.task_id}"
            )
        ordinary_correct += int(correct)
        matches += int(answer_match)
        gains += int(gain)
        verified_rows.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "depth": task.depth,
                "ordinary_correct": correct,
                "answer_match": answer_match,
                "qualified_gain_candidate": gain,
                "shadow_receipt_sha256": shadow_row["receipt_sha256"],
            }
        )

    if (
        document.get("task_count") != len(tasks)
        or document.get("ordinary_correct") != ordinary_correct
        or document.get("shadow_answer_matches") != matches
        or document.get("qualified_gain_candidates") != gains
        or document.get("all_requests_proven_ordinary_authority") is not True
    ):
        raise SemanticNeuralLiveShadowVerificationError(
            "producer aggregate counts differ"
        )
    health = document.get("boot_health")
    launch = health.get("launch_provenance") if isinstance(health, dict) else None
    if not isinstance(launch, dict) or (
        launch.get("required") is not True or launch.get("verified") is not True
    ):
        raise SemanticNeuralLiveShadowVerificationError(
            "packaged launch provenance was not proven"
        )
    sanitized = _sanitized(document)
    verification_body = {
        "schema": VERIFICATION_SCHEMA,
        "producer_result_sha256": result_sha256,
        "producer_summary_sha256": sanitized["summary_sha256"],
        "package_id": PACKAGE_ID,
        "promotion_mode": PROMOTION_MODE,
        "seed": seed,
        "task_count": len(tasks),
        "ordinary_correct": ordinary_correct,
        "shadow_answer_matches": matches,
        "qualified_gain_candidates": gains,
        "rows_sha256": _canonical_sha(verified_rows),
        "packaged_launch_provenance_verified": True,
        "ordinary_authority_verified": True,
        "raw_prompt_retained": False,
        "raw_answers_retained": False,
        "claim_boundary": (
            "live packaged resident-32B shadow comparison on the CP568 qualified "
            "four-domain surface; not active serving, open-domain reasoning, static "
            "fusion, or frontier performance"
        ),
    }
    return {
        **verification_body,
        "verification_receipt_sha256": _canonical_sha(verification_body),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_result", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = _read_private(args.private_result)
        verification = verify_document(document)
        expected_summary = _sanitized(document)
        observed_summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if observed_summary != expected_summary:
            raise SemanticNeuralLiveShadowVerificationError(
                "published producer summary differs"
            )
        atomic_write_text(
            args.output,
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"semantic neural live shadow verification failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
