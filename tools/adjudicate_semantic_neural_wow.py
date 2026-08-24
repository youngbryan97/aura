#!/usr/bin/env python3
"""Apply the frozen bounded-WOW gate to resident semantic tissue evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA: Final = "aura.rlc.semantic_neural_bounded_wow_adjudication.v1"
EXPECTED_DOMAINS: Final = (
    "coding",
    "calibration",
    "misleading_premise",
    "scientific_inference",
)
EXPECTED_FAMILIES: Final = tuple(f"frontier_{domain}" for domain in EXPECTED_DOMAINS)
EXPECTED_ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "coefficient_lesion",
    "matched_wrong_state",
)
MODEL_BOUND_CLAIM: Final = (
    "replicated lesion-dependent resident-model effective reasoning gain over "
    "ordinary decode on the frozen four-domain semantic cohort"
)


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"adjudication input is not an object: {path}")
    return value


def _verify_receipt(payload: dict[str, Any], field: str) -> None:
    body = {key: value for key, value in payload.items() if key != field}
    if payload.get(field) != _sha(body):
        raise RuntimeError(f"adjudication input receipt mismatch: {field}")


def _journal_matrix(path: Path) -> tuple[dict[str, Counter[str]], int]:
    previous = "0" * 64
    correct: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[tuple[str, str]] = set()
    decode_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise RuntimeError(f"adjudication journal line {line_number} is not an object")
        receipt = event.get("receipt_sha256")
        body = {key: value for key, value in event.items() if key != "receipt_sha256"}
        if event.get("previous_receipt_sha256") != previous or receipt != _sha(body):
            raise RuntimeError(f"adjudication journal chain broke at line {line_number}")
        previous = str(receipt)
        if event.get("event") != "decode_committed":
            continue
        row = event.get("row")
        if not isinstance(row, dict):
            raise RuntimeError("adjudication decode row is invalid")
        family, arm = row.get("family"), row.get("arm")
        pair = (str(row.get("task_id") or ""), str(arm or ""))
        if family not in EXPECTED_FAMILIES or arm not in EXPECTED_ARMS or pair in seen:
            raise RuntimeError("adjudication decode matrix identity is invalid")
        seen.add(pair)
        correct[str(family)][str(arm)] += int(row.get("correct") is True)
        decode_count += 1
    return correct, decode_count


def adjudicate(
    result_path: Path,
    verification_path: Path,
    journal_path: Path,
    supervisor_receipt_path: Path,
) -> dict[str, Any]:
    result = _read(result_path)
    verification = _read(verification_path)
    supervisor = _read(supervisor_receipt_path)
    _verify_receipt(result, "receipt_sha256")
    _verify_receipt(verification, "verification_receipt_sha256")
    _verify_receipt(supervisor, "receipt_sha256")
    matrix, decode_count = _journal_matrix(journal_path)

    task_count = int(result.get("task_count") or 0)
    result_model_identity = result.get("model_identity")
    verification_model_identity = verification.get("model_identity")
    result_manifest_identity = result.get("resident_manifest_identity")
    verification_manifest_identity = verification.get("resident_manifest_identity")
    per_family = task_count // len(EXPECTED_FAMILIES) if task_count else 0
    gains_by_family = {
        family: matrix[family]["treatment"] - matrix[family]["ordinary_base"]
        for family in EXPECTED_FAMILIES
    }
    regressions_by_family = {
        family: max(0, matrix[family]["ordinary_base"] - matrix[family]["treatment"])
        for family in EXPECTED_FAMILIES
    }
    lesion_separation = {
        family: matrix[family]["treatment"] - matrix[family]["coefficient_lesion"]
        for family in EXPECTED_FAMILIES
    }
    checks = {
        "producer_admitted": result.get("admitted") is True,
        "independent_verifier_passed": verification.get("verified") is True,
        "resident_manifest_bound": isinstance(result.get("resident_manifest_identity"), dict),
        "model_identity_match": isinstance(result_model_identity, dict)
        and result_model_identity == verification_model_identity,
        "resident_manifest_identity_match": isinstance(result_manifest_identity, dict)
        and result_manifest_identity == verification_manifest_identity,
        "frozen_profile": result.get("surface_profile") == "mixed_multidomain_v1",
        "four_domain_identity": tuple(result.get("domains") or ()) == EXPECTED_DOMAINS,
        "powered_complete_matrix": task_count == 60 and decode_count == 300,
        "treatment_perfect": verification.get("independent_exact_by_arm", {}).get("treatment")
        == 60,
        "zero_regressions": result.get("regression_count") == 0
        and not any(regressions_by_family.values()),
        "gain_in_three_domains": sum(value > 0 for value in gains_by_family.values()) >= 3,
        "all_family_lesions_separate": all(value > 0 for value in lesion_separation.values()),
        "wrong_state_never_succeeds": verification.get("independent_exact_by_arm", {}).get(
            "matched_wrong_state"
        )
        == 0,
        "paired_exact_significant": float(verification.get("paired_one_sided_exact_p") or 1.0)
        <= 0.01,
        "journal_independently_verified": verification.get("journal_decode_count") == 300,
        "supervisor_passed": supervisor.get("passed") is True
        and supervisor.get("timed_out") is False
        and supervisor.get("restart_count") == 0
        and supervisor.get("containment_verified") is True,
    }
    passed = all(checks.values())
    body = {
        "schema": SCHEMA,
        "verdict": "BOUNDED_WOW_SIGNAL" if passed else "NOT_ADMITTED",
        "passed": passed,
        "checks": checks,
        "task_count": task_count,
        "decode_count": decode_count,
        "tasks_per_family": per_family,
        "gains_by_family": gains_by_family,
        "regressions_by_family": regressions_by_family,
        "coefficient_lesion_separation_by_family": lesion_separation,
        "independent_exact_by_arm": verification.get("independent_exact_by_arm"),
        "gain_count": verification.get("gain_count"),
        "regression_count": verification.get("regression_count"),
        "paired_one_sided_exact_p": verification.get("paired_one_sided_exact_p"),
        "input_sha256s": {
            "result": _file_sha(result_path),
            "verification": _file_sha(verification_path),
            "journal": _file_sha(journal_path),
            "supervisor_receipt": _file_sha(supervisor_receipt_path),
        },
        "input_receipts": {
            "result": result["receipt_sha256"],
            "verification": verification["verification_receipt_sha256"],
            "supervisor": supervisor["receipt_sha256"],
        },
        "model_identity": result_model_identity,
        "resident_manifest_identity": result_manifest_identity,
        "claim": MODEL_BOUND_CLAIM,
        "limitations": (
            "bounded executable families; not open-domain general reasoning, static fusion, "
            "frontier performance, consciousness evidence, or unrestricted runtime promotion"
        ),
        "adjudicator_source_sha256": _file_sha(Path(__file__)),
    }
    return {**body, "adjudication_receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--supervisor-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = adjudicate(
        args.result.expanduser().resolve(strict=True),
        args.verification.expanduser().resolve(strict=True),
        args.journal.expanduser().resolve(strict=True),
        args.supervisor_receipt.expanduser().resolve(strict=True),
    )
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
