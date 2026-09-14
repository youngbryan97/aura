#!/usr/bin/env python3
"""Separate public value errors from wire-format errors in retained canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.semantic_neural_composition_decode import (  # noqa: E402
    FINAL_MARKER,
    parse_composition_response,
)
from core.runtime.atomic_writer import atomic_write_bytes_if_absent  # noqa: E402
from tools.run_semantic_neural_composition_canary import _reference, _task_document  # noqa: E402
from tools.verify_semantic_neural_composition_decode_canary import verify  # noqa: E402


def public_values(response: str, report: tuple[str, ...]) -> dict[str, int] | None:
    """Accept one JSON envelope; never repair, select, or change answer values."""
    if not isinstance(response, str) or len(response) > 16384:
        return None
    body = response.strip()
    if body.startswith(FINAL_MARKER):
        body = body[len(FINAL_MARKER):].strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1] != "```":
            return None
        body = "\n".join(lines[1:-1]).strip()

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        values = json.loads(body, object_pairs_hook=unique)
    except (ValueError, RecursionError):
        return None
    if not isinstance(values, dict) or set(values) != set(report):
        return None
    # Reuse the typed contract after removing only envelope and key-order differences.
    ordered = {key: values[key] for key in report}
    return parse_composition_response(FINAL_MARKER + json.dumps(ordered), report)


def audit(result_path: Path, journal_path: Path) -> dict[str, Any]:
    raw = result_path.read_bytes()
    payload = json.loads(raw)
    verification = verify(payload, journal_path=journal_path)
    if not verification["verified"]:
        raise ValueError("composition evidence did not verify")
    rng = random.Random(payload["seed"])
    documents = [_task_document(rng) for _ in range(payload["task_count"])]
    rows = []
    for row in payload["rows"]:
        document = documents[row["ordinal"]]
        expected = _reference(document)
        values = public_values(row["response"], tuple(document["report"]))
        rows.append({
            "ordinal": row["ordinal"], "task_id": row["task_id"], "arm": row["arm"],
            "response_sha256": row["response_sha256"],
            "wire_correct": row["correct"], "wire_parsed": row["parsed"],
            "semantic_correct": values == expected, "semantic_parsed": values is not None,
            "incorrect_fields": None if values is None else [
                key for key in expected if values[key] != expected[key]],
        })
    arms = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        arms[arm] = {"count": len(selected), **{
            key: sum(row[key] for row in selected)
            for key in ("wire_correct", "wire_parsed", "semantic_correct", "semantic_parsed")}}
    return {
        "schema": "aura.composition_decode_semantic_audit.v1",
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "result_receipt_sha256": payload["receipt_sha256"],
        "verification_receipt_sha256": verification["verification_receipt_sha256"],
        "arms": arms, "rows": rows, "serving_authority": False,
        "limitations": [
            "Post-hoc diagnostic; original grades and receipts remain unchanged.",
            "The v1/v2 query supplies operation names, not definitions of custom ratio categories.",
            "The v1/v2 runner disables native thinking in every arm.",
            "All tasks share one construction; random literals do not establish construction transfer.",
            "Treatment receives computed result state; this does not establish internalized reasoning.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.result, args.journal)
    encoded = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    if not atomic_write_bytes_if_absent(args.out, encoded, mode=0o400):
        raise FileExistsError(args.out)
    print(json.dumps(result["arms"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
