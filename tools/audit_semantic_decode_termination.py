#!/usr/bin/env python3
"""Audit retained decode termination without rewriting historical verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.answer_contract import (  # noqa: E402
    contract_answer_state,
    contract_decode_disposition,
)
from core.runtime.atomic_writer import atomic_write_bytes_if_absent  # noqa: E402
from tools.verify_semantic_neural_decode_canary import _verify_journal  # noqa: E402


def audit(result_path: Path, journal_path: Path) -> dict[str, Any]:
    raw = result_path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema") != "aura.rlc.semantic_neural_decode_canary.v1":
        raise ValueError("termination audit requires a legacy raw-text report")
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    if payload.get("receipt_sha256") != digest:
        raise ValueError("decode report seal differs")
    journal = _verify_journal(journal_path, payload=payload, raw_outputs=payload["raw_outputs"])
    rows = []
    with journal_path.open() as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("event") != "decode_committed":
                continue
            record, output = event["row"], event["raw_output"]
            text = output["response"]
            state = contract_answer_state(text)
            rows.append({
                "task_id": record["task_id"], "arm": record["arm"],
                "generated_tokens": record["generated_tokens"],
                "stopped": record["stopped"], "recorded_parsed": record["parsed"],
                "contract_reason": state["reason"],
                "contract_disposition": str(contract_decode_disposition(text)),
                "response_sha256": record["response_sha256"],
            })
    arms = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        arms[arm] = {
            "count": len(selected),
            "parsed": sum(row["recorded_parsed"] for row in selected),
            "not_stopped": sum(not row["stopped"] for row in selected),
            "stopped_unparsed": sum(row["stopped"] and not row["recorded_parsed"] for row in selected),
            "contract_reasons": dict(Counter(row["contract_reason"] for row in selected)),
            "not_stopped_token_counts": dict(Counter(
                str(row["generated_tokens"]) for row in selected if not row["stopped"])),
        }
    return {
        "schema": "aura.semantic_decode_termination_audit.v1",
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "result_receipt_sha256": payload["receipt_sha256"],
        "journal": journal, "arms": arms, "rows": rows,
        "native_channel_state_recoverable": False,
        "limitations": [
            "Legacy text discarded special tokens; native channel closure cannot be reconstructed.",
            "A stopped flag alone does not identify EOS versus a text predicate.",
            "No model rerun, correctness regrading or historical verdict replacement.",
        ],
        "serving_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.result, args.journal)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    if not atomic_write_bytes_if_absent(args.out, raw, mode=0o400):
        raise FileExistsError(args.out)
    print(json.dumps({"output": str(args.out), "arms": result["arms"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
