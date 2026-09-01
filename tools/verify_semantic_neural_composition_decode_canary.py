#!/usr/bin/env python3
"""Independently verify a resident composition decode canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.semantic_neural_composition_decode import (  # noqa: E402
    SemanticNeuralCompositionDecodeState,
    execute_composition_decode_state,
    parse_composition_response,
)
from core.learning.semantic_neural_composition import (  # noqa: E402
    render_public_typed_workflow,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from tools.run_semantic_neural_composition_canary import (  # noqa: E402
    _lesion,
    _task_document,
)
from tools.run_semantic_neural_composition_decode_canary import (  # noqa: E402
    ARMS,
    CLAIM_BOUNDARY,
    JOURNAL_SCHEMA,
    SCHEMA,
    SOURCE_PATHS,
    _arm_order,
)
from tools.verify_semantic_neural_composition_canary import (  # noqa: E402
    _reference,
)
from tools.verify_semantic_neural_composition_canary import (  # noqa: E402
    verify as verify_composition_basis,
)

VERIFICATION_SCHEMA: Final = "aura.rlc.semantic_neural_composition_decode_verification.v1"


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


def _git_blob_sha(commit: str, relative: str) -> str:
    completed = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(completed.stdout).hexdigest()


def _verify_receipt(payload: dict[str, Any], field: str) -> None:
    claimed = payload.get(field)
    body = {key: value for key, value in payload.items() if key != field}
    if claimed != _sha(body):
        raise RuntimeError(f"composition decode {field} mismatch")


def _basis(identity: dict[str, Any]) -> dict[str, Any]:
    root = Path(str(identity.get("path") or "")).expanduser().resolve(strict=True)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    stored = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    replayed = verify_composition_basis(result)
    expected = {
        "path": str(root),
        "result_receipt_sha256": result["receipt_sha256"],
        "verification_receipt_sha256": stored["verification_receipt_sha256"],
        "task_set_sha256": stored["task_set_sha256"],
        "task_count": stored["task_count"],
    }
    if stored != replayed or replayed.get("verified") is not True or identity != expected:
        raise RuntimeError("composition decode basis identity mismatch")
    return expected


def _manifest(identity: dict[str, Any], model_path: Path) -> dict[str, Any]:
    path = Path(str(identity.get("path") or "")).expanduser().resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    active = Path(str(value["active_model_path"])).expanduser().resolve(strict=True)
    expected = {
        "path": str(path),
        "sha256": _file_sha(path),
        "active_model_path": str(active),
        "schema_version": value["schema_version"],
        "base_model": str(value.get("base_model") or ""),
        "tag": str(value.get("tag") or ""),
        "fused_at": value.get("fused_at"),
    }
    if identity != expected or active != model_path:
        raise RuntimeError("composition decode resident manifest mismatch")
    return expected


def _lesion_state(
    workflow: str,
    operation: int,
    coefficient: int,
) -> SemanticNeuralCompositionDecodeState | None:
    try:
        return execute_composition_decode_state(
            workflow,
            machine=_lesion(operation, coefficient),
        )
    except (RuntimeError, ValueError):
        return None


def _states(
    documents: list[dict[str, Any]],
) -> list[dict[str, SemanticNeuralCompositionDecodeState | None]]:
    values = []
    for document in documents:
        workflow = render_public_typed_workflow(document)
        values.append(
            {
                "treatment": execute_composition_decode_state(workflow),
                "additive_lesion": _lesion_state(workflow, 0, 1),
                "multiplicative_lesion": _lesion_state(workflow, 1, 2),
            }
        )
    return values


def _wrong_state_index(
    states: list[dict[str, SemanticNeuralCompositionDecodeState | None]],
    index: int,
) -> int:
    own = states[index]["treatment"]
    assert own is not None
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        other = states[candidate]["treatment"]
        if other is not None and other.semantic_result != own.semantic_result:
            return candidate
    raise RuntimeError("composition decode verifier cannot construct state derangement")


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    body = {
        "examples": len(selected),
        "exact": sum(bool(row["correct"]) for row in selected),
        "parsed": sum(bool(row["parsed"]) for row in selected),
        "mean_prompt_tokens": round(
            sum(int(row["prompt_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in selected) / len(selected), 3
        ),
        "mean_latency_ms": round(
            sum(int(row["latency_ms"]) for row in selected) / len(selected), 3
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _verify_journal(
    payload: dict[str, Any],
    explicit_path: Path | None,
) -> dict[str, Any]:
    recorded = payload.get("journal_path")
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError("composition decode journal identity is missing")
    path = (explicit_path or Path(recorded)).expanduser().resolve(strict=True)
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"composition decode journal line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(event, dict) or event.get("schema") != JOURNAL_SCHEMA:
                raise RuntimeError(
                    f"composition decode journal line {line_number} is invalid"
                )
            receipt = event.get("receipt_sha256")
            body = {key: value for key, value in event.items() if key != "receipt_sha256"}
            if event.get("previous_receipt_sha256") != previous or receipt != _sha(body):
                raise RuntimeError(
                    f"composition decode journal chain broke at line {line_number}"
                )
            previous = str(receipt)
            events.append(event)

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(events) != len(rows) + 2:
        raise RuntimeError("composition decode journal population differs")
    started, *decode_events, completed = events
    model_identity = payload.get("model_identity")
    expected_start = {
        "event": "campaign_started",
        "source_commit": payload.get("source_commit"),
        "seed": payload.get("seed"),
        "task_count": payload.get("task_count"),
        "arm_count": len(ARMS),
        "max_tokens": payload.get("max_tokens"),
        "model_identity": model_identity,
    }
    if any(started.get(key) != value for key, value in expected_start.items()):
        raise RuntimeError("composition decode journal campaign identity differs")
    for index, (event, row) in enumerate(zip(decode_events, rows, strict=True), start=1):
        if (
            event.get("event") != "decode_committed"
            or event.get("completed") != index
            or event.get("total") != len(rows)
            or event.get("row") != row
        ):
            raise RuntimeError(f"composition decode journal row {index} differs")
    last_decode_receipt = decode_events[-1]["receipt_sha256"]
    if payload.get("journal_last_decode_receipt_sha256") != last_decode_receipt:
        raise RuntimeError("composition decode final journal decode receipt differs")
    if (
        completed.get("event") != "campaign_completed"
        or completed.get("previous_receipt_sha256") != last_decode_receipt
        or completed.get("admitted") is not payload.get("admitted")
        or completed.get("report_receipt_sha256") != payload.get("receipt_sha256")
    ):
        raise RuntimeError("composition decode journal completion differs")
    return {
        "path": str(path),
        "sha256": _file_sha(path),
        "event_count": len(events),
        "decode_count": len(decode_events),
        "final_receipt_sha256": completed["receipt_sha256"],
    }


def verify(
    payload: dict[str, Any],
    *,
    journal_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise RuntimeError("composition decode schema mismatch")
    _verify_receipt(payload, "receipt_sha256")
    source_commit = str(payload.get("source_commit") or "")
    if len(source_commit) != 40:
        raise RuntimeError("composition decode source commit is invalid")
    expected_sources = {path: _git_blob_sha(source_commit, path) for path in SOURCE_PATHS}
    if payload.get("source_sha256s") != expected_sources:
        raise RuntimeError("composition decode source identity mismatch")

    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, dict):
        raise RuntimeError("composition decode model identity is missing")
    model_path = Path(str(model_identity.get("path") or "")).expanduser().resolve(strict=True)
    expected_model = {
        "path": str(model_path),
        "config_sha256": _file_sha(model_path / "config.json"),
        "weights_index_sha256": _file_sha(model_path / "model.safetensors.index.json"),
    }
    if model_identity != expected_model:
        raise RuntimeError("composition decode model identity mismatch")
    manifest = payload.get("resident_manifest_identity")
    basis = payload.get("composition_basis")
    if not isinstance(manifest, dict) or not isinstance(basis, dict):
        raise RuntimeError("composition decode provenance is missing")
    _manifest(manifest, model_path)
    _basis(basis)

    seed = payload.get("seed")
    task_count = payload.get("task_count")
    if type(seed) is not int or type(task_count) is not int or not 2 <= task_count <= 24:
        raise RuntimeError("composition decode cohort dimensions are invalid")
    rng = random.Random(seed)
    documents = [_task_document(rng) for _index in range(task_count)]
    workflows = [render_public_typed_workflow(document) for document in documents]
    states = _states(documents)
    expected_answers = [_reference(document) for document in documents]
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != task_count * len(ARMS):
        raise RuntimeError("composition decode row population is invalid")

    replayed_rows = []
    cursor = 0
    for index, (document, workflow, answer) in enumerate(
        zip(documents, workflows, expected_answers, strict=True)
    ):
        task_id = hashlib.sha256(workflow.encode()).hexdigest()
        report = tuple(document["report"])
        for arm in _arm_order(task_id):
            row = rows[cursor]
            cursor += 1
            if not isinstance(row, dict):
                raise RuntimeError("composition decode row is invalid")
            selected = (
                states[_wrong_state_index(states, index)]["treatment"]
                if arm == "matched_wrong_state"
                else states[index].get(arm)
            )
            response = row.get("response")
            if not isinstance(response, str):
                raise RuntimeError("composition decode raw response is missing")
            parsed = parse_composition_response(response, report)
            expected_state_receipt = (
                "" if selected is None else selected.receipt()["receipt_sha256"]
            )
            replay = {
                **row,
                "ordinal": index,
                "task_id": task_id,
                "arm": arm,
                "correct": parsed == answer,
                "parsed": parsed is not None,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "state_receipt_sha256": expected_state_receipt,
            }
            if replay != row or any(
                type(row.get(field)) is not int or row[field] < 0
                for field in (
                    "prompt_tokens",
                    "generated_tokens",
                    "prefill_tokens",
                    "latency_ms",
                )
            ):
                raise RuntimeError("composition decode independent row replay differs")
            if (arm == "ordinary_base") != (row["prefill_tokens"] == 0):
                raise RuntimeError("composition decode prefill contract differs")
            replayed_rows.append(row)

    arms = {arm: _summary(replayed_rows, arm) for arm in ARMS}
    if payload.get("arms") != arms:
        raise RuntimeError("composition decode arm summaries differ")
    by_arm = {
        arm: {
            row["task_id"]: bool(row["correct"])
            for row in replayed_rows
            if row["arm"] == arm
        }
        for arm in ARMS
    }
    gain_set = sorted(
        task_id
        for task_id, correct in by_arm["treatment"].items()
        if correct and not by_arm["ordinary_base"][task_id]
    )
    regressions = sorted(
        task_id
        for task_id, correct in by_arm["ordinary_base"].items()
        if correct and not by_arm["treatment"][task_id]
    )
    treatment_exact = arms["treatment"]["exact"]
    admitted = bool(
        treatment_exact == task_count
        and gain_set
        and not regressions
        and all(arms[arm]["exact"] < treatment_exact for arm in ARMS if arm != "treatment")
    )
    if (
        payload.get("decode_calls_per_arm_per_task") != 1
        or payload.get("arm_order") != "task_hash_rotated"
        or payload.get("gain_set_sha256") != _sha(gain_set)
        or payload.get("gain_count") != len(gain_set)
        or payload.get("regression_set_sha256") != _sha(regressions)
        or payload.get("regression_count") != len(regressions)
        or payload.get("admitted") is not admitted
        or payload.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("composition decode adjudication differs")
    journal_identity = _verify_journal(payload, journal_path)
    body = {
        "schema": VERIFICATION_SCHEMA,
        "verified": admitted,
        "input_receipt_sha256": payload["receipt_sha256"],
        "source_commit": source_commit,
        "source_sha256s": expected_sources,
        "model_identity": expected_model,
        "resident_manifest_identity": manifest,
        "composition_basis": basis,
        "task_count": task_count,
        "independent_exact_by_arm": {arm: arms[arm]["exact"] for arm in ARMS},
        "gain_count": len(gain_set),
        "regression_count": len(regressions),
        "journal_identity": journal_identity,
        "paired_one_sided_exact_p": (
            math.ldexp(1.0, -len(gain_set)) if not regressions else 1.0
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "verifier_source_sha256": _file_sha(Path(__file__).resolve()),
    }
    return {**body, "verification_receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    result = verify(payload, journal_path=args.journal)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        atomic_write_text(args.out.expanduser().resolve(), encoded, power_safe=True)
    print(encoded, end="")
    return 0 if result["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
