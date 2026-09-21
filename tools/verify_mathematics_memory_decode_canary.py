#!/usr/bin/env python3
"""Independently replay a recurrent-memory language canary certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.frontier_tasks import generate_task  # noqa: E402
from core.brain.llm.latent_cortex.recurrent_memory_decode_context import (  # noqa: E402
    RecurrentMemoryDecodeState,
    execute_recurrent_memory_decode_state,
)
from core.learning.recurrent_work_memory_tissue import (  # noqa: E402
    MathematicsMemoryTissue,
    load_mathematics_memory_tissue,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402

CANARY_SCHEMA: Final = "aura.rlc.mathematics_memory_decode_canary.v1"
VERIFICATION_SCHEMA: Final = (
    "aura.rlc.mathematics_memory_decode_canary_verification.v1"
)
EXPECTED_ARMS: Final = (
    "ordinary_base",
    "matched_wire_base",
    "treatment",
    "matched_initialization",
    "no_write",
    "no_read",
    "reset_memory",
    "matched_wrong_state",
)
CAUSAL_CONTROLS: Final = EXPECTED_ARMS[1:2] + EXPECTED_ARMS[3:]


class CanaryVerificationError(ValueError):
    """Stable rejection for malformed or internally inconsistent evidence."""


def _reject(reason: str) -> None:
    raise CanaryVerificationError(reason)


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


def _bytes_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(commit: str, relative: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{commit}:{relative}"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        _reject(f"measured_source_unavailable:{relative}")
    return result.stdout


def _git_text(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _wrong_state_index(states: list[RecurrentMemoryDecodeState], index: int) -> int:
    own = (states[index].count, states[index].witness)
    for offset in range(1, len(states)):
        candidate = (index + offset) % len(states)
        if (states[candidate].count, states[candidate].witness) != own:
            return candidate
    _reject("wrong_state_derangement_unavailable")


def _summary(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm]
    if not selected:
        _reject(f"arm_rows_missing:{arm}")
    exact = sum(bool(row["correct"]) for row in selected)
    parsed = sum(bool(row["parsed"]) for row in selected)
    body = {
        "examples": len(selected),
        "exact": exact,
        "parsed": parsed,
        "exact_accuracy": round(exact / len(selected), 6),
        "parsed_accuracy": round(parsed / len(selected), 6),
        "mean_prompt_tokens": round(
            sum(int(row["prompt_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_generated_tokens": round(
            sum(int(row["generated_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_wire_prefill_tokens": round(
            sum(int(row["wire_prefill_tokens"]) for row in selected) / len(selected),
            3,
        ),
        "mean_latency_ms": round(
            sum(int(row["latency_ms"]) for row in selected) / len(selected),
            3,
        ),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanaryVerificationError("artifact_not_valid_json") from exc
    if not isinstance(payload, dict):
        _reject("artifact_root_invalid")
    return payload


def verify_canary(artifact_path: Path, *, model_path: Path) -> dict[str, Any]:
    """Reconstruct all non-model evidence and independently score raw outputs."""

    artifact = _load_json(artifact_path)
    supplied_receipt = artifact.get("receipt_sha256")
    artifact_body = {
        key: value for key, value in artifact.items() if key != "receipt_sha256"
    }
    if supplied_receipt != _sha(artifact_body):
        _reject("artifact_receipt_mismatch")
    if artifact.get("schema") != CANARY_SCHEMA:
        _reject("artifact_schema_invalid")

    protocol = artifact.get("protocol")
    if not isinstance(protocol, dict):
        _reject("protocol_invalid")
    task_count = protocol.get("task_count")
    tasks_per_difficulty = protocol.get("tasks_per_difficulty")
    seed = protocol.get("seed")
    if (
        type(task_count) is not int
        or type(tasks_per_difficulty) is not int
        or type(seed) is not int
        or task_count != tasks_per_difficulty * 3
        or tuple(protocol.get("difficulties", ())) != (1, 2, 3)
        or tuple(protocol.get("arms", ())) != EXPECTED_ARMS
        or protocol.get("ordinary_base_unconstrained") is not True
        or protocol.get("matched_wire_prefill") is not True
        or tuple(protocol.get("wire_prefill_arms", ())) != EXPECTED_ARMS[1:]
        or protocol.get("producer_available_during_decode") is not False
        or protocol.get("verifier_available_during_decode") is not False
        or protocol.get("answer_replacement_enabled") is not False
        or protocol.get("greedy_decode") is not True
    ):
        _reject("protocol_contract_invalid")

    tasks = [
        generate_task(
            "mathematics",
            seed=seed + difficulty * 10_000 + index,
            difficulty=difficulty,
        )
        for difficulty in (1, 2, 3)
        for index in range(tasks_per_difficulty)
    ]
    task_by_id = {task.task_id: task for task in tasks}
    task_index = {task.task_id: index for index, task in enumerate(tasks)}
    if len(task_by_id) != task_count:
        _reject("reconstructed_task_ids_not_unique")
    if artifact.get("task_inventory_sha256") != _sha(sorted(task_by_id)):
        _reject("task_inventory_mismatch")

    tissue = load_mathematics_memory_tissue()
    matched_initialization = MathematicsMemoryTissue(
        hidden_size=tissue.hidden_size,
        seed=tissue.seed,
    )
    treatment_states = [
        execute_recurrent_memory_decode_state(task.public.prompt, tissue=tissue)
        for task in tasks
    ]
    if artifact.get("state_inventory_sha256") != _sha(
        [state.receipt() for state in treatment_states]
    ):
        _reject("treatment_state_inventory_mismatch")

    expected_state_receipts: dict[tuple[str, str], str] = {}
    for index, task in enumerate(tasks):
        objective = task.public.prompt
        states = {
            "treatment": treatment_states[index],
            "matched_initialization": execute_recurrent_memory_decode_state(
                objective,
                tissue=matched_initialization,
            ),
            "no_write": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                write_mode="never",
            ),
            "no_read": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                read_mode="never",
            ),
            "reset_memory": execute_recurrent_memory_decode_state(
                objective,
                tissue=tissue,
                memory_mode="reset_each_step",
            ),
            "matched_wrong_state": treatment_states[
                _wrong_state_index(treatment_states, index)
            ],
        }
        for arm, state in states.items():
            expected_state_receipts[(task.task_id, arm)] = state.receipt()[
                "receipt_sha256"
            ]

    rows = artifact.get("rows")
    raw_outputs = artifact.get("raw_outputs")
    if not isinstance(rows, list) or not isinstance(raw_outputs, list):
        _reject("measurement_rows_invalid")
    expected_measurements = task_count * len(EXPECTED_ARMS)
    if len(rows) != expected_measurements or len(raw_outputs) != expected_measurements:
        _reject("measurement_count_mismatch")
    raw_by_key: dict[tuple[str, str], str] = {}
    for raw in raw_outputs:
        if not isinstance(raw, dict):
            _reject("raw_output_invalid")
        key = (raw.get("task_id"), raw.get("arm"))
        response = raw.get("response")
        if (
            not all(isinstance(value, str) for value in key)
            or key in raw_by_key
            or not isinstance(response, str)
        ):
            _reject("raw_output_key_invalid")
        raw_by_key[key] = response

    verified_rows: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            _reject("row_invalid")
        task_id = raw_row.get("task_id")
        arm = raw_row.get("arm")
        key = (task_id, arm)
        if (
            not isinstance(task_id, str)
            or not isinstance(arm, str)
            or task_id not in task_by_id
            or arm not in EXPECTED_ARMS
            or key in seen_rows
            or key not in raw_by_key
        ):
            _reject("row_key_invalid")
        seen_rows.add(key)
        task = task_by_id[task_id]
        response = raw_by_key[key]
        score = task.score(response)
        expected_receipt = (
            ""
            if arm in {"ordinary_base", "matched_wire_base"}
            else expected_state_receipts[key]
        )
        wire_tokens = raw_row.get("wire_prefill_tokens")
        generated_tokens = raw_row.get("generated_tokens")
        prompt_tokens = raw_row.get("prompt_tokens")
        latency_ms = raw_row.get("latency_ms")
        if (
            raw_row.get("difficulty") != task.public.difficulty
            or raw_row.get("correct") is not bool(score.correct)
            or raw_row.get("parsed") is not bool(score.parsed)
            or raw_row.get("response_sha256") != _bytes_sha(response.encode())
            or raw_row.get("state_receipt_sha256") != expected_receipt
            or type(wire_tokens) is not int
            or wire_tokens != (0 if arm == "ordinary_base" else protocol["wire_prefill_token_count"])
            or type(generated_tokens) is not int
            or generated_tokens < 0
            or type(prompt_tokens) is not int
            or prompt_tokens < 1
            or type(latency_ms) is not int
            or latency_ms < 0
        ):
            _reject(f"row_evidence_mismatch:{task_index[task_id]}:{arm}")
        verified_rows.append(raw_row)
    if len(seen_rows) != expected_measurements:
        _reject("measurement_matrix_incomplete")

    summaries = {arm: _summary(verified_rows, arm) for arm in EXPECTED_ARMS}
    if artifact.get("arms") != summaries:
        _reject("arm_summary_mismatch")
    ordinary = {
        row["task_id"]: bool(row["correct"])
        for row in verified_rows
        if row["arm"] == "ordinary_base"
    }
    treatment = {
        row["task_id"]: bool(row["correct"])
        for row in verified_rows
        if row["arm"] == "treatment"
    }
    gains = sorted(
        task_id for task_id, correct in treatment.items() if correct and not ordinary[task_id]
    )
    regressions = sorted(
        task_id for task_id, correct in treatment.items() if not correct and ordinary[task_id]
    )
    treatment_accuracy = summaries["treatment"]["exact_accuracy"]
    admitted = bool(
        treatment_accuracy == 1.0
        and gains
        and not regressions
        and all(summaries[arm]["exact_accuracy"] < treatment_accuracy for arm in CAUSAL_CONTROLS)
    )
    if (
        artifact.get("gain_count") != len(gains)
        or artifact.get("gain_set_sha256") != _sha(gains)
        or artifact.get("regression_count") != len(regressions)
        or artifact.get("regression_set_sha256") != _sha(regressions)
        or artifact.get("admitted") is not admitted
        or admitted is not True
    ):
        _reject("independent_verdict_mismatch")

    source = artifact.get("source_identity")
    if not isinstance(source, dict) or source.get("measured_source_clean") is not True:
        _reject("measured_source_identity_invalid")
    measured_commit = source.get("commit")
    source_sha256s = source.get("source_sha256s")
    if (
        not isinstance(measured_commit, str)
        or len(measured_commit) != 40
        or not isinstance(source_sha256s, dict)
    ):
        _reject("measured_source_identity_invalid")
    for relative, supplied_sha in source_sha256s.items():
        if (
            not isinstance(relative, str)
            or not isinstance(supplied_sha, str)
            or _bytes_sha(_git_bytes(measured_commit, relative)) != supplied_sha
        ):
            _reject(f"measured_source_hash_mismatch:{relative}")

    model_identity = artifact.get("model_identity")
    if not isinstance(model_identity, dict):
        _reject("model_identity_invalid")
    resolved_model = model_path.expanduser().resolve(strict=True)
    if (
        _file_sha(resolved_model / "config.json") != model_identity.get("config_sha256")
        or _file_sha(resolved_model / "model.safetensors.index.json")
        != model_identity.get("weights_index_sha256")
    ):
        _reject("model_identity_mismatch")

    try:
        artifact_reference = str(artifact_path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        artifact_reference = str(artifact_path.resolve())
    body = {
        "schema": VERIFICATION_SCHEMA,
        "artifact_path": artifact_reference,
        "artifact_sha256": _file_sha(artifact_path),
        "artifact_receipt_sha256": supplied_receipt,
        "measured_source_commit": measured_commit,
        "model_config_sha256": model_identity["config_sha256"],
        "verifier_source_commit": _git_text("rev-parse", "HEAD"),
        "verifier_source_clean": not bool(
            _git_text("status", "--porcelain", "--untracked-files=all")
        ),
        "verifier_source_sha256": _file_sha(Path(__file__)),
        "tissue_sha256": tissue.parameter_sha256(),
        "task_count": task_count,
        "arm_count": len(EXPECTED_ARMS),
        "measurement_count": expected_measurements,
        "treatment_exact": summaries["treatment"]["exact"],
        "ordinary_base_exact": summaries["ordinary_base"]["exact"],
        "matched_wire_base_exact": summaries["matched_wire_base"]["exact"],
        "causal_control_exacts": {
            arm: summaries[arm]["exact"] for arm in CAUSAL_CONTROLS
        },
        "gain_count": len(gains),
        "regression_count": len(regressions),
        "independently_verified": True,
        "claim_boundary": artifact.get("claim_boundary"),
    }
    return {**body, "receipt_sha256": _sha(body)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    certificate = verify_canary(
        args.artifact.expanduser().resolve(strict=True),
        model_path=args.model,
    )
    output = args.out.expanduser().resolve()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(certificate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
