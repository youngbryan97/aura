#!/usr/bin/env python3
"""Run fresh qualified tasks through Aura's real desktop chat shadow lane.

The qualified semantic-neural candidate is never serving authority in this
campaign. Every request must also prove that the sovereign CognitiveEngine ran
ordinary resident inference, and every private shadow-ledger row is reconciled
against the final user-visible answer before a sanitized verdict is emitted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.brain.llm.latent_cortex.semantic_surface_adapter import (  # noqa: E402
    SEMANTIC_SURFACE_PROFILES,
    render_scientific_surface,
)
from core.brain.llm.semantic_neural_serving import (  # noqa: E402
    DEFAULT_ACTIVATION_PATH,
    semantic_neural_serving_status,
)
from core.learning.frontier_process_supervision import (  # noqa: E402
    frontier_process_task_battery,
)
from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.state_ownership import state_root  # noqa: E402

SCHEMA: Final = "aura.semantic_neural_live_shadow.v1"
# This verifier reopens the historical CP568 shadow campaign. Keep its identity
# frozen when a later, independently qualified package becomes authoritative.
PACKAGE_ID: Final = "cp568-resident-semantic-neural-shadow"
PROMOTION_MODE: Final = "shadow"
DOMAINS: Final = (
    "coding",
    "calibration",
    "misleading_premise",
    "scientific_inference",
)
DIFFICULTIES: Final = (1, 2, 3)
DEFAULT_LEDGER: Final = state_root() / "runtime" / "semantic_neural_shadow.jsonl"


class SemanticNeuralLiveShadowError(RuntimeError):
    """The real desktop shadow path could not produce admissible evidence."""


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _private_output(path: Path, document: dict[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = Path(destination.parent.anchor)
    for part in destination.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise SemanticNeuralLiveShadowError(
                "private evidence path contains a symlink"
            )
    metadata = destination.parent.stat()
    if (
        metadata.st_uid != os.geteuid()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise SemanticNeuralLiveShadowError("private evidence custody is invalid")
    payload = json.dumps(document, sort_keys=True, indent=2) + "\n"
    descriptor = -1
    created = False
    completed = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o400,
        )
        created = True
        encoded = payload.encode("utf-8")
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise SemanticNeuralLiveShadowError("private evidence write was short")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory_fd = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        completed = True
    except OSError as exc:
        raise SemanticNeuralLiveShadowError("private evidence publication failed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if created and not completed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass


def _tasks(*, seed: int, tasks_per_difficulty: int) -> list[Any]:
    tasks = frontier_process_task_battery(
        DOMAINS,
        DIFFICULTIES,
        tasks_per_difficulty,
        seed=seed,
    )
    adapted = []
    surface_index = 0
    for index, task in enumerate(tasks):
        if task.family == "frontier_scientific_inference":
            profile = SEMANTIC_SURFACE_PROFILES[
                surface_index % len(SEMANTIC_SURFACE_PROFILES)
            ]
            surface_index += 1
            task = replace(
                task,
                prompt=render_scientific_surface(
                    task.prompt,
                    profile=profile,
                    permutation_seed=seed + index,
                ),
                transition_trace=None,
                transition_program=None,
            )
        adapted.append(task)
    return adapted


def _load_ledger_rows(path: Path, *, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
    rows = []
    for line in payload.splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SemanticNeuralLiveShadowError("shadow ledger contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise SemanticNeuralLiveShadowError("shadow ledger row is not an object")
        rows.append(row)
    return rows


def _ledger_snapshot(path: Path, *, offset: int) -> tuple[list[dict[str, Any]], int]:
    rows = _load_ledger_rows(path, offset=offset)
    return rows, path.stat().st_size if path.exists() else 0


def _ledger_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _contract_issues(payload: dict[str, Any]) -> list[str]:
    contract = payload.get("live_turn_contract")
    if not isinstance(contract, dict):
        return ["live_turn_contract_missing"]
    checks = {
        "desktop_cognitive_engine_required": True,
        "engine_think_invoked": True,
        "cognitive_engine_reply_accepted": True,
        "cognitive_engine_reply_failed": False,
        "bounded_contract_used": False,
        "legacy_fallback_used": False,
        "full_mind_path": True,
    }
    issues = [
        f"{field}:{contract.get(field)!r}"
        for field, expected in checks.items()
        if contract.get(field) is not expected
    ]
    confidence = str(payload.get("response_confidence") or "").strip().lower()
    if confidence != "high":
        issues.append(f"response_confidence:{confidence or 'missing'}")
    status = str(payload.get("status") or "").strip().lower()
    if not status or any(
        marker in status for marker in ("failed", "unavailable", "timeout", "bounded")
    ):
        issues.append(f"status:{status or 'missing'}")
    return issues


def _validate_shadow_row(
    row: dict[str, Any],
    *,
    objective: str,
    response: str,
    family: str,
    activation_sha256: str,
) -> list[str]:
    expected = {
        "schema": "aura.semantic_neural_shadow.v1",
        "objective_sha256": _sha_text(objective),
        "ordinary_answer_sha256": _sha_text(response),
        "family": family,
        "activation_sha256": activation_sha256,
        "package_id": PACKAGE_ID,
        "promotion_mode": PROMOTION_MODE,
        "raw_prompt_retained": False,
        "raw_answers_retained": False,
        "persisted": True,
    }
    issues = [
        f"{field}:{row.get(field)!r}"
        for field, value in expected.items()
        if row.get(field) != value
    ]
    receipt = row.get("receipt_sha256")
    body = {
        key: value
        for key, value in row.items()
        if key not in {"persisted", "receipt_sha256"}
    }
    if receipt != _canonical_sha(body):
        issues.append("receipt_sha256")
    if row.get("answer_match") is row.get("qualified_gain_candidate"):
        issues.append("match_gain_complement")
    return issues


def _extract_response(payload: dict[str, Any]) -> str:
    return str(
        payload.get("response")
        or payload.get("reply")
        or payload.get("message")
        or payload.get("text")
        or ""
    ).strip()


async def run_live_shadow(
    *,
    base_url: str,
    seed: int,
    tasks_per_difficulty: int,
    timeout_s: float,
    ledger_path: Path,
    private_output: Path,
) -> dict[str, Any]:
    if tasks_per_difficulty < 1 or tasks_per_difficulty > 8:
        raise SemanticNeuralLiveShadowError("tasks_per_difficulty must be in [1, 8]")
    activation = json.loads(DEFAULT_ACTIVATION_PATH.read_text(encoding="utf-8"))
    activation_sha256 = str(activation.get("activation_sha256") or "")
    model_path = str((activation.get("model_identity") or {}).get("path") or "")
    status = semantic_neural_serving_status(model_path)
    receipt = status.get("receipt")
    if (
        status.get("active") is not True
        or not isinstance(receipt, dict)
        or receipt.get("activation_sha256") != activation_sha256
        or receipt.get("package_id") != PACKAGE_ID
        or receipt.get("promotion_mode") != "shadow"
    ):
        raise SemanticNeuralLiveShadowError("CP568 shadow activation is not authoritative")

    headers = {
        "X-Aura-Surface": "desktop-ui",
        "X-Aura-Require-CognitiveEngine": "required",
        "X-Aura-Semantic-Neural-Shadow-Probe": "true",
    }
    token = os.environ.get("AURA_API_TOKEN", "").strip()
    if token:
        headers["X-Api-Token"] = token
    base = base_url.rstrip("/")
    tasks = _tasks(seed=seed, tasks_per_difficulty=tasks_per_difficulty)
    ledger_offset = await asyncio.to_thread(_ledger_size, ledger_path)
    started_at = time.time()
    transcript = []

    async with httpx.AsyncClient(headers=headers, timeout=timeout_s) as client:
        health_response = await client.get(f"{base}/api/health/boot")
        try:
            health = health_response.json()
        except json.JSONDecodeError as exc:
            raise SemanticNeuralLiveShadowError("boot health did not return JSON") from exc
        if not isinstance(health, dict) or not (
            health.get("conversation_ready") or health.get("ready")
        ):
            raise SemanticNeuralLiveShadowError("Aura is not conversation-ready")

        for index, task in enumerate(tasks):
            request_started = time.monotonic()
            response = await client.post(
                f"{base}/api/chat",
                json={
                    "message": task.prompt,
                    "session_id": f"cp568-shadow-{seed}-{index}-{uuid.uuid4().hex[:8]}",
                },
            )
            latency_s = time.monotonic() - request_started
            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise SemanticNeuralLiveShadowError("chat response did not return JSON") from exc
            if not isinstance(payload, dict):
                raise SemanticNeuralLiveShadowError("chat response is not an object")
            response_text = _extract_response(payload)
            new_rows, ledger_offset = await asyncio.to_thread(
                _ledger_snapshot,
                ledger_path,
                offset=ledger_offset,
            )
            objective_sha256 = _sha_text(task.prompt)
            matching = [
                row for row in new_rows if row.get("objective_sha256") == objective_sha256
            ]
            ledger_row = matching[-1] if len(matching) == 1 else {}
            contract_issues = _contract_issues(payload)
            shadow_issues = (
                _validate_shadow_row(
                    ledger_row,
                    objective=task.prompt,
                    response=response_text,
                    family=task.family,
                    activation_sha256=activation_sha256,
                )
                if ledger_row
                else [f"shadow_row_count:{len(matching)}"]
            )
            grade = task.grade(response_text)
            transcript.append(
                {
                    "task_id": task.task_id,
                    "family": task.family,
                    "depth": task.depth,
                    "prompt": task.prompt,
                    "response": response_text,
                    "http_status": response.status_code,
                    "latency_s": round(latency_s, 3),
                    "ordinary_correct": grade.get("correct") is True,
                    "ordinary_parsed": grade.get("parsed"),
                    "expected": grade.get("expected"),
                    "live_turn_contract": payload.get("live_turn_contract"),
                    "response_confidence": payload.get("response_confidence"),
                    "status": payload.get("status"),
                    "contract_issues": contract_issues,
                    "shadow_issues": shadow_issues,
                    "shadow_row": ledger_row,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "semantic_neural_live_shadow_progress",
                        "completed": index + 1,
                        "total": len(tasks),
                        "family": task.family,
                        "ordinary_correct": grade.get("correct") is True,
                        "answer_match": ledger_row.get("answer_match"),
                        "latency_s": round(latency_s, 3),
                        "issues": contract_issues + shadow_issues,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    rows_ok = all(
        row["http_status"] == 200
        and not row["contract_issues"]
        and not row["shadow_issues"]
        for row in transcript
    )
    ordinary_correct = sum(int(row["ordinary_correct"]) for row in transcript)
    matches = sum(int(row["shadow_row"].get("answer_match") is True) for row in transcript)
    gains = sum(
        int(row["shadow_row"].get("qualified_gain_candidate") is True)
        for row in transcript
    )
    body = {
        "schema": SCHEMA,
        "seed": seed,
        "tasks_per_difficulty": tasks_per_difficulty,
        "task_count": len(tasks),
        "domains": list(DOMAINS),
        "activation_sha256": activation_sha256,
        "package_id": PACKAGE_ID,
        "promotion_mode": PROMOTION_MODE,
        "model_path": model_path,
        "base_url": base,
        "started_at_unix": started_at,
        "completed_at_unix": time.time(),
        "boot_health": health,
        "ordinary_correct": ordinary_correct,
        "shadow_answer_matches": matches,
        "qualified_gain_candidates": gains,
        "all_requests_proven_ordinary_authority": rows_ok,
        "transcript": transcript,
    }
    document = {**body, "result_sha256": _canonical_sha(body)}
    _private_output(private_output, document)
    return document


def _sanitized(document: dict[str, Any]) -> dict[str, Any]:
    rows = document["transcript"]
    body = {
        key: value
        for key, value in document.items()
        if key not in {"transcript", "boot_health", "result_sha256"}
    }
    body["result_sha256"] = document["result_sha256"]
    body["latency_s"] = [row["latency_s"] for row in rows]
    body["rows"] = [
        {
            "task_id": row["task_id"],
            "family": row["family"],
            "depth": row["depth"],
            "prompt_sha256": _sha_text(row["prompt"]),
            "response_sha256": _sha_text(row["response"]),
            "ordinary_correct": row["ordinary_correct"],
            "answer_match": row["shadow_row"].get("answer_match"),
            "qualified_gain_candidate": row["shadow_row"].get(
                "qualified_gain_candidate"
            ),
            "shadow_receipt_sha256": row["shadow_row"].get("receipt_sha256"),
            "contract_issues": row["contract_issues"],
            "shadow_issues": row["shadow_issues"],
        }
        for row in rows
    ]
    body["rows_sha256"] = _canonical_sha(body["rows"])
    return {**body, "summary_sha256": _canonical_sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seed", type=int, default=2026081568)
    parser.add_argument("--tasks-per-difficulty", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = asyncio.run(
            run_live_shadow(
                base_url=args.base_url,
                seed=args.seed,
                tasks_per_difficulty=args.tasks_per_difficulty,
                timeout_s=args.timeout,
                ledger_path=args.ledger,
                private_output=args.private_output,
            )
        )
        summary = _sanitized(document)
        atomic_write_text(
            args.summary_output,
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"semantic neural live shadow failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_requests_proven_ordinary_authority"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
