#!/usr/bin/env python3
"""Run one crash-resumable RLC pilot under an OS-restartable controller.

The campaign runner owns the scientific journal and exact resume semantics.
This controller owns only process liveness: it verifies an immutable source
checkout, records authenticated attempt events, adopts a still-running child
after controller restart, and retries infrastructure exits until the
independent evidence verifier accepts the terminal campaign. Direct attempts
run under caffeinate here. Brokered attempts require the launchd job to run
this controller under caffeinate because the detached target is intentionally
confined by a no-fork sandbox.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    CampaignPlan,
    canonical_json_bytes,
)
from core.runtime.atomic_writer import atomic_write_bytes, ensure_private_directory  # noqa: E402
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402

CONFIG_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_config.v2"
STATE_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_state.v2"
STATUS_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_status.v2"
EVENT_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_event.v2"
PROGRESS_SCHEMA = "aura.latent_cortex.campaign_progress.v1"
CAMPAIGN_EVENT_SCHEMA = "aura.latent_cortex.campaign_event.v1"
MAX_CONFIG_BYTES = 16 * 1024 * 1024
MAX_CAMPAIGN_JOURNAL_BYTES = 1024 * 1024 * 1024
_active_pgid = 0
_stop_signal = 0
_progress_cache: dict[str, tuple[tuple[int, int, int, int], dict[str, Any]]] = {}
_progress_last_valid: dict[str, dict[str, Any]] = {}


class PilotControllerError(RuntimeError):
    pass


def _fail(code: str) -> Never:
    raise PilotControllerError(code)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_path(
    value: Any,
    *,
    role: str,
    must_exist: bool = True,
    reject_symlink: bool = True,
) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        _fail(f"{role}_path_invalid")
    path = Path(value)
    if reject_symlink and path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    if reject_symlink:
        return path.resolve(strict=must_exist)
    if must_exist and not path.exists():
        _fail(f"{role}_path_missing")
    return path


def _command(value: Any, *, role: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        _fail(f"{role}_command_invalid")
    executable = _absolute_path(value[0], role=f"{role}_executable", reject_symlink=False)
    if not os.access(executable, os.X_OK):
        _fail(f"{role}_executable_not_executable")
    return [str(executable), *value[1:]]


def _verify_mutable_output_excludes_explicit_inputs(
    *,
    source_root: Path,
    output_root: Path,
    runner: list[str],
    broker_policy: list[Any],
) -> None:
    commands: list[tuple[str, list[str], Path]] = [("runner", runner, source_root)]
    for index, specification in enumerate(broker_policy):
        if not isinstance(specification, Mapping):
            _fail("detached_broker_policy_invalid")
        command = specification.get("command")
        cwd_value = specification.get("cwd")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
            or not isinstance(cwd_value, str)
            or not cwd_value
        ):
            _fail("detached_broker_policy_invalid")
        cwd = _absolute_path(cwd_value, role=f"broker_{index}_cwd")
        if not cwd.is_dir():
            _fail("detached_broker_policy_invalid")
        commands.append((f"broker_{index}", command, cwd))

    for role, command, cwd in commands:
        try:
            explicit_inputs = detached._source_file_arguments(command, cwd)
        except (OSError, ValueError) as exc:
            raise PilotControllerError(f"{role}_explicit_input_inspection_failed") from exc
        if any(path == output_root or path.is_relative_to(output_root) for path in explicit_inputs):
            _fail(f"{role}_explicit_input_inside_execution_output_root")


def load_config(path: Path) -> dict[str, Any]:
    raw = read_stable_bytes(path.resolve(strict=True), max_bytes=MAX_CONFIG_BYTES)
    try:
        config = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PilotControllerError("controller_config_json_invalid") from exc
    required = {
        "schema",
        "campaign_name",
        "source_root",
        "source_commit",
        "campaign_dir",
        "state_dir",
        "runner_command",
        "runner_command_sha256",
        "runner_executable_sha256",
        "verifier_command",
        "verifier_command_sha256",
        "verifier_executable_sha256",
        "detached_broker_policy",
        "detached_broker_policy_sha256",
        "detached_attempt_timeout_seconds",
        "execution_output_root",
        "max_attempts",
        "retry_backoff_seconds",
        "heartbeat_seconds",
        "config_sha256",
    }
    body = dict(config) if isinstance(config, dict) else {}
    claimed = body.pop("config_sha256", None)
    if (
        not isinstance(config, dict)
        or set(config) != required
        or config.get("schema") != CONFIG_SCHEMA
        or claimed != _sha256(body)
    ):
        _fail("controller_config_invalid")
    if not _is_git_commit(config.get("source_commit")):
        _fail("controller_source_commit_invalid")
    for key in (
        "runner_command_sha256",
        "runner_executable_sha256",
        "verifier_command_sha256",
        "verifier_executable_sha256",
    ):
        if not _is_sha256(config.get(key)):
            _fail(f"controller_{key}_invalid")
    runner = _command(config.get("runner_command"), role="runner")
    verifier = _command(config.get("verifier_command"), role="verifier")
    if _sha256(runner) != config["runner_command_sha256"]:
        _fail("runner_command_hash_mismatch")
    if _sha256(verifier) != config["verifier_command_sha256"]:
        _fail("verifier_command_hash_mismatch")
    broker_policy = config.get("detached_broker_policy")
    if not isinstance(broker_policy, list) or config.get(
        "detached_broker_policy_sha256"
    ) != _sha256(broker_policy):
        _fail("detached_broker_policy_invalid")
    _verify_executables({**config, "runner_command": runner, "verifier_command": verifier})
    source_root = _absolute_path(config.get("source_root"), role="source_root")
    campaign_dir = _absolute_path(config.get("campaign_dir"), role="campaign_dir", must_exist=False)
    state_dir = _absolute_path(config.get("state_dir"), role="state_dir", must_exist=False)
    output_root = _absolute_path(
        config.get("execution_output_root"),
        role="execution_output_root",
        must_exist=False,
    )
    if output_root == source_root or not output_root.is_relative_to(source_root):
        _fail("execution_output_root_invalid")
    _verify_mutable_output_excludes_explicit_inputs(
        source_root=source_root,
        output_root=output_root,
        runner=runner,
        broker_policy=broker_policy,
    )
    for name, minimum, maximum in (
        ("max_attempts", 1, 100),
        ("retry_backoff_seconds", 1, 3600),
        ("heartbeat_seconds", 1, 300),
        ("detached_attempt_timeout_seconds", 60, 172800),
    ):
        value = config.get(name)
        if type(value) not in {int, float} or not minimum <= float(value) <= maximum:
            _fail(f"controller_{name}_invalid")
    return {
        **config,
        "source_root": str(source_root),
        "campaign_dir": str(campaign_dir),
        "state_dir": str(state_dir),
        "execution_output_root": str(output_root),
        "runner_command": runner,
        "verifier_command": verifier,
    }


def _executable_sha256(path: str) -> str:
    return hashlib.sha256(
        read_stable_bytes(Path(path).resolve(strict=True), max_bytes=1024 * 1024 * 1024)
    ).hexdigest()


def _verify_executables(config: Mapping[str, Any]) -> None:
    for role in ("runner", "verifier"):
        command = config.get(f"{role}_command")
        if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
            _fail(f"{role}_command_invalid")
        if _executable_sha256(str(command[0])) != config.get(f"{role}_executable_sha256"):
            _fail(f"{role}_executable_hash_mismatch")


def verify_source(config: Mapping[str, Any]) -> None:
    _verify_executables(config)
    source_root = str(config["source_root"])
    head = subprocess.run(
        ["/usr/bin/git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    tracked = subprocess.run(
        ["/usr/bin/git", "-c", "core.fsmonitor=false", "status", "--porcelain", "--untracked-files=no"],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != config["source_commit"]
        or tracked.returncode != 0
        or tracked.stdout.strip()
    ):
        _fail("controller_source_identity_changed")


def _default_state(config: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": STATE_SCHEMA,
        "campaign_name": config["campaign_name"],
        "config_sha256": config["config_sha256"],
        "attempts_started": 0,
        "active_child_pid": 0,
        "active_child_pgid": 0,
        "active_child_command_sha256": "",
        "active_detached_run_dir": "",
        "journal_head_sha256": "0" * 64,
        "journal_sequence": 0,
        "terminal": False,
    }
    return {**body, "state_sha256": _sha256(body)}


def _read_state(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _default_state(config)
    raw = read_stable_bytes(path.resolve(strict=True), max_bytes=4 * 1024 * 1024)
    try:
        state = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PilotControllerError("controller_state_json_invalid") from exc
    body = dict(state) if isinstance(state, dict) else {}
    claimed = body.pop("state_sha256", None)
    if (
        not isinstance(state, dict)
        or state.get("schema") != STATE_SCHEMA
        or state.get("campaign_name") != config["campaign_name"]
        or state.get("config_sha256") != config["config_sha256"]
        or claimed != _sha256(body)
    ):
        _fail("controller_state_invalid")
    return state


def _write_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in state.items() if key != "state_sha256"}
    document = {**body, "state_sha256": _sha256(body)}
    atomic_write_bytes(path, canonical_json_bytes(document) + b"\n", mode=0o600)
    return document


def _campaign_progress(campaign_dir: Path) -> dict[str, Any]:
    """Return integrity-checked progress without taking the campaign writer lock."""

    plan_path = campaign_dir / "plan.json"
    journal_path = campaign_dir / "campaign.jsonl"
    cache_key = str(journal_path)
    if not plan_path.exists() or not journal_path.exists():
        return {"schema": PROGRESS_SCHEMA, "availability": "waiting_for_plan"}
    try:
        metadata = journal_path.stat(follow_symlinks=False)
        if journal_path.is_symlink() or plan_path.is_symlink():
            _fail("campaign_progress_symlink_rejected")
        identity = (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
        )
        cached = _progress_cache.get(cache_key)
        if cached is not None and cached[0] == identity:
            return dict(cached[1])

        raw_plan = read_stable_bytes(plan_path, max_bytes=MAX_CONFIG_BYTES)
        plan_document = json.loads(raw_plan)
        if not isinstance(plan_document, dict):
            _fail("campaign_progress_plan_invalid")
        plan = CampaignPlan.from_dict(plan_document)
        cells = plan.to_dict()["cells"]
        definitions = {cell["cell_id"]: cell["definition"] for cell in cells}

        raw_journal = read_stable_bytes(
            journal_path,
            max_bytes=MAX_CAMPAIGN_JOURNAL_BYTES,
        )
        if not raw_journal.endswith(b"\n"):
            _fail("campaign_progress_journal_torn")

        previous = "0" * 64
        active: dict[str, tuple[str, str]] = {}
        attempts: dict[str, tuple[str, str]] = {}
        start_counts: dict[str, int] = {}
        result_cells: set[str] = set()
        verified_cells: set[str] = set()
        committed_cells: set[str] = set()
        failed_attempts = 0
        started_attempts = 0
        latency_by_arm: dict[str, list[float]] = {}
        records = raw_journal.splitlines()
        event_keys = {
            "schema",
            "sequence",
            "plan_sha256",
            "previous_event_sha256",
            "event",
            "cell_id",
            "attempt_id",
            "payload",
            "event_sha256",
        }
        for sequence, raw_record in enumerate(records):
            record = json.loads(raw_record)
            if (
                not isinstance(record, dict)
                or set(record) != event_keys
                or canonical_json_bytes(record) != raw_record
            ):
                _fail("campaign_progress_event_invalid")
            claimed = record["event_sha256"]
            body = {key: value for key, value in record.items() if key != "event_sha256"}
            if (
                record["schema"] != CAMPAIGN_EVENT_SCHEMA
                or record["sequence"] != sequence
                or record["plan_sha256"] != plan.plan_sha256
                or record["previous_event_sha256"] != previous
                or not _is_sha256(claimed)
                or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != claimed
            ):
                _fail("campaign_progress_hash_chain_invalid")
            previous = claimed
            event = record["event"]
            payload = record["payload"]
            cell_id = record["cell_id"]
            attempt_id = record["attempt_id"]
            if sequence == 0:
                if (
                    event != "PLAN"
                    or cell_id is not None
                    or attempt_id is not None
                    or payload != {"plan": plan.to_dict()}
                ):
                    _fail("campaign_progress_genesis_invalid")
                continue
            if (
                cell_id not in definitions
                or not isinstance(attempt_id, str)
                or not isinstance(payload, dict)
            ):
                _fail("campaign_progress_cell_event_invalid")
            if event == "STARTED":
                attempt_number = payload.get("attempt_number")
                expected_number = start_counts.get(cell_id, 0) + 1
                attempt_material = {
                    "attempt_number": expected_number,
                    "cell_id": cell_id,
                    "plan_sha256": plan.plan_sha256,
                    "schema": "aura.latent_cortex.campaign_attempt.v1",
                }
                expected_attempt_id = (
                    f"attempt-{hashlib.sha256(canonical_json_bytes(attempt_material)).hexdigest()}"
                )
                if (
                    set(payload) != {"attempt_number"}
                    or type(attempt_number) is not int
                    or attempt_number != expected_number
                    or attempt_id != expected_attempt_id
                    or cell_id in active
                    or cell_id in committed_cells
                    or attempt_id in attempts
                ):
                    _fail("campaign_progress_transition_invalid")
                start_counts[cell_id] = expected_number
                attempts[attempt_id] = (cell_id, event)
                active[cell_id] = (attempt_id, event)
                started_attempts += 1
                continue
            current = attempts.get(attempt_id)
            if current is None or current[0] != cell_id or active.get(cell_id, (None,))[0] != attempt_id:
                _fail("campaign_progress_attempt_invalid")
            prior = current[1]
            if event == "ACTION_INTERVENTION_CLAIMED" and prior == "STARTED":
                if set(payload) != {
                    "intervention_sha256",
                    "request_payload_sha256",
                    "signed_journal_event_count",
                    "signed_journal_head_sha256",
                }:
                    _fail("campaign_progress_intervention_claim_invalid")
                attempts[attempt_id] = (cell_id, event)
            elif event == "ARM_RESULT" and prior in {"STARTED", "ACTION_INTERVENTION_CLAIMED"}:
                result = payload.get("result")
                if set(payload) != {"result"} or not isinstance(result, dict):
                    _fail("campaign_progress_result_invalid")
                attempts[attempt_id] = (cell_id, event)
                active[cell_id] = (attempt_id, event)
                result_cells.add(cell_id)
                latency = result.get("latency_s")
                if type(latency) in {int, float} and math.isfinite(float(latency)) and latency >= 0:
                    arm = str(definitions[cell_id].get("arm", "unknown"))
                    latency_by_arm.setdefault(arm, []).append(float(latency))
            elif event == "VERIFIED" and prior == "ARM_RESULT":
                if set(payload) != {"verification"} or not isinstance(
                    payload.get("verification"), dict
                ):
                    _fail("campaign_progress_verification_invalid")
                attempts[attempt_id] = (cell_id, event)
                active[cell_id] = (attempt_id, event)
                verified_cells.add(cell_id)
            elif event == "COMMITTED" and prior == "VERIFIED":
                if set(payload) != {"commit"} or not isinstance(payload.get("commit"), dict):
                    _fail("campaign_progress_commit_invalid")
                attempts[attempt_id] = (cell_id, event)
                committed_cells.add(cell_id)
                active.pop(cell_id, None)
            elif event == "FAILED" and prior != "ACTION_INTERVENTION_CLAIMED":
                if (
                    set(payload) != {"details", "reason"}
                    or not isinstance(payload.get("details"), dict)
                    or not isinstance(payload.get("reason"), str)
                    or not payload["reason"]
                    or payload["reason"] != payload["reason"].strip()
                ):
                    _fail("campaign_progress_failure_invalid")
                attempts[attempt_id] = (cell_id, event)
                active.pop(cell_id, None)
                failed_attempts += 1
            else:
                _fail("campaign_progress_transition_invalid")

        by_arm: dict[str, dict[str, int]] = {}
        for cell_id, definition in definitions.items():
            arm = str(definition.get("arm", "unknown"))
            summary = by_arm.setdefault(
                arm,
                {"planned": 0, "sealed_results": 0, "verified": 0, "committed": 0},
            )
            summary["planned"] += 1
            summary["sealed_results"] += int(cell_id in result_cells)
            summary["verified"] += int(cell_id in verified_cells or cell_id in committed_cells)
            summary["committed"] += int(cell_id in committed_cells)

        observed_means = {
            arm: sum(values) / len(values) for arm, values in latency_by_arm.items() if values
        }
        global_projection = max(observed_means.values(), default=0.0)
        projected_remaining = 0.0
        if global_projection:
            for arm, summary in by_arm.items():
                remaining = summary["planned"] - summary["sealed_results"]
                projected_remaining += remaining * observed_means.get(arm, global_projection)
        all_latencies = [value for values in latency_by_arm.values() for value in values]
        active_cells = [
            {
                "arm": definitions[cell_id].get("arm"),
                "cell_id": cell_id,
                "domain": definitions[cell_id].get("domain"),
                "state": state,
                "task_id": definitions[cell_id].get("task_id"),
            }
            for cell_id, (_attempt_id, state) in active.items()
            if state in {"STARTED", "ACTION_INTERVENTION_CLAIMED"}
        ]
        progress = {
            "schema": PROGRESS_SCHEMA,
            "availability": "current",
            "plan_sha256": plan.plan_sha256,
            "journal_head_sha256": previous,
            "journal_event_count": len(records),
            "total_cells": len(definitions),
            "started_attempts": started_attempts,
            "sealed_result_cells": len(result_cells),
            "verified_cells": len(verified_cells | committed_cells),
            "committed_cells": len(committed_cells),
            "failed_attempts": failed_attempts,
            "active_cells": active_cells,
            "by_arm": by_arm,
            "observed_cell_latency_seconds": {
                "count": len(all_latencies),
                "mean": sum(all_latencies) / len(all_latencies) if all_latencies else None,
                "minimum": min(all_latencies) if all_latencies else None,
                "maximum": max(all_latencies) if all_latencies else None,
            },
            "projection": {
                "method": "per_arm_observed_mean_else_slowest_observed_arm",
                "projected_remaining_inference_seconds": (
                    projected_remaining if global_projection else None
                ),
                "confidence": "low" if len(all_latencies) < 8 else "provisional",
                "excludes": [
                    "future_model_load_and_warmup",
                    "post_seal_scoring",
                    "independent_verification",
                    "fusion_validation",
                ],
            },
        }
        _progress_cache[cache_key] = (identity, progress)
        _progress_last_valid[cache_key] = progress
        return dict(progress)
    except (OSError, TypeError, ValueError, PilotControllerError) as exc:
        prior = _progress_last_valid.get(cache_key)
        if prior is not None:
            return {
                **prior,
                "availability": "stale",
                "stale_reason": type(exc).__name__,
            }
        return {
            "schema": PROGRESS_SCHEMA,
            "availability": "unavailable",
            "reason": type(exc).__name__,
        }


def _append_event(path: Path, state: dict[str, Any], event: str, detail: Any) -> None:
    body = {
        "schema": EVENT_SCHEMA,
        "sequence": int(state["journal_sequence"]) + 1,
        "previous_event_sha256": state["journal_head_sha256"],
        "event": event,
        "recorded_at_unix": time.time(),
        "detail": detail,
    }
    document = {**body, "event_sha256": _sha256(body)}
    with path.open("ab", buffering=0) as handle:
        handle.write(canonical_json_bytes(document) + b"\n")
        os.fsync(handle.fileno())
    state["journal_sequence"] = document["sequence"]
    state["journal_head_sha256"] = document["event_sha256"]


def _reconcile_event_journal(path: Path, state: dict[str, Any]) -> bool:
    previous = "0" * 64
    sequence = 0
    documents: list[dict[str, Any]] = []
    if path.exists():
        for raw_line in path.read_bytes().splitlines():
            try:
                document = json.loads(raw_line)
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise PilotControllerError("controller_event_journal_invalid") from exc
            body = dict(document) if isinstance(document, dict) else {}
            claimed = body.pop("event_sha256", None)
            sequence += 1
            if (
                not isinstance(document, dict)
                or document.get("schema") != EVENT_SCHEMA
                or document.get("sequence") != sequence
                or document.get("previous_event_sha256") != previous
                or claimed != _sha256(body)
            ):
                _fail("controller_event_journal_invalid")
            previous = str(claimed)
            documents.append(document)
    state_sequence = state.get("journal_sequence")
    if type(state_sequence) is not int or not 0 <= state_sequence <= sequence:
        _fail("controller_event_state_mismatch")
    if state_sequence == sequence:
        if state.get("journal_head_sha256") != previous:
            _fail("controller_event_state_mismatch")
        return False
    for document in documents[state_sequence:]:
        detail = document.get("detail")
        if not isinstance(detail, Mapping):
            _fail("controller_event_recovery_invalid")
        event = document.get("event")
        if event == "ATTEMPT_RESERVED":
            state["attempts_started"] = detail.get("attempt")
            state["active_detached_run_dir"] = detail.get("detached_run_dir", "")
        elif event in {"ATTEMPT_STARTED", "ATTEMPT_ADOPTED"}:
            state["attempts_started"] = detail.get("attempt")
            state["active_child_pid"] = detail.get("pid")
            state["active_child_pgid"] = detail.get("pgid")
            state["active_child_command_sha256"] = detail.get("command_sha256")
            state["active_detached_run_dir"] = detail.get("detached_run_dir", "")
        elif event == "ATTEMPT_EXITED":
            state["active_child_pid"] = 0
            state["active_child_pgid"] = 0
            state["active_child_command_sha256"] = ""
            state["active_detached_run_dir"] = ""
        elif event == "VERIFIED_TERMINAL":
            state["terminal"] = True
        state["journal_sequence"] = document["sequence"]
        state["journal_head_sha256"] = document["event_sha256"]
    return True


def _write_status(
    path: Path,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    phase: str,
    reason: str = "",
) -> None:
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_name": config["campaign_name"],
        "controller_pid": os.getpid(),
        "phase": phase,
        "reason": reason,
        "heartbeat_at_unix": time.time(),
        "attempts_started": state["attempts_started"],
        "max_attempts": config["max_attempts"],
        "active_child_pid": state["active_child_pid"],
        "active_child_pgid": state["active_child_pgid"],
        "active_detached_run_dir": state.get("active_detached_run_dir", ""),
        "campaign_dir": config["campaign_dir"],
        "campaign_progress": _campaign_progress(Path(str(config["campaign_dir"]))),
        "source_commit": config["source_commit"],
        "config_sha256": config["config_sha256"],
    }
    atomic_write_bytes(
        path,
        canonical_json_bytes({**body, "status_sha256": _sha256(body)}) + b"\n",
        mode=0o600,
    )


def _verify_terminal(config: Mapping[str, Any], log: Any) -> bool:
    completed = subprocess.run(
        list(config["verifier_command"]),
        cwd=config["source_root"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=1800,
        check=False,
    )
    return completed.returncode == 0


def _pid_matches(pid: int, command_sha256: str) -> bool:
    if pid <= 1 or not _is_sha256(command_sha256):
        return False
    completed = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return False
    return hashlib.sha256(completed.stdout.strip().encode()).hexdigest() == command_sha256


def _signal(signum: int, _frame: Any) -> None:
    global _stop_signal
    _stop_signal = signum
    if _active_pgid > 1:
        try:
            os.killpg(_active_pgid, signum)
        except ProcessLookupError:
            pass


def _monitor_pid(
    pid: int,
    *,
    config: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    status_path: Path,
) -> None:
    while _pid_matches(pid, str(state["active_child_command_sha256"])):
        _write_status(status_path, config, state, phase="running")
        if _stop_signal:
            return
        time.sleep(float(config["heartbeat_seconds"]))
    state["active_child_pid"] = 0
    state["active_child_pgid"] = 0
    state["active_child_command_sha256"] = ""
    _write_state(state_path, state)


def _stop_detached(run_dir: Path) -> None:
    detached.main(["stop", "--run-dir", str(run_dir)])


def _run_brokered_attempt(
    *,
    config: Mapping[str, Any],
    state: dict[str, Any],
    state_path: Path,
    status_path: Path,
    event_path: Path,
) -> int:
    existing = str(state.get("active_detached_run_dir") or "")
    attempt = int(state["attempts_started"]) if existing else int(state["attempts_started"]) + 1
    run_dir = Path(str(config["state_dir"])) / "detached-attempts" / f"attempt-{attempt:04d}"
    if existing and Path(existing) != run_dir:
        _fail("detached_attempt_state_mismatch")
    if not existing:
        state["attempts_started"] = attempt
        state["active_detached_run_dir"] = str(run_dir)
        _append_event(
            event_path,
            state,
            "ATTEMPT_RESERVED",
            {"attempt": attempt, "detached_run_dir": str(run_dir)},
        )
        state.update(_write_state(state_path, state))
    launched = not (run_dir / detached.PLAN_FILE).exists()
    if launched:
        launch_args = [
            "launch",
            "--run-dir",
            str(run_dir),
            "--name",
            f"{config['campaign_name']}-attempt-{attempt}",
            "--cwd",
            str(config["source_root"]),
            "--timeout",
            str(config["detached_attempt_timeout_seconds"]),
            "--broker-policy-json",
            json.dumps(
                config["detached_broker_policy"],
                sort_keys=True,
                separators=(",", ":"),
            ),
            "--execution-output-root",
            str(config["execution_output_root"]),
            "--",
            *config["runner_command"],
        ]
        if detached.main(launch_args) != 0:
            _fail("detached_attempt_launch_failed")
        observed = detached._status(run_dir)
        state["active_child_pid"] = int(observed.get("child_pid") or 0)
        state["active_child_pgid"] = int(observed.get("child_process_group_id") or 0)
        state["active_child_command_sha256"] = ""
        _append_event(
            event_path,
            state,
            "ATTEMPT_STARTED",
            {
                "attempt": attempt,
                "pid": state["active_child_pid"],
                "pgid": state["active_child_pgid"],
                "command_sha256": "",
                "detached_run_dir": str(run_dir),
            },
        )
        state.update(_write_state(state_path, state))
    else:
        observed = detached._status(run_dir)
        state["active_child_pid"] = int(observed.get("child_pid") or 0)
        state["active_child_pgid"] = int(observed.get("child_process_group_id") or 0)
        _append_event(
            event_path,
            state,
            "ATTEMPT_ADOPTED",
            {
                "attempt": attempt,
                "pid": state["active_child_pid"],
                "pgid": state["active_child_pgid"],
                "command_sha256": "",
                "detached_run_dir": str(run_dir),
            },
        )
        state.update(_write_state(state_path, state))
    deadline = time.monotonic() + float(config["detached_attempt_timeout_seconds"]) + 120
    while True:
        observed = detached._status(run_dir)
        state["active_child_pid"] = int(observed.get("child_pid") or 0)
        state["active_child_pgid"] = int(observed.get("child_process_group_id") or 0)
        state.update(_write_state(state_path, state))
        _write_status(status_path, config, state, phase="running")
        if observed.get("terminal") is True:
            receipt = observed.get("receipt")
            if not isinstance(receipt, Mapping) or type(receipt.get("returncode")) is not int:
                _fail("detached_attempt_receipt_invalid")
            return int(receipt["returncode"])
        if observed.get("completion_indeterminate") is True:
            return 125
        if _stop_signal:
            _stop_detached(run_dir)
            return 128 + _stop_signal
        if time.monotonic() >= deadline:
            _stop_detached(run_dir)
            return 124
        time.sleep(float(config["heartbeat_seconds"]))


def run(config: Mapping[str, Any]) -> int:
    global _active_pgid
    verify_source(config)
    state_dir = ensure_private_directory(Path(str(config["state_dir"])))
    Path(str(config["campaign_dir"])).mkdir(parents=True, exist_ok=True, mode=0o700)
    state_path = state_dir / "controller-state.json"
    status_path = state_dir / "controller-status.json"
    event_path = state_dir / "controller-events.jsonl"
    log_path = state_dir / "controller.log"
    lock_path = state_dir / "controller.lock"
    lock = lock_path.open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        _fail("controller_already_running")
    state = _read_state(state_path, config)
    if _reconcile_event_journal(event_path, state):
        state = _write_state(state_path, state)
    brokered = bool(config["detached_broker_policy"])
    with lock, log_path.open("ab", buffering=0) as log:
        if not state.get("active_detached_run_dir") and _verify_terminal(config, log):
            state["terminal"] = True
            _append_event(event_path, state, "VERIFIED_TERMINAL", {})
            state = _write_state(state_path, state)
            _write_status(status_path, config, state, phase="complete")
            return 0
        active_pid = int(state.get("active_child_pid") or 0)
        if not brokered and _pid_matches(
            active_pid, str(state.get("active_child_command_sha256") or "")
        ):
            _active_pgid = int(state.get("active_child_pgid") or 0)
            _append_event(event_path, state, "ADOPTED_ACTIVE_CHILD", {"pid": active_pid})
            state = _write_state(state_path, state)
            _monitor_pid(
                active_pid,
                config=config,
                state=state,
                state_path=state_path,
                status_path=status_path,
            )
        caffeinate = shutil.which("caffeinate") if not brokered else None
        if not brokered and not caffeinate:
            _fail("caffeinate_unavailable")
        while state.get("active_detached_run_dir") or int(state["attempts_started"]) < int(
            config["max_attempts"]
        ):
            if not state.get("active_detached_run_dir") and _verify_terminal(config, log):
                state["terminal"] = True
                _append_event(event_path, state, "VERIFIED_TERMINAL", {})
                state = _write_state(state_path, state)
                _write_status(status_path, config, state, phase="complete")
                return 0
            verify_source(config)
            if brokered:
                returncode = _run_brokered_attempt(
                    config=config,
                    state=state,
                    state_path=state_path,
                    status_path=status_path,
                    event_path=event_path,
                )
                state["active_child_pid"] = 0
                state["active_child_pgid"] = 0
                state["active_child_command_sha256"] = ""
                state["active_detached_run_dir"] = ""
                _append_event(
                    event_path,
                    state,
                    "ATTEMPT_EXITED",
                    {
                        "attempt": state["attempts_started"],
                        "returncode": returncode,
                    },
                )
                state = _write_state(state_path, state)
                if _stop_signal:
                    return 128 + _stop_signal
                if _verify_terminal(config, log):
                    state["terminal"] = True
                    _append_event(event_path, state, "VERIFIED_TERMINAL", {})
                    state = _write_state(state_path, state)
                    _write_status(status_path, config, state, phase="complete")
                    return 0
                _write_status(
                    status_path,
                    config,
                    state,
                    phase="retry_wait",
                    reason=f"runner_exit_{returncode}",
                )
                time.sleep(float(config["retry_backoff_seconds"]))
                continue
            environment = dict(os.environ)
            environment["AURA_LOG_DIR"] = str(ensure_private_directory(state_dir / "aura-logs"))
            child = subprocess.Popen(
                [str(caffeinate), "-dims", *config["runner_command"]],
                cwd=config["source_root"],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _active_pgid = os.getpgid(child.pid)
            ps_command = subprocess.run(
                ["/bin/ps", "-p", str(child.pid), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
            state["attempts_started"] = int(state["attempts_started"]) + 1
            state["active_child_pid"] = child.pid
            state["active_child_pgid"] = _active_pgid
            state["active_child_command_sha256"] = hashlib.sha256(ps_command.encode()).hexdigest()
            _append_event(
                event_path,
                state,
                "ATTEMPT_STARTED",
                {
                    "attempt": state["attempts_started"],
                    "pid": child.pid,
                    "pgid": _active_pgid,
                    "command_sha256": state["active_child_command_sha256"],
                },
            )
            state = _write_state(state_path, state)
            while child.poll() is None:
                _write_status(status_path, config, state, phase="running")
                if _stop_signal:
                    child.wait(timeout=60)
                    return 128 + _stop_signal
                time.sleep(float(config["heartbeat_seconds"]))
            returncode = child.returncode
            state["active_child_pid"] = 0
            state["active_child_pgid"] = 0
            state["active_child_command_sha256"] = ""
            _append_event(
                event_path,
                state,
                "ATTEMPT_EXITED",
                {"attempt": state["attempts_started"], "returncode": returncode},
            )
            state = _write_state(state_path, state)
            if _verify_terminal(config, log):
                state["terminal"] = True
                _append_event(event_path, state, "VERIFIED_TERMINAL", {})
                state = _write_state(state_path, state)
                _write_status(status_path, config, state, phase="complete")
                return 0
            _write_status(
                status_path,
                config,
                state,
                phase="retry_wait",
                reason=f"runner_exit_{returncode}",
            )
            time.sleep(float(config["retry_backoff_seconds"]))
        _append_event(event_path, state, "ATTEMPTS_EXHAUSTED", {})
        state = _write_state(state_path, state)
        _write_status(status_path, config, state, phase="blocked", reason="attempts_exhausted")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _signal)
    try:
        return run(load_config(args.config))
    except (OSError, ValueError, subprocess.SubprocessError, PilotControllerError) as exc:
        print(f"pilot controller failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
