#!/usr/bin/env python3
"""Run all admitted candidate-cortex stages with durable stage boundaries."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.learning.candidate_cortex_admission import (  # noqa: E402
    adjudicate_checkpoint_evidence,
)
from core.learning.candidate_cortex_training import (  # noqa: E402
    ADAPTIVE_RESULT_SCHEMA,
    ADMISSION_SCHEMA,
    JOURNAL_FILE,
    OBSERVATION_SCHEMA,
    CandidateCortexTrainingError,
    StagePolicy,
    append_authenticated_event,
    build_stage_command,
    canonical_json_bytes,
    discover_exact_checkpoint,
    document_sha256,
    execution_admission,
    file_sha256,
    load_and_verify_plan,
    next_stage_plan,
    read_authenticated_journal,
    stage_adapter_root,
    validate_stage_observation,
)
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402
from tools import measure_candidate_cortex_checkpoint as measurement  # noqa: E402
from tools import run_detached_step as detached  # noqa: E402
from tools.run_candidate_cortex_canary_target import _mlx_arguments  # noqa: E402

STAGE_COMPLETION_SCHEMA = "aura.candidate_cortex_training.stage_completion.v1"
PHASE_BOUNDARY_SCHEMA = "aura.candidate_cortex_training.phase_boundary.v1"


def _key(path: Path) -> bytes:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise CandidateCortexTrainingError("journal_key_invalid")
    payload = resolved.read_bytes()
    if len(payload) < 32:
        raise CandidateCortexTrainingError("journal_key_too_short")
    return payload


def _stage_parent(plan: Mapping[str, Any], stage_index: int) -> Path:
    return stage_adapter_root(plan, stage_index).parent


def _completion_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "stage_completion.json"


def _evidence_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "checkpoint_evidence.json"


def _detail_path(plan: Mapping[str, Any], stage_index: int) -> Path:
    return _stage_parent(plan, stage_index) / "checkpoint_measurement_detail.json"


def _strict_document(path: Path) -> dict[str, Any]:
    value = measurement._strict_json(path)  # noqa: SLF001 - shared strict boundary
    if not isinstance(value, dict):
        raise CandidateCortexTrainingError("adaptive_document_invalid")
    return value


def _write_once(path: Path, value: Mapping[str, Any], *, source: str) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with local_internal_governed_scope(source, domain="file_write"):
        created = get_file_write_gateway().write_bytes_if_absent(
            path,
            payload,
            mode=0o600,
            source=source,
        )
    if not created and path.read_bytes() != payload:
        raise CandidateCortexTrainingError("adaptive_output_conflict")


def _validated_completion(
    plan: Mapping[str, Any], stage_index: int
) -> dict[str, Any] | None:
    path = _completion_path(plan, stage_index)
    if not path.is_file():
        return None
    document = _strict_document(path)
    material = dict(document)
    claimed = material.pop("completion_sha256", None)
    policy = StagePolicy(**dict(plan["stages"]))
    required = {
        "schema",
        "plan_sha256",
        "stage_index",
        "stage_iterations",
        "cumulative_iterations",
        "training_command_sha256",
        "checkpoint",
        "target_pid",
        "target_start_token",
        "host_metrics",
        "completion_sha256",
    }
    if (
        set(document) != required
        or document.get("schema") != STAGE_COMPLETION_SCHEMA
        or document.get("plan_sha256") != plan["plan_sha256"]
        or document.get("stage_index") != stage_index
        or document.get("stage_iterations") != policy.iterations(stage_index)
        or document.get("cumulative_iterations")
        != policy.cumulative_iterations(stage_index)
        or claimed != document_sha256(material)
    ):
        raise CandidateCortexTrainingError("stage_completion_invalid")
    checkpoint = discover_exact_checkpoint(
        Path(str(plan["paths"]["checkpoint_root"])),
        expected_cumulative_iterations=policy.cumulative_iterations(stage_index),
    )
    if document.get("checkpoint") != checkpoint:
        raise CandidateCortexTrainingError("stage_completion_checkpoint_drift")
    return document


def _reset_incomplete_stage(plan: Mapping[str, Any], stage_index: int) -> None:
    parent = _stage_parent(plan, stage_index)
    policy = StagePolicy(**dict(plan["stages"]))
    canonical = (
        Path(str(plan["paths"]["checkpoint_root"]))
        / f"{policy.cumulative_iterations(stage_index):07d}_adapters.safetensors"
    )
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.reset_stage", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if parent.exists() or parent.is_symlink():
            gateway.delete_path(
                parent,
                recursive=True,
                source="candidate_cortex_adaptive.reset_stage",
            )
        if canonical.exists() or canonical.is_symlink():
            gateway.delete_file(
                canonical,
                source="candidate_cortex_adaptive.reset_stage",
            )
        gateway.ensure_directory(
            stage_adapter_root(plan, stage_index),
            source="candidate_cortex_adaptive.reset_stage",
        )


def _sample_host(stop: threading.Event, state: dict[str, Any]) -> None:
    process = psutil.Process(os.getpid())
    while True:
        virtual = psutil.virtual_memory()
        try:
            rss = int(process.memory_info().rss)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            rss = 0
        state["sample_count"] += 1
        state["min_available_bytes"] = min(
            state["min_available_bytes"], int(virtual.available)
        )
        state["max_used_percent"] = max(
            state["max_used_percent"], float(virtual.percent)
        )
        state["max_process_rss_bytes"] = max(state["max_process_rss_bytes"], rss)
        if stop.wait(0.5):
            return


def _publish_checkpoint(
    plan: Mapping[str, Any],
    stage_index: int,
    command: tuple[str, ...],
    host_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    policy = StagePolicy(**dict(plan["stages"]))
    local_root = stage_adapter_root(plan, stage_index)
    local = discover_exact_checkpoint(
        local_root,
        expected_cumulative_iterations=policy.iterations(stage_index),
    )
    alias = local_root / "adapters.safetensors"
    if not alias.is_file() or file_sha256(alias) != local["sha256"]:
        raise CandidateCortexTrainingError("stage_final_adapter_mismatch")
    source_config = local_root / "adapter_config.json"
    config_payload = source_config.read_bytes()
    canonical_root = Path(str(plan["paths"]["adapter_root"])).resolve(strict=True)
    destination = (
        canonical_root
        / f"{policy.cumulative_iterations(stage_index):07d}_adapters.safetensors"
    )
    with local_internal_governed_scope(
        "candidate_cortex_adaptive.publish_stage", domain="file_write"
    ):
        gateway = get_file_write_gateway()
        if destination.exists() or destination.is_symlink():
            raise CandidateCortexTrainingError("stage_checkpoint_conflict")
        gateway.move_path(
            Path(str(local["path"])),
            destination,
            source="candidate_cortex_adaptive.publish_stage",
        )
        config_created = gateway.write_bytes_if_absent(
            canonical_root / "adapter_config.json",
            config_payload,
            mode=0o600,
            source="candidate_cortex_adaptive.publish_stage",
        )
        if (
            not config_created
            and (canonical_root / "adapter_config.json").read_bytes() != config_payload
        ):
            raise CandidateCortexTrainingError("stage_adapter_config_conflict")
        gateway.delete_file(
            alias,
            source="candidate_cortex_adaptive.publish_stage",
        )
    checkpoint = discover_exact_checkpoint(
        canonical_root,
        expected_cumulative_iterations=policy.cumulative_iterations(stage_index),
    )
    body = {
        "schema": STAGE_COMPLETION_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "stage_index": stage_index,
        "stage_iterations": policy.iterations(stage_index),
        "cumulative_iterations": policy.cumulative_iterations(stage_index),
        "training_command_sha256": document_sha256(list(command)),
        "checkpoint": checkpoint,
        "target_pid": os.getpid(),
        "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
        "host_metrics": dict(host_metrics),
    }
    completion = {**body, "completion_sha256": document_sha256(body)}
    _write_once(
        _completion_path(plan, stage_index),
        completion,
        source="candidate_cortex_adaptive.stage_completion",
    )
    return completion


def _train_stage(
    plan: Mapping[str, Any],
    stage_index: int,
    resume_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    existing = _validated_completion(plan, stage_index)
    if existing is not None:
        return existing
    _reset_incomplete_stage(plan, stage_index)
    command = build_stage_command(
        plan,
        stage_index=stage_index,
        resume_checkpoint=resume_checkpoint,
    )
    state: dict[str, Any] = {
        "sample_count": 0,
        "min_available_bytes": 2**63 - 1,
        "max_used_percent": 0.0,
        "max_process_rss_bytes": 0,
    }
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_host, args=(stop, state), daemon=True)
    sampler.start()
    started = time.monotonic()
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        with standalone_model_lane(
            owner_id=f"candidate-cortex-stage:{plan['run_id']}:{stage_index}",
            model_path=str(plan["model"]["canonical_path"]),
            purpose="train",
            priority=100,
            preemptible=False,
            require_exclusive=True,
            allow_owner_eviction=True,
            metadata={
                "tool": "run_candidate_cortex_adaptive_target",
                "plan_sha256": plan["plan_sha256"],
                "stage_index": stage_index,
            },
        ):
            from mlx_lm import lora

            try:
                lora.run(_mlx_arguments(command))
            finally:
                _release_model_memory()
    finally:
        stop.set()
        sampler.join(timeout=5.0)
    state["duration_seconds"] = max(0.0, time.monotonic() - started)
    return _publish_checkpoint(plan, stage_index, command, state)


def _release_model_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except ImportError:
        pass


def _restart_for_clean_model_phase(
    plan: Mapping[str, Any],
    *,
    run_root: Path,
    journal_key: Path,
    journal: Path,
    key: bytes,
    stage_index: int,
    next_phase: str,
) -> None:
    """Replace the process image so one model-heavy phase cannot retain another.

    MLX cache clearing only releases allocations no Python object still owns.
    Training and checkpoint measurement intentionally use different process
    images, while ``execve`` preserves the detached supervisor's target PID and
    its trainer-bound sleep inhibitor.
    """

    if next_phase not in {"measure", "decide"}:
        raise CandidateCortexTrainingError("adaptive_phase_invalid")
    launcher = Path(str(plan["python"])).expanduser()
    if not launcher.is_absolute() or not launcher.exists():
        raise CandidateCortexTrainingError("adaptive_phase_launcher_invalid")
    script = Path(__file__).resolve(strict=True)
    append_authenticated_event(
        journal,
        key=key,
        event_type="phase_restart_requested",
        payload={
            "schema": PHASE_BOUNDARY_SCHEMA,
            "plan_sha256": plan["plan_sha256"],
            "stage_index": stage_index,
            "next_phase": next_phase,
            "target_pid": os.getpid(),
            "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
        },
    )
    environment = dict(os.environ)
    environment["AURA_CANDIDATE_CORTEX_PHASE"] = next_phase
    environment["AURA_CANDIDATE_CORTEX_STAGE"] = str(stage_index)
    argv = [
        str(launcher),
        str(script),
        "--run-root",
        str(run_root.expanduser().resolve(strict=True)),
        "--journal-key",
        str(journal_key.expanduser().resolve(strict=True)),
    ]
    os.execve(str(launcher), argv, environment)
    raise CandidateCortexTrainingError("adaptive_phase_exec_returned")


def _measure_stage(
    plan: Mapping[str, Any], stage_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence_path = _evidence_path(plan, stage_index)
    detail_path = _detail_path(plan, stage_index)
    if not evidence_path.is_file() or not detail_path.is_file():
        code = measurement.main(
            [
                "--run-root",
                str(plan["paths"]["run_root"]),
                "--stage-index",
                str(stage_index),
                "--evidence-output",
                str(evidence_path),
                "--detail-output",
                str(detail_path),
            ]
        )
        if code != 0:
            raise CandidateCortexTrainingError("stage_measurement_failed")
    evidence = _strict_document(evidence_path)
    detail = _strict_document(detail_path)
    admission = adjudicate_checkpoint_evidence(
        evidence,
        plan=plan,
        stage_index=stage_index,
    )
    if (
        detail.get("evidence_sha256") != evidence.get("measurement_sha256")
        or detail.get("stage_index") != stage_index
    ):
        raise CandidateCortexTrainingError("stage_measurement_detail_invalid")
    return evidence, admission


def _observation(
    plan: Mapping[str, Any],
    stage_index: int,
    completion: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _strict_document(Path(str(plan["paths"]["adapter_identity"])))
    eval_samples = (
        int(evidence["persona"]["samples"])
        + int(evidence["retention"]["samples"])
        + len(evidence["behavior"])
    )
    raw = {
        "schema": OBSERVATION_SCHEMA,
        "stage_index": stage_index,
        "cumulative_iterations": completion["cumulative_iterations"],
        "validation_loss": float(evidence["persona"]["candidate_loss"]),
        "eval_samples": eval_samples,
        "checkpoint": completion["checkpoint"],
        "adapter_identity_sha256": document_sha256(identity),
        "model_descriptor_sha256": plan["model"]["descriptor_sha256"],
        "dataset_receipt_sha256": plan["dataset"]["receipt_sha256"],
        "trainer_pid": completion["target_pid"],
        "trainer_start_token": completion["target_start_token"],
        "trainer_exit_code": 0,
    }
    return validate_stage_observation(
        raw,
        plan=plan,
        expected_stage_index=stage_index,
        launched_identity={
            "trainer_pid": completion["target_pid"],
            "trainer_start_token": completion["target_start_token"],
        },
    )


def _stage_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = [
        dict(event["payload"])
        for event in events
        if event.get("event_type") == "stage_observed"
    ]
    admissions = [
        dict(event["payload"])
        for event in events
        if event.get("event_type") == "stage_admitted"
    ]
    return observations, admissions


def _reconcile_partial_admission(
    plan: Mapping[str, Any], journal: Path, key: bytes
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = read_authenticated_journal(journal, key=key)
    observations, admissions = _stage_events(events)
    if len(observations) == len(admissions):
        return observations, admissions
    if len(observations) != len(admissions) + 1:
        raise CandidateCortexTrainingError("stage_journal_incomplete")
    stage_index = len(admissions)
    evidence = _strict_document(_evidence_path(plan, stage_index))
    admission = adjudicate_checkpoint_evidence(
        evidence,
        plan=plan,
        stage_index=stage_index,
    )
    if admission.get("schema") != ADMISSION_SCHEMA:
        raise CandidateCortexTrainingError("stage_admission_invalid")
    append_authenticated_event(
        journal,
        key=key,
        event_type="stage_admitted",
        payload=admission,
    )
    return observations, [*admissions, admission]


def _write_result(plan: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    body = {
        "schema": ADAPTIVE_RESULT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "decision": dict(decision),
    }
    result = {**body, "result_sha256": document_sha256(body)}
    _write_once(
        Path(str(plan["paths"]["run_root"])) / "adaptive_result.json",
        result,
        source="candidate_cortex_adaptive.result",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--journal-key", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_and_verify_plan(args.run_root, verify_full_model=True)
    key = _key(args.journal_key)
    journal = Path(str(plan["paths"]["run_root"])) / JOURNAL_FILE
    events = read_authenticated_journal(journal, key=key)
    admission = execution_admission(
        plan,
        execute=True,
        authenticated_events=events,
    )
    if admission.get("execution_authorized") is not True:
        raise CandidateCortexTrainingError("adaptive_execution_not_authorized")

    while True:
        observations, admissions = _reconcile_partial_admission(plan, journal, key)
        next_stage = next_stage_plan(
            plan,
            observations=observations,
            admissions=admissions,
        )
        if next_stage.get("decision") != "CONTINUE":
            _write_result(plan, next_stage)
            print(json.dumps(next_stage, indent=2, sort_keys=True), flush=True)
            return 0
        stage_index = int(next_stage["stage_index"])
        current_plan = load_and_verify_plan(args.run_root, verify_full_model=True)
        if current_plan != plan:
            raise CandidateCortexTrainingError("adaptive_plan_or_input_drift")
        completion = _validated_completion(plan, stage_index)
        trained_in_this_process = completion is None
        if completion is None:
            append_authenticated_event(
                journal,
                key=key,
                event_type="stage_started",
                payload={
                    "stage_index": stage_index,
                    "target_pid": os.getpid(),
                    "target_start_token": detached._process_start_token(os.getpid()),  # noqa: SLF001
                    "command_sha256": document_sha256(next_stage["command"]),
                },
            )
            completion = _train_stage(
                plan,
                stage_index,
                next_stage.get("resume_checkpoint"),
            )
        if trained_in_this_process and (
            not _evidence_path(plan, stage_index).is_file()
            or not _detail_path(plan, stage_index).is_file()
        ):
            _restart_for_clean_model_phase(
                plan,
                run_root=args.run_root,
                journal_key=args.journal_key,
                journal=journal,
                key=key,
                stage_index=stage_index,
                next_phase="measure",
            )
        evidence, stage_admission = _measure_stage(plan, stage_index)
        observation = _observation(
            plan,
            stage_index,
            completion,
            evidence,
        )
        append_authenticated_event(
            journal,
            key=key,
            event_type="stage_observed",
            payload=observation,
        )
        append_authenticated_event(
            journal,
            key=key,
            event_type="stage_admitted",
            payload=stage_admission,
        )
        _restart_for_clean_model_phase(
            plan,
            run_root=args.run_root,
            journal_key=args.journal_key,
            journal=journal,
            key=key,
            stage_index=stage_index,
            next_phase="decide",
        )


if __name__ == "__main__":
    raise SystemExit(main())
