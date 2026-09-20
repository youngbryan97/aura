#!/usr/bin/env python3
"""Finish a resident RLC directional campaign without a human handoff gap.

The long-running campaign controller owns inference, retries, and its generic
independent evidence verdict.  This source-bound supervisor waits for that
controller's authenticated terminal state, independently recomputes the
directional interaction gate, and materializes the exact powered-campaign
handoff only when every preregistered directional rule passes.

It never loads a model, signs evidence, activates an adapter, or fuses weights.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
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
from tools.materialize_powered_latent_cortex_handoff import materialize  # noqa: E402
from tools.verify_latent_cortex_directional_gate import verify as verify_directional  # noqa: E402

CONFIG_SCHEMA = "aura.latent_cortex.directional_closeout_config.v1"
STATUS_SCHEMA = "aura.latent_cortex.directional_closeout_status.v1"
RECEIPT_SCHEMA = "aura.latent_cortex.directional_closeout_receipt.v1"
CONTROLLER_CONFIG_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_config.v2"
CONTROLLER_STATUS_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_status.v2"
CONTROLLER_STATE_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_state.v2"
CONTROLLER_EVENT_SCHEMA = "aura.latent_cortex.resumable_pilot_controller_event.v2"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_EVENT_JOURNAL_BYTES = 64 * 1024 * 1024
_LABEL = re.compile(r"[A-Za-z0-9.-]{1,180}")


class DirectionalCloseoutError(RuntimeError):
    """A closeout input or terminal claim failed closed."""


def _fail(reason: str) -> Never:
    raise DirectionalCloseoutError(reason)


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _bytes_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _absolute_path(value: Any, *, role: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        _fail(f"{role}_path_invalid")
    path = Path(value)
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    return path.resolve(strict=must_exist)


def _read_json(path: Path, *, role: str, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    payload = read_stable_bytes(path.resolve(strict=True), max_bytes=max_bytes)
    try:
        document = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DirectionalCloseoutError(f"{role}_json_invalid") from exc
    if not isinstance(document, dict):
        _fail(f"{role}_document_invalid")
    return document


def _verified_hashed_document(
    path: Path,
    *,
    role: str,
    schema: str,
    digest_key: str,
) -> dict[str, Any]:
    document = _read_json(path, role=role)
    material = dict(document)
    claimed = material.pop(digest_key, None)
    if document.get("schema") != schema or claimed != _sha(material):
        _fail(f"{role}_integrity_invalid")
    return document


def _file_binding(path: Path, *, role: str) -> dict[str, Any]:
    if path.is_symlink():
        _fail(f"{role}_symlink_rejected")
    resolved = path.resolve(strict=True)
    payload = read_stable_bytes(resolved, max_bytes=MAX_JSON_BYTES)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": _bytes_sha(payload),
        "size_bytes": len(payload),
    }


def _git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-c", "core.fsmonitor=false", *args],
        cwd=source_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        _fail(f"source_git_command_failed:{args[0]}")
    return completed.stdout.strip()


def _verify_source_identity(config: Mapping[str, Any]) -> None:
    source_root = Path(str(config["source_root"]))
    if _git(source_root, "rev-parse", "HEAD") != config["source_commit"]:
        _fail("source_commit_changed")
    if _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        _fail("source_tracked_files_dirty")
    tool = Path(__file__).resolve(strict=True)
    if tool != source_root / "tools" / tool.name:
        _fail("closeout_tool_not_executed_from_bound_source")
    if _bytes_sha(read_stable_bytes(tool, max_bytes=MAX_JSON_BYTES)) != config[
        "closeout_tool_sha256"
    ]:
        _fail("closeout_tool_changed")


def _verify_bound_inputs(config: Mapping[str, Any]) -> None:
    for role in ("controller_config", "contamination_trust_root"):
        binding = config[role]
        actual = _file_binding(Path(str(binding["path"])), role=role)
        if actual != dict(binding):
            _fail(f"{role}_binding_changed")
    controller = _verified_hashed_document(
        Path(str(config["controller_config"]["path"])),
        role="controller_config",
        schema=CONTROLLER_CONFIG_SCHEMA,
        digest_key="config_sha256",
    )
    if (
        controller.get("config_sha256")
        != config["controller_config_identity_sha256"]
        or controller.get("source_commit") != config["controller_source_commit"]
    ):
        _fail("controller_config_binding_invalid")


def _write_once(path: Path, document: Mapping[str, Any], *, conflict: str) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    destination = path.expanduser().resolve(strict=False)
    ensure_private_directory(destination.parent)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or destination.read_bytes() != payload:
            _fail(conflict)
        return
    descriptor = os.open(
        destination,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("closeout_short_write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_status(
    config: Mapping[str, Any],
    *,
    phase: str,
    reason: str = "",
    controller: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
) -> None:
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_name": config["campaign_name"],
        "phase": phase,
        "reason": reason,
        "heartbeat_at_unix": time.time(),
        "source_commit": config["source_commit"],
        "config_sha256": config["config_sha256"],
        "controller": dict(controller or {}),
        "result": dict(result or {}),
    }
    path = Path(str(config["state_dir"])) / "closeout-status.json"
    atomic_write_bytes(
        path,
        canonical_json_bytes({**body, "status_sha256": _sha(body)}) + b"\n",
        mode=0o600,
    )


def build_config(
    *,
    source_root: Path,
    controller_config_path: Path,
    output_root: Path,
    contamination_trust_root: Path,
    target_campaign_name: str,
    launch_label: str,
    poll_seconds: int = 15,
    stale_after_seconds: int = 180,
    max_wait_seconds: int = 86400,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve(strict=True)
    controller_config_path = controller_config_path.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    contamination_trust_root = contamination_trust_root.expanduser().resolve(strict=True)
    if not _LABEL.fullmatch(launch_label):
        _fail("launch_label_invalid")
    if not target_campaign_name or len(target_campaign_name) > 256:
        _fail("target_campaign_name_invalid")
    controller_config = _verified_hashed_document(
        controller_config_path,
        role="controller_config",
        schema=CONTROLLER_CONFIG_SCHEMA,
        digest_key="config_sha256",
    )
    campaign_dir = _absolute_path(
        controller_config.get("campaign_dir"), role="campaign_dir"
    )
    state_dir = _absolute_path(controller_config.get("state_dir"), role="controller_state_dir")
    plan_path = campaign_dir / "plan.json"
    plan = CampaignPlan.from_dict(_read_json(plan_path, role="campaign_plan"))
    if plan.campaign_name != controller_config.get("campaign_name"):
        _fail("controller_campaign_name_mismatch")
    source_commit = _git(source_root, "rev-parse", "HEAD")
    tool_path = source_root / "tools" / Path(__file__).name
    if not tool_path.is_file() or tool_path.is_symlink():
        _fail("closeout_tool_missing_from_source")
    for name, value, minimum, maximum in (
        ("poll_seconds", poll_seconds, 1, 300),
        ("stale_after_seconds", stale_after_seconds, 30, 3600),
        ("max_wait_seconds", max_wait_seconds, 60, 604800),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            _fail(f"{name}_invalid")
    body = {
        "schema": CONFIG_SCHEMA,
        "campaign_name": plan.campaign_name,
        "source_root": str(source_root),
        "source_commit": source_commit,
        "closeout_tool_sha256": _bytes_sha(
            read_stable_bytes(tool_path, max_bytes=MAX_JSON_BYTES)
        ),
        "controller_config": _file_binding(
            controller_config_path, role="controller_config"
        ),
        "controller_config_identity_sha256": controller_config["config_sha256"],
        "controller_source_commit": controller_config["source_commit"],
        "controller_status_path": str(state_dir / "controller-status.json"),
        "controller_state_path": str(state_dir / "controller-state.json"),
        "controller_events_path": str(state_dir / "controller-events.jsonl"),
        "campaign_dir": str(campaign_dir),
        "plan_sha256": plan.plan_sha256,
        "independent_verdict_path": str(
            Path(str(controller_config["execution_output_root"])) / "independent-verdict.json"
        ),
        "contamination_trust_root": _file_binding(
            contamination_trust_root, role="contamination_trust_root"
        ),
        "output_root": str(output_root),
        "state_dir": str(output_root / "supervisor"),
        "directional_verdict_path": str(output_root / "directional-verdict.json"),
        "powered_handoff_path": str(output_root / "powered-campaign-handoff.json"),
        "receipt_path": str(output_root / "closeout-receipt.json"),
        "target_campaign_name": target_campaign_name,
        "launch_label": launch_label,
        "poll_seconds": poll_seconds,
        "stale_after_seconds": stale_after_seconds,
        "max_wait_seconds": max_wait_seconds,
    }
    return {**body, "config_sha256": _sha(body)}


def write_config(path: Path, config: Mapping[str, Any]) -> None:
    _write_once(path, config, conflict="closeout_config_conflict")


def load_config(path: Path) -> dict[str, Any]:
    document = _read_json(path.expanduser().resolve(strict=True), role="closeout_config")
    material = dict(document)
    claimed = material.pop("config_sha256", None)
    required = {
        "schema",
        "campaign_name",
        "source_root",
        "source_commit",
        "closeout_tool_sha256",
        "controller_config",
        "controller_config_identity_sha256",
        "controller_source_commit",
        "controller_status_path",
        "controller_state_path",
        "controller_events_path",
        "campaign_dir",
        "plan_sha256",
        "independent_verdict_path",
        "contamination_trust_root",
        "output_root",
        "state_dir",
        "directional_verdict_path",
        "powered_handoff_path",
        "receipt_path",
        "target_campaign_name",
        "launch_label",
        "poll_seconds",
        "stale_after_seconds",
        "max_wait_seconds",
        "config_sha256",
    }
    if (
        set(document) != required
        or document.get("schema") != CONFIG_SCHEMA
        or claimed != _sha(material)
        or not _is_git_commit(document.get("source_commit"))
        or not _is_sha256(document.get("closeout_tool_sha256"))
        or not _is_sha256(document.get("controller_config_identity_sha256"))
        or not _is_git_commit(document.get("controller_source_commit"))
        or not _is_sha256(document.get("plan_sha256"))
        or not _LABEL.fullmatch(str(document.get("launch_label", "")))
    ):
        _fail("closeout_config_invalid")
    for role in (
        "source_root",
        "controller_status_path",
        "controller_state_path",
        "controller_events_path",
        "campaign_dir",
        "independent_verdict_path",
        "output_root",
        "state_dir",
        "directional_verdict_path",
        "powered_handoff_path",
        "receipt_path",
    ):
        must_exist = role in {"source_root", "campaign_dir"}
        _absolute_path(document[role], role=role, must_exist=must_exist)
    for role in ("controller_config", "contamination_trust_root"):
        binding = document.get(role)
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"role", "path", "sha256", "size_bytes"}
            or binding.get("role") != role
            or not _is_sha256(binding.get("sha256"))
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] < 0
        ):
            _fail(f"{role}_binding_invalid")
        actual = _file_binding(Path(str(binding["path"])), role=role)
        if actual != dict(binding):
            _fail(f"{role}_binding_changed")
    controller = _verified_hashed_document(
        Path(str(document["controller_config"]["path"])),
        role="controller_config",
        schema=CONTROLLER_CONFIG_SCHEMA,
        digest_key="config_sha256",
    )
    if (
        controller.get("config_sha256")
        != document["controller_config_identity_sha256"]
        or controller.get("source_commit") != document["controller_source_commit"]
        or controller.get("campaign_name") != document["campaign_name"]
        or controller.get("campaign_dir") != document["campaign_dir"]
    ):
        _fail("controller_config_binding_invalid")
    plan = CampaignPlan.from_dict(
        _read_json(Path(str(document["campaign_dir"])) / "plan.json", role="campaign_plan")
    )
    if (
        plan.campaign_name != document["campaign_name"]
        or plan.plan_sha256 != document["plan_sha256"]
    ):
        _fail("campaign_plan_changed")
    return document


def _verify_controller_events(path: Path) -> dict[str, Any]:
    payload = read_stable_bytes(path.resolve(strict=True), max_bytes=MAX_EVENT_JOURNAL_BYTES)
    previous = "0" * 64
    sequence = 0
    last_event = ""
    for raw in payload.splitlines():
        try:
            document = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise DirectionalCloseoutError("controller_event_json_invalid") from exc
        material = dict(document) if isinstance(document, dict) else {}
        claimed = material.pop("event_sha256", None)
        if (
            not isinstance(document, dict)
            or document.get("schema") != CONTROLLER_EVENT_SCHEMA
            or document.get("sequence") != sequence + 1
            or document.get("previous_event_sha256") != previous
            or claimed != _sha(material)
        ):
            _fail("controller_event_chain_invalid")
        sequence = document["sequence"]
        previous = document["event_sha256"]
        last_event = str(document.get("event", ""))
    return {"events": sequence, "head_sha256": previous, "last_event": last_event}


def _controller_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    status = _verified_hashed_document(
        Path(str(config["controller_status_path"])),
        role="controller_status",
        schema=CONTROLLER_STATUS_SCHEMA,
        digest_key="status_sha256",
    )
    if (
        status.get("campaign_name") != config["campaign_name"]
        or status.get("campaign_dir") != config["campaign_dir"]
        or status.get("config_sha256") != config["controller_config_identity_sha256"]
        or status.get("source_commit") != config["controller_source_commit"]
    ):
        _fail("controller_status_binding_invalid")
    progress = status.get("campaign_progress")
    snapshot = {
        "phase": status.get("phase"),
        "reason": status.get("reason"),
        "heartbeat_at_unix": status.get("heartbeat_at_unix"),
        "status_sha256": status.get("status_sha256"),
        "sealed_result_cells": (
            progress.get("sealed_result_cells") if isinstance(progress, Mapping) else None
        ),
        "total_cells": progress.get("total_cells") if isinstance(progress, Mapping) else None,
        "failed_attempts": (
            progress.get("failed_attempts") if isinstance(progress, Mapping) else None
        ),
    }
    if status.get("phase") != "complete":
        return snapshot
    state = _verified_hashed_document(
        Path(str(config["controller_state_path"])),
        role="controller_state",
        schema=CONTROLLER_STATE_SCHEMA,
        digest_key="state_sha256",
    )
    events = _verify_controller_events(Path(str(config["controller_events_path"])))
    if (
        state.get("terminal") is not True
        or state.get("journal_sequence") != events["events"]
        or state.get("journal_head_sha256") != events["head_sha256"]
        or events["last_event"] != "VERIFIED_TERMINAL"
    ):
        _fail("controller_terminal_receipt_invalid")
    return {**snapshot, "terminal": True, "controller_events": events}


def _artifact_summary(path: Path, *, role: str) -> dict[str, Any]:
    document = _read_json(path, role=role)
    return {
        **_file_binding(path, role=role),
        "schema": document.get("schema"),
        "decision": document.get("decision"),
        "directional_gate_passed": document.get("directional_gate_passed"),
        "diagnoses": document.get("diagnoses"),
    }


def closeout_once(config: Mapping[str, Any]) -> dict[str, Any] | None:
    _verify_source_identity(config)
    _verify_bound_inputs(config)
    controller = _controller_snapshot(config)
    phase = controller.get("phase")
    if phase == "blocked":
        _write_status(
            config,
            phase="blocked",
            reason=f"campaign_controller_blocked:{controller.get('reason', '')}",
            controller=controller,
        )
        return {
            "terminal": True,
            "exit_code": 0,
            "decision": "repair_campaign_controller",
        }
    if phase != "complete":
        heartbeat = controller.get("heartbeat_at_unix")
        stale = type(heartbeat) not in {int, float} or (
            time.time() - float(heartbeat) > int(config["stale_after_seconds"])
        )
        _write_status(
            config,
            phase="waiting",
            reason="controller_heartbeat_stale" if stale else "campaign_running",
            controller=controller,
        )
        return None

    directional_path = Path(str(config["directional_verdict_path"]))
    verdict = verify_directional(
        campaign_dir=Path(str(config["campaign_dir"])),
        independent_verdict_path=Path(str(config["independent_verdict_path"])),
        contamination_trust_root=Path(str(config["contamination_trust_root"]["path"])),
    )
    _write_once(
        directional_path,
        verdict,
        conflict="directional_verdict_output_conflict",
    )
    handoff_summary: dict[str, Any] | None = None
    if verdict.get("directional_gate_passed") is True:
        handoff_path = Path(str(config["powered_handoff_path"]))
        materialize(
            directional_verdict_path=directional_path,
            campaign_dir=Path(str(config["campaign_dir"])),
            target_campaign_name=str(config["target_campaign_name"]),
            output=handoff_path,
        )
        handoff_summary = _artifact_summary(handoff_path, role="powered_handoff")
    result = {
        "decision": verdict.get("decision"),
        "directional_gate_passed": verdict.get("directional_gate_passed"),
        "directional_verdict": _artifact_summary(
            directional_path, role="directional_verdict"
        ),
        "powered_handoff": handoff_summary,
    }
    receipt_material = {
        "schema": RECEIPT_SCHEMA,
        "campaign_name": config["campaign_name"],
        "plan_sha256": config["plan_sha256"],
        "source_commit": config["source_commit"],
        "config_sha256": config["config_sha256"],
        "controller_terminal": controller,
        "result": result,
        "nonclaims": {
            "reasoning_gain_proven": False,
            "frontier_gain_proven": False,
            "production_activation_authorized": False,
            "static_weight_fusion_authorized": False,
        },
    }
    receipt = {**receipt_material, "receipt_sha256": _sha(receipt_material)}
    _write_once(
        Path(str(config["receipt_path"])),
        receipt,
        conflict="closeout_receipt_conflict",
    )
    _write_status(config, phase="complete", controller=controller, result=result)
    return {"terminal": True, "exit_code": 0, **result}


def run(config_path: Path) -> int:
    config = load_config(config_path)
    state_dir = ensure_private_directory(Path(str(config["state_dir"])))
    lock_path = state_dir / "closeout.lock"
    lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DirectionalCloseoutError("closeout_supervisor_already_running") from exc
        deadline = time.monotonic() + int(config["max_wait_seconds"])
        while True:
            result = closeout_once(config)
            if result is not None:
                return int(result["exit_code"])
            if time.monotonic() >= deadline:
                _write_status(config, phase="retry", reason="bounded_wait_elapsed")
                return 3
            time.sleep(int(config["poll_seconds"]))
    finally:
        os.close(lock_descriptor)


def _launchd_payload(config_path: Path, config: Mapping[str, Any]) -> bytes:
    payload = {
        "Label": config["launch_label"],
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-i",
            sys.executable,
            str(Path(__file__).resolve(strict=True)),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
        ],
        "WorkingDirectory": config["source_root"],
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(Path(str(config["state_dir"])) / "launchd.log"),
        "StandardErrorPath": str(Path(str(config["state_dir"])) / "launchd.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    _verify_source_identity(config)
    state_dir = ensure_private_directory(Path(str(config["state_dir"])))
    launch_agents = ensure_private_directory(Path.home() / "Library" / "LaunchAgents")
    plist_path = launch_agents / f"{config['launch_label']}.plist"
    payload = _launchd_payload(config_path, config)
    atomic_write_bytes(plist_path, payload, mode=0o600)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    completed = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        _fail(f"launchd_bootstrap_failed:{completed.returncode}:{completed.stderr.strip()}")
    material = {
        "schema": "aura.latent_cortex.directional_closeout_launchd.v1",
        "label": config["launch_label"],
        "config_sha256": config["config_sha256"],
        "plist_path": str(plist_path),
        "plist_sha256": _bytes_sha(payload),
    }
    receipt = {**material, "launch_sha256": _sha(material)}
    _write_once(
        state_dir / "launchd-receipt.json",
        receipt,
        conflict="closeout_launchd_receipt_conflict",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--controller-config", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--contamination-trust-root", type=Path, required=True)
    prepare.add_argument("--target-campaign-name", required=True)
    prepare.add_argument("--launch-label", required=True)
    prepare.add_argument("--poll-seconds", type=int, default=15)
    prepare.add_argument("--stale-after-seconds", type=int, default=180)
    prepare.add_argument("--max-wait-seconds", type=int, default=86400)
    prepare.add_argument("--output", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    install = commands.add_parser("install-launchd")
    install.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            config = build_config(
                source_root=args.source_root,
                controller_config_path=args.controller_config,
                output_root=args.output_root,
                contamination_trust_root=args.contamination_trust_root,
                target_campaign_name=args.target_campaign_name,
                launch_label=args.launch_label,
                poll_seconds=args.poll_seconds,
                stale_after_seconds=args.stale_after_seconds,
                max_wait_seconds=args.max_wait_seconds,
            )
            write_config(args.output, config)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.action == "run":
            return run(args.config)
        print(json.dumps(install_launchd(args.config), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(
            f"run_latent_cortex_directional_closeout: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
