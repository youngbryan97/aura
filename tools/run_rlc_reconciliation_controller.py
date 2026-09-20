#!/usr/bin/env python3
"""Durable OS-supervised controller for the resident RLC reconciliation sweep.

The sweep owns scientific cell resumption. This controller owns the process
lifecycle around it: immutable source verification, one exclusive model owner,
bounded exact-run retries, process-group wedge recovery, authenticated liveness,
and launchd/caffeinate custody. It never changes tasks, arms, grading, or gates.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import plistlib
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA: Final = "aura.rlc_reconciliation_controller.v1"
SOURCE_SCHEMA: Final = "aura.rlc_reconciliation_source_manifest.v1"
SOURCE_GIT_SCHEMA: Final = "aura.rlc_reconciliation_source_git_identity.v1"
HEARTBEAT_SCHEMA: Final = "aura.rlc_reconciliation_controller_heartbeat.v1"
STATUS_SCHEMA: Final = "aura.rlc_reconciliation_controller_status.v1"
LAUNCH_SCHEMA: Final = "aura.rlc_reconciliation_controller_launchd.v1"
RECOVERY_SCHEMA: Final = "aura.rlc_reconciliation_controller_recovery.v1"
TERMINAL_PHASES: Final = frozenset({"complete", "yielded", "blocked"})
CONFIG_SUFFIXES: Final = frozenset({".json", ".toml", ".yaml", ".yml", ".jinja"})
GLOBAL_MODEL_LOCK: Final = Path.home() / ".aura/state/rlc-reconciliation-model.lock"
CLAIM_TASK_REGISTRY_VERSION: Final = "2026.08.06.1"
CAMPAIGN_STAGES: Final = frozenset({"component", "pilot", "certificate"})
COMPOSED_RECURRENT_ARM: Final = "complete_system_recurrent_composed"


class ControllerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: Mapping[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        payload = json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(payload)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ControllerError(f"{role}_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{role}_not_object:{path}")
    return value


def _terminal_verdict(path: Path) -> dict[str, Any] | None:
    """Return a verdict only when the scientific sweep finished its coverage.

    The sweep deliberately writes interim verdicts after durable cells so an
    interrupted campaign remains inspectable.  File existence therefore says
    evidence is available, not that the campaign is complete.
    """

    futility_path = path.with_name("terminal_futility.json")
    if futility_path.is_file():
        futility = _read_json(futility_path, role="terminal_futility")
        body = {
            key: value for key, value in futility.items() if key != "receipt_sha256"
        }
        if (
            futility.get("schema")
            != "aura.rlc.composed_terminal_futility.v1"
            or futility.get("receipt_sha256") != _sha(body)
            or futility.get("terminal") is not True
            or futility.get("decision")
            != "terminal_preregistered_zero_regression_futility"
            or futility.get("positive_claim_authority") is not False
            or futility.get("fusion_authorized") is not False
            or futility.get("wow_signal_authorized") is not False
            or not futility.get("fatal_regression_contrasts")
        ):
            raise ControllerError("terminal_futility_receipt_invalid")
        return futility
    if not path.is_file():
        return None
    verdict = _read_json(path, role="sweep_verdict")
    decision = verdict.get("decision")
    if (
        verdict.get("coverage_complete") is not True
        or verdict.get("arms_complete") is not True
        or not isinstance(decision, str)
        or not decision.strip()
        or decision == "inconclusive_campaign_incomplete"
    ):
        return None
    return verdict


def _append_event(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(document, sort_keys=True, allow_nan=False) + "\n")
        sink.flush()
        os.fsync(sink.fileno())


def _source_paths(root: Path) -> list[Path]:
    paths: set[Path] = set()
    for base in (root / "core", root / "config"):
        if not base.is_dir():
            raise ControllerError(f"source_root_incomplete:{base}")
    # Every Python file is executable input because Python can import a module
    # from any package on the source root (including sitecustomize at startup).
    for candidate in root.rglob("*.py"):
        relative = candidate.relative_to(root)
        if candidate.is_file() and not any(
            part in {".git", "__pycache__", ".pytest_cache"}
            for part in relative.parts
        ):
            paths.add(relative)
    for candidate in (root / "config").rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in CONFIG_SUFFIXES:
            paths.add(candidate.relative_to(root))
    for relative in (
        Path("tools/run_rlc_reconciliation_sweep.py"),
        Path("tools/run_rlc_reconciliation_controller.py"),
        Path("pyproject.toml"),
        Path("requirements_lock.txt"),
    ):
        path = root / relative
        if not path.is_file():
            raise ControllerError(f"required_source_missing:{relative}")
        paths.add(relative)
    return sorted(paths, key=lambda item: os.fsencode(item.as_posix()))


def _git_bytes(root: Path, *arguments: str) -> bytes:
    observed = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
        capture_output=True,
        timeout=15.0,
        check=False,
    )
    if observed.returncode != 0:
        detail = observed.stderr.decode("utf-8", errors="replace").strip()
        raise ControllerError(
            f"source_git_identity_unavailable:{arguments[0]}:{detail}"
        )
    return observed.stdout


def build_source_git_identity(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Prove a campaign capsule is one clean detached Git worktree.

    A source manifest proves the bytes the controller chose to hash, but an
    archive has no independent authority for the commit label attached to
    those bytes.  The worker's runtime identity is Git-backed too, so accepting
    an archive here only defers an inevitable source-unbound failure until
    after an expensive model load.
    """

    root = root.expanduser().resolve(strict=True)
    normalized_commit = str(source_commit).strip().lower()
    if len(normalized_commit) != 40 or any(
        character not in "0123456789abcdef" for character in normalized_commit
    ):
        raise ControllerError("source_commit_invalid")
    top_level = Path(
        _git_bytes(root, "rev-parse", "--show-toplevel")
        .decode("utf-8", errors="strict")
        .strip()
    ).expanduser().resolve(strict=True)
    if top_level != root:
        raise ControllerError("source_git_root_mismatch")
    observed_commit = (
        _git_bytes(root, "rev-parse", "HEAD")
        .decode("ascii", errors="strict")
        .strip()
        .lower()
    )
    if observed_commit != normalized_commit:
        raise ControllerError("source_git_commit_mismatch")
    branch = (
        _git_bytes(root, "rev-parse", "--abbrev-ref", "HEAD")
        .decode("utf-8", errors="strict")
        .strip()
    )
    if branch != "HEAD":
        raise ControllerError("source_capsule_not_detached")
    status = _git_bytes(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        raise ControllerError("source_capsule_dirty")
    body = {
        "schema": SOURCE_GIT_SCHEMA,
        "source_root": str(root),
        "source_commit": observed_commit,
        "source_branch": "DETACHED",
        "workspace_status_sha256": hashlib.sha256(status).hexdigest(),
    }
    return {**body, "identity_sha256": _sha(body)}


def verify_source_git_identity(
    root: Path,
    identity: Mapping[str, Any],
) -> None:
    if not isinstance(identity, Mapping):
        raise ControllerError("source_git_identity_invalid")
    body = {key: value for key, value in identity.items() if key != "identity_sha256"}
    if (
        identity.get("schema") != SOURCE_GIT_SCHEMA
        or identity.get("identity_sha256") != _sha(body)
        or identity.get("source_branch") != "DETACHED"
        or identity.get("workspace_status_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ControllerError("source_git_identity_invalid")
    observed = build_source_git_identity(
        root,
        source_commit=str(identity.get("source_commit") or ""),
    )
    if observed != dict(identity):
        raise ControllerError("source_git_identity_drift")


def build_source_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = [
        {
            "path": path.as_posix(),
            "size": (root / path).stat().st_size,
            "sha256": _sha_file(root / path),
        }
        for path in _source_paths(root)
    ]
    body = {
        "schema": SOURCE_SCHEMA,
        "source_commit": source_commit,
        "files": files,
    }
    return {**body, "manifest_sha256": _sha(body)}


def verify_source_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("schema") != SOURCE_SCHEMA
        or manifest.get("manifest_sha256") != _sha(body)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ControllerError("source_manifest_invalid")
    root = root.expanduser().resolve(strict=True)
    recorded_paths = [Path(str(item.get("path", ""))) for item in manifest["files"]]
    if recorded_paths != _source_paths(root):
        raise ControllerError("source_file_set_drift")
    for item in manifest["files"]:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise ControllerError("source_manifest_entry_invalid")
        relative = Path(str(item["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ControllerError("source_manifest_path_invalid")
        path = root / relative
        try:
            metadata = path.stat()
        except OSError as exc:
            raise ControllerError(f"source_file_missing:{relative}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != item["size"]
            or _sha_file(path) != item["sha256"]
        ):
            raise ControllerError(f"source_file_drift:{relative}")


def build_model_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: os.fsencode(path.relative_to(root).as_posix()),
    )
    if not files or not (root / "config.json").is_file():
        raise ControllerError("model_directory_incomplete")
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha_file(path),
        }
        for path in files
    ]
    body = {"root": str(root), "files": entries}
    return {**body, "manifest_sha256": _sha(body)}


def verify_model_manifest(manifest: Mapping[str, Any]) -> None:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256") != _sha(body)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise ControllerError("model_manifest_invalid")
    root = Path(str(manifest.get("root"))).expanduser().resolve(strict=True)
    observed = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )
    recorded = [str(item.get("path")) for item in manifest["files"]]
    if observed != recorded:
        raise ControllerError("model_file_set_drift")
    for item in manifest["files"]:
        path = root / str(item["path"])
        metadata = path.stat()
        if metadata.st_size != item["size"] or _sha_file(path) != item["sha256"]:
            raise ControllerError(f"model_file_drift:{item['path']}")


def _model_checkpoint_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the full-SHA checkpoint identity used by the scientific sweep."""

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ControllerError("model_manifest_invalid")
    weights = sorted(
        (
            item
            for item in files
            if isinstance(item, Mapping)
            and Path(str(item.get("path") or "")).parent == Path(".")
            and Path(str(item.get("path") or "")).suffix
            in {".safetensors", ".npz", ".gguf"}
        ),
        key=lambda item: str(item["path"]),
    )
    if not weights:
        raise ControllerError("model_checkpoint_weights_missing")
    combined = hashlib.sha256()
    for item in weights:
        digest = str(item.get("sha256") or "")
        if len(digest) != 64:
            raise ControllerError("model_checkpoint_weight_identity_invalid")
        combined.update(f"{Path(str(item['path'])).name}:{digest};".encode())
    return {
        "fingerprint": combined.hexdigest(),
        "method": "sha256",
        "files": len(weights),
    }


def _verified_adapter_input(
    adapter_root: Path,
    *,
    model_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    from core.brain.llm.latent_cortex.campaign_launch_bundle import (
        verify_adapter_freeze,
    )

    supplied = adapter_root.expanduser()
    if supplied.is_symlink():
        raise ControllerError("adapter_root_symlink_rejected")
    root = supplied.resolve(strict=True)
    certificate = verify_adapter_freeze(root)
    receipt = certificate.get("identity_receipt")
    if not isinstance(receipt, Mapping):
        raise ControllerError("adapter_identity_receipt_invalid")
    if receipt.get("base_checkpoint_fingerprint") != model_checkpoint.get(
        "fingerprint"
    ):
        raise ControllerError("adapter_base_checkpoint_mismatch")
    if receipt.get("training_objective_learned") is not True:
        raise ControllerError("adapter_training_objective_not_learned")
    adapter = (root / "adapter.safetensors").resolve(strict=True)
    manifest = (root / "recurrence_adapter_manifest.json").resolve(strict=True)
    if adapter.parent != root or manifest.parent != root:
        raise ControllerError("adapter_launch_path_invalid")
    return {
        "adapter_root": str(root),
        "adapter": str(adapter),
        "adapter_manifest": str(manifest),
        "adapter_freeze": certificate,
    }


def _verified_integrated_recurrent_package_input(
    package_root: Path,
) -> dict[str, Any]:
    """Freeze the complete recurrent package and its activation authority."""

    supplied = package_root.expanduser()
    if supplied.is_symlink():
        raise ControllerError("integrated_recurrent_package_symlink_rejected")
    root = supplied.resolve(strict=True)
    from core.brain.llm.unified_recurrent_qualified_activation_store import (
        read_qualified_activation,
    )
    from core.brain.llm.unified_recurrent_shadow import inspect_shadow_package

    verified = inspect_shadow_package(root)
    manifest = verified["manifest"]
    activation = read_qualified_activation()
    if (
        activation.get("mode") != "qualified_typed_only"
        or activation.get("serving_authority") is not True
        or activation.get("package_id") != manifest.get("package_id")
        or activation.get("manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise ControllerError("integrated_recurrent_package_activation_mismatch")
    files: list[dict[str, Any]] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: os.fsencode(candidate.relative_to(root).as_posix()),
    ):
        if path.is_symlink():
            raise ControllerError("integrated_recurrent_package_file_symlink_rejected")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha_file(path),
            }
        )
    if not files:
        raise ControllerError("integrated_recurrent_package_empty")
    body = {
        "root": str(root),
        "package_id": str(manifest["package_id"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "controller_sha256": str(activation["controller_sha256"]),
        "activation_sha256": str(activation["activation_sha256"]),
        "activation": activation,
        "files": files,
    }
    return {**body, "input_sha256": _sha(body)}


def _verify_integrated_recurrent_package_input(
    expected: Mapping[str, Any],
) -> None:
    if not isinstance(expected, Mapping) or not isinstance(expected.get("root"), str):
        raise ControllerError("integrated_recurrent_package_identity_invalid")
    observed = _verified_integrated_recurrent_package_input(Path(expected["root"]))
    if observed != dict(expected):
        raise ControllerError("integrated_recurrent_package_identity_drift")


def _config_body(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "config_sha256"}


def _controller_program(config: Mapping[str, Any]) -> Path:
    configured = config.get("controller_program")
    if configured:
        return Path(str(configured)).expanduser().absolute()
    return Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_controller.py"


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path.expanduser().resolve(strict=True), role="controller_config")
    if config.get("schema") != SCHEMA or config.get("config_sha256") != _sha(_config_body(config)):
        raise ControllerError("controller_config_invalid")
    required = {
        "campaign_id",
        "source_root",
        "source_commit",
        "source_git_identity",
        "source_manifest_path",
        "python",
        "python_sha256",
        "model",
        "model_manifest",
        "out_dir",
        "arms",
        "seed",
        "per_domain",
        "difficulty",
        "campaign_stage",
        "domains",
        "task_registry_version",
        "n_slots",
        "max_tokens",
        "memory_fraction",
        "episode_wall_s",
        "attempt_wall_s",
        "max_attempts",
        "poll_s",
        "stale_after_s",
        "retry_backoff_s",
        "heartbeat_key_path",
        "launch_label",
    }
    if not required.issubset(config):
        raise ControllerError("controller_config_incomplete")
    if not 1 <= int(config["max_attempts"]) <= 32:
        raise ControllerError("controller_attempt_budget_invalid")
    if type(config["difficulty"]) is not int or config["difficulty"] not in {1, 2, 3}:
        raise ControllerError("controller_difficulty_invalid")
    if config["campaign_stage"] not in CAMPAIGN_STAGES:
        raise ControllerError("controller_campaign_stage_invalid")
    domains = config["domains"]
    if not isinstance(domains, list) or any(
        not isinstance(domain, str) or not domain for domain in domains
    ):
        raise ControllerError("controller_domains_invalid")
    if len(set(domains)) != len(domains):
        raise ControllerError("controller_domains_invalid")
    from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS

    if any(domain not in FRONTIER_DOMAINS for domain in domains):
        raise ControllerError("controller_domains_invalid")
    if config["task_registry_version"] != CLAIM_TASK_REGISTRY_VERSION:
        raise ControllerError("controller_task_registry_not_contamination_safe")
    if float(config["stale_after_s"]) <= float(config["episode_wall_s"]):
        raise ControllerError("controller_stale_budget_too_short")
    if not isinstance(config.get("source_git_identity"), Mapping):
        raise ControllerError("source_git_identity_invalid")
    if config["source_git_identity"].get("source_commit") != config["source_commit"]:
        raise ControllerError("source_git_commit_binding_invalid")
    if "controller_program" in config and "controller_program_sha256" not in config:
        raise ControllerError("controller_program_identity_incomplete")
    adapter_fields = {
        "adapter_root",
        "adapter",
        "adapter_manifest",
        "adapter_freeze",
    }
    present_adapter_fields = adapter_fields.intersection(config)
    if present_adapter_fields and present_adapter_fields != adapter_fields:
        raise ControllerError("controller_adapter_identity_incomplete")
    recurrent_fields = {
        "integrated_recurrent_package",
        "integrated_recurrent_max_tokens",
    }
    present_recurrent_fields = recurrent_fields.intersection(config)
    if present_recurrent_fields and present_recurrent_fields != recurrent_fields:
        raise ControllerError("integrated_recurrent_package_identity_incomplete")
    if present_recurrent_fields:
        recurrent = config["integrated_recurrent_package"]
        if not isinstance(recurrent, Mapping):
            raise ControllerError("integrated_recurrent_package_identity_invalid")
        budget = config["integrated_recurrent_max_tokens"]
        if type(budget) is not int or not 16 <= budget <= 2048:
            raise ControllerError("integrated_recurrent_budget_invalid")
    requested_arms = {
        value.strip() for value in str(config["arms"]).split(",") if value.strip()
    }
    if (COMPOSED_RECURRENT_ARM in requested_arms) != bool(present_recurrent_fields):
        raise ControllerError("integrated_recurrent_arm_package_contract_mismatch")
    return config


def build_config(
    *,
    source_root: Path,
    source_commit: str,
    model: Path,
    out_dir: Path,
    python: Path,
    arms: str,
    seed: int,
    per_domain: int,
    difficulty: int,
    n_slots: int,
    max_tokens: int,
    memory_fraction: float,
    episode_wall_s: float,
    attempt_wall_s: float,
    max_attempts: int,
    poll_s: float,
    stale_after_s: float,
    retry_backoff_s: float,
    campaign_stage: str = "certificate",
    domains: Sequence[str] = (),
    adapter_root: Path | None = None,
    integrated_recurrent_package: Path | None = None,
    integrated_recurrent_max_tokens: int = 2048,
    task_registry_version: str = CLAIM_TASK_REGISTRY_VERSION,
    controller_program: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if type(difficulty) is not int or difficulty not in {1, 2, 3}:
        raise ControllerError("controller_difficulty_invalid")
    if campaign_stage not in CAMPAIGN_STAGES:
        raise ControllerError("controller_campaign_stage_invalid")
    from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS

    normalized_domains = tuple(str(domain).strip() for domain in domains)
    if (
        any(not domain or domain not in FRONTIER_DOMAINS for domain in normalized_domains)
        or len(set(normalized_domains)) != len(normalized_domains)
    ):
        raise ControllerError("controller_domains_invalid")
    if task_registry_version != CLAIM_TASK_REGISTRY_VERSION:
        raise ControllerError("controller_task_registry_not_contamination_safe")
    requested_arms = {value.strip() for value in arms.split(",") if value.strip()}
    if (COMPOSED_RECURRENT_ARM in requested_arms) != (
        integrated_recurrent_package is not None
    ):
        raise ControllerError("integrated_recurrent_arm_package_contract_mismatch")
    source_root = source_root.expanduser().resolve(strict=True)
    source_commit = str(source_commit).strip().lower()
    model = model.expanduser().resolve(strict=True)
    # Preserve the venv entrypoint for execution. Resolving its symlink to the
    # Homebrew base binary drops pyvenv.cfg discovery and therefore the exact
    # dependency environment, even though both paths hash the same executable.
    python = python.expanduser().absolute()
    resolved_python = python.resolve(strict=True)
    out_dir = out_dir.expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    campaign_id = out_dir.name
    source_git_identity = build_source_git_identity(
        source_root,
        source_commit=source_commit,
    )
    controller_program = (
        controller_program.expanduser().resolve(strict=True)
        if controller_program is not None
        else source_root / "tools/run_rlc_reconciliation_controller.py"
    )
    manifest = build_source_manifest(source_root, source_commit=source_commit)
    manifest_path = out_dir / "source_manifest.json"
    key_path = out_dir / ".heartbeat.key"
    key = secrets.token_bytes(32)
    model_manifest = build_model_manifest(model)
    model_checkpoint = _model_checkpoint_identity(model_manifest)
    adapter_input = (
        _verified_adapter_input(adapter_root, model_checkpoint=model_checkpoint)
        if adapter_root is not None
        else {}
    )
    if type(integrated_recurrent_max_tokens) is not int or not (
        16 <= integrated_recurrent_max_tokens <= 2048
    ):
        raise ControllerError("integrated_recurrent_budget_invalid")
    integrated_recurrent_input = (
        {
            "integrated_recurrent_package": (
                _verified_integrated_recurrent_package_input(
                    integrated_recurrent_package
                )
            ),
            "integrated_recurrent_max_tokens": integrated_recurrent_max_tokens,
        }
        if integrated_recurrent_package is not None
        else {}
    )
    body = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "source_root": str(source_root),
        "source_commit": source_commit,
        "source_git_identity": source_git_identity,
        "source_manifest_path": str(manifest_path),
        "controller_program": str(controller_program),
        "controller_program_sha256": _sha_file(controller_program),
        "python": str(python),
        "python_sha256": _sha_file(resolved_python),
        "model": str(model),
        "model_manifest": model_manifest,
        "model_checkpoint": model_checkpoint,
        "out_dir": str(out_dir / "sweep"),
        "arms": arms,
        "seed": int(seed),
        "per_domain": int(per_domain),
        "difficulty": int(difficulty),
        "campaign_stage": campaign_stage,
        "domains": list(normalized_domains),
        "task_registry_version": task_registry_version,
        "n_slots": int(n_slots),
        "max_tokens": int(max_tokens),
        "memory_fraction": float(memory_fraction),
        "episode_wall_s": float(episode_wall_s),
        "attempt_wall_s": float(attempt_wall_s),
        "max_attempts": int(max_attempts),
        "poll_s": float(poll_s),
        "stale_after_s": float(stale_after_s),
        "retry_backoff_s": float(retry_backoff_s),
        "heartbeat_key_path": str(key_path),
        "launch_label": f"com.aura.rlc-reconciliation.{campaign_id}",
        **adapter_input,
        **integrated_recurrent_input,
    }
    config = {**body, "config_sha256": _sha(body)}
    return config, manifest, key


def write_prepared_campaign(
    config_path: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    key: bytes,
) -> None:
    config_path = config_path.expanduser().absolute()
    manifest_path = Path(str(config["source_manifest_path"]))
    key_path = Path(str(config["heartbeat_key_path"]))
    if config_path.exists() or manifest_path.exists() or key_path.exists():
        raise ControllerError("prepared_campaign_artifact_exists")
    _atomic_json(manifest_path, manifest)
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as sink:
        sink.write(key)
        sink.flush()
        os.fsync(sink.fileno())
    _atomic_json(config_path, config)


def _copy_file_durable(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    with source.open("rb") as source_handle, temporary.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "size": destination.stat().st_size,
        "sha256": _sha_file(destination),
    }


def _validated_recovery_files(previous_sweep: Path) -> tuple[int, list[Path]]:
    fingerprint_path = previous_sweep / "decode_fingerprint.json"
    task_path = previous_sweep / "task_commitment.json"
    journal_path = previous_sweep / "journal.jsonl"
    recorded = _read_json(fingerprint_path, role="decode_fingerprint")
    fingerprints = recorded.get("decode_fingerprint")
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ControllerError("recovery_decode_fingerprint_invalid")
    required: set[Path] = {fingerprint_path, task_path, journal_path}
    keys: set[tuple[str, str]] = set()
    cells = 0
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ControllerError("recovery_journal_unavailable") from exc
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ControllerError("recovery_journal_torn_or_invalid") from exc
        if record.get("event") != "CELL":
            continue
        arm = str(record.get("arm") or "")
        task_id = str(record.get("task_id") or "")
        key = (arm, task_id)
        if not arm or not task_id or key in keys:
            raise ControllerError("recovery_journal_duplicate_or_unbound_cell")
        if record.get("decode_fingerprint") != fingerprints.get(arm):
            raise ControllerError("recovery_journal_fingerprint_mismatch")
        keys.add(key)
        cells += 1
        relative_receipt = record.get("runtime_receipt_path")
        if relative_receipt:
            relative = Path(str(relative_receipt))
            if relative.is_absolute() or ".." in relative.parts:
                raise ControllerError("recovery_runtime_receipt_path_invalid")
            receipt_path = (previous_sweep / relative).resolve(strict=True)
            if previous_sweep.resolve() not in receipt_path.parents:
                raise ControllerError("recovery_runtime_receipt_path_invalid")
            receipt = _read_json(receipt_path, role="recovery_runtime_receipt")
            if _sha(receipt) != record.get("runtime_receipt_sha256"):
                raise ControllerError("recovery_runtime_receipt_hash_mismatch")
            required.add(receipt_path)
    if cells == 0:
        raise ControllerError("recovery_journal_empty")
    if not task_path.is_file():
        raise ControllerError("recovery_task_commitment_missing")
    return cells, sorted(required, key=lambda path: os.fsencode(str(path)))


def recover_campaign(
    previous_config_path: Path,
    *,
    out_dir: Path,
    output: Path,
    controller_program: Path,
) -> dict[str, Any]:
    """Recover admitted cells under a new lifecycle controller, unchanged."""

    previous_config = load_config(previous_config_path)
    _source_is_current(previous_config)
    previous_root = Path(str(previous_config["out_dir"])).parent
    lock_descriptor = os.open(
        previous_root / ".controller.lock", os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("recovery_source_campaign_is_active") from exc
        cells, source_files = _validated_recovery_files(
            Path(str(previous_config["out_dir"]))
        )
        config, manifest, key = build_config(
            source_root=Path(str(previous_config["source_root"])),
            source_commit=str(previous_config["source_commit"]),
            model=Path(str(previous_config["model"])),
            out_dir=out_dir,
            python=Path(str(previous_config["python"])),
            arms=str(previous_config["arms"]),
            seed=int(previous_config["seed"]),
            per_domain=int(previous_config["per_domain"]),
            difficulty=int(previous_config["difficulty"]),
            campaign_stage=str(previous_config["campaign_stage"]),
            domains=tuple(str(value) for value in previous_config["domains"]),
            task_registry_version=str(previous_config["task_registry_version"]),
            n_slots=int(previous_config["n_slots"]),
            max_tokens=int(previous_config["max_tokens"]),
            memory_fraction=float(previous_config["memory_fraction"]),
            episode_wall_s=float(previous_config["episode_wall_s"]),
            attempt_wall_s=float(previous_config["attempt_wall_s"]),
            max_attempts=int(previous_config["max_attempts"]),
            poll_s=float(previous_config["poll_s"]),
            stale_after_s=float(previous_config["stale_after_s"]),
            retry_backoff_s=float(previous_config["retry_backoff_s"]),
            adapter_root=(
                Path(str(previous_config["adapter_root"]))
                if previous_config.get("adapter_root")
                else None
            ),
            integrated_recurrent_package=(
                Path(str(previous_config["integrated_recurrent_package"]["root"]))
                if previous_config.get("integrated_recurrent_package")
                else None
            ),
            integrated_recurrent_max_tokens=int(
                previous_config.get("integrated_recurrent_max_tokens") or 2048
            ),
            controller_program=controller_program,
        )
        write_prepared_campaign(output, config, manifest, key)
        previous_sweep = Path(str(previous_config["out_dir"])).resolve()
        destination_sweep = Path(str(config["out_dir"])).resolve()
        copied: list[dict[str, Any]] = []
        for source in source_files:
            relative = source.resolve().relative_to(previous_sweep)
            identity = _copy_file_durable(source, destination_sweep / relative)
            identity["relative_path"] = relative.as_posix()
            copied.append(identity)
        body = {
            "schema": RECOVERY_SCHEMA,
            "previous_campaign_id": previous_config["campaign_id"],
            "previous_config_sha256": previous_config["config_sha256"],
            "campaign_id": config["campaign_id"],
            "config_sha256": config["config_sha256"],
            "source_commit": config["source_commit"],
            "controller_program": config["controller_program"],
            "controller_program_sha256": config["controller_program_sha256"],
            "preserved_cells": cells,
            "copied_files": copied,
            "scientific_parameters_changed": False,
            "prepared_unix": time.time(),
        }
        receipt = {**body, "recovery_sha256": _sha(body)}
        _atomic_json(out_dir / "recovery_receipt.json", receipt)
        return receipt
    finally:
        os.close(lock_descriptor)


def _heartbeat_key(config: Mapping[str, Any]) -> bytes:
    path = Path(str(config["heartbeat_key_path"]))
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ControllerError("heartbeat_key_custody_invalid")
    key = path.read_bytes()
    if len(key) != 32:
        raise ControllerError("heartbeat_key_invalid")
    return key


def _signed_heartbeat(config: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": HEARTBEAT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        **body,
    }
    payload["hmac_sha256"] = hmac.new(
        _heartbeat_key(config), _canonical(payload), hashlib.sha256
    ).hexdigest()
    return payload


def verify_heartbeat(config: Mapping[str, Any], heartbeat: Mapping[str, Any]) -> None:
    provided = heartbeat.get("hmac_sha256")
    body = {key: value for key, value in heartbeat.items() if key != "hmac_sha256"}
    expected = hmac.new(_heartbeat_key(config), _canonical(body), hashlib.sha256).hexdigest()
    if (
        heartbeat.get("schema") != HEARTBEAT_SCHEMA
        or heartbeat.get("campaign_id") != config["campaign_id"]
        or heartbeat.get("config_sha256") != config["config_sha256"]
        or not isinstance(provided, str)
        or not hmac.compare_digest(provided, expected)
    ):
        raise ControllerError("controller_heartbeat_invalid")


def _journal_cells(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                if json.loads(line).get("event") == "CELL":
                    count += 1
            except (AttributeError, json.JSONDecodeError):
                continue
    return count


def _progress_snapshot(config: Mapping[str, Any], log_path: Path) -> dict[str, Any]:
    out_dir = Path(str(config["out_dir"]))
    status_path = out_dir / "status.json"
    journal_path = out_dir / "journal.jsonl"
    mtimes = [
        path.stat().st_mtime
        for path in (status_path, journal_path, log_path)
        if path.is_file()
    ]
    sweep_status: dict[str, Any] = {}
    if status_path.is_file():
        try:
            sweep_status = _read_json(status_path, role="sweep_status")
        except ControllerError:
            sweep_status = {"phase": "unreadable"}
    return {
        "cells": _journal_cells(journal_path),
        "last_progress_unix": max(mtimes, default=0.0),
        "sweep_phase": sweep_status.get("phase"),
        "sweep_arm": sweep_status.get("arm"),
        "sweep_arm_progress": sweep_status.get("arm_progress"),
    }


def _parse_ps_cpu_time(value: str) -> float:
    day_parts = value.strip().split("-", 1)
    days = 0
    clock = day_parts[0]
    if len(day_parts) == 2:
        days = int(day_parts[0])
        clock = day_parts[1]
    parts = clock.split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError("invalid process CPU time")
    return (
        days * 86_400
        + int(hours) * 3_600
        + int(minutes) * 60
        + float(seconds)
    )


def _process_group_activity(process_group: int) -> dict[str, Any]:
    observed = subprocess.run(
        ["/bin/ps", "-axo", "pgid=,time=,rss=,state="],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0:
        return {"available": False, "member_count": 0}
    cpu_seconds = 0.0
    rss_kib = 0
    states: list[str] = []
    members = 0
    for line in observed.stdout.splitlines():
        fields = line.strip().split()
        if len(fields) != 4 or not fields[0].isdigit():
            continue
        if int(fields[0]) != process_group:
            continue
        try:
            cpu_seconds += _parse_ps_cpu_time(fields[1])
            rss_kib += int(fields[2])
        except ValueError:
            return {"available": False, "member_count": members}
        states.append(fields[3])
        members += 1
    return {
        "available": members > 0,
        "member_count": members,
        "cpu_seconds": round(cpu_seconds, 2),
        "rss_kib": rss_kib,
        "states": sorted(states),
    }


def _progress_is_stale(
    snapshot: Mapping[str, Any],
    *,
    attempt_started_unix: float,
    process_activity_unix: float,
    observed_unix: float,
    stale_after_s: float,
) -> bool:
    """Judge this attempt only from progress it had an opportunity to make."""

    effective_progress_unix = max(
        float(snapshot.get("last_progress_unix") or 0.0),
        float(attempt_started_unix),
        float(process_activity_unix),
    )
    return observed_unix - effective_progress_unix > stale_after_s


def _sweep_command(config: Mapping[str, Any]) -> list[str]:
    command = [
        str(config["python"]),
        str(Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_sweep.py"),
        "--model",
        str(config["model"]),
        "--out-dir",
        str(config["out_dir"]),
        "--seed",
        str(config["seed"]),
        "--per-domain",
        str(config["per_domain"]),
        "--difficulty",
        str(config["difficulty"]),
        "--campaign-stage",
        str(config["campaign_stage"]),
        "--domains",
        ",".join(str(value) for value in config["domains"]),
        "--task-registry-version",
        str(config["task_registry_version"]),
        "--n-slots",
        str(config["n_slots"]),
        "--max-tokens",
        str(config["max_tokens"]),
        "--memory-fraction",
        str(config["memory_fraction"]),
        "--episode-wall-s",
        str(config["episode_wall_s"]),
        "--max-wall-s",
        str(config["attempt_wall_s"]),
        "--arms",
        str(config["arms"]),
    ]
    if config.get("adapter_root"):
        command.extend(
            [
                "--adapter",
                str(config["adapter_root"]),
                "--adapter-manifest",
                str(config["adapter_manifest"]),
            ]
        )
    if config.get("integrated_recurrent_package"):
        command.extend(
            [
                "--integrated-recurrent-package",
                str(config["integrated_recurrent_package"]["root"]),
                "--integrated-recurrent-max-tokens",
                str(config["integrated_recurrent_max_tokens"]),
            ]
        )
    return command


def _terminate_exact_group(process: subprocess.Popen[Any]) -> None:
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    if group != process.pid:
        raise ControllerError("sweep_process_group_identity_invalid")
    os.killpg(group, signal.SIGTERM)
    try:
        process.wait(timeout=15.0)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(group, signal.SIGKILL)
    process.wait(timeout=10.0)


def _write_status(config: Mapping[str, Any], *, phase: str, **fields: Any) -> None:
    out_root = Path(str(config["out_dir"])).parent
    body = {
        "schema": STATUS_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "phase": phase,
        "updated_unix": time.time(),
        **fields,
    }
    _atomic_json(out_root / "controller_status.json", {**body, "status_sha256": _sha(body)})


def _source_is_current(config: Mapping[str, Any]) -> None:
    manifest = _read_json(Path(str(config["source_manifest_path"])), role="source_manifest")
    if manifest.get("source_commit") != config["source_commit"]:
        raise ControllerError("source_commit_binding_invalid")
    verify_source_git_identity(
        Path(str(config["source_root"])),
        config["source_git_identity"],
    )
    verify_source_manifest(Path(str(config["source_root"])), manifest)
    python = Path(str(config["python"])).expanduser().absolute()
    if _sha_file(python.resolve(strict=True)) != config["python_sha256"]:
        raise ControllerError("interpreter_identity_drift")
    controller_program = _controller_program(config).resolve(strict=True)
    expected_controller_sha256 = config.get("controller_program_sha256")
    if expected_controller_sha256 is not None:
        if _sha_file(controller_program) != expected_controller_sha256:
            raise ControllerError("controller_program_identity_drift")
    elif controller_program != (
        Path(str(config["source_root"])) / "tools/run_rlc_reconciliation_controller.py"
    ).resolve(strict=True):
        raise ControllerError("controller_program_identity_incomplete")
    verify_model_manifest(config["model_manifest"])
    observed_checkpoint = _model_checkpoint_identity(config["model_manifest"])
    if config.get("model_checkpoint") != observed_checkpoint:
        raise ControllerError("model_checkpoint_identity_drift")
    if config.get("adapter_root"):
        observed_adapter = _verified_adapter_input(
            Path(str(config["adapter_root"])),
            model_checkpoint=observed_checkpoint,
        )
        expected_adapter = {
            key: config[key]
            for key in (
                "adapter_root",
                "adapter",
                "adapter_manifest",
                "adapter_freeze",
            )
        }
        if observed_adapter != expected_adapter:
            raise ControllerError("adapter_identity_drift")
    if config.get("integrated_recurrent_package"):
        _verify_integrated_recurrent_package_input(
            config["integrated_recurrent_package"]
        )


def _process_record(pid: int) -> tuple[int, str]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-o", "ppid=", "-o", "command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0 or not observed.stdout.strip():
        raise ControllerError(f"process_lineage_unavailable:{pid}")
    fields = observed.stdout.strip().split(None, 1)
    if len(fields) != 2 or not fields[0].isdigit():
        raise ControllerError(f"process_lineage_invalid:{pid}")
    return int(fields[0]), fields[1]


def _process_table() -> list[tuple[int, int, str]]:
    observed = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=,ppid=,command="],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if observed.returncode != 0:
        raise ControllerError("process_table_unavailable")
    rows: list[tuple[int, int, str]] = []
    for line in observed.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit():
            rows.append((int(fields[0]), int(fields[1]), fields[2]))
    return rows


def _verify_launchd_lineage(
    config: Mapping[str, Any], *, caffeinate_wait_s: float = 5.0
) -> dict[str, Any]:
    controller_pid = os.getpid()
    controller_parent, controller_command = _process_record(controller_pid)
    controller_program = str(_controller_program(config))
    controller_required = (
        controller_program,
        " run ",
        "--config",
        str(config["campaign_id"]),
        "--launchd-supervised",
    )
    if controller_parent != 1 or any(
        value not in controller_command for value in controller_required
    ):
        raise ControllerError("controller_launchd_caffeinate_lineage_invalid")
    caffeinate_required = (
        "/usr/bin/caffeinate",
        "-dims",
        str(config["python"]),
        controller_program,
        str(config["campaign_id"]),
    )
    deadline = time.monotonic() + max(0.0, float(caffeinate_wait_s))
    while True:
        caffeinate_children = [
            pid
            for pid, parent, command in _process_table()
            if parent == controller_pid
            and all(value in command for value in caffeinate_required)
        ]
        if len(caffeinate_children) == 1:
            break
        if len(caffeinate_children) > 1 or time.monotonic() >= deadline:
            raise ControllerError("controller_launchd_caffeinate_lineage_invalid")
        time.sleep(0.05)
    return {
        "launchd_pid": 1,
        "caffeinate_pid": caffeinate_children[0],
        "controller_pid": controller_pid,
    }


def run(config_path: Path, *, launchd_supervised: bool = False) -> int:
    config = load_config(config_path)
    if not launchd_supervised:
        raise ControllerError("controller_requires_launchd_supervision")
    lineage = _verify_launchd_lineage(config)
    root = Path(str(config["out_dir"])).parent
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_descriptor = os.open(root / ".controller.lock", os.O_RDWR | os.O_CREAT, 0o600)
    GLOBAL_MODEL_LOCK.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    global_lock_descriptor = os.open(
        GLOBAL_MODEL_LOCK, os.O_RDWR | os.O_CREAT, 0o600
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("controller_already_running") from exc
        try:
            fcntl.flock(global_lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ControllerError("another_reconciliation_model_owner_is_active") from exc
        _source_is_current(config)
        log_path = root / "sweep.log"
        events_path = root / "controller_attempts.jsonl"
        heartbeat_path = root / "controller_heartbeat.json"
        for attempt in range(1, int(config["max_attempts"]) + 1):
            if (Path(str(config["out_dir"])) / "YIELD").exists():
                _write_status(config, phase="yielded", attempt=attempt - 1)
                return 0
            if _terminal_verdict(Path(str(config["out_dir"])) / "verdict.json"):
                _write_status(config, phase="complete", attempt=attempt - 1)
                return 0
            _source_is_current(config)
            command = _sweep_command(config)
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=str(config["source_root"]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "AURA_LOG_DIR": str(root / "logs")},
                )
                _append_event(
                    events_path,
                    {
                        "event": "ATTEMPT_STARTED",
                        "attempt": attempt,
                        "pid": process.pid,
                        "process_group": process.pid,
                        "command_sha256": _sha(command),
                        "source_commit": config["source_commit"],
                        "started_unix": started,
                    },
                )
                reason = ""
                last_cpu_seconds = -1.0
                process_activity_unix = started
                while process.poll() is None:
                    snapshot = _progress_snapshot(config, log_path)
                    observed_unix = time.time()
                    process_activity = _process_group_activity(process.pid)
                    cpu_seconds = float(process_activity.get("cpu_seconds") or 0.0)
                    if process_activity.get("available") and cpu_seconds > last_cpu_seconds:
                        process_activity_unix = observed_unix
                        last_cpu_seconds = cpu_seconds
                    heartbeat = _signed_heartbeat(
                        config,
                        {
                            "controller_pid": os.getpid(),
                            "sweep_pid": process.pid,
                            "sweep_process_group": process.pid,
                            "attempt": attempt,
                            "observed_unix": observed_unix,
                            "attempt_started_unix": started,
                            "process_activity_unix": process_activity_unix,
                            "process_group_activity": process_activity,
                            "lineage": lineage,
                            **snapshot,
                        },
                    )
                    _atomic_json(heartbeat_path, heartbeat)
                    if _progress_is_stale(
                        snapshot,
                        attempt_started_unix=started,
                        process_activity_unix=process_activity_unix,
                        observed_unix=observed_unix,
                        stale_after_s=float(config["stale_after_s"]),
                    ):
                        reason = "progress_stalled"
                        _terminate_exact_group(process)
                        break
                    time.sleep(float(config["poll_s"]))
                returncode = process.poll()
            snapshot = _progress_snapshot(config, log_path)
            _append_event(
                events_path,
                {
                    "event": "ATTEMPT_FINISHED",
                    "attempt": attempt,
                    "pid": process.pid,
                    "returncode": returncode,
                    "reason": reason or "process_exited",
                    "finished_unix": time.time(),
                    **snapshot,
                },
            )
            if returncode == 4 or (Path(str(config["out_dir"])) / "YIELD").exists():
                _write_status(config, phase="yielded", attempt=attempt, **snapshot)
                return 0
            if _terminal_verdict(Path(str(config["out_dir"])) / "verdict.json"):
                _write_status(config, phase="complete", attempt=attempt, **snapshot)
                return 0
            if attempt < int(config["max_attempts"]):
                _write_status(
                    config,
                    phase="retrying",
                    attempt=attempt,
                    returncode=returncode,
                    reason=reason or "incomplete_exit",
                    **snapshot,
                )
                time.sleep(float(config["retry_backoff_s"]))
        _write_status(config, phase="blocked", reason="attempt_budget_exhausted")
        return 2
    finally:
        os.close(global_lock_descriptor)
        os.close(lock_descriptor)


def _launch_payload(config_path: Path, config: Mapping[str, Any]) -> bytes:
    root = Path(str(config["out_dir"])).parent
    payload = {
        "Label": config["launch_label"],
        "ProgramArguments": [
            "/usr/bin/caffeinate",
            "-dims",
            str(config["python"]),
            str(_controller_program(config)),
            "run",
            "--config",
            str(config_path.expanduser().resolve(strict=True)),
            "--launchd-supervised",
        ],
        "WorkingDirectory": str(config["source_root"]),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "StandardOutPath": str(root / "controller.log"),
        "StandardErrorPath": str(root / "controller.log"),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)


def install_launchd(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    _source_is_current(config)
    root = Path(str(config["out_dir"])).parent
    launch_agents = Path.home() / "Library/LaunchAgents"
    launch_agents.mkdir(parents=True, exist_ok=True, mode=0o700)
    plist_path = launch_agents / f"{config['launch_label']}.plist"
    payload = _launch_payload(config_path, config)
    temporary = plist_path.with_suffix(".tmp")
    temporary.write_bytes(payload)
    os.chmod(temporary, 0o600)
    os.replace(temporary, plist_path)
    domain = f"gui/{os.getuid()}"
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    started = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if started.returncode != 0:
        raise ControllerError(f"launchd_bootstrap_failed:{started.returncode}:{started.stderr.strip()}")
    body = {
        "schema": LAUNCH_SCHEMA,
        "campaign_id": config["campaign_id"],
        "config_sha256": config["config_sha256"],
        "label": config["launch_label"],
        "domain": domain,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(payload).hexdigest(),
        "launchd_keepalive": True,
        "caffeinate": True,
        "installed_unix": time.time(),
    }
    receipt = {**body, "launch_sha256": _sha(body)}
    _atomic_json(root / "launchd_receipt.json", receipt)
    return receipt


def status(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = Path(str(config["out_dir"])).parent
    result: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "source_commit": config["source_commit"],
        "controller_status": None,
        "heartbeat": None,
    }
    status_path = root / "controller_status.json"
    if status_path.is_file():
        controller_status = _read_json(status_path, role="controller_status")
        body = {key: value for key, value in controller_status.items() if key != "status_sha256"}
        if controller_status.get("status_sha256") != _sha(body):
            raise ControllerError("controller_status_invalid")
        result["controller_status"] = controller_status
    heartbeat_path = root / "controller_heartbeat.json"
    if heartbeat_path.is_file():
        heartbeat = _read_json(heartbeat_path, role="controller_heartbeat")
        verify_heartbeat(config, heartbeat)
        result["heartbeat"] = heartbeat
    result["progress"] = _progress_snapshot(config, root / "sweep.log")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--source-commit", required=True)
    prepare.add_argument("--model", type=Path, required=True)
    prepare.add_argument("--adapter-root", type=Path)
    prepare.add_argument("--integrated-recurrent-package", type=Path)
    prepare.add_argument("--integrated-recurrent-max-tokens", type=int, default=2048)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--python", type=Path, required=True)
    prepare.add_argument("--controller-program", type=Path)
    prepare.add_argument("--arms", default="full_stack,full_stack_oracle")
    prepare.add_argument("--seed", type=int, default=20260808)
    prepare.add_argument("--per-domain", type=int, default=4)
    prepare.add_argument("--difficulty", type=int, choices=(1, 2, 3), default=2)
    prepare.add_argument(
        "--campaign-stage",
        choices=tuple(sorted(CAMPAIGN_STAGES)),
        default="certificate",
    )
    prepare.add_argument(
        "--domains",
        default="",
        help="comma-separated frontier task domains; empty selects all domains",
    )
    prepare.add_argument(
        "--task-registry-version",
        default=CLAIM_TASK_REGISTRY_VERSION,
    )
    prepare.add_argument("--n-slots", type=int, default=16)
    prepare.add_argument("--max-tokens", type=int, default=512)
    prepare.add_argument("--memory-fraction", type=float, default=0.40)
    prepare.add_argument("--episode-wall-s", type=float, default=900.0)
    prepare.add_argument("--attempt-wall-s", type=float, default=32_400.0)
    prepare.add_argument("--max-attempts", type=int, default=8)
    prepare.add_argument("--poll-s", type=float, default=15.0)
    prepare.add_argument("--stale-after-s", type=float, default=1_800.0)
    prepare.add_argument("--retry-backoff-s", type=float, default=30.0)
    prepare.add_argument("--output", type=Path, required=True)
    recover = commands.add_parser("recover")
    recover.add_argument("--previous-config", type=Path, required=True)
    recover.add_argument("--out-dir", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    recover.add_argument("--controller-program", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--launchd-supervised", action="store_true")
    install = commands.add_parser("install-launchd")
    install.add_argument("--config", type=Path, required=True)
    inspect = commands.add_parser("status")
    inspect.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "prepare":
            config, manifest, key = build_config(
                source_root=args.source_root,
                source_commit=args.source_commit,
                model=args.model,
                adapter_root=args.adapter_root,
                integrated_recurrent_package=args.integrated_recurrent_package,
                integrated_recurrent_max_tokens=(
                    args.integrated_recurrent_max_tokens
                ),
                out_dir=args.out_dir,
                python=args.python,
                arms=args.arms,
                seed=args.seed,
                per_domain=args.per_domain,
                difficulty=args.difficulty,
                campaign_stage=args.campaign_stage,
                domains=tuple(
                    value.strip() for value in args.domains.split(",") if value.strip()
                ),
                task_registry_version=args.task_registry_version,
                n_slots=args.n_slots,
                max_tokens=args.max_tokens,
                memory_fraction=args.memory_fraction,
                episode_wall_s=args.episode_wall_s,
                attempt_wall_s=args.attempt_wall_s,
                max_attempts=args.max_attempts,
                poll_s=args.poll_s,
                stale_after_s=args.stale_after_s,
                retry_backoff_s=args.retry_backoff_s,
                controller_program=args.controller_program,
            )
            write_prepared_campaign(args.output, config, manifest, key)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        if args.action == "recover":
            print(
                json.dumps(
                    recover_campaign(
                        args.previous_config,
                        out_dir=args.out_dir,
                        output=args.output,
                        controller_program=args.controller_program,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.action == "run":
            return run(args.config, launchd_supervised=args.launchd_supervised)
        if args.action == "install-launchd":
            print(json.dumps(install_launchd(args.config), indent=2, sort_keys=True))
            return 0
        print(json.dumps(status(args.config), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"run_rlc_reconciliation_controller: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
