#!/usr/bin/env python3
"""Promote a supported resident replication through every serving gate.

This controller deliberately owns no scientific decision.  It waits for the
frozen replication controller, independently recomputes its terminal verdict,
and only then composes materialization, rollback lifecycle proof, and verified
qualified activation.  Every path is deterministic so launchd can resume the
pipeline without repeating or skipping a completed gate.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import plistlib
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Never

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.unified_recurrent_qualified_activation_store import (  # noqa: E402
    default_qualified_activation_path,
)
from core.brain.llm.unified_recurrent_shadow_pointer import (  # noqa: E402
    deactivate_shadow_pointer,
    default_shadow_activation_paths,
    read_shadow_pointer,
    resolve_shadow_pointer,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_bytes_if_absent,
    ensure_private_directory,
)
from tools import adjudicate_unified_intrinsic_resident_replication as replication  # noqa: E402
from tools import manage_unified_recurrent_qualified_activation as activation  # noqa: E402
from tools import materialize_unified_intrinsic_shadow_package as materializer  # noqa: E402
from tools import run_unified_intrinsic_resident_campaign as resident  # noqa: E402
from tools import run_unified_recurrent_shadow_lifecycle as lifecycle  # noqa: E402
from tools.unified_intrinsic_resident_identity import (  # noqa: E402
    build_source_git_identity,
    canonical_bytes,
    canonical_sha256,
    verify_source_git_identity,
)

CONFIG_SCHEMA: Final = "aura.unified_intrinsic.promotion_pipeline_config.v2"
STATUS_SCHEMA: Final = "aura.unified_intrinsic.promotion_pipeline_status.v1"
COMPLETION_SCHEMA: Final = "aura.unified_intrinsic.promotion_pipeline_complete.v2"
INTENT_SCHEMA: Final = "aura.unified_intrinsic.promotion_pipeline_launch_intent.v1"
LAUNCH_SCHEMA: Final = "aura.unified_intrinsic.promotion_pipeline_launchd.v1"
ACTIVE_STAGE_SCHEMA: Final = "aura.unified_intrinsic.promotion_active_stage.v2"
SOURCE_PATHS: Final = (
    "tools/run_unified_recurrent_promotion_pipeline.py",
    "tools/adjudicate_unified_intrinsic_resident_replication.py",
    "tools/materialize_unified_intrinsic_shadow_package.py",
    "tools/run_unified_recurrent_shadow_lifecycle.py",
    "tools/run_unified_recurrent_shadow_live_canary.py",
    "tools/manage_unified_recurrent_qualified_activation.py",
    "core/brain/llm/unified_recurrent_shadow.py",
    "core/brain/llm/unified_recurrent_shadow_pointer.py",
    "core/brain/llm/unified_recurrent_qualified_activation.py",
    "core/brain/llm/unified_recurrent_qualified_activation_store.py",
    "core/brain/llm/unified_recurrent_shadow_battery.py",
)
LAUNCH_AGENTS_ROOT: Final = Path.home() / "Library/LaunchAgents"
CAPSULES_ROOT: Final = Path.home() / ".aura/training-capsules"
_PACKAGE_ID: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,119}")
_STAGE_TIMEOUTS: Final = {
    "materialize": 30 * 60.0,
    "lifecycle": 90 * 60.0,
    "activate": 45 * 60.0,
}


class UnifiedRecurrentPromotionError(RuntimeError):
    """The supported-evidence promotion contract could not be preserved."""


def _fail(message: str) -> Never:
    raise UnifiedRecurrentPromotionError(message)


def _read_canonical(path: Path) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UnifiedRecurrentPromotionError(
            f"promotion document is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict) or payload != canonical_bytes(value) + b"\n":
        _fail(f"promotion document is not canonical: {path}")
    return value


def _source_identity(root: Path = REPO_ROOT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    commit = result.stdout.strip().lower()
    if result.returncode != 0:
        _fail("promotion source commit is unavailable")
    try:
        git = build_source_git_identity(root, source_commit=commit)
        verify_source_git_identity(root, git)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion source is not a clean detached capsule"
        ) from exc
    digests: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = root / relative
        try:
            digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise UnifiedRecurrentPromotionError(
                f"promotion source is unavailable: {relative}"
            ) from exc
    body = {"root": str(root), "git": git, "source_sha256s": digests}
    return {**body, "identity_sha256": canonical_sha256(body)}


def _pipeline_root(campaign: Path, requested: Path | None) -> Path:
    lexical = (
        requested.expanduser().absolute()
        if requested is not None
        else campaign / "resident-promotion"
    )
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                _fail("promotion state path contains a symlink")
        except OSError as exc:
            raise UnifiedRecurrentPromotionError(
                "promotion state path is unavailable"
            ) from exc
    root = lexical.resolve(strict=False)
    if root == campaign or not root.is_relative_to(campaign):
        _fail("promotion state must be a strict campaign child")
    return root


def _replication_arguments(config: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        campaign=Path(str(config["campaign"])),
        output=Path(str(config["replication_root"])),
        verdict_output=None,
    )


def _adjudicate_available_replication(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    arguments = _replication_arguments(config)
    verdict_path = Path(str(config["replication_root"])) / "replication-verdict.json"
    arguments.verdict_output = verdict_path
    try:
        verdict = replication.adjudicate(arguments)
    except replication.ResidentReplicationIncompleteError:
        return None
    stored = _read_canonical(verdict_path)
    if stored != verdict:
        _fail("promotion stored and recomputed replication verdicts differ")
    return verdict


def _key(config: Mapping[str, Any]) -> bytes:
    campaign_config = resident._load_config(  # noqa: SLF001
        Path(str(config["campaign"])) / "campaign.json"
    )
    if campaign_config.get("config_sha256") != config.get("campaign_config_sha256"):
        _fail("promotion campaign identity differs")
    return resident._key(  # noqa: SLF001
        Path(campaign_config["paths"]["heartbeat_key"]),
        expected_sha256=str(campaign_config["heartbeat_key_sha256"]),
    )


def _signature(body: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, canonical_bytes(dict(body)), hashlib.sha256).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_canonical(path.expanduser().absolute())
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("config_sha256") != canonical_sha256(body)
        or config.get("source") != _source_identity()
    ):
        _fail("promotion config or source identity differs")
    campaign = Path(str(config.get("campaign") or "")).resolve(strict=True)
    if _pipeline_root(campaign, Path(str(config.get("pipeline_root") or ""))) != Path(
        str(config.get("pipeline_root") or "")
    ):
        _fail("promotion pipeline path identity differs")
    replication_root = _pipeline_root(
        campaign,
        Path(str(config.get("replication_root") or "")),
    )
    if replication_root != Path(str(config.get("replication_root") or "")):
        _fail("promotion replication path identity differs")
    _key(config)
    return config


def _status_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["pipeline_root"])) / "controller-status.json"


def _read_status(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _status_path(config)
    if not path.exists():
        return None
    status = _read_canonical(path)
    body = {key: value for key, value in status.items() if key != "hmac_sha256"}
    if (
        status.get("schema") != STATUS_SCHEMA
        or status.get("config_sha256") != config.get("config_sha256")
        or type(status.get("sequence")) is not int
        or int(status["sequence"]) < 1
        or not isinstance(status.get("hmac_sha256"), str)
        or not hmac.compare_digest(status["hmac_sha256"], _signature(body, _key(config)))
    ):
        _fail("promotion controller status authentication failed")
    return status


def _publish_status(
    config: Mapping[str, Any],
    state: str,
    details: Mapping[str, Any],
    *,
    sleep_inhibitor_pid: int | None = None,
) -> dict[str, Any]:
    previous = _read_status(config)
    body = {
        "schema": STATUS_SCHEMA,
        "config_sha256": config["config_sha256"],
        "sequence": 1 if previous is None else int(previous["sequence"]) + 1,
        "state": state,
        "controller_pid": os.getpid(),
        "controller_start_token": replication.launcher.detached._process_start_token(  # noqa: SLF001
            os.getpid()
        ),
        "sleep_inhibitor_pid": sleep_inhibitor_pid,
        "heartbeat_at": time.time(),
        "details": dict(details),
    }
    status = {**body, "hmac_sha256": _signature(body, _key(config))}
    atomic_write_bytes(_status_path(config), canonical_bytes(status) + b"\n", mode=0o600)
    return status


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    if _PACKAGE_ID.fullmatch(str(arguments.package_id)) is None:
        _fail("promotion package id is invalid")
    campaign = arguments.campaign.expanduser().resolve(strict=True)
    campaign_config = resident._load_config(campaign / "campaign.json")  # noqa: SLF001
    if Path(campaign_config["paths"]["campaign_root"]).resolve(strict=True) != campaign:
        _fail("promotion campaign root differs")
    replication_root = _pipeline_root(
        campaign,
        arguments.replication_root
        if arguments.replication_root is not None
        else campaign / "resident-replication",
    )
    replication_args = argparse.Namespace(
        campaign=campaign,
        output=replication_root,
        verdict_output=None,
    )
    _campaign, _config, plan = replication._load_plan(replication_args)  # noqa: SLF001
    pipeline_root = _pipeline_root(campaign, arguments.output)
    ensure_private_directory(pipeline_root)
    pointer, releases = default_shadow_activation_paths()
    model_path = Path(str(campaign_config["model"]["root"])).resolve(strict=True)
    body = {
        "schema": CONFIG_SCHEMA,
        "campaign": str(campaign),
        "campaign_id": campaign_config["campaign_id"],
        "campaign_config_sha256": campaign_config["config_sha256"],
        "campaign_source_commit": campaign_config["source"]["git"]["commit"],
        "replication_root": str(replication_root),
        "replication_plan_sha256": plan["plan_sha256"],
        "pipeline_root": str(pipeline_root),
        "package_id": arguments.package_id,
        "package_root": str(releases),
        "package": str(releases / arguments.package_id),
        "lifecycle_output": str(pipeline_root / "lifecycle"),
        "qualified_canary_output": str(pipeline_root / "qualified-canary.json"),
        "completion_output": str(pipeline_root / "promotion-complete.json"),
        "model_path": str(model_path),
        "model_manifest_sha256": campaign_config["model"]["manifest_sha256"],
        "pointer_path": str(pointer),
        "activation_path": str(default_qualified_activation_path()),
        "stage_timeouts": dict(_STAGE_TIMEOUTS),
        "source": _source_identity(),
    }
    config = {**body, "config_sha256": canonical_sha256(body)}
    path = pipeline_root / "promotion-config.json"
    payload = canonical_bytes(config) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != config:
            _fail("promotion config already differs")
    return config


def _published_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    commit = result.stdout.strip().lower()
    if (
        result.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        _fail("promotion source commit is invalid")
    published = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), "merge-base", "--is-ancestor", commit, "origin/main"],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if published.returncode != 0:
        _fail("promotion source commit is not published on origin/main")
    return commit


def _capsule(commit: str) -> Path:
    ensure_private_directory(CAPSULES_ROOT)
    capsule = CAPSULES_ROOT / f"unified-recurrent-promotion-{commit[:12]}"
    if capsule.exists():
        try:
            identity = build_source_git_identity(capsule, source_commit=commit)
            verify_source_git_identity(capsule, identity)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise UnifiedRecurrentPromotionError(
                "existing promotion capsule identity differs"
            ) from exc
        return capsule
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(capsule), commit],
        capture_output=True,
        text=True,
        timeout=300.0,
        check=False,
    )
    if result.returncode != 0:
        _fail(f"promotion capsule creation failed: {result.stderr.strip()[:300]}")
    try:
        identity = build_source_git_identity(capsule, source_commit=commit)
        verify_source_git_identity(capsule, identity)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedRecurrentPromotionError(
            "new promotion capsule identity differs"
        ) from exc
    return capsule


def install_published(arguments: argparse.Namespace) -> dict[str, Any]:
    """Create a clean published capsule, prepare there, and install its controller."""

    if _PACKAGE_ID.fullmatch(str(arguments.package_id)) is None:
        _fail("promotion package id is invalid")
    commit = _published_commit(REPO_ROOT)
    capsule = _capsule(commit)
    campaign = arguments.campaign.expanduser().resolve(strict=True)
    campaign_config = resident._load_config(campaign / "campaign.json")  # noqa: SLF001
    python, _interpreter = replication._runtime_python(campaign_config)  # noqa: SLF001
    script = capsule / "tools/run_unified_recurrent_promotion_pipeline.py"
    command = [
        str(python),
        str(script),
        "prepare",
        str(campaign),
        "--package-id",
        str(arguments.package_id),
    ]
    if arguments.replication_root is not None:
        command.extend(("--replication-root", str(arguments.replication_root)))
    if arguments.output is not None:
        command.extend(("--output", str(arguments.output)))
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    prepared = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300.0,
        check=False,
        env=environment,
    )
    if prepared.returncode != 0:
        _fail(f"promotion capsule preparation failed: {prepared.stderr.strip()[:500]}")
    try:
        config = json.loads(prepared.stdout)
    except json.JSONDecodeError as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion capsule preparation output is invalid"
        ) from exc
    if not isinstance(config, dict) or config.get("source", {}).get("git", {}).get(
        "commit"
    ) != commit:
        _fail("promotion capsule preparation identity differs")
    config_path = Path(str(config["pipeline_root"])) / "promotion-config.json"
    installed = subprocess.run(
        [
            str(python),
            str(script),
            "install-launchd",
            str(config_path),
            "--poll-interval",
            str(float(arguments.poll_interval)),
            "--controller-timeout",
            str(float(arguments.controller_timeout)),
        ],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
        env=environment,
    )
    if installed.returncode != 0:
        _fail(f"promotion capsule launch failed: {installed.stderr.strip()[:500]}")
    try:
        receipt = json.loads(installed.stdout)
    except json.JSONDecodeError as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion capsule launch output is invalid"
        ) from exc
    return {"capsule": str(capsule), "config": config, "launch": receipt}


def _verify_package_binding(
    config: Mapping[str, Any],
    package: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    package_path = Path(str(package.get("package") or ""))
    try:
        manifest = materializer._read_document(package_path / "manifest.json")  # noqa: SLF001
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion package manifest is unavailable"
        ) from exc
    if manifest.get("manifest_sha256") != package.get("manifest_sha256"):
        _fail("promotion package inspection and manifest identities differ")
    expected = {
        "package_id": config["package_id"],
        "campaign_id": config["campaign_id"],
        "campaign_config_sha256": config["campaign_config_sha256"],
        "source_commit": config["campaign_source_commit"],
        "model_manifest_sha256": config["model_manifest_sha256"],
        "replication_plan_sha256": config["replication_plan_sha256"],
        "replication_verdict_sha256": verdict["verdict_sha256"],
        "checkpoint_sha256": verdict["checkpoint_sha256"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        _fail("promotion package does not bind the current replication")
    return dict(manifest)


def _materialize_or_reopen(
    config: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    package = Path(str(config["package"]))
    if package.exists():
        result = materializer.inspect_shadow_package(
            package,
            expected_package_id=str(config["package_id"]),
        )
    else:
        result = materializer.materialize(
            Path(str(config["campaign"])),
            output_root=Path(str(config["package_root"])),
            package_id=str(config["package_id"]),
            replication_root=Path(str(config["replication_root"])),
        )
    manifest = _verify_package_binding(config, result, verdict)
    return {**result, "manifest": manifest}


def _read_bound_canary_battery(
    package_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts")
    binding = artifacts.get("canary_battery") if isinstance(artifacts, Mapping) else None
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"path", "sha256", "size_bytes"}
        or not isinstance(binding.get("path"), str)
        or Path(binding["path"]).name != binding["path"]
    ):
        _fail("promotion canary battery binding is unavailable")
    battery_path = package_path / binding["path"]
    identity = materializer._stable_identity(battery_path)  # noqa: SLF001
    battery = materializer._read_document(battery_path)  # noqa: SLF001
    try:
        verified = materializer.validate_shadow_canary_battery(battery)
    except (TypeError, ValueError) as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion canary battery commitment differs"
        ) from exc
    if (
        verified.get("battery_sha256") != manifest.get("canary_battery_sha256")
        or identity != dict(binding)
    ):
        _fail("promotion canary battery artifact identity differs")
    return verified


def _lifecycle_or_reopen(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(config["lifecycle_output"]))
    result_path = output / "lifecycle-result.json"
    if result_path.exists():
        result = activation._read_lifecycle(result_path)  # noqa: SLF001
        if result.get("supported") is not True:
            _fail("promotion lifecycle receipt is not supported")
        return result
    if output.exists() and any(output.iterdir()):
        pointer_path = Path(str(config["pointer_path"]))
        activation_path = Path(str(config["activation_path"]))
        if activation_path.exists() or activation_path.is_symlink():
            _fail("promotion lifecycle interruption left qualified authority")
        if pointer_path.exists() or pointer_path.is_symlink():
            pointer = read_shadow_pointer(pointer_path)
            package = resolve_shadow_pointer(
                pointer_path,
                releases_root=Path(str(config["package_root"])),
            )
            if package != Path(str(config["package"])):
                _fail("promotion lifecycle interruption selected another package")
            deactivate_shadow_pointer(
                pointer_path=pointer_path,
                releases_root=Path(str(config["package_root"])),
                expected_current_sha256=pointer["pointer_sha256"],
            )
        quarantine = output.with_name(
            f"{output.name}.interrupted-{time.time_ns()}"
        )
        os.replace(output, quarantine)
    return asyncio.run(
        lifecycle.run_lifecycle(
            Path(str(config["package"])),
            model_path=Path(str(config["model_path"])),
            output_directory=output,
            minimum_wrong_to_right=1,
            maximum_shadow_latency_ms=120_000,
            maximum_latency_ratio_numerator=8,
            maximum_latency_ratio_denominator=1,
        )
    )


def _activation_arguments(config: Mapping[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        package=Path(str(config["package"])),
        lifecycle_result=Path(str(config["lifecycle_output"])) / "lifecycle-result.json",
        model=Path(str(config["model_path"])),
        canary_output=Path(str(config["qualified_canary_output"])),
        case_timeout=180.0,
        pointer=None,
        releases_root=None,
        activation=None,
        expected_current_pointer_sha256=None,
        expected_current_activation_sha256=None,
    )


def _activate_or_reopen(config: Mapping[str, Any]) -> dict[str, Any]:
    arguments = _activation_arguments(config)
    canary_path = Path(str(config["qualified_canary_output"]))
    pointer_path = Path(str(config["pointer_path"]))
    activation_path = Path(str(config["activation_path"]))
    candidate_canary_path = activation._candidate_canary_path(canary_path)  # noqa: SLF001
    if (activation_path.exists() or activation_path.is_symlink()) and not (
        pointer_path.exists() or pointer_path.is_symlink()
    ):
        orphaned = activation.read_qualified_activation(activation_path)
        activation.deactivate_qualified_activation(
            activation_path=activation_path,
            expected_current_sha256=orphaned["activation_sha256"],
        )
        for stale in (candidate_canary_path, canary_path):
            activation._quarantine_existing_canary(stale)  # noqa: SLF001
    if activation_path.exists() or activation_path.is_symlink():
        observed = activation._status(arguments)  # noqa: SLF001
        if Path(str(observed.get("package") or "")) != Path(str(config["package"])):
            _fail("promotion refuses another active qualified package")
        manifest = materializer._read_document(  # noqa: SLF001
            Path(str(config["package"])) / "manifest.json"
        )
        battery = _read_bound_canary_battery(Path(str(config["package"])), manifest)
        durable = activation.read_qualified_activation(activation_path)
        candidate_canary = (
            activation._read_lifecycle(candidate_canary_path)  # noqa: SLF001
            if candidate_canary_path.exists()
            else None
        )
        canary = (
            activation._read_lifecycle(canary_path)  # noqa: SLF001
            if canary_path.exists()
            else None
        )
        if isinstance(canary, Mapping) and isinstance(candidate_canary, Mapping):
            try:
                pending = activation.pending_activation_from_serving(durable)
                candidate = activation.candidate_activation_from_pending(pending)
            except ValueError:
                pending = None
                candidate = None
            if (
                isinstance(pending, Mapping)
                and isinstance(candidate, Mapping)
                and not activation.qualified_serving_canary_errors(
                    canary,
                    expected_activation=pending,
                    expected_battery=battery,
                )
                and not activation.qualified_serving_canary_errors(
                    candidate_canary,
                    expected_activation=candidate,
                    expected_battery=battery,
                )
                and canary.get("manifest_sha256")
                == manifest.get("manifest_sha256")
                and canary.get("authority_remains_active") is False
                and canary.get("canary_authority_was_request_scoped") is True
                and candidate_canary.get("serving_authority") is False
                and candidate_canary.get("authority_remains_active") is False
                and candidate_canary.get("canary_authority_was_request_scoped") is True
                and candidate_canary.get("package_id") == durable.get("package_id")
                and candidate_canary.get("manifest_sha256")
                == durable.get("manifest_sha256")
                and candidate_canary.get("checkpoint_sha256")
                == durable.get("checkpoint_sha256")
                and candidate_canary.get("controller_sha256")
                == durable.get("controller_sha256")
                and candidate_canary.get("result_sha256")
                == durable.get("candidate_canary_sha256")
                and canary.get("result_sha256")
                == durable.get("qualified_canary_sha256")
            ):
                return {
                    **observed,
                    "action": "activate_verified",
                    "canary": canary,
                    "candidate_canary": candidate_canary,
                }
        activation.deactivate_qualified_activation(
            activation_path=activation_path,
            expected_current_sha256=str(observed.get("activation_sha256") or ""),
        )
        if pointer_path.exists() or pointer_path.is_symlink():
            pointer = read_shadow_pointer(pointer_path)
            deactivate_shadow_pointer(
                pointer_path=pointer_path,
                releases_root=Path(str(config["package_root"])),
                expected_current_sha256=pointer["pointer_sha256"],
            )
        _fail("active qualified authority lacked this pipeline's verified canary and was revoked")
    if pointer_path.exists() or pointer_path.is_symlink():
        pointer = read_shadow_pointer(pointer_path)
        selected = resolve_shadow_pointer(
            pointer_path,
            releases_root=Path(str(config["package_root"])),
        )
        if selected != Path(str(config["package"])):
            _fail("another shadow package owns the inactive pointer")
        deactivate_shadow_pointer(
            pointer_path=pointer_path,
            releases_root=Path(str(config["package_root"])),
            expected_current_sha256=pointer["pointer_sha256"],
        )
        if canary_path.exists():
            os.replace(
                canary_path,
                canary_path.with_name(
                    f"{canary_path.name}.interrupted-{time.time_ns()}"
                ),
            )
    return activation._activate_verified(arguments)  # noqa: SLF001


def _write_completion(
    config: Mapping[str, Any],
    verdict: Mapping[str, Any],
    package: Mapping[str, Any],
    lifecycle_result: Mapping[str, Any],
    activated: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = package.get("manifest")
    canary = activated.get("canary")
    candidate_canary = activated.get("candidate_canary")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(canary, Mapping)
        or not isinstance(candidate_canary, Mapping)
    ):
        _fail("promotion completion identity is unavailable")
    try:
        durable = activation.read_qualified_activation(
            Path(str(config["activation_path"]))
        )
        pending = activation.pending_activation_from_serving(durable)
        candidate = activation.candidate_activation_from_pending(pending)
        battery = _read_bound_canary_battery(
            Path(str(config["package"])),
            manifest,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion completion active authority is invalid"
        ) from exc
    if (
        verdict.get("supported") is not True
        or lifecycle_result.get("supported") is not True
        or canary.get("supported") is not True
        or canary.get("authority_remains_active") is not False
        or canary.get("canary_authority_was_request_scoped") is not True
        or candidate_canary.get("supported") is not True
        or candidate_canary.get("authority_remains_active") is not False
        or candidate_canary.get("canary_authority_was_request_scoped") is not True
        or activated.get("active") is not True
        or activated.get("activation_sha256") != durable.get("activation_sha256")
        or durable.get("qualified_canary_sha256") != canary.get("result_sha256")
        or durable.get("candidate_canary_sha256")
        != candidate_canary.get("result_sha256")
        or activation.qualified_serving_canary_errors(
            canary,
            expected_activation=pending,
            expected_battery=battery,
        )
        or activation.qualified_serving_canary_errors(
            candidate_canary,
            expected_activation=candidate,
            expected_battery=battery,
        )
    ):
        _fail("promotion completion requires every supported serving gate")
    path = Path(str(config["completion_output"]))
    if path.exists():
        existing = _read_canonical(path)
        existing_body = {
            key: value for key, value in existing.items() if key != "completion_sha256"
        }
        if (
            existing.get("schema") != COMPLETION_SCHEMA
            or existing.get("completion_sha256") != canonical_sha256(existing_body)
            or existing.get("config_sha256") != config.get("config_sha256")
            or existing.get("replication_verdict_sha256")
            != verdict.get("verdict_sha256")
            or existing.get("package_id") != manifest.get("package_id")
            or existing.get("manifest_sha256") != manifest.get("manifest_sha256")
            or existing.get("lifecycle_result_sha256")
            != lifecycle_result.get("result_sha256")
            or existing.get("activation_sha256")
            != activated.get("activation_sha256")
            or existing.get("qualified_canary_sha256")
            != canary.get("result_sha256")
            or existing.get("candidate_canary_sha256")
            != candidate_canary.get("result_sha256")
            or existing.get("claim_boundary") != verdict.get("claim_boundary")
            or existing.get("supported") is not True
            or existing.get("serving_authority") is not True
        ):
            _fail("promotion completion receipt identity differs")
        return existing
    body = {
        "schema": COMPLETION_SCHEMA,
        "config_sha256": config["config_sha256"],
        "replication_verdict_sha256": verdict["verdict_sha256"],
        "package_id": manifest["package_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "lifecycle_result_sha256": lifecycle_result["result_sha256"],
        "activation_sha256": activated["activation_sha256"],
        "qualified_canary_sha256": canary["result_sha256"],
        "candidate_canary_sha256": candidate_canary["result_sha256"],
        "supported": True,
        "serving_authority": True,
        "claim_boundary": verdict["claim_boundary"],
        "completed_at_unix": time.time(),
    }
    result = {**body, "completion_sha256": canonical_sha256(body)}
    payload = canonical_bytes(result) + b"\n"
    if not atomic_write_bytes_if_absent(path, payload, mode=0o400):
        if _read_canonical(path) != result:
            _fail("promotion completion receipt already differs")
    return result


def _start_sleep_inhibitor() -> subprocess.Popen[bytes]:
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed system executable and argv
            ["/usr/bin/caffeinate", "-dims", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion sleep inhibitor could not start"
        ) from exc
    time.sleep(0.05)
    if process.poll() is not None:
        _fail("promotion sleep inhibitor exited during startup")
    return process


def _launchd_job(label: str) -> dict[str, Any]:
    target = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", target],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("promotion launchd job is unavailable")
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("pid = "):
            try:
                pid = int(stripped.removeprefix("pid = "))
            except ValueError:
                break
            if pid > 1:
                return {"target": target, "pid": pid}
    _fail("promotion launchd pid is unavailable")


def _process_row(pid: int) -> tuple[int, str] | None:
    result = subprocess.run(
        ["/bin/ps", "-o", "ppid=,command=", "-p", str(pid)],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parent, separator, command = result.stdout.strip().partition(" ")
    if not separator:
        return None
    try:
        return int(parent), command.strip()
    except ValueError:
        return None


def _process_presence_state(pid: int) -> str:
    """Distinguish an absent PID from a live replacement without signaling it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "absent"
    except PermissionError:
        return "present"
    except OSError:
        return "unknown"
    return "present"


def _verify_supervision(config: Mapping[str, Any], inhibitor_pid: int) -> dict[str, Any]:
    job = _launchd_job(_label(config))
    if job["pid"] != os.getpid():
        _fail("promotion launchd controller pid differs")
    inhibitor = _process_row(inhibitor_pid)
    expected = f"/usr/bin/caffeinate -dims -w {os.getpid()}"
    if inhibitor is None or inhibitor[0] != os.getpid() or inhibitor[1] != expected:
        _fail("promotion sleep inhibitor lineage differs")
    return {
        "target": job["target"],
        "controller_pid": job["pid"],
        "sleep_inhibitor_pid": inhibitor_pid,
    }


def _installed_lineage_is_valid(
    status: Mapping[str, Any] | None,
    job: Mapping[str, Any] | None,
) -> bool:
    return _installed_lineage_state(status, job) == "valid"


def _installed_lineage_state(
    status: Mapping[str, Any] | None,
    job: Mapping[str, Any] | None,
) -> str:
    if status is None or job is None:
        return "unknown"
    controller_pid = status.get("controller_pid")
    inhibitor_pid = status.get("sleep_inhibitor_pid")
    if type(controller_pid) is not int or type(inhibitor_pid) is not int:
        return "unknown"
    if job.get("pid") != controller_pid:
        return "conflict"
    owner_state = replication.launcher.detached._identity_state(  # noqa: SLF001
        controller_pid,
        str(status.get("controller_start_token") or ""),
    )
    if owner_state == "dead":
        presence = _process_presence_state(controller_pid)
        if presence == "present":
            return "conflict"
        if presence != "absent":
            return "unknown"
    if owner_state != "alive":
        return owner_state
    inhibitor = _process_row(inhibitor_pid)
    if (
        inhibitor is not None
        and inhibitor[0] == controller_pid
        and inhibitor[1] == f"/usr/bin/caffeinate -dims -w {controller_pid}"
    ):
        return "valid"
    return "conflict"


def _terminal_refutation_is_valid(
    config: Mapping[str, Any],
    status: Mapping[str, Any] | None,
) -> bool:
    if status is None or status.get("state") != "refuted":
        return False
    controller_pid = status.get("controller_pid")
    inhibitor_pid = status.get("sleep_inhibitor_pid")
    start_token = status.get("controller_start_token")
    details = status.get("details")
    if (
        type(controller_pid) is not int
        or controller_pid <= 1
        or type(inhibitor_pid) is not int
        or inhibitor_pid <= 1
        or not isinstance(start_token, str)
        or not start_token
        or not isinstance(details, Mapping)
    ):
        return False
    verdict = _adjudicate_available_replication(config)
    return bool(
        verdict is not None
        and verdict.get("supported") is False
        and details.get("verdict") == verdict.get("verdict")
        and details.get("verdict_sha256") == verdict.get("verdict_sha256")
    )


def _active_stage_path(config: Mapping[str, Any]) -> Path:
    return Path(str(config["pipeline_root"])) / "active-stage.json"


def _read_active_stage(config: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _active_stage_path(config)
    if not path.exists():
        return None
    value = _read_canonical(path)
    body = {key: item for key, item in value.items() if key != "hmac_sha256"}
    members = value.get("group_members")
    if (
        value.get("schema") != ACTIVE_STAGE_SCHEMA
        or value.get("config_sha256") != config.get("config_sha256")
        or value.get("stage") not in _STAGE_TIMEOUTS
        or type(value.get("controller_pid")) is not int
        or int(value["controller_pid"]) <= 1
        or not isinstance(value.get("controller_start_token"), str)
        or not value["controller_start_token"]
        or type(value.get("child_pid")) is not int
        or int(value["child_pid"]) <= 1
        or not isinstance(value.get("child_start_token"), str)
        or not value["child_start_token"]
        or value.get("process_group_id") != value.get("child_pid")
        or not isinstance(value.get("hmac_sha256"), str)
        or not isinstance(members, list)
        or not members
        or any(
            not isinstance(member, Mapping)
            or set(member) != {"pid", "start_token"}
            or type(member.get("pid")) is not int
            or member["pid"] <= 1
            or not isinstance(member.get("start_token"), str)
            or not member["start_token"]
            for member in members
        )
        or [member["pid"] for member in members]
        != sorted({member["pid"] for member in members})
        or not any(
            member["pid"] == value.get("child_pid")
            and member["start_token"] == value.get("child_start_token")
            for member in members
        )
        or not hmac.compare_digest(value["hmac_sha256"], _signature(body, _key(config)))
    ):
        _fail("promotion active-stage authentication failed")
    return value


def _process_group_members(process_group_id: int) -> list[dict[str, Any]]:
    """Snapshot exact identities for every member of one Unix process group."""

    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,pgid="],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if result.returncode != 0:
        _fail("promotion process-group inventory is unavailable")
    members: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, pgid = (int(field) for field in fields)
        except ValueError:
            continue
        if pgid != process_group_id or pid <= 1:
            continue
        token = replication.launcher.detached._process_start_token(pid)  # noqa: SLF001
        if not token:
            try:
                if os.getpgid(pid) != process_group_id:
                    continue
            except ProcessLookupError:
                continue
            _fail("promotion process-group member identity is unavailable")
        members.append({"pid": pid, "start_token": token})
    return sorted(members, key=lambda member: member["pid"])


def _publish_active_stage(
    config: Mapping[str, Any],
    *,
    controller_pid: int,
    controller_start_token: str,
    stage: str,
    child_pid: int,
    child_start_token: str,
    result_path: Path,
    log_path: Path,
    timeout_s: float,
) -> dict[str, Any]:
    group_members = _process_group_members(child_pid)
    if not any(
        member["pid"] == child_pid and member["start_token"] == child_start_token
        for member in group_members
    ):
        _fail("promotion stage leader is absent from its process group")
    body = {
        "schema": ACTIVE_STAGE_SCHEMA,
        "config_sha256": config["config_sha256"],
        "stage": stage,
        "controller_pid": controller_pid,
        "controller_start_token": controller_start_token,
        "child_pid": child_pid,
        "child_start_token": child_start_token,
        "process_group_id": child_pid,
        "group_members": group_members,
        "result_path": str(result_path),
        "log_path": str(log_path),
        "timeout_s": timeout_s,
        "started_at_unix": time.time(),
    }
    result = {**body, "hmac_sha256": _signature(body, _key(config))}
    atomic_write_bytes(
        _active_stage_path(config),
        canonical_bytes(result) + b"\n",
        mode=0o600,
    )
    return result


def _refresh_active_stage_members(
    config: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_start_token: str,
) -> dict[str, Any]:
    observed = _read_active_stage(config)
    if observed is None:
        _fail("promotion active-stage receipt disappeared")
    if (
        observed.get("child_pid") != expected_pid
        or observed.get("child_start_token") != expected_start_token
    ):
        _fail("promotion active-stage identity changed during refresh")
    current = _process_group_members(int(observed["process_group_id"]))
    known = {member["pid"]: member["start_token"] for member in observed["group_members"]}
    for member in current:
        previous = known.get(member["pid"])
        if previous is not None and previous != member["start_token"]:
            _fail("promotion process-group member identity was reused")
        known[member["pid"]] = member["start_token"]
    body = {
        key: value
        for key, value in observed.items()
        if key != "hmac_sha256"
    }
    body["group_members"] = [
        {"pid": pid, "start_token": token}
        for pid, token in sorted(known.items())
    ]
    refreshed = {**body, "hmac_sha256": _signature(body, _key(config))}
    atomic_write_bytes(
        _active_stage_path(config),
        canonical_bytes(refreshed) + b"\n",
        mode=0o600,
    )
    return refreshed


def _clear_active_stage(
    config: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_start_token: str,
) -> None:
    observed = _read_active_stage(config)
    if observed is None:
        return
    if (
        observed.get("child_pid") != expected_pid
        or observed.get("child_start_token") != expected_start_token
    ):
        _fail("promotion active-stage identity changed before cleanup")
    _active_stage_path(config).unlink()


def _terminate_exact_stage_process(
    process: subprocess.Popen[bytes] | None,
    *,
    pid: int,
    start_token: str,
    group_members: Sequence[Mapping[str, Any]] | None = None,
    grace_s: float = 10.0,
) -> None:
    known = {
        int(member["pid"]): str(member["start_token"])
        for member in (group_members or ({"pid": pid, "start_token": start_token},))
    }
    if known.get(pid) != start_token:
        _fail("promotion stage leader is absent from authenticated group membership")

    def current_exact_members() -> list[dict[str, Any]]:
        current = _process_group_members(pid)
        for member in current:
            if known.get(member["pid"]) != member["start_token"]:
                _fail("promotion process group contains an unauthenticated member")
        return current

    current = current_exact_members()
    if not current:
        if process is not None:
            process.wait(timeout=grace_s)
        return
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not current_exact_members():
            if process is not None:
                process.wait(timeout=grace_s)
            return
        time.sleep(0.1)
    if current_exact_members():
        os.killpg(pid, signal.SIGKILL)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not current_exact_members():
            if process is not None:
                process.wait(timeout=grace_s)
            return
        time.sleep(0.1)
    _fail("promotion process group remained alive after forced termination")


def _retire_interrupted_stage(config: Mapping[str, Any]) -> None:
    observed = _read_active_stage(config)
    if observed is None:
        return
    owner_pid = observed.get("controller_pid")
    owner_token = str(observed.get("controller_start_token") or "")
    current_pid = os.getpid()
    current_token = replication.launcher.detached._process_start_token(  # noqa: SLF001
        current_pid
    )
    if (owner_pid, owner_token) != (current_pid, current_token):
        owner_state = replication.launcher.detached._identity_state(  # noqa: SLF001
            int(owner_pid),
            owner_token,
        )
        if owner_state != "dead":
            _fail(
                "promotion active stage owner is not proven dead: "
                f"{owner_state}"
            )
    _terminate_exact_stage_process(
        None,
        pid=int(observed["child_pid"]),
        start_token=str(observed["child_start_token"]),
        group_members=observed["group_members"],
    )
    _clear_active_stage(
        config,
        expected_pid=int(observed["child_pid"]),
        expected_start_token=str(observed["child_start_token"]),
    )


def _stage_result_path(config: Mapping[str, Any], stage: str) -> tuple[Path, Path]:
    root = Path(str(config["pipeline_root"])) / "stage-attempts"
    ensure_private_directory(root)
    attempt = f"{stage}-{time.time_ns()}"
    return root / f"{attempt}.json", root / f"{attempt}.log"


def _runtime_python(config: Mapping[str, Any]) -> Path:
    campaign_config = resident._load_config(  # noqa: SLF001
        Path(str(config["campaign"])) / "campaign.json"
    )
    python, interpreter = replication._runtime_python(campaign_config)  # noqa: SLF001
    # Preserve the attested virtualenv entrypoint for execution. Resolving its
    # symlink selects the base Homebrew binary, bypasses pyvenv.cfg discovery,
    # and drops venv-only dependencies such as MLX. The resolved binary remains
    # part of the interpreter attestation; it is not the executable contract.
    if str(python) != interpreter.get("executable"):
        _fail("promotion runtime interpreter entrypoint differs from attestation")
    return python


def _stage_cleanup_members(
    config: Mapping[str, Any],
    active: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_start_token: str,
) -> list[dict[str, Any]]:
    """Recover exact stage membership even when receipt reopening fails."""

    try:
        refreshed = _refresh_active_stage_members(
            config,
            expected_pid=expected_pid,
            expected_start_token=expected_start_token,
        )
        return list(refreshed["group_members"])
    except UnifiedRecurrentPromotionError:
        known = {
            int(member["pid"]): str(member["start_token"])
            for member in active["group_members"]
        }
        for member in _process_group_members(expected_pid):
            previous = known.get(member["pid"])
            if previous is not None and previous != member["start_token"]:
                _fail("promotion cleanup observed process identity reuse")
            known[member["pid"]] = member["start_token"]
        return [
            {"pid": pid, "start_token": token}
            for pid, token in sorted(known.items())
        ]


def _run_bounded_stage(
    config_path: Path,
    config: Mapping[str, Any],
    stage: str,
    *,
    sleep_inhibitor_pid: int,
    remaining_controller_s: float,
) -> dict[str, Any]:
    if stage not in _STAGE_TIMEOUTS:
        _fail("promotion stage name is invalid")
    configured_timeouts = config.get("stage_timeouts")
    if not isinstance(configured_timeouts, Mapping):
        _fail("promotion stage timeout contract is unavailable")
    try:
        stage_timeout = float(configured_timeouts[stage])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise UnifiedRecurrentPromotionError(
            "promotion stage timeout contract is invalid"
        ) from exc
    timeout_s = min(stage_timeout, float(remaining_controller_s))
    if not (0.0 < timeout_s <= _STAGE_TIMEOUTS[stage]):
        _fail("promotion stage has no remaining bounded runtime")

    _retire_interrupted_stage(config)
    result_path, log_path = _stage_result_path(config, stage)
    controller_pid = os.getpid()
    controller_start_token = replication.launcher.detached._process_start_token(  # noqa: SLF001
        controller_pid
    )
    if (
        not controller_start_token
        or replication.launcher.detached._identity_state(  # noqa: SLF001
            controller_pid,
            controller_start_token,
        )
        != "alive"
    ):
        _fail("promotion controller identity is not proven alive before stage launch")
    command = [
        str(_runtime_python(config)),
        str(Path(__file__).resolve(strict=True)),
        "run-stage",
        str(config_path),
        stage,
        "--result-output",
        str(result_path),
        "--controller-pid",
        str(controller_pid),
        "--controller-start-token",
        controller_start_token,
    ]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    process: subprocess.Popen[bytes] | None = None
    child_token = ""
    active: dict[str, Any] | None = None
    started = time.monotonic()
    with log_path.open("ab", buffering=0) as log:
        try:
            process = subprocess.Popen(  # noqa: S603 - attested interpreter and capsule script
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = time.monotonic() + min(5.0, timeout_s)
            while time.monotonic() < deadline:
                child_token = replication.launcher.detached._process_start_token(  # noqa: SLF001
                    process.pid
                )
                if child_token:
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.05)
            if not child_token or process.poll() is not None:
                _fail(f"promotion {stage} stage exited before supervision")
            active = _publish_active_stage(
                config,
                controller_pid=controller_pid,
                controller_start_token=controller_start_token,
                stage=stage,
                child_pid=process.pid,
                child_start_token=child_token,
                result_path=result_path,
                log_path=log_path,
                timeout_s=timeout_s,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                active = _refresh_active_stage_members(
                    config,
                    expected_pid=process.pid,
                    expected_start_token=child_token,
                )
                if elapsed >= timeout_s:
                    _terminate_exact_stage_process(
                        process,
                        pid=process.pid,
                        start_token=child_token,
                        group_members=active["group_members"],
                    )
                    _fail(f"promotion {stage} stage exceeded {timeout_s:.1f}s")
                _publish_status(
                    config,
                    f"{stage}_running",
                    {
                        "stage": stage,
                        "child_pid": process.pid,
                        "child_start_token": child_token,
                        "active_stage_hmac_sha256": active["hmac_sha256"],
                        "elapsed_s": round(elapsed, 3),
                        "timeout_s": timeout_s,
                    },
                    sleep_inhibitor_pid=sleep_inhibitor_pid,
                )
                time.sleep(min(5.0, max(0.1, timeout_s - elapsed)))
            if process.returncode != 0:
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                except OSError:
                    tail = "stage log unavailable"
                _fail(
                    f"promotion {stage} stage failed with exit {process.returncode}: "
                    f"{tail.strip()}"
                )
            result = _read_canonical(result_path)
            return result
        finally:
            if process is not None and child_token:
                if active is not None:
                    members = _stage_cleanup_members(
                        config,
                        active,
                        expected_pid=process.pid,
                        expected_start_token=child_token,
                    )
                else:
                    members = _process_group_members(process.pid)
                    if not any(member["pid"] == process.pid for member in members):
                        members = [
                            *members,
                            {"pid": process.pid, "start_token": child_token},
                        ]
                _terminate_exact_stage_process(
                    process,
                    pid=process.pid,
                    start_token=child_token,
                    group_members=members,
                )
                if _process_group_members(process.pid):
                    _fail("promotion stage process group was not empty at cleanup")
                if active is not None:
                    _clear_active_stage(
                        config,
                        expected_pid=process.pid,
                        expected_start_token=child_token,
                    )


def _monitor_controller(controller_pid: int, controller_start_token: str) -> None:
    while True:
        time.sleep(1.0)
        owner_state = replication.launcher.detached._identity_state(  # noqa: SLF001
            controller_pid,
            controller_start_token,
        )
        if owner_state == "dead":
            pid = os.getpid()
            process_group = os.getpgrp()
            if process_group != pid:
                os.kill(pid, signal.SIGTERM)
                return
            os.killpg(process_group, signal.SIGTERM)
            return


def run_stage(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(arguments.config)
    if arguments.stage not in _STAGE_TIMEOUTS:
        _fail("promotion stage name is invalid")
    if arguments.controller_pid <= 1 or not arguments.controller_start_token:
        _fail("promotion stage controller identity is invalid")
    if os.getpid() != os.getpgrp():
        _fail("promotion stage requires its own process group")
    if (
        replication.launcher.detached._identity_state(  # noqa: SLF001
            arguments.controller_pid,
            arguments.controller_start_token,
        )
        != "alive"
    ):
        _fail("promotion stage controller is not alive")
    result_path = _pipeline_root(
        Path(str(config["campaign"])),
        arguments.result_output,
    )
    if not result_path.is_relative_to(Path(str(config["pipeline_root"]))):
        _fail("promotion stage result must stay inside the pipeline root")
    monitor = threading.Thread(
        target=_monitor_controller,
        args=(arguments.controller_pid, arguments.controller_start_token),
        name=f"promotion-{arguments.stage}-parent-monitor",
        daemon=True,
    )
    monitor.start()
    if arguments.stage == "materialize":
        verdict_path = Path(str(config["replication_root"])) / "replication-verdict.json"
        result = _materialize_or_reopen(config, _read_canonical(verdict_path))
    elif arguments.stage == "lifecycle":
        result = _lifecycle_or_reopen(config)
    else:
        result = _activate_or_reopen(config)
    atomic_write_bytes(result_path, canonical_bytes(result) + b"\n", mode=0o600)
    return {
        "stage": arguments.stage,
        "result_path": str(result_path),
        "result_sha256": canonical_sha256(result),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = arguments.config.expanduser().absolute()
    config = _load_config(config_path)
    inhibitor = _start_sleep_inhibitor()
    deadline = time.monotonic() + float(arguments.controller_timeout)
    try:
        supervision = _verify_supervision(config, inhibitor.pid)
        while time.monotonic() < deadline:
            observed = replication.status(_replication_arguments(config))
            controller = observed.get("controller") or {}
            state = str(controller.get("state") or observed.get("state") or "unknown")
            evaluations = observed.get("evaluations")
            evaluation_rows = evaluations if isinstance(evaluations, list) else []
            has_completed_evidence = observed.get("complete") is True or any(
                isinstance(row, Mapping) and row.get("state") == "completed"
                for row in evaluation_rows
            )
            verdict = (
                _adjudicate_available_replication(config)
                if state in {"completed", "refuted"} or has_completed_evidence
                else None
            )
            if verdict is None:
                if state in {"failed", "not_admitted"}:
                    _fail(f"replication terminated before adjudication: {state}")
                failed_evaluations = [
                    row
                    for row in evaluation_rows
                    if isinstance(row, Mapping)
                    and row.get("state") in {"failed", "stopped"}
                ]
                if failed_evaluations:
                    _fail("replication evaluator terminated before adjudication")
                _publish_status(
                    config,
                    "waiting_for_replication",
                    {"replication_state": state, "supervision": supervision},
                    sleep_inhibitor_pid=inhibitor.pid,
                )
                time.sleep(float(arguments.poll_interval))
                continue

            if verdict.get("plan_sha256") != config.get("replication_plan_sha256"):
                _fail("promotion replication plan identity differs")
            if verdict.get("supported") is not True:
                status = _publish_status(
                    config,
                    "refuted",
                    {
                        "verdict": verdict.get("verdict"),
                        "verdict_sha256": verdict.get("verdict_sha256"),
                    },
                    sleep_inhibitor_pid=inhibitor.pid,
                )
                return {"state": "refuted", "supported": False, "status": status}

            _publish_status(
                config,
                "materializing",
                {"verdict_sha256": verdict["verdict_sha256"]},
                sleep_inhibitor_pid=inhibitor.pid,
            )
            package = _run_bounded_stage(
                config_path,
                config,
                "materialize",
                sleep_inhibitor_pid=inhibitor.pid,
                remaining_controller_s=max(0.0, deadline - time.monotonic()),
            )
            manifest = package.get("manifest") or {}
            _publish_status(
                config,
                "proving_lifecycle",
                {"manifest_sha256": manifest.get("manifest_sha256")},
                sleep_inhibitor_pid=inhibitor.pid,
            )
            lifecycle_result = _run_bounded_stage(
                config_path,
                config,
                "lifecycle",
                sleep_inhibitor_pid=inhibitor.pid,
                remaining_controller_s=max(0.0, deadline - time.monotonic()),
            )
            _publish_status(
                config,
                "activating_verified",
                {"lifecycle_result_sha256": lifecycle_result["result_sha256"]},
                sleep_inhibitor_pid=inhibitor.pid,
            )
            activated = _run_bounded_stage(
                config_path,
                config,
                "activate",
                sleep_inhibitor_pid=inhibitor.pid,
                remaining_controller_s=max(0.0, deadline - time.monotonic()),
            )
            completion = _write_completion(
                config, verdict, package, lifecycle_result, activated
            )
            status = _publish_status(
                config,
                "completed",
                {
                    "completion_sha256": completion["completion_sha256"],
                    "activation_sha256": completion["activation_sha256"],
                },
                sleep_inhibitor_pid=inhibitor.pid,
            )
            return {
                "state": "completed",
                "supported": True,
                "completion": completion,
                "status": status,
            }
        _fail("promotion controller timed out")
    except BaseException as exc:
        _publish_status(
            config,
            "failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
            sleep_inhibitor_pid=inhibitor.pid,
        )
        raise
    finally:
        inhibitor.terminate()
        try:
            inhibitor.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            inhibitor.kill()


def _label(config: Mapping[str, Any]) -> str:
    digest = str(config.get("config_sha256") or "")
    if len(digest) != 64:
        _fail("promotion launch label identity is invalid")
    return f"com.aura.unified-recurrent-promotion.{digest[:16]}"


def _launch_contract(
    config_path: Path,
    config: Mapping[str, Any],
    arguments: argparse.Namespace,
) -> tuple[Path, bytes, dict[str, Any]]:
    campaign_config = resident._load_config(  # noqa: SLF001
        Path(str(config["campaign"])) / "campaign.json"
    )
    python, interpreter = replication._runtime_python(campaign_config)  # noqa: SLF001
    script = Path(__file__).resolve(strict=True)
    command = [
        str(python),
        str(script),
        "run",
        str(config_path.expanduser().absolute()),
        "--poll-interval",
        str(float(arguments.poll_interval)),
        "--controller-timeout",
        str(float(arguments.controller_timeout)),
        "--launchd-supervised",
    ]
    label = _label(config)
    pipeline_root = Path(str(config["pipeline_root"]))
    payload = {
        "Label": label,
        "ProgramArguments": command,
        "WorkingDirectory": str(pipeline_root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 30,
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "VIRTUAL_ENV": interpreter["sys_prefix"],
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "StandardOutPath": str(pipeline_root / "controller-launchd.log"),
        "StandardErrorPath": str(pipeline_root / "controller-launchd.log"),
    }
    plist = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    plist_path = LAUNCH_AGENTS_ROOT / f"{label}.plist"
    body = {
        "schema": INTENT_SCHEMA,
        "config_sha256": config["config_sha256"],
        "label": label,
        "plist_path": str(plist_path),
        "plist_sha256": hashlib.sha256(plist).hexdigest(),
        "program_arguments": command,
        "controller_source_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "interpreter": interpreter,
    }
    return plist_path, plist, {**body, "intent_sha256": canonical_sha256(body)}


def install_launchd(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = arguments.config.expanduser().absolute()
    config = _load_config(config_path)
    root = Path(str(config["pipeline_root"]))
    plist_path, plist, intent = _launch_contract(config_path, config, arguments)
    intent_path = root / "launch-intent.json"
    payload = canonical_bytes(intent) + b"\n"
    if not atomic_write_bytes_if_absent(intent_path, payload, mode=0o400):
        if _read_canonical(intent_path) != intent:
            _fail("promotion launch intent already differs")
    ensure_private_directory(LAUNCH_AGENTS_ROOT)
    atomic_write_bytes(plist_path, plist, mode=0o600)
    domain = f"gui/{os.getuid()}"
    try:
        existing_job = _launchd_job(_label(config))
    except UnifiedRecurrentPromotionError:
        existing_job = None
    existing_status = _read_status(config)
    existing_state = _installed_lineage_state(existing_status, existing_job)
    bootstrapped_here = False
    terminal_refutation = _terminal_refutation_is_valid(config, existing_status)
    if existing_state == "valid":
        status = existing_status
        job = existing_job
        validated = True
    elif existing_job is None and terminal_refutation:
        status = existing_status
        job = {
            "target": f"{domain}/{_label(config)}",
            "pid": status["controller_pid"],
        }
        validated = True
    else:
        if existing_job is not None:
            if existing_state != "dead":
                _fail(
                    "promotion existing launchd controller is not proven stale: "
                    f"{existing_state}"
                )
            subprocess.run(
                ["/bin/launchctl", "bootout", f"{domain}/{_label(config)}"],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        started = subprocess.run(
            ["/bin/launchctl", "bootstrap", domain, str(plist_path)],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        if started.returncode != 0:
            _fail(f"promotion launchd bootstrap failed: {started.stderr.strip()[:300]}")
        bootstrapped_here = True
        status = None
        job = None
        validated = False
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if validated:
            break
        try:
            job = _launchd_job(_label(config))
            status = _read_status(config)
            if _installed_lineage_is_valid(status, job):
                validated = True
                break
            if (
                isinstance(status, Mapping)
                and job.get("pid") == status.get("controller_pid")
                and _terminal_refutation_is_valid(config, status)
            ):
                terminal_refutation = True
                validated = True
                break
        except UnifiedRecurrentPromotionError:
            try:
                status = _read_status(config)
                if _terminal_refutation_is_valid(config, status):
                    terminal_refutation = True
                    job = {
                        "target": f"{domain}/{_label(config)}",
                        "pid": status["controller_pid"],
                    }
                    validated = True
                    break
            except UnifiedRecurrentPromotionError:
                status = None
        time.sleep(0.25)
    if not validated or status is None or job is None:
        if bootstrapped_here:
            subprocess.run(
                ["/bin/launchctl", "bootout", f"{domain}/{_label(config)}"],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        _fail("promotion launchd authenticated start timed out")
    body = {
        "schema": LAUNCH_SCHEMA,
        "config_sha256": config["config_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "target": job["target"],
        "pid": status["controller_pid"],
        "start_token": status["controller_start_token"],
        "sleep_inhibitor_pid": status["sleep_inhibitor_pid"],
        "terminal_refutation": terminal_refutation,
        "initial_status_sha256": canonical_sha256(status),
        "installed_at_unix_ns": time.time_ns(),
    }
    receipt = {**body, "launch_sha256": canonical_sha256(body)}
    atomic_write_bytes(root / "launchd-receipt.json", canonical_bytes(receipt) + b"\n")
    return receipt


def status(arguments: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(arguments.config)
    observed = _read_status(config)
    if observed is None:
        return {"state": "not_started", "config_sha256": config["config_sha256"]}
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("campaign", type=Path)
    prepare_parser.add_argument("--replication-root", type=Path)
    prepare_parser.add_argument("--output", type=Path)
    prepare_parser.add_argument("--package-id", required=True)
    published = subparsers.add_parser("install-published")
    published.add_argument("campaign", type=Path)
    published.add_argument("--replication-root", type=Path)
    published.add_argument("--output", type=Path)
    published.add_argument("--package-id", required=True)
    published.add_argument("--poll-interval", type=float, default=15.0)
    published.add_argument("--controller-timeout", type=float, default=16 * 60 * 60)
    stage = subparsers.add_parser("run-stage")
    stage.add_argument("config", type=Path)
    stage.add_argument("stage", choices=tuple(_STAGE_TIMEOUTS))
    stage.add_argument("--result-output", type=Path, required=True)
    stage.add_argument("--controller-pid", type=int, required=True)
    stage.add_argument("--controller-start-token", required=True)
    for name in ("run", "status", "install-launchd"):
        child = subparsers.add_parser(name)
        child.add_argument("config", type=Path)
        if name != "status":
            child.add_argument("--poll-interval", type=float, default=15.0)
            child.add_argument("--controller-timeout", type=float, default=16 * 60 * 60)
            if name == "run":
                child.add_argument("--launchd-supervised", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.action != "status" and hasattr(arguments, "poll_interval") and (
        arguments.poll_interval <= 0.0
        or arguments.controller_timeout <= arguments.poll_interval
    ):
        parser.error("promotion numeric contract is invalid")
    try:
        result = {
            "prepare": prepare,
            "install-published": install_published,
            "run-stage": run_stage,
            "run": run,
            "status": status,
            "install-launchd": install_launchd,
        }[arguments.action](arguments)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"unified recurrent promotion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
