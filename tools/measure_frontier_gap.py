#!/usr/bin/env python3
"""Run Aura's auditable candidate battery and update its evidence artifact.

Candidate scoring is always available. A gap-to-frontier and capability-ledger
entry require a real Aura model, complete verified execution, a matched named
reference artifact, and a clean exact source provenance record.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IMMUTABLE_CHILD_ENV = "AURA_FRONTIER_IMMUTABLE_CHILD"
_ORIGINAL_REPO_ENV = "AURA_FRONTIER_ORIGINAL_REPO"
_IMMUTABLE_CHILD_TIMEOUT_S = 1_800.0


def _absolute_cli_path(argv: list[str], option: str, *, base: Path) -> list[str]:
    result = list(argv)
    for index, value in enumerate(result[:-1]):
        if value == option and result[index + 1]:
            result[index + 1] = str((base / result[index + 1]).resolve())
    return result


async def _stdlib_exec_async(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE if capture_output else None,
        stderr=asyncio.subprocess.PIPE if capture_output else None,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise subprocess.TimeoutExpired(argv, timeout_s) from None
    return subprocess.CompletedProcess(
        argv,
        int(process.returncode or 0),
        stdout.decode("utf-8", errors="replace") if stdout is not None else "",
        stderr.decode("utf-8", errors="replace") if stderr is not None else "",
    )


def _stdlib_exec(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    return asyncio.run(
        _stdlib_exec_async(
            argv,
            cwd=cwd,
            env=env,
            capture_output=capture_output,
            timeout_s=timeout_s,
        )
    )


def _stdlib_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = _stdlib_exec(
        ["git", *args],
        cwd=root,
        timeout_s=60.0,
    )
    if check and completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "git failed").strip())
    return completed


def _make_tree_read_only(root: Path) -> None:
    root = root.resolve(strict=True)
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        relative_parts = path.relative_to(root).parts
        if ".git" in relative_parts:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            try:
                resolved = path.resolve(strict=True)
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(
                    f"source checkout contains a broken symlink: {path}"
                ) from exc
            if not resolved.is_relative_to(root) or ".git" in resolved.relative_to(root).parts:
                raise RuntimeError(f"source checkout contains an unsafe symlink: {path}")
            continue
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        if path.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"source checkout path remained writable: {path}")
    root_mode = root.lstat().st_mode
    root.chmod(root_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    if root.lstat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("source checkout root remained writable")


def _restore_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in (root, *root.rglob("*")):
        try:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                continue
            path.chmod(mode | stat.S_IWUSR)
        except OSError:
            continue


def _immutable_child_env(
    *,
    temporary: Path,
    original_root: Path,
    commit: str,
    tree: str,
    canonical_remote_sha256: str,
) -> dict[str, str]:
    """Build a hermetic child environment without ambient credentials or state."""

    runtime = temporary / "runtime"
    home = runtime / "home"
    tmp = runtime / "tmp"
    aura_root = runtime / "aura"
    test_root = runtime / "test"
    for directory in (home, tmp, aura_root, test_root):
        directory.mkdir(parents=True, mode=0o700, exist_ok=True)

    allowed_exact = {"LANG", "LC_ALL", "LC_CTYPE", "PATH", "TZ"}
    role_prefixes = (
        "AURA_FRONTIER_COMMON_",
        "AURA_FRONTIER_WORKER_",
        "AURA_FRONTIER_VERIFIER_",
        "AURA_FRONTIER_RUN_SIGNER_",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed_exact or key.startswith(role_prefixes)
    }
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(tmp),
            "AURA_ROOT": str(aura_root),
            "AURA_TEST_RUNTIME_ROOT": str(test_root),
            "AURA_LOG_SQLITE_ENABLED": "0",
            _IMMUTABLE_CHILD_ENV: "1",
            _ORIGINAL_REPO_ENV: str(original_root),
            "AURA_FRONTIER_BOOTSTRAP_COMMIT": commit,
            "AURA_FRONTIER_BOOTSTRAP_TREE": tree,
            "AURA_FRONTIER_CANONICAL_REMOTE_SHA256": canonical_remote_sha256,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _immutable_bootstrap() -> int:
    """Launch the measurement from a fresh verified checkout before imports."""

    original_script = Path(__file__).resolve()
    original_root = original_script.parent.parent
    status = _stdlib_git(
        original_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).stdout
    if status:
        print(
            "[frontier-gap] refusing measurement bootstrap from a dirty source tree",
            file=sys.stderr,
        )
        return 3
    commit = _stdlib_git(original_root, "rev-parse", "HEAD").stdout.strip().lower()
    tree = _stdlib_git(original_root, "rev-parse", "HEAD^{tree}").stdout.strip().lower()
    remote_url = _stdlib_git(
        original_root,
        "config",
        "--get",
        "remote.origin.url",
        check=False,
    ).stdout.strip()
    canonical_remote_sha256 = (
        hashlib.sha256(remote_url.encode("utf-8")).hexdigest() if remote_url else ""
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(
        r"[0-9a-f]{40}", tree
    ):
        print("[frontier-gap] exact source identity is unavailable", file=sys.stderr)
        return 3
    argv = list(sys.argv[1:])
    for option in (
        "--out",
        "--reference-artifact",
        "--release-attestation",
    ):
        argv = _absolute_cli_path(argv, option, base=original_root)
    temporary = Path(tempfile.mkdtemp(prefix="aura-frontier-v5-"))
    checkout = temporary / "checkout"
    try:
        clone = _stdlib_exec(
            [
                "git",
                "clone",
                "--quiet",
                "--no-hardlinks",
                "--no-checkout",
                str(original_root),
                str(checkout),
            ],
            cwd=original_root,
            timeout_s=180.0,
        )
        if clone.returncode != 0:
            raise RuntimeError((clone.stderr or clone.stdout or "git clone failed").strip())
        _stdlib_git(checkout, "checkout", "--quiet", "--detach", commit)
        cloned_tree = _stdlib_git(checkout, "rev-parse", "HEAD^{tree}").stdout.strip().lower()
        if cloned_tree != tree or _stdlib_git(
            checkout,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).stdout:
            raise RuntimeError("fresh checkout does not reproduce the verified source tree")
        _make_tree_read_only(checkout)
        env = _immutable_child_env(
            temporary=temporary,
            original_root=original_root,
            commit=commit,
            tree=tree,
            canonical_remote_sha256=canonical_remote_sha256,
        )
        completed = _stdlib_exec(
            [sys.executable, str(checkout / "tools" / original_script.name), *argv],
            cwd=checkout,
            env=env,
            capture_output=False,
            timeout_s=_IMMUTABLE_CHILD_TIMEOUT_S,
        )
        return int(completed.returncode)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(
            f"[frontier-gap] immutable bootstrap failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 3
    finally:
        _restore_tree_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__" and os.environ.get(_IMMUTABLE_CHILD_ENV) != "1":
    raise SystemExit(_immutable_bootstrap())

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.frontier_evidence_v5 import (  # noqa: E402
    CORRECTNESS_RECEIPT_SCHEMA,
    EFFECTIVE_RUNTIME_MANIFEST_SCHEMA,
    PROTOCOL_MANIFEST,
    PROTOCOL_MANIFEST_SHA256,
    RUN_ENVELOPE_SCHEMA,
    SOURCE_IDENTITY_SCHEMA,
    SUPERVISOR_OBSERVATION_SCHEMA,
    WORKER_RECEIPT_SCHEMA,
    build_trust_basis,
    expected_request_id,
    require_sha256,
    validate_effective_runtime_manifest,
    validate_source_identity,
)
from core.brain.frontier_gap import (  # noqa: E402
    CAPABILITY_EVIDENCE_CLASS,
    CONTROL_EVIDENCE_CLASS,
    DISQUALIFYING_FALLBACK_MARKERS,
    MATCHED_BUDGET,
    MODEL_MANIFEST_SCHEMA,
    MODEL_STABILITY_SCHEMA,
    REJECTED_EVIDENCE_CLASS,
    SOURCE_PROVENANCE_SCHEMA,
    SOURCE_STABILITY_SCHEMA,
    TRUSTED_REFERENCE_BASIS,
    GapLedger,
    ReferenceEvidence,
    SolverObservation,
    build_battery,
    canonical_json_bytes,
    identity_freeze_sha256,
    run_battery,
    sha256_json,
    validate_capability_report,
    validate_reference_artifact,
)
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

Generate = Callable[[str, float], Awaitable[str]]


@dataclass
class ExecutionStats:
    attempted: int = 0
    completed: int = 0
    failed: int = 0
    invalid: int = 0
    empty: int = 0
    unverified: int = 0
    fallback_items: int = 0
    disqualifying_fallbacks: int = 0
    errors: list[str] = field(default_factory=list)

    def _error(self, message: str) -> None:
        if len(self.errors) < 12:
            self.errors.append(message[:300])

    def record_failure(self, exc: BaseException) -> None:
        self.failed += 1
        self._error(f"{type(exc).__name__}: {exc}")

    def record_result(self, observation: SolverObservation, *, task_type: str) -> None:
        self.completed += 1
        answer = observation.answer.strip()
        receipt = dict(observation.receipt or {})
        if not answer:
            self.empty += 1
            self._error(f"empty answer for {task_type}")
        if observation.verified is not True:
            self.unverified += 1
            self._error(f"unverified amplifier result for {task_type}")
        signed_worker = receipt.get("schema") == WORKER_RECEIPT_SCHEMA
        if not receipt or (
            not signed_worker and receipt.get("task_type") != task_type
        ):
            self.invalid += 1
            self._error(f"missing or mismatched reasoning receipt for {task_type}")
        fallbacks = tuple(str(item) for item in observation.fallbacks_used if str(item))
        if fallbacks:
            self.fallback_items += 1
        if any(
            any(marker in fallback for marker in DISQUALIFYING_FALLBACK_MARKERS)
            for fallback in fallbacks
        ):
            self.disqualifying_fallbacks += 1
            self._error(f"disqualifying fallback for {task_type}: {','.join(fallbacks)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "completed": self.completed,
            "failed": self.failed,
            "invalid": self.invalid,
            "empty": self.empty,
            "unverified": self.unverified,
            "fallback_items": self.fallback_items,
            "disqualifying_fallbacks": self.disqualifying_fallbacks,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class SolverSelection:
    generate: Generate
    mode: str
    capability_candidate: bool
    subject: str
    reason: str
    execution: ExecutionStats = field(default_factory=ExecutionStats)
    model_manifest_before: dict[str, Any] | None = None
    model_path_supplier: Callable[[], str] | None = None
    effective_runtime_manifest: dict[str, Any] | None = None
    worker_client: SignedWorkerClient | None = None
    model_stability_override: dict[str, Any] | None = None


def _git(*args: str) -> str:
    completed = get_subprocess_gateway().run(
        ["git", *args],
        capture_output=True,
        read_only=True,
        source="proof_tooling:frontier_gap_git",
        timeout=30,
        cwd=REPO_ROOT,
        accelerator_capability="none",
    )
    if completed.returncode != 0:
        raise RuntimeError(str(completed.stderr or completed.stdout or "git failed").strip())
    return str(completed.stdout or "")


def _resolve_commit_tree(commit_sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit_sha or "")):
        raise ValueError("commit SHA is invalid")
    tree_sha = _git("rev-parse", "--verify", f"{commit_sha}^{{tree}}").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise ValueError("resolved tree SHA is invalid")
    return tree_sha


def _resolve_commit_component_sha256(commit_sha: str, relative_path: str) -> str:
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(commit_sha or ""))
        or not relative_path
        or relative_path.startswith("/")
        or ".." in relative_path.split("/")
    ):
        raise ValueError("commit component identity is invalid")
    content = _git("show", f"{commit_sha}:{relative_path}")
    return hashlib.sha256(content.encode("utf-8", errors="surrogateescape")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_regular_file_beneath(base: Path, relative_path: str) -> tuple[str, int]:
    """Hash one regular file through no-follow descriptor traversal."""

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("model artifact relative path is unsafe")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    file_flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        file_flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        file_flags |= os.O_NOFOLLOW

    descriptors: list[int] = []
    try:
        current = os.open(str(base), directory_flags)
        descriptors.append(current)
        for component in relative.parts[:-1]:
            current = os.open(component, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_descriptor = os.open(relative.parts[-1], file_flags, dir_fd=current)
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"model artifact entry is not a regular file: {relative_path}")
        digest = hashlib.sha256()
        observed_size = 0
        remaining = before.st_size
        while remaining > 0:
            chunk = os.read(file_descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"model artifact truncated while hashing: {relative_path}")
            digest.update(chunk)
            observed_size += len(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_descriptor)
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if not stable_identity or observed_size != after.st_size:
            raise RuntimeError(f"model artifact changed while hashing: {relative_path}")
        return digest.hexdigest(), int(after.st_size)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


_MODEL_WEIGHT_SUFFIXES = {".bin", ".gguf", ".npz", ".pt", ".pth", ".safetensors"}


def collect_model_manifest(path_text: str) -> dict[str, Any]:
    supplied = Path(path_text).expanduser()
    if supplied.is_symlink():
        raise ValueError("model artifact root cannot be a symlink")
    root = supplied.resolve(strict=True)
    if root.is_file():
        candidates = [root]
    else:
        discovered = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        symlinks = [path for path in discovered if path.is_symlink()]
        if symlinks:
            raise ValueError(f"model artifact contains a symlink: {symlinks[0]}")
        candidates = sorted(
            (path for path in discovered if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    if not candidates:
        raise ValueError("model artifact contains no files")
    entries: list[dict[str, Any]] = []
    roles: dict[str, list[str]] = {
        "weights": [],
        "configuration": [],
        "tokenizer": [],
        "adapters": [],
    }
    base = root.parent if root.is_file() else root
    for file_path in candidates:
        if file_path.is_symlink():
            raise ValueError(f"model artifact contains a symlink: {file_path}")
        relative = file_path.relative_to(base).as_posix()
        digest, size = _hash_regular_file_beneath(base, relative)
        entry = {
            "path": relative,
            "size": size,
            "sha256": digest,
        }
        entries.append(entry)
        lowered = relative.lower()
        suffix = file_path.suffix.lower()
        if suffix in _MODEL_WEIGHT_SUFFIXES:
            roles["weights"].append(relative)
        if file_path.name.lower() in {
            "config.json",
            "generation_config.json",
            "model_config.json",
        }:
            roles["configuration"].append(relative)
        if any(marker in lowered for marker in ("tokenizer", "vocab", "merges.txt")):
            roles["tokenizer"].append(relative)
        if any(marker in lowered for marker in ("adapter", "lora")):
            roles["adapters"].append(relative)
    for required_role in ("weights", "configuration", "tokenizer"):
        if not roles[required_role]:
            raise ValueError(f"model artifact is missing {required_role} material")
    body = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "model_path": str(root),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
        "files": entries,
        "roles": roles,
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def _digest_role_files(model_manifest: dict[str, Any], role: str) -> str:
    by_path = {entry["path"]: entry["sha256"] for entry in model_manifest.get("files", [])}
    material = [
        {"path": path, "sha256": by_path[path]}
        for path in model_manifest.get("roles", {}).get(role, [])
        if path in by_path
    ]
    return sha256_json(material)


def collect_effective_runtime_manifest(
    *,
    subject_id: str,
    model_manifest: dict[str, Any],
    sealed_evaluation_enforced: bool,
    fresh_process: bool,
    immutable_source: bool,
    modifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    libraries: dict[str, str] = {"python": platform.python_version()}
    for distribution in ("mlx", "mlx-lm", "transformers", "tokenizers"):
        try:
            libraries[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            libraries[distribution] = "not-installed"
    adapters = [
        entry["sha256"]
        for entry in model_manifest.get("files", [])
        if entry["path"] in set(model_manifest.get("roles", {}).get("adapters", []))
    ]
    steering: list[str] = []
    for supplied in filter(None, os.environ.get("AURA_STEERING_VECTOR_PATHS", "").split(os.pathsep)):
        path = Path(supplied).expanduser()
        if path.is_file() and not path.is_symlink():
            steering.append(_hash_file(path))
    body = {
        "schema": EFFECTIVE_RUNTIME_MANIFEST_SCHEMA,
        "subject_id": subject_id,
        "base_model_manifest_sha256": model_manifest["manifest_sha256"],
        "tokenizer_sha256": _digest_role_files(model_manifest, "tokenizer"),
        "prompt_template_sha256": sha256_json(
            {
                "configuration": _digest_role_files(model_manifest, "configuration"),
                "tokenizer": _digest_role_files(model_manifest, "tokenizer"),
                "template_policy": "model_material_exact",
            }
        ),
        "execution_identity": {
            "worker_implementation_sha256": _hash_file(Path(__file__).resolve()),
            "python_executable_sha256": _hash_file(
                Path(sys.executable).resolve(strict=True)
            ),
            "library_lock_sha256": sha256_json(libraries),
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "inference_backend": "mlx",
        },
        "inference_libraries": libraries,
        "adapters_sha256": sorted(set(adapters)),
        "steering_sha256": sorted(set(steering)),
        "modifiers": copy.deepcopy(modifiers or {}),
        "cache_policy": {
            "prompt_cache": "disabled",
            "result_cache": "disabled",
            "playbook_cache": "disabled",
            "clear_before_run": True,
        },
        "generation_parameters": copy.deepcopy(MATCHED_BUDGET),
        "runtime_isolation": {
            "fresh_process": fresh_process,
            "immutable_source": immutable_source,
            "network_enabled": False,
            "tools_enabled": False,
            "sealed_evaluation_enforced": sealed_evaluation_enforced,
        },
    }
    return {**body, "manifest_sha256": sha256_json(body)}


def collect_source_identity(
    *,
    repository_id: str,
    release_ref: str,
    release_attestation: dict[str, Any] | None,
    trusted_release_keys: dict[str, str],
) -> dict[str, Any] | None:
    if not repository_id or not release_ref or not isinstance(release_attestation, dict):
        return None
    try:
        commit = _git("rev-parse", "HEAD").strip().lower()
        tree = _git("rev-parse", "HEAD^{tree}").strip().lower()
        release_commit = _git("rev-parse", "--verify", release_ref).strip().lower()
        ancestry = get_subprocess_gateway().run(
            ["git", "merge-base", "--is-ancestor", release_commit, commit],
            capture_output=True,
            read_only=True,
            source="proof_tooling:frontier_gap_release_ancestry",
            timeout=30,
            cwd=REPO_ROOT,
            accelerator_capability="none",
        )
        if ancestry.returncode != 0:
            return None
        remote_digest = os.environ.get("AURA_FRONTIER_CANONICAL_REMOTE_SHA256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", remote_digest):
            return None
        body = {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "repository_id": repository_id,
            "canonical_remote_sha256": remote_digest,
            "commit_sha": commit,
            "tree_sha": tree,
            "release_ref": release_ref,
            "release_commit_sha": release_commit,
            "release_attestation": copy.deepcopy(release_attestation),
            "release_attestation_sha256": sha256_json(release_attestation),
            "head_descends_from_release": True,
            "clean": not bool(
                _git("status", "--porcelain=v1", "--untracked-files=all")
            ),
            "immutable_checkout": os.environ.get(_IMMUTABLE_CHILD_ENV) == "1",
            "imports_after_verification": os.environ.get(_IMMUTABLE_CHILD_ENV) == "1",
        }
        identity = {**body, "identity_sha256": sha256_json(body)}
        return validate_source_identity(
            identity,
            trusted_release_keys=trusted_release_keys,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _model_stability_window(selection: SolverSelection) -> dict[str, Any]:
    before = selection.model_manifest_before
    supplier = selection.model_path_supplier
    if before is None or supplier is None:
        return {
            "schema": MODEL_STABILITY_SCHEMA,
            "before": before,
            "after": None,
            "stable": False,
            "window_sha256": "",
        }
    current_path = supplier()
    after = collect_model_manifest(current_path)
    window_body = {"before": before, "after": after}
    return {
        "schema": MODEL_STABILITY_SCHEMA,
        **window_body,
        "stable": canonical_json_bytes(before) == canonical_json_bytes(after),
        "window_sha256": sha256_json(window_body),
    }


def collect_model_stability_window(selection: SolverSelection) -> dict[str, Any]:
    if selection.model_stability_override is not None:
        return copy.deepcopy(selection.model_stability_override)
    try:
        return _model_stability_window(selection)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "schema": MODEL_STABILITY_SCHEMA,
            "before": selection.model_manifest_before,
            "after": None,
            "stable": False,
            "window_sha256": "",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
        }


def collect_source_provenance(
    source_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        commit_sha = _git("rev-parse", "HEAD").strip().lower()
        tree_sha = _git("rev-parse", "HEAD^{tree}").strip().lower()
        status_text = _git(
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        diff_text = _git("diff", "--binary", "--no-ext-diff", "HEAD", "--")
        index_diff_text = _git(
            "diff",
            "--cached",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "schema": SOURCE_PROVENANCE_SCHEMA,
            "commit_sha": "",
            "tree_sha": "",
            "clean": False,
            "workspace_diff_sha256": "",
            "index_diff_sha256": "",
            "untracked_content_sha256": "",
            "workspace_state_sha256": "",
            "issues": [f"git_identity_failed:{type(exc).__name__}"],
        }

    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        issues.append("invalid_commit_sha")
    if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        issues.append("invalid_tree_sha")
    status_bytes = status_text.encode("utf-8", errors="surrogateescape")
    diff_bytes = diff_text.encode("utf-8", errors="surrogateescape")
    index_diff_bytes = index_diff_text.encode("utf-8", errors="surrogateescape")
    untracked_digest = hashlib.sha256()
    for record in status_text.split("\0"):
        if not record.startswith("?? "):
            continue
        relative = record[3:]
        path = REPO_ROOT / relative
        untracked_digest.update(relative.encode("utf-8", errors="surrogateescape"))
        try:
            untracked_digest.update(_hash_file(path).encode())
        except OSError as exc:
            issues.append(f"untracked_hash_failed:{relative}:{type(exc).__name__}")
    component_paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "core/brain/frontier_gap.py",
        REPO_ROOT / "core/brain/frontier_evidence_v5.py",
        REPO_ROOT / "core/brain/reasoning_amplifier_v2.py",
        REPO_ROOT / "core/brain/verifiers/registry.py",
        REPO_ROOT / "core/brain/llm/mlx_client.py",
        REPO_ROOT / "core/brain/llm/model_registry.py",
        REPO_ROOT / "core/runtime/dynamic_execution_gateway.py",
    )
    component_sha256: dict[str, str] = {}
    for component in component_paths:
        try:
            relative = component.relative_to(REPO_ROOT).as_posix()
            component_sha256[relative] = _hash_file(component)
        except (OSError, ValueError) as exc:
            issues.append(f"component_hash_failed:{component.name}:{type(exc).__name__}")

    workspace_digest = hashlib.sha256()
    workspace_digest.update(commit_sha.encode())
    workspace_digest.update(tree_sha.encode())
    workspace_digest.update(status_bytes)
    workspace_digest.update(diff_bytes)
    workspace_digest.update(index_diff_bytes)
    workspace_digest.update(untracked_digest.digest())
    clean = not status_bytes and not issues
    return {
        "schema": SOURCE_PROVENANCE_SCHEMA,
        "commit_sha": commit_sha,
        "tree_sha": tree_sha,
        "clean": clean,
        "workspace_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "index_diff_sha256": hashlib.sha256(index_diff_bytes).hexdigest(),
        "untracked_content_sha256": untracked_digest.hexdigest(),
        "workspace_state_sha256": workspace_digest.hexdigest(),
        "execution_component_sha256": component_sha256,
        "source_identity": copy.deepcopy(source_identity),
        "issues": issues,
    }


def source_stability_window(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    body = {"before": before, "after": after}
    return {
        "schema": SOURCE_STABILITY_SCHEMA,
        **body,
        "stable": canonical_json_bytes(before) == canonical_json_bytes(after),
        "window_sha256": sha256_json(body),
    }


def _live_instance_up(port: int = 8000) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


_CONTROL_FACTS = {
    "chemical symbol for gold": "Au",
    "known as the red planet": "Mars",
    "sides does a hexagon": "6",
    "capital of japan": "Tokyo",
    "absorb for photosynthesis": "carbon dioxide",
    "chemical symbol for sodium": "Na",
    "largest ocean on earth": "Pacific Ocean",
    "degrees are in a right angle": "90",
    "capital of kenya": "Nairobi",
    "element has atomic number 8": "oxygen",
    "instrument measures atmospheric pressure": "barometer",
    "smallest prime number": "2",
    "continent contains peru": "South America",
    "si unit of electric current": "ampere",
    "changes liquid water into vapor": "evaporation",
    "capital of new zealand": "Wellington",
}


async def _synthetic_control_generate(prompt: str, temperature: float = 0.0) -> str:
    del temperature
    lowered = prompt.lower()
    multiplication = re.search(r"compute (\d+) \* (\d+)", lowered)
    if multiplication:
        return str(int(multiplication.group(1)) * int(multiplication.group(2)))
    for key, answer in _CONTROL_FACTS.items():
        if key in lowered:
            return answer
    if "who is oldest" in lowered:
        names = re.findall(r"([A-Z][a-z]+) is older than", prompt)
        return names[0] if names else ""
    function = re.search(
        r"function `([A-Za-z_][A-Za-z0-9_]*)\(xs\)` returning the "
        r"(sum|maximum|minimum) of",
        prompt,
        re.IGNORECASE,
    )
    if function:
        function_name = function.group(1)
        operation = {"sum": "sum", "maximum": "max", "minimum": "min"}[
            function.group(2).lower()
        ]
        return f"```python\ndef {function_name}(xs):\n    return {operation}(xs)\n```"
    return ""


async def _invoke_json_command(
    command: str,
    request: dict[str, Any],
    *,
    role: str,
    timeout_s: float,
) -> dict[str, Any]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("signed evidence command is empty")
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=REPO_ROOT,
        env=_sealed_command_env(role),
    )
    payload = canonical_json_bytes(request) + b"\n"
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(payload),
            timeout=timeout_s,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"signed evidence command failed: {detail}")
    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("signed evidence command returned malformed JSON") from exc
    if not isinstance(response, dict):
        raise ValueError("signed evidence command response must be an object")
    return response


def _sealed_command_env(role: str) -> dict[str, str]:
    if role not in {"worker", "verifier", "run_signer"}:
        raise ValueError("frontier command role is invalid")
    allowed_exact = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "TMPDIR",
    }
    role_prefix = f"AURA_FRONTIER_{role.upper()}_"
    env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed_exact
        or key.startswith("AURA_FRONTIER_COMMON_")
        or key.startswith(role_prefix)
    }
    env.update(
        {
            "AURA_FRONTIER_PROTOCOL": "5",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


class SignedWorkerClient:
    """One fresh JSON-lines worker process for an entire sealed battery run."""

    def __init__(self, command: str) -> None:
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self.stderr_tail: list[str] = []
        self.runtime_manifest: dict[str, Any] | None = None
        self.candidate_model: dict[str, Any] | None = None

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        async for line in self.process.stderr:
            self.stderr_tail.append(line.decode("utf-8", errors="replace")[:300])
            self.stderr_tail[:] = self.stderr_tail[-20:]

    async def _exchange(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("generation worker is not running")
        process.stdin.write(canonical_json_bytes(request) + b"\n")
        await process.stdin.drain()
        line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout_s)
        if not line:
            detail = "".join(self.stderr_tail)[-500:]
            raise RuntimeError(f"generation worker exited without a receipt: {detail}")
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("generation worker emitted malformed JSON") from exc
        if not isinstance(response, dict):
            raise ValueError("generation worker response must be an object")
        return response

    async def start(self, handshake: dict[str, Any]) -> dict[str, Any]:
        argv = shlex.split(self.command)
        if not argv:
            raise ValueError("generation worker command is empty")
        self.process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=REPO_ROOT,
            env={
                **_sealed_command_env("worker"),
                "AURA_FRONTIER_GENERATION_WORKER": "1",
            },
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        response = await self._exchange(
            {"action": "handshake", **handshake},
            timeout_s=120.0,
        )
        if set(response) != {
            "status",
            "effective_runtime_manifest",
            "candidate_model",
        } or response.get(
            "status"
        ) != "ready":
            raise ValueError("generation worker handshake is incomplete")
        self.runtime_manifest = validate_effective_runtime_manifest(
            response["effective_runtime_manifest"]
        )
        candidate_model = response.get("candidate_model")
        if not isinstance(candidate_model, dict):
            raise ValueError("generation worker omitted model stability evidence")
        self.candidate_model = copy.deepcopy(candidate_model)
        return copy.deepcopy(self.runtime_manifest)

    async def solve(
        self, request: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        started = time.monotonic()
        response = await self._exchange(
            {"action": "generate", **request},
            timeout_s=MATCHED_BUDGET["hard_timeout_s"],
        )
        observed_wall_time_s = time.monotonic() - started
        if observed_wall_time_s > MATCHED_BUDGET["hard_timeout_s"]:
            raise TimeoutError("generation worker exceeded the supervised hard deadline")
        if set(response) != {"answer", "worker_receipt"}:
            raise ValueError("generation worker response lacks signed execution evidence")
        answer = response.get("answer")
        receipt = response.get("worker_receipt")
        if not isinstance(answer, str) or not isinstance(receipt, dict):
            raise ValueError("generation worker answer or receipt is malformed")
        process = self.process
        if process is None or process.pid is None:
            raise RuntimeError("generation worker process identity disappeared")
        request_id = expected_request_id(
            run_id=str(request["run_id"]),
            run_nonce_sha256=str(request["run_nonce_sha256"]),
            item_id=str(request["item_id"]),
            attempt_index=int(request["attempt_index"]),
        )
        supervisor_observation = {
            "schema": SUPERVISOR_OBSERVATION_SCHEMA,
            "run_id": request["run_id"],
            "run_nonce_sha256": request["run_nonce_sha256"],
            "item_id": request["item_id"],
            "request_id": request_id,
            "attempt_index": request["attempt_index"],
            "prompt_sha256": request["prompt_sha256"],
            "output_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "observed_wall_time_s": round(observed_wall_time_s, 9),
            "deadline_s": MATCHED_BUDGET["hard_timeout_s"],
            "deadline_exceeded": False,
            "process_pid": int(process.pid),
            "process_running_after_response": process.returncode is None,
            "observed_at_unix": time.time(),
        }
        return answer, copy.deepcopy(receipt), supervisor_observation

    async def close(self) -> None:
        process = self.process
        if process is not None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b'{"action":"close"}\n')
                    await process.stdin.drain()
                    process.stdin.close()
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except (BrokenPipeError, ConnectionError, TimeoutError):
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            await asyncio.gather(self._stderr_task, return_exceptions=True)


def build_signed_worker_solver(
    *,
    selection: SolverSelection,
    items: list[Any],
    run_id: str,
    run_nonce_b64: str,
    run_nonce_sha256: str,
    source_identity: dict[str, Any],
    challenge_bundle_sha256: str,
) -> Callable[[str, str], Awaitable[SolverObservation]]:
    client = selection.worker_client
    runtime = selection.effective_runtime_manifest
    if client is None or runtime is None:
        raise ValueError("signed worker selection is incomplete")
    model_window = selection.model_stability_override
    if not isinstance(model_window, dict):
        raise ValueError("signed worker omitted its model stability window")
    model_stability_sha256 = require_sha256(
        model_window.get("window_sha256"), field_name="candidate model stability"
    )
    cursor = 0

    async def solve(prompt: str, task_type: str) -> SolverObservation:
        nonlocal cursor
        index = cursor
        cursor += 1
        selection.execution.attempted += 1
        if index >= len(items):
            exc = RuntimeError("battery attempted more items than the signed task specification")
            selection.execution.record_failure(exc)
            raise exc
        item = items[index]
        if item.prompt != prompt or item.task_type != task_type:
            exc = RuntimeError("battery item order differs from the signed task specification")
            selection.execution.record_failure(exc)
            raise exc
        try:
            answer, receipt, supervisor_observation = await client.solve(
                {
                    "run_id": run_id,
                    "run_nonce_b64": run_nonce_b64,
                    "run_nonce_sha256": run_nonce_sha256,
                    "attempt_index": index,
                    "item_id": item.item_id,
                    "task_type": item.task_type,
                    "prompt": item.prompt,
                    "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest(),
                    "source_identity_sha256": source_identity["identity_sha256"],
                    "runtime_manifest_sha256": runtime["manifest_sha256"],
                    "model_stability_sha256": model_stability_sha256,
                    "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
                    "challenge_bundle_sha256": challenge_bundle_sha256,
                    "sealed_evaluation": True,
                }
            )
            observation = SolverObservation(
                answer=answer,
                verified=True,
                receipt=receipt,
                supervisor_observation=supervisor_observation,
                fallbacks_used=tuple(
                    receipt.get("signed_payload", {}).get("fallbacks_used") or ()
                ),
            )
            selection.execution.record_result(observation, task_type=task_type)
            return observation
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            selection.execution.record_failure(exc)
            raise

    return solve


async def build_solver() -> tuple[Callable[..., Awaitable[SolverObservation]], SolverSelection]:
    from core.brain.reasoning_amplifier_v2 import (
        AmplificationRequest,
        ReasoningAmplifierV2,
        ReasoningMode,
    )

    selection = await _resolve_generate()
    amplifier = ReasoningAmplifierV2(selection.generate)

    async def solve(prompt: str, task_type: str) -> SolverObservation:
        attempt_index = selection.execution.attempted
        selection.execution.attempted += 1
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        request = AmplificationRequest(
            objective=prompt,
            task_type=task_type,
            time_budget_s=20.0,
            mode=ReasoningMode.NORMAL,
            sample_budget=3,
            context={
                "skip_evidence": True,
                "skip_cache": True,
                "skip_precompute_enqueue": True,
                "disable_batched_candidates": True,
                "read_only_evaluation": True,
                "sealed_evaluation": True,
                "evaluation_request_id": sha256_json(
                    {
                        "diagnostic_subject": selection.subject,
                        "attempt_index": attempt_index,
                        "task_type": task_type,
                        "prompt_sha256": prompt_sha256,
                    }
                ),
            },
        )
        try:
            result = await asyncio.wait_for(
                amplifier.amplify(request),
                timeout=MATCHED_BUDGET["hard_timeout_s"],
            )
            receipt = result.receipt.to_dict() if result.receipt is not None else {}
            observation = SolverObservation(
                answer=str(result.answer or ""),
                verified=False,
                receipt=receipt,
                fallbacks_used=tuple(receipt.get("fallbacks_used") or ()),
                diagnostics=tuple(receipt.get("known_failures") or ()),
            )
            selection.execution.record_result(observation, task_type=task_type)
            return observation
        except (TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            selection.execution.record_failure(exc)
            raise

    return solve, selection


async def _resolve_generate() -> SolverSelection:
    if _live_instance_up():
        return SolverSelection(
            generate=_synthetic_control_generate,
            mode="synthetic_control_live_instance_up",
            capability_candidate=False,
            subject="deterministic_pipeline_control",
            reason=(
                "the desktop runtime owns the model lane; this CLI has no "
                "authenticated resident-amplifier proof channel"
            ),
        )
    resolution_error = "model loading was not explicitly enabled"
    try:
        from core.brain.llm.model_registry import get_runtime_model_path

        model_path = str(get_runtime_model_path() or "")
        if os.environ.get("AURA_FRONTIER_LOAD_MODEL") == "1" and model_path:
            from core.brain.llm.mlx_client import get_mlx_client

            model_manifest = await asyncio.to_thread(collect_model_manifest, model_path)
            client = get_mlx_client(model_path)

            async def generate(prompt: str, temperature: float = 0.0) -> str:
                output = await client.generate(
                    prompt,
                    max_tokens=MATCHED_BUDGET["max_tokens"],
                    temperature=MATCHED_BUDGET["temperature"],
                    top_p=MATCHED_BUDGET["top_p"],
                    top_k=MATCHED_BUDGET["top_k"],
                    benchmark_request=True,
                    # One output contract, not two.
                    #
                    # Both were asked for, and the worker refuses that: two
                    # exclusive contracts leave it with no single one to
                    # honour. So every item of the diagnostic was refused with
                    # ambiguous_output_contract and the run scored 0.0 — which
                    # is why this mode has never produced a number, on top of
                    # every reason of policy. The battery is four deterministic
                    # classes wanting exact short answers, and both flags carry
                    # the same sealed, uncached handling, so the strict answer
                    # contract alone loses nothing.
                    strict_answer_contract=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    sealed_evaluation=True,
                    timeout=MATCHED_BUDGET["hard_timeout_s"],
                )
                return output or ""

            runtime_manifest = collect_effective_runtime_manifest(
                subject_id=f"aura-unattested:{model_manifest['manifest_sha256']}",
                model_manifest=model_manifest,
                sealed_evaluation_enforced=False,
                fresh_process=False,
                immutable_source=os.environ.get(_IMMUTABLE_CHILD_ENV) == "1",
                modifiers={
                    "latent_bridge": "benchmark_contract_requested",
                    "worker_receipts": "unavailable",
                    "reason": "in-process diagnostic cannot attest worker enforcement",
                },
            )

            return SolverSelection(
                generate=generate,
                mode="amplifier_mlx_unattested_diagnostic",
                capability_candidate=False,
                subject=f"aura_unattested:{runtime_manifest['manifest_sha256']}",
                reason="in-process model execution lacks independent v5 receipts",
                model_manifest_before=model_manifest,
                model_path_supplier=lambda: str(
                    Path(str(client.model_path)).expanduser().resolve(strict=True)
                ),
                effective_runtime_manifest=runtime_manifest,
            )
        if not model_path:
            resolution_error = "no runtime model path is configured"
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        resolution_error = f"model resolution failed: {type(exc).__name__}: {exc}"
    return SolverSelection(
        generate=_synthetic_control_generate,
        mode="synthetic_control",
        capability_candidate=False,
        subject="deterministic_pipeline_control",
        reason=resolution_error,
    )


def _load_reference(
    path_text: str,
    *,
    seed: int,
    per_class: int,
    trusted_evaluator_keys: dict[str, str],
    trusted_worker_keys: dict[str, str],
    trusted_verifiers: dict[str, dict[str, str]],
    trusted_run_keys: dict[str, str],
    trusted_release_keys: dict[str, str],
    expected_identity_freeze_sha256: str | None = None,
    verification_time_unix: float | None = None,
    require_fresh_challenge: bool = False,
) -> tuple[ReferenceEvidence | None, str]:
    if not path_text:
        return None, "matched named reference artifact not supplied"
    path = Path(path_text).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            validate_reference_artifact(
                payload,
                seed=seed,
                per_class=per_class,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                expected_identity_freeze_sha256=expected_identity_freeze_sha256,
                verification_time_unix=verification_time_unix,
                require_fresh_challenge=require_fresh_challenge,
            ),
            "",
        )
    except (OSError, ValueError, TypeError) as exc:
        return None, f"reference validation failed: {type(exc).__name__}: {exc}"


def _parse_trusted_evaluator_keys(values: list[str]) -> dict[str, str]:
    trusted: dict[str, str] = {}
    for value in values:
        evaluator_id, separator, public_key = str(value or "").partition("=")
        evaluator_id = evaluator_id.strip()
        public_key = public_key.strip()
        if not separator or not evaluator_id or not public_key:
            raise ValueError("trusted evaluator keys must use EVALUATOR_ID=BASE64_PUBLIC_KEY")
        if evaluator_id in trusted and trusted[evaluator_id] != public_key:
            raise ValueError(f"conflicting trusted evaluator key: {evaluator_id}")
        trusted[evaluator_id] = public_key
    return trusted


def _parse_trusted_verifiers(values: list[str]) -> dict[str, dict[str, str]]:
    trusted: dict[str, dict[str, str]] = {}
    for value in values:
        verifier_id, separator, material = str(value or "").partition("=")
        parts = [part.strip() for part in material.split(",")]
        if not separator or not verifier_id.strip() or len(parts) != 3:
            raise ValueError(
                "trusted verifiers must use ID=BASE64_KEY,IMPLEMENTATION_SHA256,RELEASE_SHA256"
            )
        public_key, implementation, release = parts
        require_sha256(implementation, field_name="trusted verifier implementation")
        require_sha256(release, field_name="trusted verifier release")
        pin = {
            "public_key_b64": public_key,
            "implementation_sha256": implementation,
            "release_sha256": release,
        }
        verifier_id = verifier_id.strip()
        if verifier_id in trusted and trusted[verifier_id] != pin:
            raise ValueError(f"conflicting trusted verifier pin: {verifier_id}")
        trusted[verifier_id] = pin
    return trusted


class PriorArtifactCorruptionError(ValueError):
    def __init__(self, message: str, *, raw_bytes: bytes) -> None:
        super().__init__(message)
        self.raw_bytes = raw_bytes


class ConcurrentEvidenceUpdateError(RuntimeError):
    """The ledger head changed after this run loaded its prior snapshot."""


def _evidence_lock_path(path: Path) -> Path:
    identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    return Path.home() / ".aura" / "runtime" / "frontier-locks" / f"{identity}.lock"


def _artifact_state_sha256(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"aura.frontier_artifact.absent.v1").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_artifact_sha256(prior: dict[str, Any], path: Path) -> str | None:
    if not prior:
        return None
    if prior.get("schema") != "aura.frontier_gap_report.v5":
        if not path.exists():
            return None
        digest, _ = _hash_regular_file_beneath(path.parent, path.name)
        return digest

    inherited = prior.get("legacy_artifact_sha256")
    if inherited:
        return require_sha256(inherited, field_name="legacy frontier artifact")

    prefix = f"{path.name}.legacy.v4."
    recovered: set[str] = set()
    if path.parent.exists():
        for candidate in path.parent.iterdir():
            if not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name.removeprefix(prefix)
            if not re.fullmatch(r"[0-9a-f]{12}", suffix):
                raise ValueError("preserved legacy artifact name is malformed")
            digest, _ = _hash_regular_file_beneath(path.parent, candidate.name)
            if not digest.startswith(suffix):
                raise ValueError("preserved legacy artifact digest does not match its name")
            recovered.add(digest)
    if len(recovered) > 1:
        raise ValueError("legacy frontier artifact lineage is ambiguous")
    return next(iter(recovered), None)


def _read_prior_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    from core.runtime.atomic_writer import interprocess_file_lock

    with interprocess_file_lock(_evidence_lock_path(path)):
        prior = _read_prior_strict(path)
        return prior, _artifact_state_sha256(path)


@asynccontextmanager
async def _evidence_persistence_lock(path: Path):
    from core.runtime.atomic_writer import interprocess_file_lock

    manager = interprocess_file_lock(_evidence_lock_path(path))
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="frontier-evidence-lock",
    )
    entered = False

    async def await_worker(future: asyncio.Future[Any]) -> tuple[Any, bool]:
        cancellation_requested = False
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                if future.cancelled():
                    raise
                cancellation_requested = True
        return future.result(), cancellation_requested

    try:
        _, cancelled_during_enter = await await_worker(
            loop.run_in_executor(executor, manager.__enter__)
        )
        entered = True
        if cancelled_during_enter:
            raise asyncio.CancelledError
        yield
    finally:
        cancelled_during_exit = False
        try:
            if entered:
                _, cancelled_during_exit = await await_worker(
                    loop.run_in_executor(
                        executor,
                        manager.__exit__,
                        None,
                        None,
                        None,
                    )
                )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if cancelled_during_exit:
            raise asyncio.CancelledError


def _read_prior_strict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise PriorArtifactCorruptionError(
            f"existing frontier artifact is unreadable: {exc}", raw_bytes=b""
        ) from exc
    try:
        envelope = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PriorArtifactCorruptionError(
            "existing frontier artifact contains malformed JSON",
            raw_bytes=raw_bytes,
        ) from exc
    payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else None
    if not isinstance(payload, dict):
        raise PriorArtifactCorruptionError(
            "existing frontier artifact payload is not an object",
            raw_bytes=raw_bytes,
        )
    schema = payload.get("schema")
    if schema not in {
        "aura.frontier_gap_report.v1",
        "aura.frontier_gap_report.v2",
        "aura.frontier_gap_report.v3",
        "aura.frontier_gap_report.v4",
        "aura.frontier_gap_report.v5",
    }:
        raise PriorArtifactCorruptionError(
            "existing frontier artifact schema is unsupported",
            raw_bytes=raw_bytes,
        )
    return payload


class EvidenceBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pending: dict[str, dict[str, Any]] = {}

    def _path(self, digest: str) -> Path:
        require_sha256(digest, field_name="evidence blob")
        return self.root / f"{digest}.json"

    def _resolve_disk(self, digest: str) -> dict[str, Any]:
        path = self._path(digest)
        try:
            envelope = json.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"evidence blob cannot be read: {digest}") from exc
        if (
            isinstance(envelope, dict)
            and set(envelope) == {"schema", "schema_name", "schema_version", "payload"}
            and envelope.get("schema") == "frontier_evidence_blob"
            and envelope.get("schema_name") == "frontier_evidence_blob"
            and envelope.get("schema_version") == 1
        ):
            envelope = envelope.get("payload")
        if not isinstance(envelope, dict) or set(envelope) != {
            "schema",
            "evidence_sha256",
            "payload",
        }:
            raise ValueError("evidence blob envelope is malformed")
        if envelope.get("schema") != "aura.frontier_evidence_blob.v1":
            raise ValueError("evidence blob schema is invalid")
        payload = envelope.get("payload")
        if (
            envelope.get("evidence_sha256") != digest
            or not isinstance(payload, dict)
            or sha256_json(payload) != digest
        ):
            raise ValueError("evidence blob content digest mismatch")
        return payload

    def resolve(self, digest: str) -> dict[str, Any]:
        if digest in self.pending:
            return copy.deepcopy(self.pending[digest])
        return self._resolve_disk(digest)

    def stage(self, digest: str, payload: dict[str, Any]) -> None:
        require_sha256(digest, field_name="evidence blob")
        if sha256_json(payload) != digest:
            raise ValueError("attempted to stage evidence under the wrong digest")
        path = self._path(digest)
        if path.exists():
            existing = self.resolve(digest)
            if canonical_json_bytes(existing) != canonical_json_bytes(payload):
                raise ValueError("content-addressed evidence blob collision")
            return
        self.pending[digest] = copy.deepcopy(payload)

    async def flush(self, gateway: Any) -> None:
        await gateway.ensure_directory_async(self.root, source="frontier_gap.evidence")
        for digest, payload in sorted(self.pending.items()):
            envelope = {
                "schema": "aura.frontier_evidence_blob.v1",
                "evidence_sha256": digest,
                "payload": payload,
            }
            await gateway.write_bytes_if_absent_async(
                self._path(digest),
                canonical_json_bytes(envelope) + b"\n",
                source="frontier_gap.evidence",
            )
            committed = await asyncio.to_thread(self._resolve_disk, digest)
            if canonical_json_bytes(committed) != canonical_json_bytes(payload):
                raise RuntimeError(f"evidence blob commit verification failed: {digest}")
        self.pending.clear()

    async def prune_unreferenced(self, gateway: Any, referenced: set[str]) -> None:
        if not self.root.exists():
            return
        for path in self.root.glob("*.json"):
            if path.stem not in referenced:
                await gateway.delete_path_async(
                    path,
                    source="frontier_gap.evidence_retention",
                )


def _load_ledger(
    prior: dict[str, Any],
    *,
    key: str,
    evidence_class: str,
    trusted_evaluator_keys: dict[str, str],
    trusted_worker_keys: dict[str, str],
    trusted_verifiers: dict[str, dict[str, str]],
    trusted_run_keys: dict[str, str],
    trusted_release_keys: dict[str, str],
    evidence_store: EvidenceBlobStore,
) -> GapLedger:
    raw = prior.get(key)
    if prior.get("schema") == "aura.frontier_gap_report.v5":
        if not isinstance(raw, dict):
            raise ValueError(f"v5 frontier artifact is missing {key}")
        ledger = GapLedger.from_dict(
            raw,
            evidence_class=evidence_class,
            evidence_blob_resolver=evidence_store.resolve,
            trusted_evaluator_keys=trusted_evaluator_keys,
            trusted_worker_keys=trusted_worker_keys,
            trusted_verifiers=trusted_verifiers,
            trusted_run_keys=trusted_run_keys,
            trusted_release_keys=trusted_release_keys,
            source_tree_resolver=_resolve_commit_tree,
            source_component_resolver=_resolve_commit_component_sha256,
        )
    else:
        ledger = GapLedger(
            evidence_class=evidence_class,
            capability_claim_eligible=(evidence_class == CAPABILITY_EVIDENCE_CLASS),
        )
    return ledger


def _capability_eligibility(
    selection: SolverSelection,
    report: dict[str, Any],
    source_window: dict[str, Any],
    model_window: dict[str, Any],
    reference: ReferenceEvidence | None,
    source_identity: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = int(report.get("expected_item_count") or 0)
    execution = selection.execution
    if not selection.capability_candidate or selection.mode != "amplifier_mlx_worker_v5":
        reasons.append("solver_is_not_isolated_signed_v5_worker")
    if reference is None:
        reasons.append("validated_matched_reference_missing")
    if source_identity is None:
        reasons.append("trusted_repository_release_identity_missing")
    if selection.effective_runtime_manifest is None:
        reasons.append("effective_runtime_manifest_missing")
    if not isinstance(report.get("trust_basis"), dict):
        reasons.append("external_trust_basis_missing")
    before = source_window.get("before")
    after = source_window.get("after")
    if source_window.get("stable") is not True or not isinstance(before, dict) or not isinstance(after, dict):
        reasons.append("source_changed_during_measurement")
    else:
        if before.get("clean") is not True or after.get("clean") is not True:
            reasons.append("source_tree_not_clean")
        commit_sha = str(before.get("commit_sha") or "")
        tree_sha = str(before.get("tree_sha") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
            reasons.append("exact_commit_unavailable")
        if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
            reasons.append("exact_tree_unavailable")
        if re.fullmatch(r"[0-9a-f]{40}", commit_sha) and re.fullmatch(
            r"[0-9a-f]{40}", tree_sha
        ):
            try:
                if _resolve_commit_tree(commit_sha) != tree_sha:
                    reasons.append("commit_tree_binding_invalid")
                components = before.get("execution_component_sha256")
                if not isinstance(components, dict) or not components:
                    reasons.append("source_component_binding_invalid")
                elif any(
                    _resolve_commit_component_sha256(commit_sha, path) != digest
                    for path, digest in components.items()
                ):
                    reasons.append("source_component_binding_invalid")
            except (OSError, RuntimeError, TypeError, ValueError):
                reasons.append("source_commit_binding_invalid")
    model_before = model_window.get("before")
    model_after = model_window.get("after")
    if (
        model_window.get("stable") is not True
        or not isinstance(model_before, dict)
        or not isinstance(model_after, dict)
    ):
        reasons.append("candidate_model_changed_or_unavailable")
    else:
        manifest_sha = str(model_before.get("manifest_sha256") or "")
        runtime = selection.effective_runtime_manifest or {}
        if runtime.get("base_model_manifest_sha256") != manifest_sha:
            reasons.append("effective_runtime_base_model_mismatch")
        if selection.subject != f"aura_model:{runtime.get('manifest_sha256', '')}":
            reasons.append("candidate_effective_runtime_identity_mismatch")
    if execution.attempted != expected or execution.completed != expected:
        reasons.append("incomplete_execution_count")
    if execution.failed:
        reasons.append("execution_failures_present")
    if execution.invalid:
        reasons.append("invalid_receipts_present")
    if execution.empty:
        reasons.append("empty_answers_present")
    if execution.unverified:
        reasons.append("unverified_answers_present")
    if execution.disqualifying_fallbacks:
        reasons.append("disqualifying_fallbacks_present")
    items = report.get("items")
    if not isinstance(items, list) or len(items) != expected:
        reasons.append("item_evidence_incomplete")
    elif any(
        not item.get("receipt")
        or not item.get("supervisor_observation")
        or not item.get("correctness_receipt")
        or item.get("execution_error")
        or not str(item.get("answer") or "").strip()
        for item in items
    ):
        reasons.append("signed_worker_or_correctness_receipts_incomplete")
    if not isinstance(report.get("correctness_receipts"), list) or len(
        report.get("correctness_receipts") or []
    ) != expected:
        reasons.append("independent_correctness_receipts_incomplete")
    if not isinstance(report.get("run_envelope"), dict):
        reasons.append("independent_run_envelope_missing")
    if not isinstance(report.get("task_spec"), dict) or not isinstance(
        report.get("challenge"), dict
    ):
        reasons.append("signed_task_or_commit_reveal_challenge_missing")
    if report.get("reference_basis") != TRUSTED_REFERENCE_BASIS or report.get("overall_gap") is None:
        reasons.append("frontier_gap_not_computable")
    return not reasons, list(dict.fromkeys(reasons))


def _capability_measurement(
    *,
    selection: SolverSelection,
    report: dict[str, Any],
    eligible: bool,
    reasons: list[str],
) -> dict[str, Any]:
    if eligible:
        return {
            "status": "measured",
            "subject": selection.subject,
            "overall_candidate_score": report["overall_candidate_score"],
            "overall_gap": report["overall_gap"],
            "claim_eligible": True,
        }
    if selection.capability_candidate:
        return {
            "status": "rejected",
            "subject": selection.subject,
            "claim_eligible": False,
            "reasons": reasons,
        }
    return {
        "status": "not_measured",
        "subject": "aura_resident_model",
        "claim_eligible": False,
        "reason": selection.reason,
        "required_next_evidence": (
            "run amplifier_mlx with complete verified receipts, a clean exact source, "
            "a stable content-addressed model, and a trusted signed matched-budget "
            "reference artifact"
        ),
    }


def _load_json_object(path_text: str, *, label: str) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} cannot be read as JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


async def _attach_independent_correctness(
    report: dict[str, Any],
    *,
    command: str,
    run_id: str,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    if not command:
        return receipts
    for item in report["items"]:
        response = await _invoke_json_command(
            command,
            {
                "action": "grade",
                "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
                "task_spec": copy.deepcopy(report["task_spec"]),
                "challenge": copy.deepcopy(report["challenge"]),
                "run_id": run_id,
                "item": {
                    key: item[key]
                    for key in (
                        "index",
                        "item_id",
                        "task_class",
                        "task_type",
                        "prompt_sha256",
                        "grader_implementation_sha256",
                        "expected_answer_commitment_sha256",
                        "hidden_case_commitment_sha256",
                        "answer",
                        "output_sha256",
                    )
                },
            },
            role="verifier",
            timeout_s=MATCHED_BUDGET["hard_timeout_s"],
        )
        receipt = response.get("correctness_receipt", response)
        if not isinstance(receipt, dict) or receipt.get("schema") != CORRECTNESS_RECEIPT_SCHEMA:
            raise ValueError("independent verifier omitted a v5 correctness receipt")
        item["correctness_receipt"] = copy.deepcopy(receipt)
        receipts.append(copy.deepcopy(receipt))
    report["correctness_receipts"] = receipts
    return receipts


async def _attach_run_envelope(
    report: dict[str, Any],
    *,
    command: str,
    run_id: str,
    run_nonce: bytes,
    run_nonce_sha256: str,
    started_at_unix: float,
    completed_at_unix: float,
    source_identity: dict[str, Any],
    runtime_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if not command:
        return None
    worker_payloads = [item["receipt"]["signed_payload"] for item in report["items"]]
    supervisor_observations = [item["supervisor_observation"] for item in report["items"]]
    correctness = report.get("correctness_receipts") or []
    if len(correctness) != len(worker_payloads):
        raise ValueError("cannot sign a run with incomplete correctness receipts")
    trust_basis = report.get("trust_basis")
    if not isinstance(trust_basis, dict):
        raise ValueError("cannot sign a run without an exact external trust basis")
    outputs = [
        {
            "index": item["index"],
            "item_id": item["item_id"],
            "answer": item["answer"],
            "output_sha256": item["output_sha256"],
        }
        for item in report["items"]
    ]
    task_spec_sha256 = sha256_json(report["task_spec"])
    challenge_sha256 = sha256_json(report["challenge"])
    # The protocol validators recompute these exact digests; the coordinator
    # signs only after all worker and verifier envelopes exist.
    budget_summary = {
        "item_count": len(worker_payloads),
        "total_input_tokens": sum(
            item["resource_usage"]["input_tokens"] for item in worker_payloads
        ),
        "total_output_tokens": sum(
            item["resource_usage"]["output_tokens"] for item in worker_payloads
        ),
        "total_candidate_count": sum(
            item["resource_usage"]["candidate_count"] for item in worker_payloads
        ),
        "total_generation_calls": sum(
            item["resource_usage"]["generation_calls"] for item in worker_payloads
        ),
        "maximum_item_wall_time_s": max(
            (item["resource_usage"]["wall_time_s"] for item in worker_payloads),
            default=0,
        ),
        "maximum_supervisor_wall_time_s": max(
            (
                float(item["observed_wall_time_s"])
                for item in supervisor_observations
            ),
            default=0,
        ),
        "all_within_budget": True,
    }
    payload = {
        "run_id": run_id,
        "run_nonce_b64": base64.b64encode(run_nonce).decode("ascii"),
        "run_nonce_sha256": run_nonce_sha256,
        "task_spec_sha256": task_spec_sha256,
        "challenge_bundle_sha256": challenge_sha256,
        "protocol_manifest_sha256": PROTOCOL_MANIFEST_SHA256,
        "source_identity_sha256": source_identity["identity_sha256"],
        "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
        "reference_artifact_sha256": report["reference_artifact_sha256"],
        "trust_basis_sha256": trust_basis["manifest_sha256"],
        "worker_receipt_sha256": [sha256_json(item["receipt"]) for item in report["items"]],
        "supervisor_observation_sha256": [
            sha256_json(item) for item in supervisor_observations
        ],
        "correctness_receipt_sha256": [sha256_json(item) for item in correctness],
        "outputs_sha256": sha256_json(outputs),
        "worker_signer_ids": [item["receipt"]["signer"]["signer_id"] for item in report["items"]],
        "verifier_id": report["task_spec"]["signed_payload"]["verifier_identity"][
            "verifier_id"
        ],
        "started_at_unix": started_at_unix,
        "completed_at_unix": completed_at_unix,
        "budget_summary": budget_summary,
    }
    response = await _invoke_json_command(
        command,
        {
            "action": "sign_run",
            "schema": RUN_ENVELOPE_SCHEMA,
            "signed_payload": payload,
        },
        role="run_signer",
        timeout_s=30.0,
    )
    envelope = response.get("run_envelope", response)
    if not isinstance(envelope, dict) or envelope.get("schema") != RUN_ENVELOPE_SCHEMA:
        raise ValueError("independent run signer omitted a v5 run envelope")
    report["run_envelope"] = copy.deepcopy(envelope)
    return envelope


def _resolve_output_path(path_text: str) -> Path:
    out = Path(path_text).expanduser()
    if not out.is_absolute():
        out = Path(os.environ.get(_ORIGINAL_REPO_ENV, str(REPO_ROOT))) / out
    return out.resolve()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    parser.add_argument("--out", default="artifacts/frontier_gap/latest.json")
    parser.add_argument("--reference-artifact", default="")
    parser.add_argument("--trusted-reference-key", action="append", default=[])
    parser.add_argument("--trusted-worker-key", action="append", default=[])
    parser.add_argument("--trusted-run-key", action="append", default=[])
    parser.add_argument("--trusted-release-key", action="append", default=[])
    parser.add_argument(
        "--trusted-verifier",
        action="append",
        default=[],
        metavar="ID=KEY,IMPLEMENTATION_SHA256,RELEASE_SHA256",
    )
    parser.add_argument("--repository-id", default="")
    parser.add_argument("--trusted-release-ref", default="")
    parser.add_argument("--release-attestation", default="")
    parser.add_argument(
        "--worker-command",
        default="",
        help="fresh JSON-lines generation worker implementing frontier protocol v5",
    )
    parser.add_argument(
        "--verifier-command",
        default="",
        help="independent correctness verifier accepting one JSON request",
    )
    parser.add_argument(
        "--run-signer-command",
        default="",
        help="independent run coordinator signer accepting one JSON request",
    )
    parser.add_argument("--require-capability-evidence", action="store_true")
    args = parser.parse_args()
    if args.per_class <= 0 or args.per_class > 16:
        parser.error("--per-class must be between 1 and 16")

    try:
        trusted_evaluator_keys = _parse_trusted_evaluator_keys(
            args.trusted_reference_key
        )
        trusted_worker_keys = _parse_trusted_evaluator_keys(args.trusted_worker_key)
        trusted_run_keys = _parse_trusted_evaluator_keys(args.trusted_run_key)
        trusted_release_keys = _parse_trusted_evaluator_keys(args.trusted_release_key)
        trusted_verifiers = _parse_trusted_verifiers(args.trusted_verifier)
        trust_material_supplied = any(
            (
                trusted_evaluator_keys,
                trusted_worker_keys,
                trusted_run_keys,
                trusted_release_keys,
                trusted_verifiers,
            )
        )
        trust_basis = (
            build_trust_basis(
                evaluator_keys=trusted_evaluator_keys,
                worker_keys=trusted_worker_keys,
                verifiers=trusted_verifiers,
                run_keys=trusted_run_keys,
                release_keys=trusted_release_keys,
            )
            if trust_material_supplied
            else None
        )
        release_attestation = _load_json_object(
            args.release_attestation,
            label="release attestation",
        )
    except ValueError as exc:
        parser.error(str(exc))

    source_identity = await asyncio.to_thread(
        collect_source_identity,
        repository_id=args.repository_id,
        release_ref=args.trusted_release_ref,
        release_attestation=release_attestation,
        trusted_release_keys=trusted_release_keys,
    )
    source_before = await asyncio.to_thread(collect_source_provenance, source_identity)
    run_nonce = secrets.token_bytes(32)
    run_nonce_sha256 = hashlib.sha256(run_nonce).hexdigest()
    run_id = sha256_json(
        {
            "run_nonce_sha256": run_nonce_sha256,
            "source_identity_sha256": (
                source_identity.get("identity_sha256") if source_identity else None
            ),
            "started_at_unix": time.time_ns(),
        }
    )
    reference: ReferenceEvidence | None = None
    reference_error = ""
    protocol_error = ""
    selection: SolverSelection
    solve: Callable[..., Awaitable[SolverObservation]]
    worker_client: SignedWorkerClient | None = None

    if args.worker_command:
        if source_identity is None or trust_basis is None:
            print(
                "[frontier-gap] signed worker requires complete external trust pins and a trusted repository/release identity",
                file=sys.stderr,
            )
            return 2
        worker_client = SignedWorkerClient(args.worker_command)
        try:
            runtime_manifest = await worker_client.start(
                {
                    "protocol_manifest": copy.deepcopy(PROTOCOL_MANIFEST),
                    "run_id": run_id,
                    "run_nonce_b64": base64.b64encode(run_nonce).decode("ascii"),
                    "run_nonce_sha256": run_nonce_sha256,
                    "source_identity": copy.deepcopy(source_identity),
                    "sealed_evaluation": True,
                }
            )
            reference_payload = _load_json_object(
                args.reference_artifact,
                label="reference artifact",
            )
            if reference_payload is None:
                raise ValueError("signed worker requires a reference artifact")
            raw_reference = reference_payload.get("payload", reference_payload)
            raw_signed = raw_reference.get("signed_payload") if isinstance(raw_reference, dict) else None
            if not isinstance(raw_signed, dict):
                raise ValueError("reference artifact lacks a signed payload")
            reference_runtime = validate_effective_runtime_manifest(
                raw_signed.get("effective_runtime_manifest")
            )
            expected_freeze = identity_freeze_sha256(
                source_identity_sha256=source_identity["identity_sha256"],
                candidate_runtime_sha256=runtime_manifest["manifest_sha256"],
                reference_runtime_sha256=reference_runtime["manifest_sha256"],
            )
            reference, reference_error = _load_reference(
                args.reference_artifact,
                seed=args.seed,
                per_class=args.per_class,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                expected_identity_freeze_sha256=expected_freeze,
                verification_time_unix=time.time(),
                require_fresh_challenge=True,
            )
            if reference is None:
                raise ValueError(reference_error or "reference validation failed")
            selection = SolverSelection(
                generate=_synthetic_control_generate,
                mode="amplifier_mlx_worker_v5",
                capability_candidate=True,
                subject=f"aura_model:{runtime_manifest['manifest_sha256']}",
                reason="fresh isolated generation worker with signed v5 receipts",
                effective_runtime_manifest=runtime_manifest,
                worker_client=worker_client,
                model_stability_override=copy.deepcopy(worker_client.candidate_model),
            )
            signed_items = build_battery(
                seed=args.seed,
                per_class=args.per_class,
                challenge_nonce=reference.challenge_nonce,
            )
            solve = build_signed_worker_solver(
                selection=selection,
                items=signed_items,
                run_id=run_id,
                run_nonce_b64=base64.b64encode(run_nonce).decode("ascii"),
                run_nonce_sha256=run_nonce_sha256,
                source_identity=source_identity,
                challenge_bundle_sha256=sha256_json(reference.challenge),
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            await worker_client.close()
            print(
                f"[frontier-gap] signed worker admission failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 2
    else:
        reference, reference_error = _load_reference(
            args.reference_artifact,
            seed=args.seed,
            per_class=args.per_class,
            trusted_evaluator_keys=trusted_evaluator_keys,
            trusted_worker_keys=trusted_worker_keys,
            trusted_verifiers=trusted_verifiers,
            trusted_run_keys=trusted_run_keys,
            trusted_release_keys=trusted_release_keys,
        )
        solve, selection = await build_solver()

    print(
        f"[frontier-gap] mode={selection.mode} seed={args.seed} "
        f"per_class={args.per_class} reference="
        f"{reference.model_id if reference else 'unavailable'}"
    )
    run_started = time.time()
    try:
        report = await run_battery(
            solve,
            seed=args.seed,
            per_class=args.per_class,
            reference=reference,
            grade_to_foundry=False,
        )
        if selection.capability_candidate:
            await _attach_independent_correctness(
                report,
                command=args.verifier_command,
                run_id=run_id,
            )
            await _attach_run_envelope(
                report,
                command=args.run_signer_command,
                run_id=run_id,
                run_nonce=run_nonce,
                run_nonce_sha256=run_nonce_sha256,
                started_at_unix=run_started,
                completed_at_unix=time.time(),
                source_identity=source_identity or {},
                runtime_manifest=selection.effective_runtime_manifest or {},
            )
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
        protocol_error = f"{type(exc).__name__}: {str(exc)[:500]}"
        if "report" not in locals():
            report = await run_battery(
                _synthetic_control_generate,
                seed=args.seed,
                per_class=args.per_class,
                grade_to_foundry=False,
            )
    finally:
        if worker_client is not None:
            await worker_client.close()

    source_after = await asyncio.to_thread(collect_source_provenance, source_identity)
    source_window = source_stability_window(source_before, source_after)
    model_window = await asyncio.to_thread(collect_model_stability_window, selection)
    report.update(
        {
            "solver_mode": selection.mode,
            "measurement_subject": selection.subject,
            "execution": selection.execution.to_dict(),
            "source_provenance": source_after,
            "source_stability": source_window,
            "source_identity": copy.deepcopy(source_identity),
            "candidate_model": model_window,
            "effective_runtime_manifest": copy.deepcopy(
                selection.effective_runtime_manifest
            ),
            "trust_basis": copy.deepcopy(trust_basis),
        }
    )
    eligible, eligibility_reasons = _capability_eligibility(
        selection,
        report,
        source_window,
        model_window,
        reference,
        source_identity,
    )
    for reason in (reference_error, protocol_error):
        if reason:
            eligibility_reasons.append(reason)
    eligibility_reasons = list(dict.fromkeys(eligibility_reasons))
    eligible = eligible and not eligibility_reasons
    evidence_class = (
        CAPABILITY_EVIDENCE_CLASS
        if eligible
        else REJECTED_EVIDENCE_CLASS
        if selection.capability_candidate
        else CONTROL_EVIDENCE_CLASS
    )
    report.update(
        {
            "evidence_class": evidence_class,
            "capability_claim_eligible": eligible,
            "eligibility_reasons": [] if eligible else eligibility_reasons,
            "reference_validation_error": reference_error or None,
        }
    )
    if eligible:
        try:
            validate_capability_report(
                report,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                source_tree_resolver=_resolve_commit_tree,
                source_component_resolver=_resolve_commit_component_sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            eligible = False
            evidence_class = REJECTED_EVIDENCE_CLASS
            eligibility_reasons.append(
                f"final_protocol_validation_failed:{type(exc).__name__}:{str(exc)[:300]}"
            )
            report.update(
                {
                    "evidence_class": evidence_class,
                    "capability_claim_eligible": False,
                    "eligibility_reasons": list(dict.fromkeys(eligibility_reasons)),
                }
            )

    out = _resolve_output_path(args.out)
    store = EvidenceBlobStore(out.parent / "evidence-v5")
    try:
        prior, prior_state_sha256 = await asyncio.to_thread(
            _read_prior_snapshot, out
        )
        ledgers = {
            CAPABILITY_EVIDENCE_CLASS: _load_ledger(
                prior,
                key="capability_ledger",
                evidence_class=CAPABILITY_EVIDENCE_CLASS,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                evidence_store=store,
            ),
            CONTROL_EVIDENCE_CLASS: _load_ledger(
                prior,
                key="control_ledger",
                evidence_class=CONTROL_EVIDENCE_CLASS,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                evidence_store=store,
            ),
            REJECTED_EVIDENCE_CLASS: _load_ledger(
                prior,
                key="rejected_ledger",
                evidence_class=REJECTED_EVIDENCE_CLASS,
                trusted_evaluator_keys=trusted_evaluator_keys,
                trusted_worker_keys=trusted_worker_keys,
                trusted_verifiers=trusted_verifiers,
                trusted_run_keys=trusted_run_keys,
                trusted_release_keys=trusted_release_keys,
                evidence_store=store,
            ),
        }
    except (
        PriorArtifactCorruptionError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        raw_bytes = exc.raw_bytes if isinstance(exc, PriorArtifactCorruptionError) else (
            out.read_bytes() if out.exists() else b""
        )
        quarantine = out.with_name(
            f"{out.name}.corrupt.{int(time.time())}.{hashlib.sha256(raw_bytes).hexdigest()[:12]}"
        )
        with local_internal_governed_scope("frontier_gap", domain="state_mutation"):
            gateway = get_file_write_gateway()
            await gateway.ensure_directory_async(out.parent, source="frontier_gap.quarantine")
            await gateway.write_bytes_async(
                quarantine,
                raw_bytes,
                source="frontier_gap.quarantine",
            )
        print(
            f"[frontier-gap] existing evidence failed closed and was preserved at {quarantine}: {exc}",
            file=sys.stderr,
        )
        return 3

    ledger = ledgers[evidence_class]
    ledger.add(
        report,
        evidence_blob_writer=store.stage,
        trusted_evaluator_keys=trusted_evaluator_keys,
        trusted_worker_keys=trusted_worker_keys,
        trusted_verifiers=trusted_verifiers,
        trusted_run_keys=trusted_run_keys,
        trusted_release_keys=trusted_release_keys,
        source_tree_resolver=_resolve_commit_tree,
        source_component_resolver=_resolve_commit_component_sha256,
    )
    latest_entry = ledger.runs[-1]
    claim = (
        "Admissible class-scoped v5 diagnostic evidence; no general frontier claim."
        if eligible
        else "Rejected model attempt retained in a content-addressed evidence blob; no capability claim."
        if selection.capability_candidate
        else "Pipeline-control evidence only; no Aura capability claim."
    )
    legacy_digest = _legacy_artifact_sha256(prior, out)
    body = {
        "schema": "aura.frontier_gap_report.v5",
        "measurement_scope": "four_class_deterministic_diagnostic",
        "general_frontier_claim_eligible": False,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "git_commit": source_after.get("commit_sha"),
        "solver_mode": selection.mode,
        "measurement_subject": selection.subject,
        "evidence_class": evidence_class,
        "capability_claim_eligible": eligible,
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "latest_evidence": {
            "evidence_sha256": latest_entry["evidence_sha256"],
            "entry_sha256": latest_entry["entry_sha256"],
            "evidence_class": evidence_class,
            "overall_candidate_score": report["overall_candidate_score"],
            "overall_gap": report["overall_gap"] if eligible else None,
            "challenge_id": report.get("challenge_id"),
            "outputs_retained": True,
        },
        "capability_ledger": ledgers[CAPABILITY_EVIDENCE_CLASS].to_dict(),
        "control_ledger": ledgers[CONTROL_EVIDENCE_CLASS].to_dict(),
        "rejected_ledger": ledgers[REJECTED_EVIDENCE_CLASS].to_dict(),
        "legacy_artifact_sha256": legacy_digest,
        "claim": claim,
        "capability_measurement": _capability_measurement(
            selection=selection,
            report=report,
            eligible=eligible,
            reasons=report["eligibility_reasons"],
        ),
    }

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    referenced = {
        entry["evidence_sha256"]
        for current in ledgers.values()
        for entry in current.runs
    }
    try:
        async with _evidence_persistence_lock(out):
            current_state = await asyncio.to_thread(_artifact_state_sha256, out)
            if current_state != prior_state_sha256:
                raise ConcurrentEvidenceUpdateError(
                    "frontier ledger head changed before commit"
                )
            with local_internal_governed_scope(
                "frontier_gap", domain="state_mutation"
            ):
                gateway = get_file_write_gateway()
                await gateway.ensure_directory_async(out.parent, source="frontier_gap")
                await store.flush(gateway)
                if legacy_digest and out.exists():
                    legacy_path = out.with_name(
                        f"{out.name}.legacy.v4.{legacy_digest[:12]}"
                    )
                    if not legacy_path.exists():
                        await gateway.copy_path_async(
                            out,
                            legacy_path,
                            source="frontier_gap.legacy_preservation",
                        )
                await gateway.write_json_async(
                    out,
                    body,
                    schema_version=5,
                    schema_name="frontier_gap_report",
                    source="frontier_gap",
                )
                await store.prune_unreferenced(gateway, referenced)
    except ConcurrentEvidenceUpdateError as exc:
        print(
            f"[frontier-gap] evidence commit lost its compare-and-swap race: {exc}",
            file=sys.stderr,
        )
        return 4

    print("=" * 68)
    print(
        f"FOUR-CLASS DIAGNOSTIC - {selection.mode} - "
        f"candidate {report['overall_candidate_score']} - "
        f"gap {report['overall_gap'] if eligible else 'ineligible'}"
    )
    print(f"evidence class: {evidence_class}")
    print(f"claim eligible: {eligible}")
    print(f"evidence blob: {latest_entry['evidence_sha256']}")
    print(f"artifact: {out}")
    if args.require_capability_evidence and not eligible:
        print(
            "[frontier-gap] capability evidence rejected: "
            + "; ".join(report["eligibility_reasons"]),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
