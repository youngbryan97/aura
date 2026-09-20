"""Immutable identities for a resident unified-recurrence campaign."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

SOURCE_GIT_SCHEMA: Final = "aura.unified_intrinsic.source_git.v1"
SOURCE_MANIFEST_SCHEMA: Final = "aura.unified_intrinsic.source_manifest.v1"
MODEL_MANIFEST_SCHEMA: Final = "aura.unified_intrinsic.model_manifest.v1"
RUNTIME_SCHEMA: Final = "aura.unified_intrinsic.runtime.v1"
CAMPAIGN_BINDING_SCHEMA: Final = "aura.unified_intrinsic.campaign_binding.v1"

class UnifiedResidentIdentityError(RuntimeError):
    """A campaign input cannot be bound to stable bytes."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def campaign_checkpoint_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the immutable campaign identity into every durable checkpoint."""

    source = config.get("source")
    source_git = source.get("git") if isinstance(source, Mapping) else None
    source_manifest = source.get("manifest") if isinstance(source, Mapping) else None
    model = config.get("model")
    runtime = config.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (source_git, source_manifest, model, runtime)
    ):
        raise UnifiedResidentIdentityError("campaign_binding_inputs_invalid")
    body = {
        "schema": CAMPAIGN_BINDING_SCHEMA,
        "campaign_id": config.get("campaign_id"),
        "campaign_config_sha256": config.get("config_sha256"),
        "source_commit": source_git.get("commit"),
        "source_tree": source_git.get("tree"),
        "source_manifest_sha256": source_manifest.get("manifest_sha256"),
        "model_manifest_sha256": model.get("manifest_sha256"),
        "runtime_identity_sha256": runtime.get("identity_sha256"),
        "dataset_identity_sha256": config.get("dataset", {}).get("identity_sha256"),
        "tokenizer_identity_sha256": config.get("tokenizer", {}).get(
            "identity_sha256"
        ),
        "tokenized_dataset_identity_sha256": config.get(
            "tokenized_dataset", {}
        ).get("identity_sha256"),
        "training_profile_sha256": canonical_sha256(
            {
                "profile": config.get("profile"),
                "training": config.get("training"),
                "training_args": config.get("training_args"),
            }
        ),
    }
    required = tuple(value for key, value in body.items() if key != "schema")
    if any(not isinstance(value, str) or not value for value in required):
        raise UnifiedResidentIdentityError("campaign_binding_inputs_invalid")
    return {**body, "binding_sha256": canonical_sha256(body)}


def _stable_file(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise UnifiedResidentIdentityError(f"identity_symlink_rejected:{path}")
    try:
        before = path.stat()
    except OSError as exc:
        raise UnifiedResidentIdentityError(f"identity_file_unavailable:{path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise UnifiedResidentIdentityError(f"identity_file_not_regular:{path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise UnifiedResidentIdentityError(f"identity_file_unreadable:{path}") from exc
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if tuple(getattr(before, field) for field in fields) != tuple(
        getattr(after, field) for field in fields
    ):
        raise UnifiedResidentIdentityError(f"identity_file_changed:{path}")
    label = path.name if root is None else path.relative_to(root).as_posix()
    return {
        "path": label,
        "size_bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        # Identity reads only, and the daemon makes a read take twenty
        # seconds on this checkout against a thirty-second bound.
        ["git", "-C", str(root), "-c", "core.fsmonitor=false", *arguments],
        capture_output=True,
        check=False,
        timeout=30.0,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise UnifiedResidentIdentityError(
            f"source_git_identity_unavailable:{arguments[0]}:{detail}"
        )
    return result.stdout


def build_source_git_identity(root: Path, *, source_commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    commit = str(source_commit).strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise UnifiedResidentIdentityError("source_commit_invalid")
    top = Path(
        _git(root, "rev-parse", "--show-toplevel")
        .decode("utf-8", errors="strict")
        .strip()
    ).resolve(strict=True)
    observed_commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip().lower()
    observed_tree = (
        _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip().lower()
    )
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    visible_paths = _git(
        root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    )
    for encoded in visible_paths.split(b"\0"):
        if not encoded:
            continue
        try:
            relative = Path(encoded.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise UnifiedResidentIdentityError(
                "source_git_path_not_utf8"
            ) from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise UnifiedResidentIdentityError("source_git_path_invalid")
        if (root / relative).is_symlink():
            raise UnifiedResidentIdentityError(
                f"identity_symlink_rejected:{relative.as_posix()}"
            )
    if top != root:
        raise UnifiedResidentIdentityError("source_git_root_mismatch")
    if observed_commit != commit:
        raise UnifiedResidentIdentityError("source_git_commit_mismatch")
    if branch != "HEAD":
        raise UnifiedResidentIdentityError("source_capsule_not_detached")
    if status:
        raise UnifiedResidentIdentityError("source_capsule_dirty")
    body = {
        "schema": SOURCE_GIT_SCHEMA,
        "root": str(root),
        "commit": observed_commit,
        "tree": observed_tree,
        "branch": "DETACHED",
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def verify_source_git_identity(root: Path, expected: Mapping[str, Any]) -> None:
    if not isinstance(expected, Mapping):
        raise UnifiedResidentIdentityError("source_git_identity_invalid")
    body = {key: value for key, value in expected.items() if key != "identity_sha256"}
    if (
        expected.get("schema") != SOURCE_GIT_SCHEMA
        or expected.get("identity_sha256") != canonical_sha256(body)
    ):
        raise UnifiedResidentIdentityError("source_git_identity_invalid")
    if build_source_git_identity(
        root,
        source_commit=str(expected.get("commit") or ""),
    ) != dict(expected):
        raise UnifiedResidentIdentityError("source_git_identity_drift")


def _source_paths(root: Path) -> list[Path]:
    payload = _git(root, "ls-files", "-z")
    result: list[Path] = []
    for encoded in payload.split(b"\0"):
        if not encoded:
            continue
        try:
            relative = Path(encoded.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise UnifiedResidentIdentityError(
                "source_manifest_path_not_utf8"
            ) from exc
        if relative.is_absolute() or ".." in relative.parts:
            raise UnifiedResidentIdentityError("source_manifest_path_invalid")
        path = root / relative
        if not path.exists() and not path.is_symlink():
            raise UnifiedResidentIdentityError(f"source_file_missing:{relative}")
        result.append(relative)
    if not result:
        raise UnifiedResidentIdentityError("source_manifest_empty")
    return sorted(result, key=lambda path: os.fsencode(path.as_posix()))


def build_source_manifest(root: Path, *, source_commit: str) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files = [_stable_file(root / relative, root=root) for relative in _source_paths(root)]
    body = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source_commit": source_commit,
        "file_count": len(files),
        "files": files,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def verify_source_manifest(root: Path, expected: Mapping[str, Any]) -> None:
    root = root.expanduser().resolve(strict=True)
    if not isinstance(expected, Mapping):
        raise UnifiedResidentIdentityError("source_manifest_invalid")
    body = {key: value for key, value in expected.items() if key != "manifest_sha256"}
    if (
        expected.get("schema") != SOURCE_MANIFEST_SCHEMA
        or expected.get("manifest_sha256") != canonical_sha256(body)
        or expected.get("file_count") != len(expected.get("files", ()))
    ):
        raise UnifiedResidentIdentityError("source_manifest_invalid")
    observed_paths = _source_paths(root)
    recorded = [Path(str(row.get("path") or "")) for row in expected["files"]]
    if recorded != observed_paths:
        raise UnifiedResidentIdentityError("source_manifest_file_set_drift")
    for relative, expected_row in zip(observed_paths, expected["files"], strict=True):
        if relative.is_absolute() or ".." in relative.parts:
            raise UnifiedResidentIdentityError("source_manifest_path_invalid")
        if _stable_file(root / relative, root=root) != expected_row:
            raise UnifiedResidentIdentityError(f"source_manifest_file_drift:{relative}")


def build_model_manifest(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    files: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: os.fsencode(item.name)):
        if path.is_symlink():
            raise UnifiedResidentIdentityError(f"identity_symlink_rejected:{path}")
        if path.is_file():
            files.append(_stable_file(path, root=root))
        elif not path.is_dir():
            raise UnifiedResidentIdentityError(f"model_artifact_not_regular:{path}")
    names = {row["path"] for row in files}
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    weight_names = sorted(
        name
        for name in names
        if PurePosixPath(name).suffix in {".safetensors", ".npz", ".gguf"}
    )
    if not required.issubset(names) or not weight_names:
        raise UnifiedResidentIdentityError("model_manifest_incomplete")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise UnifiedResidentIdentityError("model_config_invalid") from exc
    if not isinstance(config, dict):
        raise UnifiedResidentIdentityError("model_config_invalid")
    text_config = config.get("text_config")
    shape = text_config if isinstance(text_config, dict) else config
    dimensions = {
        "model_type": config.get("model_type"),
        "num_hidden_layers": shape.get("num_hidden_layers"),
        "hidden_size": shape.get("hidden_size"),
        "vocab_size": shape.get("vocab_size"),
        "quantization": config.get("quantization") or config.get("quantization_config"),
    }
    if any(
        type(dimensions[name]) is not int or int(dimensions[name]) < 1
        for name in ("num_hidden_layers", "hidden_size", "vocab_size")
    ):
        raise UnifiedResidentIdentityError("model_dimensions_invalid")
    shard_index: dict[str, Any] | None = None
    index_path = root / "model.safetensors.index.json"
    if len(weight_names) > 1:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeError) as exc:
            raise UnifiedResidentIdentityError("model_shard_index_invalid") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        referenced = (
            sorted(set(weight_map.values()))
            if isinstance(weight_map, dict)
            and weight_map
            and all(isinstance(value, str) for value in weight_map.values())
            else []
        )
        if referenced != weight_names:
            raise UnifiedResidentIdentityError("model_shard_inventory_mismatch")
        shard_index = {
            "path": index_path.name,
            "tensor_count": len(weight_map),
            "shards": referenced,
        }
    body = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "root": str(root),
        "file_count": len(files),
        "files": files,
        "weights": weight_names,
        "shard_index": shard_index,
        "dimensions": dimensions,
    }
    return {**body, "manifest_sha256": canonical_sha256(body)}


def verify_model_manifest(expected: Mapping[str, Any]) -> None:
    if not isinstance(expected, Mapping):
        raise UnifiedResidentIdentityError("model_manifest_invalid")
    body = {key: value for key, value in expected.items() if key != "manifest_sha256"}
    if (
        expected.get("schema") != MODEL_MANIFEST_SCHEMA
        or expected.get("manifest_sha256") != canonical_sha256(body)
        or expected.get("file_count") != len(expected.get("files", ()))
    ):
        raise UnifiedResidentIdentityError("model_manifest_invalid")
    root = Path(str(expected.get("root") or "")).expanduser().resolve(strict=True)
    observed = build_model_manifest(root)
    if observed != dict(expected):
        raise UnifiedResidentIdentityError("model_manifest_drift")


def trainer_model_identity_from_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the trainer's historical identity shape from the full manifest."""

    if not isinstance(expected, Mapping):
        raise UnifiedResidentIdentityError("model_manifest_invalid")
    manifest_body = {
        key: value for key, value in expected.items() if key != "manifest_sha256"
    }
    files = expected.get("files")
    if (
        expected.get("schema") != MODEL_MANIFEST_SCHEMA
        or expected.get("manifest_sha256") != canonical_sha256(manifest_body)
        or not isinstance(files, list)
        or expected.get("file_count") != len(files)
    ):
        raise UnifiedResidentIdentityError("model_manifest_invalid")
    rows: list[dict[str, Any]] = []
    for row in files:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("size_bytes"), int)
            or not _is_manifest_sha(row.get("sha256"))
        ):
            raise UnifiedResidentIdentityError("model_manifest_file_invalid")
        if row["path"] == "README.md":
            continue
        rows.append(
            {
                "name": row["path"],
                "size": row["size_bytes"],
                "sha256": row["sha256"],
            }
        )
    by_name = {row["name"]: row for row in rows}
    weight_names = expected.get("weights")
    if (
        not isinstance(weight_names, list)
        or not weight_names
        or any(
            not isinstance(name, str)
            or not name.endswith(".safetensors")
            or name not in by_name
            for name in weight_names
        )
        or "config.json" not in by_name
    ):
        raise UnifiedResidentIdentityError("trainer_model_manifest_incompatible")
    weight_rows = [by_name[name] for name in weight_names]
    behavior_rows = [
        row for row in rows if not row["name"].endswith(".safetensors")
    ]
    body = {
        "canonical_path": str(expected.get("root") or ""),
        "config_sha256": by_name["config.json"]["sha256"],
        "weights": weight_rows,
        "behavior_files": behavior_rows,
        "behavior_sha256": canonical_sha256(behavior_rows),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def _is_manifest_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def runtime_identity() -> dict[str, Any]:
    from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (
        runtime_environment_identity,
    )

    executable = Path(os.path.abspath(sys.executable))
    real_executable = executable.resolve(strict=True)
    dependencies: dict[str, Any] = {}
    for distribution_name in (
        "huggingface-hub",
        "psutil",
        "safetensors",
        "tokenizers",
        "transformers",
    ):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise UnifiedResidentIdentityError(
                f"runtime_dependency_missing:{distribution_name}"
            ) from exc
        rows: list[dict[str, Any]] = []
        for declared in distribution.files or ():
            path = Path(str(distribution.locate_file(declared)))
            if path.is_dir():
                continue
            row = _stable_file(path)
            row["path"] = str(declared)
            rows.append(row)
        rows.sort(key=lambda row: os.fsencode(row["path"]))
        if not rows:
            raise UnifiedResidentIdentityError(
                f"runtime_dependency_files_missing:{distribution_name}"
            )
        dependency_body = {
            "distribution": distribution_name,
            "version": distribution.version,
            "file_count": len(rows),
            "total_bytes": sum(int(row["size_bytes"]) for row in rows),
            "files": rows,
        }
        dependencies[distribution_name] = {
            **dependency_body,
            "tree_sha256": canonical_sha256(dependency_body),
        }
    import mlx.core as mx

    device = dict(mx.device_info())
    body = {
        "schema": RUNTIME_SCHEMA,
        "environment": runtime_environment_identity(),
        "additional_dependencies": dependencies,
        "platform": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "mlx_device": device,
        "behavior_environment": {
            name: os.environ.get(name)
            for name in (
                "HF_HUB_OFFLINE",
                "HF_HOME",
                "AURA_LANE_ADMISSION",
                "AURA_LANE_BUDGET_FRACTION",
                "AURA_LANE_BUDGET_GB",
                "AURA_LANE_EVICTION_SHIELD_S",
                "AURA_MODEL_LANE_COMPENSATION_TIMEOUT_S",
                "AURA_MODEL_LANE_EVICTION_TIMEOUT_S",
                "AURA_MODEL_LANE_OWNER_LEASE_TTL_S",
                "AURA_MODEL_LANE_RESERVATION_TTL_S",
                "AURA_MODEL_LANE_STATE_PATH",
                "MLX_METAL_DEBUG",
                "MLX_METAL_JIT",
                "PYTHONHASHSEED",
                "TOKENIZERS_PARALLELISM",
                "TRANSFORMERS_OFFLINE",
            )
        },
        "interpreter": {
            **_stable_file(real_executable),
            "executable": str(executable),
            "real_executable": str(real_executable),
            "sys_prefix": str(Path(sys.prefix).resolve(strict=True)),
            "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        },
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


__all__ = [
    "CAMPAIGN_BINDING_SCHEMA",
    "UnifiedResidentIdentityError",
    "build_model_manifest",
    "build_source_git_identity",
    "build_source_manifest",
    "campaign_checkpoint_binding",
    "canonical_bytes",
    "canonical_sha256",
    "runtime_identity",
    "trainer_model_identity_from_manifest",
    "verify_model_manifest",
    "verify_source_git_identity",
    "verify_source_manifest",
]
