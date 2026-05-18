"""Bounded cognitive patch runner.

This module is the narrow runtime surface for LLM-proposed optimizer patches.
It accepts structured patch envelopes, validates them, writes only to approved
project locations, and returns receipts that callers can persist or route
through additional governance.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Optimizer.CognitivePatchRunner")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_GENERATED_SKILL_DIR = Path("skills/generated")
_GENERATED_RECIPE_DIR = Path("skills/generated/recipes")
_ALLOWED_CONFIG_PREFIXES = ("config/", "configs/", ".aura/config/")
_SAFE_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
_MAX_CODE_BYTES = 128_000
_MAX_RECIPE_STEPS = 64
_BLOCKED_IMPORT_ROOTS = {
    "ctypes",
    "ftplib",
    "multiprocessing",
    "os",
    "paramiko",
    "pty",
    "requests",
    "shlex",
    "shutil",
    "socket",
    "subprocess",
    "telnetlib",
    "urllib",
}
_BLOCKED_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
_SENSITIVE_CONFIG_KEY_MARKERS = ("api_key", "password", "secret", "token")


class PatchValidationError(ValueError):
    """Raised when a patch envelope fails local safety validation."""


_RECOVERABLE_ERRORS = (
    FileExistsError,
    FileNotFoundError,
    IsADirectoryError,
    json.JSONDecodeError,
    OSError,
    PatchValidationError,
    PermissionError,
    SyntaxError,
    TypeError,
    UnicodeError,
    ValueError,
)


@dataclass(frozen=True)
class PatchReceipt:
    applied: bool
    patch_type: str
    reason: str
    path: str = ""
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "type": self.patch_type,
            "reason": self.reason,
            "path": self.path,
            "sha256": self.sha256,
            "timestamp": datetime.now(UTC).isoformat(),
        }


def run_cognitive_patch(patch: Mapping[str, Any], *, project_root: Path | None = None) -> dict[str, Any]:
    """Validate and apply a structured cognitive patch.

    Supported patch types:
      - ``skill_install`` with ``payload.name`` and either ``payload.code`` or
        ``payload.steps``.
      - ``config_update`` with ``payload.path`` and ``payload.updates``.
    """
    root = (project_root or _PROJECT_ROOT).resolve()
    try:
        if not isinstance(patch, Mapping):
            raise PatchValidationError("patch envelope must be a mapping")
        if patch.get("ok") is not True:
            reason = str(patch.get("error") or "patch envelope was not approved by its producer")
            return PatchReceipt(False, str(patch.get("type") or "unknown"), reason).to_dict()

        patch_type = _required_text(patch, "type")
        payload = patch.get("payload")
        if not isinstance(payload, Mapping):
            raise PatchValidationError("payload must be a mapping")

        if patch_type == "skill_install":
            return _apply_skill_install(payload, root).to_dict()
        if patch_type == "config_update":
            return _apply_config_update(payload, root).to_dict()
        return PatchReceipt(False, patch_type, "unsupported_patch_type").to_dict()
    except _RECOVERABLE_ERRORS as exc:
        record_degradation("cognitive_patch_runner", exc)
        logger.warning("Cognitive patch rejected: %s", exc)
        return PatchReceipt(False, str(patch.get("type") if isinstance(patch, Mapping) else "unknown"), str(exc)).to_dict()


def _apply_skill_install(payload: Mapping[str, Any], root: Path) -> PatchReceipt:
    name = _safe_artifact_name(_required_text(payload, "name"))
    overwrite = bool(payload.get("overwrite", False))
    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        rel_path = _GENERATED_SKILL_DIR / f"{name}.py"
        absolute_path = _resolve_inside(root, rel_path)
        if absolute_path.exists() and not overwrite:
            raise FileExistsError(f"{rel_path.as_posix()} already exists")
        _validate_skill_code(code, rel_path.as_posix())
        body = code.rstrip() + "\n"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(absolute_path, body)
        return PatchReceipt(
            True,
            "skill_install",
            "installed_python_skill",
            rel_path.as_posix(),
            _sha256_text(body),
        )

    steps = payload.get("steps")
    if isinstance(steps, list) and steps:
        rel_path = _GENERATED_RECIPE_DIR / f"{name}.json"
        absolute_path = _resolve_inside(root, rel_path)
        if absolute_path.exists() and not overwrite:
            raise FileExistsError(f"{rel_path.as_posix()} already exists")
        recipe = _validated_recipe_payload(name, steps, payload)
        body = json.dumps(recipe, sort_keys=True, indent=2) + "\n"
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(absolute_path, body)
        return PatchReceipt(
            True,
            "skill_install",
            "installed_step_recipe",
            rel_path.as_posix(),
            _sha256_text(body),
        )

    raise PatchValidationError("skill_install requires non-empty code or steps")


def _apply_config_update(payload: Mapping[str, Any], root: Path) -> PatchReceipt:
    rel_path = _safe_config_path(_required_text(payload, "path"))
    updates = payload.get("updates")
    if not isinstance(updates, Mapping) or not updates:
        raise PatchValidationError("config_update requires a non-empty updates mapping")
    _reject_sensitive_config_keys(updates)

    create = bool(payload.get("create", False))
    absolute_path = _resolve_inside(root, Path(rel_path))
    if not absolute_path.exists():
        if not create:
            raise FileNotFoundError(f"{rel_path} does not exist")
        current: dict[str, Any] = {}
    else:
        current_raw = absolute_path.read_text(encoding="utf-8")
        current = json.loads(current_raw) if current_raw.strip() else {}
        if not isinstance(current, dict):
            raise PatchValidationError("config file must contain a JSON object")

    merged = _deep_merge(current, dict(updates))
    body = json.dumps(merged, sort_keys=True, indent=2) + "\n"
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(absolute_path, body)
    return PatchReceipt(True, "config_update", "updated_json_config", rel_path, _sha256_text(body))


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PatchValidationError(f"{key} must be a non-empty string")
    return value.strip()


def _safe_artifact_name(name: str) -> str:
    normalized = name.strip().lower()
    if re.fullmatch(_SAFE_NAME_PATTERN, normalized) is None:
        raise PatchValidationError("name must be lowercase letters, digits, and underscores")
    return normalized


def _safe_config_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("/")
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise PatchValidationError("config path may not traverse outside the project")
    if not normalized.endswith(".json"):
        raise PatchValidationError("config_update only writes JSON files")
    if not normalized.startswith(_ALLOWED_CONFIG_PREFIXES):
        raise PatchValidationError("config_update path must be under config/, configs/, or .aura/config/")
    return normalized


def _resolve_inside(root: Path, rel_path: Path) -> Path:
    target = (root / rel_path).resolve()
    target.relative_to(root)
    return target


def _validate_skill_code(code: str, filename: str) -> None:
    if len(code.encode("utf-8")) > _MAX_CODE_BYTES:
        raise PatchValidationError("skill code exceeds maximum size")
    tree = ast.parse(code, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _validate_import(node)
        elif isinstance(node, ast.Call):
            _validate_call(node)


def _validate_import(node: ast.Import | ast.ImportFrom) -> None:
    if isinstance(node, ast.Import):
        imported_roots = {alias.name.split(".", 1)[0] for alias in node.names}
    else:
        imported_roots = {(node.module or "").split(".", 1)[0]}
    blocked = sorted(root for root in imported_roots if root in _BLOCKED_IMPORT_ROOTS)
    if blocked:
        raise PatchValidationError(f"blocked import: {', '.join(blocked)}")


def _validate_call(node: ast.Call) -> None:
    name = ""
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
        root = _attribute_root(node.func)
        if root in _BLOCKED_IMPORT_ROOTS:
            raise PatchValidationError(f"blocked call root: {root}")
    if name in _BLOCKED_CALLS:
        raise PatchValidationError(f"blocked call: {name}")


def _attribute_root(node: ast.Attribute) -> str:
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _validated_recipe_payload(name: str, steps: list[Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    if len(steps) > _MAX_RECIPE_STEPS:
        raise PatchValidationError("step recipe exceeds maximum length")
    validated_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if isinstance(step, str) and step.strip():
            validated_steps.append({"index": index, "instruction": step.strip()})
        elif isinstance(step, Mapping) and isinstance(step.get("instruction"), str) and step["instruction"].strip():
            validated_steps.append({"index": index, "instruction": step["instruction"].strip()})
        else:
            raise PatchValidationError("each recipe step must include an instruction")
    return {
        "name": name,
        "description": str(payload.get("description") or ""),
        "installed_at": datetime.now(UTC).isoformat(),
        "steps": validated_steps,
    }


def _reject_sensitive_config_keys(mapping: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in mapping.items():
        key_text = str(key)
        path = f"{prefix}.{key_text}" if prefix else key_text
        lowered = key_text.lower()
        if any(marker in lowered for marker in _SENSITIVE_CONFIG_KEY_MARKERS):
            raise PatchValidationError(f"sensitive config key requires a secret store: {path}")
        if isinstance(value, Mapping):
            _reject_sensitive_config_keys(value, path)
    json.dumps(mapping)


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
