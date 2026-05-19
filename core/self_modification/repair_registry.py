"""Runtime repair registry for autonomous self-modification evidence."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation

_REGISTRY_SUBSYSTEM = "self_modification_repair_registry"
_REGISTRY_BOUNDARY_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)


def repair_registry_path(code_base: Path | None = None) -> Path:
    override = os.environ.get("AURA_PENDING_PATCH_REGISTRY")
    if override:
        return Path(override).expanduser()

    try:
        return config.paths.data_dir / "selfmod" / "pending_patch_registry.jsonl"
    except _REGISTRY_BOUNDARY_ERRORS as exc:
        record_degradation(_REGISTRY_SUBSYSTEM, exc)
        base = code_base.resolve() if code_base is not None else Path.cwd().resolve()
        return base / ".aura_runtime" / "data" / "selfmod" / "pending_patch_registry.jsonl"


def repair_entry_for_fix(fix: Any, test_results: dict[str, Any]) -> dict[str, Any]:
    fixed_code = str(getattr(fix, "fixed_code", ""))
    original_code = str(getattr(fix, "original_code", ""))
    return {
        "recorded_at": time.time(),
        "target_file": str(getattr(fix, "target_file", "")),
        "target_line": getattr(fix, "target_line", None),
        "explanation": str(getattr(fix, "explanation", "")),
        "hypothesis": str(getattr(fix, "hypothesis", "")),
        "confidence": str(getattr(fix, "confidence", "")),
        "original_code": original_code,
        "original_code_sha256": hashlib.sha256(original_code.encode("utf-8")).hexdigest(),
        "fixed_code": fixed_code,
        "fixed_code_sha256": hashlib.sha256(fixed_code.encode("utf-8")).hexdigest(),
        "test_results": test_results,
    }


def append_repair_entry(fix: Any, test_results: dict[str, Any], code_base: Path) -> Path:
    registry_path = repair_registry_path(code_base)
    entry = repair_entry_for_fix(fix, test_results)
    validate_repair_entry(entry, line_no=0)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, default=str))
        handle.write("\n")
    return registry_path


def validate_repair_entry(entry: Any, *, line_no: int) -> None:
    label = f"registry line {line_no}" if line_no else "registry entry"
    if not isinstance(entry, dict):
        raise ValueError(f"{label} is not an object")

    missing = {"target_file", "fixed_code", "fixed_code_sha256", "test_results"} - set(entry)
    if missing:
        raise ValueError(f"{label} missing {sorted(missing)}")

    target = Path(str(entry["target_file"]))
    if target.is_absolute() or ".." in target.parts:
        raise ValueError(f"{label} has unsafe target path")

    fixed_code = str(entry["fixed_code"])
    fixed_hash = hashlib.sha256(fixed_code.encode("utf-8")).hexdigest()
    if fixed_hash != entry["fixed_code_sha256"]:
        raise ValueError(f"{label} fixed_code_sha256 mismatch")

    if "original_code" in entry and "original_code_sha256" in entry:
        original_code = str(entry["original_code"])
        original_hash = hashlib.sha256(original_code.encode("utf-8")).hexdigest()
        if original_hash != entry["original_code_sha256"]:
            raise ValueError(f"{label} original_code_sha256 mismatch")


def validate_repair_registry(registry_file: Path) -> None:
    if not registry_file.exists():
        return

    with registry_file.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            validate_repair_entry(json.loads(line), line_no=line_no)
