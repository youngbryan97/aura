"""Writable runtime workspace helpers for live environment adapters.

Environment adapters often need small sidecar files such as rc/config files,
trace handles, or driver state. Those files must not assume direct access to a
user home directory: headless runs, app sandboxes, and CI harnesses may only
permit writes inside Aura's runtime workspace or a temporary directory.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from core.runtime.errors import record_degradation

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("_", value.strip()).strip("._-")
    return cleaned or fallback


def environment_runtime_dir(environment_id: str, *, purpose: str = "runtime") -> Path:
    """Return a writable directory for one adapter's runtime sidecar files.

    ``AURA_ENV_RUNTIME_DIR`` can override the root for harnesses. Otherwise we
    use the canonical Aura data directory, which already falls back to
    ``.aura_runtime`` inside the project when the configured home is not
    writable. If even that fails, use the system temp directory and surface a
    degradation receipt instead of failing silently.
    """
    env_name = _safe_component(environment_id, fallback="environment")
    purpose_name = _safe_component(purpose, fallback="runtime")
    override = os.environ.get("AURA_ENV_RUNTIME_DIR")

    try:
        if override:
            root = Path(override).expanduser().resolve()
        else:
            from core.config import config

            root = config.paths.data_dir / "environment_runtime"
        target = root / env_name / purpose_name
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception as exc:
        record_degradation("environment_runtime_workspace", exc)
        fallback = Path(tempfile.gettempdir()) / "aura_environment_runtime" / env_name / purpose_name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def environment_runtime_file(environment_id: str, filename: str, *, purpose: str = "runtime") -> Path:
    """Return a writable sidecar file path without allowing path traversal."""
    safe_filename = _safe_component(Path(filename).name, fallback="sidecar")
    return environment_runtime_dir(environment_id, purpose=purpose) / safe_filename


__all__ = ["environment_runtime_dir", "environment_runtime_file"]
