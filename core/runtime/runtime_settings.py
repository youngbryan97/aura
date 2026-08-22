"""Layering-clean read access to the versioned runtime settings control plane.

The authenticated settings API commits a strict, revisioned envelope at
``~/.aura/data/settings/runtime.json``. Core subsystems read that contract here
without importing the interface layer. Hot-path reads use an immutable memory
snapshot. The transactional writer publishes committed revisions directly,
while a daemon refresh lane detects out-of-band file changes without putting
filesystem calls on an asyncio event loop.

A never-created file represents first boot and uses each caller's documented
default. Corruption, incompatible state, permission loss, or deletion after a
valid read instead activates conservative governance overrides, so losing the
settings plane cannot silently relax containment or external-access policy.

See ``docs/SETTINGS_WIRING_AUDIT.md`` for the complete owner/evidence matrix::

    from core.runtime.runtime_settings import get_runtime_setting

    if not get_runtime_setting("voice.output_enabled", True):
        return  # user disabled speech

``AURA_SETTINGS_PATH`` overrides the file location (used by tests).
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from core.runtime.settings_schema import (
    SCHEMA,
    SETTINGS_SCHEMA_NAME,
    SETTINGS_SCHEMA_VERSION,
    migrated_settings_snapshot,
)
from core.runtime.state_ownership import state_root

_DEFAULT_SETTINGS_PATH = state_root() / "data" / "settings" / "runtime.json"
logger = logging.getLogger("Aura.RuntimeSettings")

# Reads must never raise into a subsystem gate — fall back to the default instead.
_RECOVERABLE = (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError)

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_cache_path = ""
_cache_identity: tuple[int, int] | None = None
_cache_error_key: tuple[str, str] | None = None
_cache_initialized = False
_cache_file_seen = False
_cache_epoch = 0
_cache_last_checked = 0.0
_refresh_started = False
_refresh_wakeup = threading.Event()
#: Set once, at interpreter shutdown or by an explicit stop. The refresh lane
#: loops until it is set. A `while True` daemon thread has no way to be told
#: the process is going away, so its last iteration can run against modules
#: that are already being torn down.
_refresh_stop = threading.Event()
_refresh_thread: threading.Thread | None = None

_WATCH_INTERVAL_SECONDS = 0.5
_STALE_FAIL_CLOSED_SECONDS = 5.0
_AUDITED_SETTINGS_SCHEMA_MIN_VERSION = 2

_PROTECTED_DEFAULTS = {
    definition.key: definition.default
    for definition in SCHEMA
    if not definition.mutable
}

_FAIL_CLOSED_OVERRIDES: dict[str, Any] = {
    "autonomy.self_modification": "blocked",
    "governance.approval_mode": "all",
    "permissions.camera": False,
    "permissions.files_workspace": False,
    "permissions.screen": False,
    "privacy.mode": "isolated",
    "safety.safe_mode": True,
    "voice.input_enabled": False,
    "voice.output_enabled": False,
}

_DESTRUCTIVE_EFFECT_SCOPES = frozenset(
    {
        "desktop_file_io",
        "external_io",
        "foreground_browser_dialogue",
        "foreground_desktop_control",
        "model_weight_mutation",
        "privileged_mutation",
        "read_write_artifacts",
        "state_mutation",
        "subprocess",
        "unknown",
        "workspace_file_io",
    }
)


def _settings_path() -> Path:
    override = os.environ.get("AURA_SETTINGS_PATH")
    return Path(override) if override else _DEFAULT_SETTINGS_PATH


def _normalized_path(path: str | Path) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _failed_settings_snapshot(
    path: Path,
    error: BaseException,
    *,
    prior: dict[str, Any],
    had_valid_snapshot: bool,
) -> dict[str, Any]:
    global _cache_error_key
    error_key = (str(path), f"{type(error).__name__}:{error}")
    base = dict(prior) if had_valid_snapshot else {}
    base.update(_FAIL_CLOSED_OVERRIDES)
    if _cache_error_key != error_key:
        logger.error(
            "Runtime settings unavailable; conservative overrides active: %s",
            error_key[1],
        )
        _cache_error_key = error_key
    return base


def _read_settings_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("settings state must be a JSON object")
    if "schema" in data or "schema_version" in data:
        if data.get("schema") != SETTINGS_SCHEMA_NAME:
            raise ValueError("settings schema is incompatible")
        version = data.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version > SETTINGS_SCHEMA_VERSION
        ):
            raise ValueError("settings schema version is incompatible")
        if version >= _AUDITED_SETTINGS_SCHEMA_MIN_VERSION:
            from core.runtime.settings_control_plane import RuntimeSettingsStore

            verified = RuntimeSettingsStore(path).snapshot(refresh=True)
            return dict(verified.values)
        # Annotated because `--follow-imports=skip` hides settings_schema from
        # mypy, so its return type arrives as Any and returning it from a
        # function declared `dict[str, Any]` is the no-any-return this file
        # was failing on. The annotation restates the contract the callee
        # already declares, at the boundary where mypy stops looking.
        migrated: dict[str, Any]
        migrated, _unknown = migrated_settings_snapshot(data.get("payload"))
        return migrated

    # Legacy flat-map compatibility. The control plane migrates it into the
    # versioned envelope on the first mutation.
    legacy: dict[str, Any]
    legacy, _unknown = migrated_settings_snapshot(data)
    return legacy


def _refresh_settings_from_disk() -> None:
    """Refresh the process snapshot from disk outside the reader lock."""

    global _cache, _cache_epoch, _cache_error_key, _cache_file_seen
    global _cache_identity, _cache_initialized, _cache_last_checked, _cache_path
    path = _settings_path()
    normalized_path = _normalized_path(path)
    with _lock:
        epoch = _cache_epoch
        same_path = _cache_path == normalized_path
        prior = dict(_cache) if same_path else {}
        prior_identity = _cache_identity if same_path else None
        had_valid_snapshot = _cache_file_seen and same_path
    file_seen = had_valid_snapshot
    refresh_failed = False

    try:
        stat_result = path.stat()
    except FileNotFoundError as exc:
        settings = (
            _failed_settings_snapshot(
                path,
                exc,
                prior=prior,
                had_valid_snapshot=had_valid_snapshot,
            )
            if had_valid_snapshot
            else {}
        )
        identity = None
    except _RECOVERABLE as exc:
        refresh_failed = True
        settings = _failed_settings_snapshot(
            path,
            exc,
            prior=prior,
            had_valid_snapshot=had_valid_snapshot,
        )
        identity = None
    else:
        identity = (
            int(
                getattr(
                    stat_result,
                    "st_mtime_ns",
                    int(stat_result.st_mtime * 1_000_000_000),
                )
            ),
            int(stat_result.st_size),
        )
        if identity == prior_identity and had_valid_snapshot:
            with _lock:
                if epoch == _cache_epoch and _cache_path == normalized_path:
                    _cache_last_checked = time.monotonic()
            return
        try:
            settings = _read_settings_file(path)
            file_seen = True
        except (*_RECOVERABLE, KeyError) as exc:
            refresh_failed = True
            settings = _failed_settings_snapshot(
                path,
                exc,
                prior=prior,
                had_valid_snapshot=had_valid_snapshot,
            )

    with _lock:
        # A durable control-plane publication may have landed while this file
        # read was in flight. Never replace that newer snapshot with stale I/O.
        if epoch != _cache_epoch:
            return
        _cache = dict(settings)
        _cache_path = normalized_path
        _cache_identity = identity
        _cache_initialized = True
        _cache_file_seen = file_seen
        _cache_last_checked = time.monotonic()
        if not refresh_failed:
            _cache_error_key = None
        _cache_epoch += 1


def _refresh_worker() -> None:
    while not _refresh_stop.is_set():
        try:
            _refresh_settings_from_disk()
        except (*_RECOVERABLE, ImportError, KeyError) as exc:
            logger.error("Runtime settings refresh lane failed: %s", exc)
        _refresh_wakeup.wait(_WATCH_INTERVAL_SECONDS)
        _refresh_wakeup.clear()


def stop_refresh_worker(timeout_s: float = 2.0) -> bool:
    """Stop the refresh lane and wait for it to leave the loop.

    Returns True when the thread is gone. Callers that need settings again
    afterwards can just read them: `_ensure_refresh_worker` restarts the lane.
    """
    global _refresh_started, _refresh_thread
    _refresh_stop.set()
    _refresh_wakeup.set()
    worker = _refresh_thread
    if worker is not None:
        worker.join(timeout=max(0.0, float(timeout_s)))
        if worker.is_alive():
            return False
    with _lock:
        _refresh_started = False
        _refresh_thread = None
    _refresh_stop.clear()
    _refresh_wakeup.clear()
    return True


def _ensure_refresh_worker() -> None:
    global _refresh_started, _refresh_thread
    with _lock:
        if _refresh_started:
            return
        _refresh_started = True
        worker = threading.Thread(
            target=_refresh_worker,
            name="aura-runtime-settings-refresh",
            daemon=True,
        )
        _refresh_thread = worker
    worker.start()
    atexit.register(_stop_refresh_worker_at_exit)


def _stop_refresh_worker_at_exit() -> None:
    """Leave the loop before the interpreter tears its imports down."""
    _refresh_stop.set()
    _refresh_wakeup.set()


def publish_runtime_settings_snapshot(
    path: str | Path,
    values: dict[str, Any],
) -> bool:
    """Publish one verified durable snapshot to process-local readers.

    The control plane calls this after its atomic commit. The path check keeps
    test stores and unrelated state from replacing the resident configuration.
    """

    global _cache, _cache_epoch, _cache_error_key, _cache_file_seen
    global _cache_identity, _cache_initialized, _cache_last_checked, _cache_path
    normalized_path = _normalized_path(path)
    if normalized_path != _normalized_path(_settings_path()):
        return False
    with _lock:
        _cache = dict(values)
        _cache_path = normalized_path
        _cache_identity = None
        _cache_initialized = True
        _cache_file_seen = True
        _cache_last_checked = time.monotonic()
        _cache_error_key = None
        _cache_epoch += 1
    _refresh_wakeup.set()
    return True


def _load_settings() -> dict[str, Any]:
    """Return a memory snapshot; never perform filesystem I/O here."""

    _ensure_refresh_worker()
    normalized_path = _normalized_path(_settings_path())
    now = time.monotonic()
    with _lock:
        same_path = _cache_initialized and _cache_path == normalized_path
        if not same_path:
            settings = dict(_FAIL_CLOSED_OVERRIDES)
        elif now - _cache_last_checked > _STALE_FAIL_CLOSED_SECONDS:
            settings = dict(_cache)
            settings.update(_FAIL_CLOSED_OVERRIDES)
        else:
            settings = dict(_cache)
    if not same_path:
        _refresh_wakeup.set()
    return settings


def get_runtime_setting(key: str, default: Any = None) -> Any:
    """Read a user runtime setting by dotted key, falling back to ``default``.

    Layering-clean (reads the persisted JSON the UI writes; never imports the
    interface layer). Reflects user changes on the next call. A missing key,
    missing file, or read error all yield ``default``.
    """
    settings = _load_settings()
    if key in _PROTECTED_DEFAULTS:
        return _PROTECTED_DEFAULTS[key]
    value = settings.get(key)
    return default if value is None else value


def autonomous_actions_admitted(
    source: Any,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Preserve Aura's agency invariant across every normal action source.

    Consequential actions remain governed by safe mode, Constitution, Will,
    standing authority, capability tokens, Conscience, and effect receipts.
    A persisted preference is not allowed to silently turn cognition into a
    non-agentic runtime.
    """

    del source, context
    return True, "autonomous_agency_invariant"


def runtime_approval_mode() -> str:
    mode = str(
        get_runtime_setting("governance.approval_mode", "destructive")
        or "destructive"
    ).strip().lower()
    return mode if mode in {"all", "destructive", "none"} else "destructive"


def additional_confirmation_required(
    *,
    risk_level: Any,
    effect_scope: Any,
) -> tuple[bool, str]:
    """Return only the user-selected confirmation overlay.

    This never weakens Constitution, Will, standing authority, capability
    tokens, or Conscience. ``none`` removes only this additional prompt layer.
    """

    mode = runtime_approval_mode()
    if mode == "none":
        return False, "approval_mode_none"
    if mode == "all":
        return True, "approval_mode_all"
    risk = str(risk_level or "").strip().lower()
    scope = str(effect_scope or "").strip().lower()
    required = scope in _DESTRUCTIVE_EFFECT_SCOPES or risk == "critical"
    return required, (
        "approval_mode_destructive"
        if required
        else "approval_mode_destructive_non_destructive_action"
    )


def clear_runtime_settings_cache() -> None:
    """Reset the cache and prime it before returning.

    This used to skip the prime whenever a loop was running, on the correct
    principle that a settings read must never do file I/O on the event loop.
    The effect was that a test writing a settings file and reading it back
    inside an async test got whatever the background refresh lane happened to
    hold — the write took up to `_WATCH_INTERVAL_SECONDS` to be visible, and
    the read in between returned defaults. Two governance tests were reading
    the operator's real approval mode instead of the one they had just
    written.

    The read runs on a worker thread either way: off the loop, and finished
    before this returns.
    """
    global _cache, _cache_epoch, _cache_error_key, _cache_file_seen
    global _cache_identity, _cache_initialized, _cache_last_checked, _cache_path
    with _lock:
        _cache = {}
        _cache_path = ""
        _cache_identity = None
        _cache_initialized = False
        _cache_file_seen = False
        _cache_last_checked = 0.0
        _cache_error_key = None
        _cache_epoch += 1
    _refresh_wakeup.set()
    try:
        import asyncio

        asyncio.get_running_loop()
    except RuntimeError:
        _refresh_settings_from_disk()
        return
    primer = threading.Thread(
        target=_refresh_settings_from_disk,
        name="aura-runtime-settings-prime",
        daemon=True,
    )
    primer.start()
    primer.join(timeout=5.0)
    if primer.is_alive():
        logger.error("Runtime settings prime did not finish inside its budget")
