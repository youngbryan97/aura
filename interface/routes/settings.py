"""interface/routes/settings.py
─────────────────────────────────
Single source of truth for runtime-configurable settings. Every major
behavior has a visible setting, sane default, and explanation. The
settings panel binds 1:1 with this API.

Storage: ``~/.aura/data/settings/runtime.json`` (atomic write-tmp + rename).

Endpoints:
    GET    /api/settings           — full schema + current values
    PATCH  /api/settings           — partial update (per-key validation)
    POST   /api/settings/reset     — restore defaults for a section
    POST   /api/settings/auth/fresh — register a fresh user authorization
                                    (used by Conscience for destructive ops)
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse

from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Settings")

router = APIRouter(prefix="/settings", tags=["settings"])


_SETTINGS_DIR = Path.home() / ".aura" / "data" / "settings"
_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
_SETTINGS_PATH = _SETTINGS_DIR / "runtime.json"


# ─── schema ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SettingDef:
    key: str
    label: str
    section: str
    default: Any
    explanation: str
    type_: str  # "bool" | "int" | "float" | "string" | "enum"
    choices: Optional[Tuple[str, ...]] = None
    min_: Optional[float] = None
    max_: Optional[float] = None


SCHEMA: Tuple[SettingDef, ...] = (
    # ── Models ────────────────────────────────────────────────────────
    SettingDef("model.local_path", "Local model path", "models", "", "Where the local MLX worker loads weights from. Empty falls back to the runtime default.", "string"),
    SettingDef("model.deep_path", "Deep model path", "models", "", "Path for the heavy 72B-class lane.", "string"),
    SettingDef("model.cloud_fallback_enabled", "Enable cloud fallback", "models", False, "When local cortex is unavailable, may route requests to a configured cloud provider.", "bool"),

    # ── Voice / IO ────────────────────────────────────────────────────
    SettingDef("voice.input_enabled", "Voice input", "voice", True, "Microphone input is allowed when granted.", "bool"),
    SettingDef("voice.output_enabled", "Voice output", "voice", True, "Speech synthesis is allowed.", "bool"),
    SettingDef("voice.output_rate", "Speech rate", "voice", 1.0, "Multiplier for synthesis speed.", "float", min_=0.5, max_=2.0),

    # ── Permissions ──────────────────────────────────────────────────
    SettingDef("permissions.camera", "Camera access", "permissions", False, "Per-session camera access.", "bool"),
    SettingDef("permissions.screen", "Screen perception", "permissions", False, "Read pixels of an explicitly granted window.", "bool"),
    SettingDef("permissions.files_workspace", "Workspace files", "permissions", True, "Sandbox dir at ~/.aura/data/world/workspace.", "bool"),

    # ── Autonomy ─────────────────────────────────────────────────────
    SettingDef("autonomy.level", "Autonomy level", "autonomy", "balanced", "How freely Aura initiates actions on her own.", "enum", choices=("paused", "minimal", "balanced", "full")),
    SettingDef("autonomy.proactive_messaging", "Proactive messaging", "autonomy", "minimal", "How often Aura starts conversations on her own.", "enum", choices=("never", "minimal", "balanced", "frequent")),
    SettingDef("autonomy.self_modification", "Self-modification", "autonomy", "staged", "Whether structural self-modification is allowed and how.", "enum", choices=("blocked", "staged", "open")),

    # ── Memory ───────────────────────────────────────────────────────
    SettingDef("memory.retention_days", "Retention (days)", "memory", 365, "How long episodic memories stay before reaper consideration.", "int", min_=7, max_=3650),
    SettingDef("memory.review_window", "Review window (days)", "memory", 30, "Time window for narrative-arc consolidation.", "int", min_=1, max_=365),

    # ── Privacy / Safety ─────────────────────────────────────────────
    SettingDef("privacy.mode", "Privacy mode", "privacy", "standard", "Tightens telemetry, pauses external posting, narrows world bridge.", "enum", choices=("standard", "private", "isolated")),
    SettingDef("safety.safe_mode", "Safe mode", "privacy", False, "Disables all destructive primitives + outgoing world ops.", "bool"),

    # ── Developer / Diagnostics ──────────────────────────────────────
    SettingDef("dev.developer_mode", "Developer mode", "dev", False, "Exposes /api/trace, raw subsystem panels, and additional logs.", "bool"),
    SettingDef("dev.diagnostics_enabled", "Diagnostics on/off", "dev", True, "When on, Aura runs the boot self-diagnostic on startup.", "bool"),

    # ── Theme ────────────────────────────────────────────────────────
    SettingDef("theme.mode", "Theme", "theme", "auto", "Light, dark, or follow OS.", "enum", choices=("auto", "light", "dark", "high_contrast")),
    SettingDef("theme.reduced_motion", "Reduced motion", "theme", False, "Honor OS reduced-motion preference for animations.", "bool"),

    # ── Notifications ────────────────────────────────────────────────
    SettingDef("notify.enabled", "Notifications", "notify", True, "Allow Aura to send local OS notifications.", "bool"),
    SettingDef("notify.quiet_hours_start", "Quiet hours start", "notify", "22:00", "HH:MM at which proactive notifications go quiet.", "string"),
    SettingDef("notify.quiet_hours_end", "Quiet hours end", "notify", "08:00", "HH:MM when proactive notifications resume.", "string"),
)


# ─── store ─────────────────────────────────────────────────────────────────


class SettingsStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._data: Dict[str, Any] = {s.key: s.default for s in SCHEMA}
        self._load()
        self._subscribers: List[Callable[[str, Any, Any], None]] = []

    def _load(self) -> None:
        if not _SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._data.update({k: v for k, v in data.items() if k in self._data})
        except Exception as exc:
            record_degradation('settings', exc)
            logger.warning("settings load failed: %s", exc)

    def _save(self) -> None:
        tmp = _SETTINGS_PATH.with_suffix(".json.tmp")
        atomic_write_text(tmp, json.dumps(self._data, indent=2), encoding="utf-8")
        os.replace(tmp, _SETTINGS_PATH)

    # ── public ───────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def set(self, key: str, value: Any) -> Any:
        defn = self._defn(key)
        coerced = self._validate(defn, value)
        with self._lock:
            previous = self._data.get(key)
            self._data[key] = coerced
            self._save()
        for cb in self._subscribers:
            try:
                cb(key, previous, coerced)
            except Exception:
                pass  # no-op: intentional
        return coerced

    def reset_section(self, section: str) -> Dict[str, Any]:
        with self._lock:
            for s in SCHEMA:
                if s.section == section:
                    self._data[s.key] = s.default
            self._save()
            return self.all()

    def subscribe(self, cb: Callable[[str, Any, Any], None]) -> None:
        self._subscribers.append(cb)

    @staticmethod
    def _defn(key: str) -> SettingDef:
        for s in SCHEMA:
            if s.key == key:
                return s
        raise KeyError(key)

    @staticmethod
    def _validate(defn: SettingDef, value: Any) -> Any:
        if defn.type_ == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)
        if defn.type_ == "int":
            v = int(value)
            if defn.min_ is not None:
                v = max(int(defn.min_), v)
            if defn.max_ is not None:
                v = min(int(defn.max_), v)
            return v
        if defn.type_ == "float":
            v = float(value)
            if defn.min_ is not None:
                v = max(float(defn.min_), v)
            if defn.max_ is not None:
                v = min(float(defn.max_), v)
            return v
        if defn.type_ == "enum":
            v = str(value)
            if defn.choices and v not in defn.choices:
                raise ValueError(f"invalid_enum:{v}")
            return v
        return str(value)


_STORE: Optional[SettingsStore] = None


def get_settings() -> SettingsStore:
    global _STORE
    if _STORE is None:
        _STORE = SettingsStore()
    return _STORE


# ─── routes ────────────────────────────────────────────────────────────────


@router.get("")
async def get_all(_: None = Depends(_require_internal)) -> JSONResponse:
    store = get_settings()
    return JSONResponse({
        "schema": [
            {
                "key": s.key,
                "label": s.label,
                "section": s.section,
                "default": s.default,
                "explanation": s.explanation,
                "type": s.type_,
                "choices": list(s.choices) if s.choices else None,
                "min": s.min_,
                "max": s.max_,
            }
            for s in SCHEMA
        ],
        "values": store.all(),
    })


@router.patch("")
async def patch_settings(
    payload: Dict[str, Any] = Body(...),
    _: None = Depends(_require_internal),
) -> JSONResponse:
    store = get_settings()
    applied: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for k, v in payload.items():
        try:
            applied[k] = store.set(k, v)
        except Exception as exc:
            record_degradation('settings', exc)
            errors[k] = str(exc)
    return JSONResponse({"applied": applied, "errors": errors, "values": store.all()})


@router.post("/reset")
async def reset_section(
    payload: Dict[str, Any] = Body(...),
    _: None = Depends(_require_internal),
) -> JSONResponse:
    section = str(payload.get("section", ""))
    if not section:
        raise HTTPException(status_code=400, detail="section_required")
    store = get_settings()
    store.reset_section(section)
    return JSONResponse({"reset": section, "values": store.all()})


@router.post("/auth/fresh")
async def acknowledge_fresh_auth(_: None = Depends(_require_internal)) -> JSONResponse:
    """Tell the Conscience that the user just issued a fresh authorization.

    Called from the UI's destructive-action confirmation modal. The
    Conscience's destructive-op rules require a fresh auth within a
    rolling 60-second window.
    """
    try:
        from core.ethics.conscience import get_conscience
        get_conscience().acknowledge_user_authorization()
        return JSONResponse({"ok": True, "when": time.time()})
    except Exception as exc:
        record_degradation('settings', exc)
        raise HTTPException(status_code=500, detail=str(exc))


__all__ = ["router", "SettingsStore", "SCHEMA", "get_settings"]
