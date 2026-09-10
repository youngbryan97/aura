"""Canonical schema for Aura's runtime settings control plane."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

SETTINGS_SCHEMA_NAME = "aura.runtime_settings"
SETTINGS_SCHEMA_VERSION = 3
SETTINGS_AUDIT_SCHEMA = "aura.runtime_settings.audit.v1"
SETTINGS_APPLICATION_AUDIT_SCHEMA = "aura.runtime_settings.application.v1"

_CLOCK_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")


@dataclass(frozen=True, slots=True)
class SettingDef:
    key: str
    label: str
    section: str
    default: Any
    explanation: str
    type_: str
    choices: tuple[str, ...] | None = None
    min_: float | None = None
    max_: float | None = None
    owner: str = "runtime"
    apply_mode: str = "read_at_gate"
    mutable: bool = True


SCHEMA: tuple[SettingDef, ...] = (
    SettingDef(
        "model.local_path",
        "Local model path",
        "models",
        "",
        "Where the primary local MLX lane loads weights from.",
        "string",
        owner="model_registry",
    ),
    SettingDef(
        "model.deep_path",
        "Deep model path",
        "models",
        "",
        "Where the deep local solver lane loads weights from.",
        "string",
        owner="model_registry",
    ),
    SettingDef(
        "voice.input_enabled",
        "Microphone input",
        "voice",
        True,
        "Permit microphone capture when the operating system grant is present.",
        "bool",
        owner="voice_input",
    ),
    SettingDef(
        "voice.output_enabled",
        "Spoken replies",
        "voice",
        True,
        "Permit speech synthesis and streamed spoken replies.",
        "bool",
        owner="voice_output",
    ),
    SettingDef(
        "voice.auto_listen",
        "Auto-listen",
        "voice",
        False,
        "Start the canonical desktop microphone lane automatically when input is permitted.",
        "bool",
        owner="voice_input",
        apply_mode="live_bridge",
    ),
    SettingDef(
        "voice.output_rate",
        "Speech rate",
        "voice",
        1.0,
        "Multiplier applied to the configured synthesis rate.",
        "float",
        min_=0.5,
        max_=2.0,
        owner="voice_output",
    ),
    SettingDef(
        "permissions.camera",
        "Camera access",
        "permissions",
        True,
        "Permit camera reads when the operating system grant is present.",
        "bool",
        owner="permission_gates",
    ),
    SettingDef(
        "permissions.screen",
        "Screen perception",
        "permissions",
        True,
        "Permit screen perception when the operating system grant is present.",
        "bool",
        owner="permission_gates",
    ),
    SettingDef(
        "permissions.files_workspace",
        "Workspace files",
        "permissions",
        True,
        "Permit file effects within Aura's governed workspace boundary.",
        "bool",
        owner="permission_gates",
    ),
    SettingDef(
        "autonomy.actions_enabled",
        "Autonomous agency",
        "autonomy",
        True,
        (
            "Protected agency invariant. Emergency containment uses safe mode "
            "without converting Aura's normal operating posture into a persistent kill switch."
        ),
        "bool",
        owner="authority_gateway",
        mutable=False,
    ),
    SettingDef(
        "autonomy.level",
        "Autonomous nature",
        "autonomy",
        "full",
        (
            "Protected agency invariant. Operators may constrain particular "
            "external effects or enter emergency safe mode, but cannot turn "
            "Aura's agency into a user-selected operating level."
        ),
        "enum",
        # Legacy values remain parseable so the control plane can audit and
        # reconcile old state. ``mutable=False`` rejects every non-default
        # mutation before commit.
        choices=("paused", "minimal", "balanced", "full"),
        owner="agency_invariant",
        mutable=False,
    ),
    SettingDef(
        "autonomy.proactive_messaging",
        "Proactive messaging",
        "autonomy",
        "minimal",
        "Select how often Aura may initiate a conversation.",
        "enum",
        choices=("never", "minimal", "balanced", "frequent"),
        owner="proactive_communication",
    ),
    SettingDef(
        "autonomy.self_modification",
        "Self-modification",
        "autonomy",
        "staged",
        "Select whether structural self-modification is blocked, staged, or open to governed promotion.",
        "enum",
        choices=("blocked", "staged", "open"),
        owner="growth_ladder",
    ),
    SettingDef(
        "governance.approval_mode",
        "Require approval",
        "autonomy",
        "destructive",
        "Add explicit user confirmation for all actions, destructive actions, or no additional actions.",
        "enum",
        choices=("destructive", "all", "none"),
        owner="authority_gateway",
    ),
    SettingDef(
        "learning.auto_enrichment_enabled",
        "Auto-enrichment",
        "memory",
        True,
        "Permit background extraction of durable knowledge from completed conversations.",
        "bool",
        owner="knowledge_enrichment",
    ),
    SettingDef(
        "learning.reflection_enabled",
        "Reflection learning",
        "memory",
        True,
        (
            "Permit automatic conversation reflection and reflection-derived "
            "learning without disabling direct or autonomous internal thought."
        ),
        "bool",
        owner="reflection_learning",
    ),
    SettingDef(
        "memory.retention_days",
        "Retention (days)",
        "memory",
        365,
        "Recency horizon used by governed memory pruning.",
        "int",
        min_=7,
        max_=3650,
        owner="sovereign_pruner",
    ),
    SettingDef(
        "memory.review_window",
        "Review window (days)",
        "memory",
        30,
        "Narrative review horizon for memory consolidation.",
        "int",
        min_=1,
        max_=365,
        owner="memory_consolidation",
    ),
    SettingDef(
        "privacy.mode",
        "Privacy mode",
        "privacy",
        "standard",
        "Select the runtime's external-data and world-bridge posture.",
        "enum",
        choices=("standard", "private", "isolated"),
        owner="world_bridge",
    ),
    SettingDef(
        "safety.safe_mode",
        "Safe mode",
        "privacy",
        False,
        "Restrict self-directed mutation and outgoing world operations.",
        "bool",
        owner="safe_mode",
        apply_mode="live_bridge",
    ),
    SettingDef(
        "onboarding.completed",
        "First-run setup done",
        "dev",
        False,
        "Whether the first-run wizard has been finished on this machine.",
        "bool",
        owner="first_run_wizard",
    ),
    SettingDef(
        "dev.developer_mode",
        "Developer mode",
        "dev",
        False,
        "Expose authenticated developer diagnostics.",
        "bool",
        owner="developer_routes",
    ),
    SettingDef(
        "dev.diagnostics_enabled",
        "Boot diagnostics",
        "dev",
        True,
        "Run optional boot diagnostics in addition to required health checks.",
        "bool",
        owner="boot_diagnostics",
    ),
    SettingDef(
        "theme.mode",
        "Theme",
        "theme",
        "auto",
        "Select the desktop shell theme.",
        "enum",
        choices=("auto", "light", "dark", "high_contrast"),
        owner="desktop_shell",
        apply_mode="frontend_runtime",
    ),
    SettingDef(
        "theme.reduced_motion",
        "Reduced motion",
        "theme",
        False,
        "Reduce non-essential desktop-shell animation.",
        "bool",
        owner="desktop_shell",
        apply_mode="frontend_runtime",
    ),
    SettingDef(
        "notify.enabled",
        "Notifications",
        "notify",
        True,
        "Permit local operating-system notifications.",
        "bool",
        owner="desktop_notifications",
    ),
    SettingDef(
        "notify.quiet_hours_start",
        "Quiet hours start",
        "notify",
        "22:00",
        "Start of the local notification quiet window.",
        "string",
        owner="desktop_notifications",
    ),
    SettingDef(
        "notify.quiet_hours_end",
        "Quiet hours end",
        "notify",
        "08:00",
        "End of the local notification quiet window.",
        "string",
        owner="desktop_notifications",
    ),
)

SCHEMA_BY_KEY = {setting.key: setting for setting in SCHEMA}
DEFAULT_VALUES = {setting.key: setting.default for setting in SCHEMA}


def validate_setting_value(key: str, value: Any) -> Any:
    """Validate one JSON setting without lossy coercion or silent clamping."""

    try:
        definition = SCHEMA_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown_setting:{key}") from exc

    if definition.type_ == "bool":
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be a boolean")
        return value
    if definition.type_ == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{key} must be an integer")
        numeric = int(value)
        if definition.min_ is not None and numeric < int(definition.min_):
            raise ValueError(f"{key} must be >= {int(definition.min_)}")
        if definition.max_ is not None and numeric > int(definition.max_):
            raise ValueError(f"{key} must be <= {int(definition.max_)}")
        return numeric
    if definition.type_ == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key} must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{key} must be a finite number")
        if definition.min_ is not None and numeric < float(definition.min_):
            raise ValueError(f"{key} must be >= {float(definition.min_)}")
        if definition.max_ is not None and numeric > float(definition.max_):
            raise ValueError(f"{key} must be <= {float(definition.max_)}")
        return numeric
    if definition.type_ == "enum":
        if not isinstance(value, str):
            raise TypeError(f"{key} must be a string enum")
        if definition.choices and value not in definition.choices:
            choices = ",".join(definition.choices)
            raise ValueError(f"{key} must be one of: {choices}")
        return value
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    if len(value) > 4096:
        raise ValueError(f"{key} exceeds 4096 characters")
    if key in {"notify.quiet_hours_start", "notify.quiet_hours_end"} and not _CLOCK_RE.fullmatch(value):
        raise ValueError(f"{key} must use 24-hour HH:MM format")
    return value


def validate_settings_patch(changes: Any) -> dict[str, Any]:
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty object")
    validated: dict[str, Any] = {}
    for key, value in changes.items():
        normalized_key = str(key)
        normalized_value = validate_setting_value(normalized_key, value)
        definition = SCHEMA_BY_KEY[normalized_key]
        if not definition.mutable and normalized_value != definition.default:
            raise ValueError(f"protected_runtime_invariant:{normalized_key}")
        validated[normalized_key] = normalized_value
    return validated


def validated_settings_snapshot(values: Any) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Validate a stored partial map and return defaults plus known values."""

    if not isinstance(values, dict):
        raise TypeError("settings values must be an object")
    unknown = tuple(sorted(str(key) for key in values if str(key) not in SCHEMA_BY_KEY))
    validated = dict(DEFAULT_VALUES)
    for key, value in values.items():
        normalized = str(key)
        if normalized in SCHEMA_BY_KEY:
            validated[normalized] = validate_setting_value(normalized, value)
    return validated, unknown


def migrated_settings_snapshot(
    values: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize explicitly supported pre-v2 representations, then validate."""

    if not isinstance(values, dict):
        raise TypeError("settings values must be an object")
    migrated = dict(values)
    proactive = migrated.get("autonomy.proactive_messaging")
    if isinstance(proactive, bool):
        migrated["autonomy.proactive_messaging"] = (
            "minimal" if proactive else "never"
        )
    return validated_settings_snapshot(migrated)


__all__ = [
    "DEFAULT_VALUES",
    "SCHEMA",
    "SCHEMA_BY_KEY",
    "SETTINGS_APPLICATION_AUDIT_SCHEMA",
    "SETTINGS_AUDIT_SCHEMA",
    "SETTINGS_SCHEMA_NAME",
    "SETTINGS_SCHEMA_VERSION",
    "SettingDef",
    "validate_setting_value",
    "validate_settings_patch",
    "migrated_settings_snapshot",
    "validated_settings_snapshot",
]
