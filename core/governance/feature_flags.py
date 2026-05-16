"""core/governance/feature_flags.py
====================================
Feature flag system for Aura runtime.

Provides a centralized, observable mechanism for enabling/disabling features
at runtime without code changes. Critical for safe deployment and rollback.

Flags can be:
- Set via environment variables (AURA_FLAG_<NAME>=1/0)
- Set via config file (~/.aura/feature_flags.json)
- Set programmatically at runtime

All flag changes are logged for audit trail.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.Governance.FeatureFlags")


# Default flag definitions with descriptions
_DEFAULT_FLAGS: Dict[str, Dict[str, Any]] = {
    "boring_mode_auto_enter": {
        "default": True,
        "description": "Auto-enter Boring Mode on critical substrate/model failures",
    },
    "will_strict_enforcement": {
        "default": True,
        "description": "Require Will approval for all gated actions (vs advisory-only)",
    },
    "incident_manager_enabled": {
        "default": True,
        "description": "Enable structured incident tracking",
    },
    "prometheus_metrics_enabled": {
        "default": True,
        "description": "Enable Prometheus /metrics/prometheus endpoint",
    },
    "workspace_jail_enabled": {
        "default": True,
        "description": "Enable path traversal prevention for file skills",
    },
    "substrate_nan_guard": {
        "default": True,
        "description": "Enable NaN/Inf detection and rollback in substrate ODE",
    },
    "exponential_backoff_restarts": {
        "default": True,
        "description": "Use exponential backoff for process restarts",
    },
    "stale_lock_reclamation": {
        "default": True,
        "description": "Auto-reclaim stale singleton locks from dead processes",
    },
    "structured_logging": {
        "default": True,
        "description": "Use JSON structured logging (vs plain text)",
    },
    "adapter_truth_gate": {
        "default": True,
        "description": "Return WorldResult instead of [] for adapter failures",
    },
    "phi_warmup_gate": {
        "default": True,
        "description": "Require 50+ transitions before trusting phi values",
    },
    "memory_dedup_on_write": {
        "default": True,
        "description": "Check for semantic duplicates before writing memory",
    },
    "initiative_overflow_tracking": {
        "default": True,
        "description": "Track and alert on initiative queue overflow",
    },
    "proactive_backoff": {
        "default": True,
        "description": "Back off proactive presence based on user response rate",
    },
    "thermal_throttle_integration": {
        "default": True,
        "description": "Integrate thermal pressure into inference decisions",
    },
}


class FeatureFlags:
    """Centralized feature flag manager."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._flags: Dict[str, bool] = {}
        self._overrides: Dict[str, bool] = {}
        self._config_path = config_path or (Path.home() / ".aura" / "feature_flags.json")
        self._change_log: list[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load flags from defaults, config file, and environment."""
        # 1. Defaults
        for name, spec in _DEFAULT_FLAGS.items():
            self._flags[name] = spec["default"]

        # 2. Config file
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text())
                if isinstance(data, dict):
                    for name, value in data.items():
                        if isinstance(value, bool):
                            self._flags[name] = value
                    logger.info(
                        "Loaded %d feature flags from %s",
                        len(data),
                        self._config_path,
                    )
            except Exception as e:
                logger.warning("Failed to load feature flags: %s", e)

        # 3. Environment overrides (highest priority)
        for name in list(self._flags.keys()):
            env_key = f"AURA_FLAG_{name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                self._flags[name] = env_val.lower() in ("1", "true", "yes")
                logger.info(
                    "Feature flag %s overridden by env %s=%s",
                    name, env_key, self._flags[name],
                )

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled."""
        if flag_name in self._overrides:
            return self._overrides[flag_name]
        return self._flags.get(flag_name, False)

    def set_flag(self, flag_name: str, value: bool, reason: str = "") -> None:
        """Set a flag at runtime (programmatic override)."""
        old = self.is_enabled(flag_name)
        self._overrides[flag_name] = value
        self._change_log.append({
            "flag": flag_name,
            "old": old,
            "new": value,
            "reason": reason,
            "timestamp": time.time(),
        })
        logger.info(
            "Feature flag %s changed: %s -> %s (reason: %s)",
            flag_name, old, value, reason or "programmatic",
        )

    def get_all(self) -> Dict[str, bool]:
        """Get all flag states."""
        result = dict(self._flags)
        result.update(self._overrides)
        return result

    def get_descriptions(self) -> Dict[str, str]:
        """Get flag descriptions."""
        return {
            name: spec["description"]
            for name, spec in _DEFAULT_FLAGS.items()
        }

    def save(self) -> None:
        """Persist current flag state to config file."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            current = self.get_all()
            self._config_path.write_text(
                json.dumps(current, indent=2, sort_keys=True) + "\n"
            )
            logger.info("Feature flags saved to %s", self._config_path)
        except Exception as e:
            logger.warning("Failed to save feature flags: %s", e)

    def get_change_log(self, n: int = 20) -> list[Dict[str, Any]]:
        """Get recent flag changes."""
        return self._change_log[-n:]


# Singleton
_flags_instance: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    global _flags_instance
    if _flags_instance is None:
        _flags_instance = FeatureFlags()
    return _flags_instance
