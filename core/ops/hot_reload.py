"""core/ops/hot_reload.py
─────────────────────────
Live code reload engine for Aura's cognitive pipeline.

Reloads Python modules in topological (leaf → root) order so that
code changes take effect without restarting the kernel, LLM servers,
or dropping active state. Preserves:
  • Running event loop & asyncio tasks
  • ServiceContainer registrations (instances stay alive)
  • Active LLM server processes (llama-server / MLX worker)
  • Loaded model weights in GPU/unified memory
  • Conversation history & episodic memory

Important safety rule:
    The default "all" scope is a curated live-safe union of reload scopes.
    It is intentionally not "every loaded core module", because reloading
    runtime-owned infrastructure such as actor buses, state vault plumbing,
    supervision, or sensory task engines can partially tear down the session
    while leaving the process alive.

Usage:
    POST /api/system/hot-reload          → reload all live-safe scopes
    POST /api/system/hot-reload?scope=X  → reload only scope X
"""
from __future__ import annotations
from core.runtime.errors import record_degradation


import importlib
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("Aura.HotReload")

# ── Scopes ─────────────────────────────────────────────────────
# Each scope maps to a set of module prefixes that will be reloaded.
# Scopes are intentionally conservative — we never reload the kernel
# itself, the running server, or the ServiceContainer.

RELOAD_SCOPES: Dict[str, List[str]] = {
    "phases": [
        "core.phases.",
    ],
    "skills": [
        "core.skills.",
    ],
    "consciousness": [
        "core.consciousness.",
    ],
    "llm": [
        "core.brain.llm.context_assembler",
        "core.brain.llm.inference_gate",
        "core.brain.llm.local_server_client",
        "core.brain.llm.model_registry",
    ],
    "affect": [
        "core.affect.",
    ],
    "memory": [
        "core.memory.",
    ],
    "identity": [
        "core.identity",
        "core.will",
    ],
    "resilience": [
        "core.resilience.",
        "core.ops.",
    ],
    "orchestrator_mixins": [
        "core.orchestrator.mixins.",
    ],
    "learning": [
        "core.learning.",
    ],
    "agency": [
        "core.agency.",
    ],
}

# "all" is intentionally a safe union rather than "reload every loaded core
# module". These scopes cover the parts users typically expect to live-refresh
# while leaving runtime-owned infrastructure alone.
LIVE_SAFE_ALL_SCOPES: tuple[str, ...] = (
    "phases",
    "skills",
    "consciousness",
    "llm",
    "affect",
    "memory",
    "identity",
    "orchestrator_mixins",
    "learning",
    "agency",
)

# Modules that must NEVER be reloaded — reloading them would
# destroy running state or break the process.
PROTECTED_MODULES: Set[str] = {
    "core.container",
    "core.config",
    "core.event_bus",
    "core.kernel.aura_kernel",
    "core.kernel.kernel_interface",
    "core.service_registration",
    "interface.server",
    "interface.websocket_manager",
    "interface.event_bridge",
    "interface.auth",
    "core.reliability_engine",
    "core.resilience.circuit_breaker",
    "core.resilience.metrics_exporter",
}

# Prefixes that own live subprocesses, transports, background tasks, or other
# non-idempotent runtime state. These can still be changed on disk, but must be
# picked up via a full Aura reboot instead of in-process hot reload.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "core.bus.",
    "core.executive.",
    "core.ops.",
    "core.providers.",
    "core.reaper",
    "core.senses.",
    "core.state.",
    "core.supervisor.",
)


@dataclass
class ReloadResult:
    """Outcome of a single hot-reload operation."""

    scope: str
    reloaded: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Dict[str, str]] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return len(self.failed) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "scope": self.scope,
            "reloaded_count": len(self.reloaded),
            "skipped_count": len(self.skipped),
            "failed_count": len(self.failed),
            "reloaded": self.reloaded,
            "failed": self.failed,
            "duration_ms": round(self.duration_ms, 1),
        }


class HotReloader:
    """Live code reload engine.

    Reloads modules matching a scope prefix in reverse-sorted order
    (deepest submodules first) to ensure dependent modules pick up
    changes from their imports.
    """

    def __init__(self, project_root: Optional[str] = None):
        self._project_root = Path(
            project_root
            or os.getenv("AURA_ROOT", "")
            or Path(__file__).resolve().parents[2]
        )
        self._reload_count = 0
        self._last_reload_at = 0.0
        self._last_result: Optional[ReloadResult] = None

    @property
    def last_result(self) -> Optional[ReloadResult]:
        return self._last_result

    def _is_protected(self, module_name: str) -> bool:
        """Check if a module is in the protected set."""
        for protected in PROTECTED_MODULES:
            if module_name == protected or module_name.startswith(protected + "."):
                return True
        for prefix in PROTECTED_PREFIXES:
            if module_name == prefix.rstrip(".") or module_name.startswith(prefix):
                return True
        return False

    def _collect_modules_for_scope(self, scope: str) -> List[str]:
        """Find all currently-loaded modules matching the given scope."""
        if scope not in RELOAD_SCOPES:
            return []

        prefixes = RELOAD_SCOPES[scope]
        matched: List[str] = []

        for module_name in list(sys.modules.keys()):
            for prefix in prefixes:
                if prefix.endswith("."):
                    # Prefix match (e.g., "core.phases." matches "core.phases.foo")
                    if module_name.startswith(prefix) or module_name == prefix.rstrip("."):
                        matched.append(module_name)
                        break
                else:
                    # Exact match
                    if module_name == prefix:
                        matched.append(module_name)
                        break

        # Sort deepest-first for safe reload order
        matched.sort(key=lambda m: m.count("."), reverse=True)
        return matched

    def _collect_live_safe_all_modules(self) -> List[str]:
        """Collect the curated live-safe union used by the default `all` scope."""
        matched: Set[str] = set()
        for scope in LIVE_SAFE_ALL_SCOPES:
            matched.update(self._collect_modules_for_scope(scope))
        return sorted(
            (name for name in matched if not self._is_protected(name)),
            key=lambda m: m.count("."),
            reverse=True,
        )

    def reload_scope(self, scope: str) -> ReloadResult:
        """Reload all modules in a specific scope."""
        start = time.monotonic()
        result = ReloadResult(scope=scope)

        if scope == "all":
            modules = self._collect_live_safe_all_modules()
        else:
            modules = self._collect_modules_for_scope(scope)

        if not modules:
            result.duration_ms = (time.monotonic() - start) * 1000
            logger.info("♻️ HotReload[%s]: No loaded modules matched.", scope)
            self._last_result = result
            return result

        logger.info(
            "♻️ HotReload[%s]: Reloading %d module(s)...",
            scope,
            len(modules),
        )

        for module_name in modules:
            if self._is_protected(module_name):
                result.skipped.append(module_name)
                continue

            module = sys.modules.get(module_name)
            if module is None:
                result.skipped.append(module_name)
                continue

            try:
                importlib.reload(module)
                result.reloaded.append(module_name)
            except Exception as exc:
                record_degradation('hot_reload', exc)
                tb = traceback.format_exc()
                logger.error(
                    "♻️ HotReload: Failed to reload %s: %s",
                    module_name,
                    exc,
                )
                result.failed.append({
                    "module": module_name,
                    "error": str(exc),
                    "traceback": tb[-500:],
                })

        result.duration_ms = (time.monotonic() - start) * 1000
        self._reload_count += 1
        self._last_reload_at = time.time()
        self._last_result = result

        status = "✅" if result.ok else "⚠️"
        logger.info(
            "%s HotReload[%s]: %d reloaded, %d skipped, %d failed (%.1fms)",
            status,
            scope,
            len(result.reloaded),
            len(result.skipped),
            len(result.failed),
            result.duration_ms,
        )

        return result

    def reload_all(self) -> ReloadResult:
        """Reload all core modules (full refresh)."""
        return self.reload_scope("all")

    def reload_file(self, filepath: str) -> ReloadResult:
        """Reload the module corresponding to a specific file path.

        Useful for IDE integrations or file-watcher hooks.
        """
        start = time.monotonic()
        result = ReloadResult(scope=f"file:{Path(filepath).name}")

        try:
            abs_path = Path(filepath).resolve()
            rel_path = abs_path.relative_to(self._project_root)
        except (ValueError, RuntimeError):
            result.failed.append({
                "module": filepath,
                "error": f"File is not under project root: {self._project_root}",
                "traceback": "",
            })
            result.duration_ms = (time.monotonic() - start) * 1000
            self._last_result = result
            return result

        # Convert path to module name: core/phases/foo.py → core.phases.foo
        parts = list(rel_path.parts)
        if parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts and parts[-1] == "__init__":
            parts.pop()
        module_name = ".".join(parts)

        if self._is_protected(module_name):
            result.skipped.append(module_name)
            result.duration_ms = (time.monotonic() - start) * 1000
            self._last_result = result
            return result

        module = sys.modules.get(module_name)
        if module is None:
            result.skipped.append(module_name)
            result.duration_ms = (time.monotonic() - start) * 1000
            self._last_result = result
            return result

        try:
            importlib.reload(module)
            result.reloaded.append(module_name)
        except Exception as exc:
            record_degradation('hot_reload', exc)
            tb = traceback.format_exc()
            result.failed.append({
                "module": module_name,
                "error": str(exc),
                "traceback": tb[-500:],
            })

        result.duration_ms = (time.monotonic() - start) * 1000
        self._reload_count += 1
        self._last_reload_at = time.time()
        self._last_result = result
        return result

    def get_status(self) -> Dict[str, Any]:
        """Return the current state of the hot-reload engine."""
        return {
            "reload_count": self._reload_count,
            "last_reload_at": self._last_reload_at,
            "last_result": self._last_result.to_dict() if self._last_result else None,
            "available_scopes": list(RELOAD_SCOPES.keys()) + ["all"],
            "live_safe_all_scopes": list(LIVE_SAFE_ALL_SCOPES),
            "protected_modules": sorted(PROTECTED_MODULES),
            "protected_prefixes": sorted(PROTECTED_PREFIXES),
        }


# ── Singleton ──────────────────────────────────────────────────

_instance: Optional[HotReloader] = None


def get_hot_reloader() -> HotReloader:
    """Get or create the singleton HotReloader instance."""
    global _instance
    if _instance is None:
        _instance = HotReloader()
    return _instance
