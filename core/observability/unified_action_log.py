"""core/observability/unified_action_log.py — Unified Behavioral Assertion Log

Every time ANY subsystem proposes, executes, or blocks an action,
it gets logged here with its source generation, gate status, and outcome.
This makes the three-generation overlap (VolitionEngine, AgencyCore,
Gen3 constitutional) visible and debuggable.
"""
import asyncio
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from core.memory.retention_policy import working_history_retention_policy
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.ActionLog")

_MAX_ENTRIES = working_history_retention_policy("AURA_UNIFIED_ACTION_LOG_MAX_ENTRIES").max_items

_DIAGNOSTIC_ACTION_PREFIXES = (
    "ROUTER_ERROR:",
    "TRACEBACK",
    "EXCEPTION:",
)
_DIAGNOSTIC_ACTION_FRAGMENTS = (
    "client_returned_no_text",
    "foreground_owner_timeout",
    "warmup_foreground_owner_timeout",
    "all_failed",
)


def _is_diagnostic_noise_action(action: str, source: str = "", outcome: str = "") -> bool:
    """Return True for diagnostics that must never enter behavioral history.

    The unified action log is a causal/action stream. Router failures and
    foreground timeout diagnostics belong in incident/degradation telemetry,
    not in the stream that drives the neural feed and self-model. Persisting
    them as actions makes Aura later "remember" infrastructure errors as if
    they were meaningful behavior.
    """
    text = " ".join(str(part or "") for part in (action, source, outcome)).strip()
    if not text:
        return True
    upper = text.upper()
    if upper.startswith(_DIAGNOSTIC_ACTION_PREFIXES):
        return True
    lower = text.lower()
    return any(fragment in lower for fragment in _DIAGNOSTIC_ACTION_FRAGMENTS)


class UnifiedActionLog:
    """Single stream of all behavioral assertions across all subsystems."""

    def __init__(self):
        self._entries: deque = deque(maxlen=_MAX_ENTRIES)
        self._lock = threading.Lock()
        self._persist_path: Path | None = None
        try:
            from core.config import config
            self._persist_path = config.paths.data_dir / "unified_action_log.jsonl"
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_recent_entries()
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation('unified_action_log', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

    def _load_recent_entries(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            lines = self._persist_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            record_degradation("unified_action_log", exc)
            logger.debug("Failed to read persisted action log %s: %s", self._persist_path, exc)
            return

        restored = []
        for raw in lines[-_MAX_ENTRIES:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                record_degradation("unified_action_log", exc)
                logger.debug("Skipping corrupt action log line: %s", exc)
                continue
            if isinstance(entry, dict):
                if _is_diagnostic_noise_action(
                    str(entry.get("action", "") or ""),
                    str(entry.get("source", "") or ""),
                    str(entry.get("outcome", "") or ""),
                ):
                    continue
                restored.append(entry)

        with self._lock:
            self._entries.extend(restored)

    def record(
        self,
        action: str,
        source: str,
        generation: str,
        gate_status: str = "approved",
        outcome: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """Record a behavioral assertion.

        Args:
            action: What was proposed/executed (e.g. "speak", "web_search", "goal_genesis")
            source: Which module (e.g. "VolitionEngine", "AgencyCore.social_hunger", "mycelium.reflex_identity")
            generation: "gen1_volition", "gen2_agency", "gen3_constitutional", "reflex"
            gate_status: "approved", "blocked", "bypassed", "pending"
            outcome: Result description
            metadata: Extra context
        """
        if _is_diagnostic_noise_action(action, source, outcome):
            logger.debug(
                "Dropped diagnostic noise from unified action log: action=%r source=%r outcome=%r",
                str(action)[:120],
                str(source)[:120],
                str(outcome)[:120],
            )
            return

        entry = {
            "t": time.time(),
            "action": action,
            "source": source,
            "gen": generation,
            "gate": gate_status,
            "outcome": outcome[:200] if outcome else "",
        }
        if metadata:
            entry["meta"] = {k: str(v)[:100] for k, v in metadata.items()}

        with self._lock:
            self._entries.append(entry)

        # Keep the in-memory event synchronous, but never fsync on the event
        # loop. The async task owns the durable append and inherits the
        # governance context from this call.
        if self._persist_path:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                from core.governance_context import local_internal_governed_scope
                with local_internal_governed_scope("unified_action_log.record", domain="file_write"):
                    get_file_write_gateway().append_text(
                        self._persist_path,
                        json.dumps(entry) + "\n",
                        encoding="utf-8",
                        source="unified_action_log.record",
                    )
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as _exc:
                record_degradation('unified_action_log', _exc)
                logger.debug("Suppressed Exception: %s", _exc)
            else:
                loop.create_task(self._persist_entry(entry))

    async def _persist_entry(self, entry: dict[str, Any]) -> None:
        try:
            from core.governance_context import local_internal_governed_scope
            with local_internal_governed_scope("unified_action_log.record", domain="file_write"):
                await get_file_write_gateway().append_text_async(
                    self._persist_path,
                    json.dumps(entry) + "\n",
                    encoding="utf-8",
                    source="unified_action_log.record",
                )
        except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("unified_action_log", exc)
            logger.debug("Deferred action-log append failed: %s", exc)

    def recent(self, limit: int = 20):
        with self._lock:
            items = list(self._entries)
        return items[-limit:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._entries)
        if not items:
            return {"total": 0}
        by_gen = {}
        by_gate = {}
        for e in items:
            g = e.get("gen", "unknown")
            by_gen[g] = by_gen.get(g, 0) + 1
            s = e.get("gate", "unknown")
            by_gate[s] = by_gate.get(s, 0) + 1
        return {
            "total": len(items),
            "by_generation": by_gen,
            "by_gate_status": by_gate,
        }


_instance: UnifiedActionLog | None = None


def get_action_log() -> UnifiedActionLog:
    global _instance
    if _instance is None:
        _instance = UnifiedActionLog()
    return _instance
