"""core/unified_action_log.py — Unified Behavioral Assertion Log

Every time ANY subsystem proposes, executes, or blocks an action,
it gets logged here with its source generation, gate status, and outcome.
This makes the three-generation overlap (VolitionEngine, AgencyCore,
Gen3 constitutional) visible and debuggable.
"""
import json
import logging
import time
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.ActionLog")

_MAX_ENTRIES = 500


class UnifiedActionLog:
    """Single stream of all behavioral assertions across all subsystems."""

    def __init__(self):
        self._entries: deque = deque(maxlen=_MAX_ENTRIES)
        self._lock = threading.Lock()
        self._persist_path: Optional[Path] = None
        try:
            from core.config import config
            self._persist_path = config.paths.data_dir / "unified_action_log.jsonl"
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_recent_entries()
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

    def _load_recent_entries(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            lines = self._persist_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return

        restored = []
        for raw in lines[-_MAX_ENTRIES:]:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except Exception:
                continue
            if isinstance(entry, dict):
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
        metadata: Optional[Dict[str, Any]] = None,
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

        # Async-safe file append
        if self._persist_path:
            try:
                with open(self._persist_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

    def recent(self, limit: int = 20):
        with self._lock:
            items = list(self._entries)
        return items[-limit:]

    def stats(self) -> Dict[str, Any]:
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


_instance: Optional[UnifiedActionLog] = None


def get_action_log() -> UnifiedActionLog:
    global _instance
    if _instance is None:
        _instance = UnifiedActionLog()
    return _instance
