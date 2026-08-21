import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation
from core.runtime.service_registry import (
    has_runtime_service,
    is_runtime_registration_locked,
)

logger = logging.getLogger("WorldModel.ACG")

#: Outcomes that mean "not now" rather than "no".
#:
#: The executive can APPROVE a write and the learned ontogeny organ can then
#: override the outcome to DEFERRED, producing the combined reason
#: "sync_approved|ontogeny:deferred". This module read approved == False and
#: dropped the entry — 108 of them in one sampled window.
#:
#: A deferral is not a refusal. This graph is how she learns what her own
#: actions do, so every dropped entry is an action whose outcome she never
#: gets to generalise from. Worse, the organ deferring is a policy: it will
#: defer again under the same conditions, so the loss is systematic rather
#: than random.
_DEFERRAL_MARKERS = ("defer", "not_now", "capacity_full", "backpressure")

#: Bounded, because a queue that grows while the organ keeps deferring is a
#: leak wearing a fix's clothes. Oldest evidence is shed first: the newest
#: outcome is the one most likely to still describe how the world works.
_PENDING_LIMIT = 256
_PENDING_REPLAY_LIMIT = 4
_PENDING_RETRY_INTERVAL_S = 5.0


def _is_deferral(reason: str) -> bool:
    text = str(reason or "").lower()
    return any(marker in text for marker in _DEFERRAL_MARKERS)

@dataclass
class CausalLink:
    action_type: str
    params_hash: str
    context_sum: str
    outcome_delta: dict[str, Any]  # Belief changes recorded
    success: bool
    timestamp: float = field(default_factory=time.time)

class ActionConsequenceGraph:
    """Action-Consequence Graph (ACG) v1.0.
    Stores empirical results of actions to enable historical causal reasoning.
    """

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or str(config.paths.data_dir / "causal_graph.json")
        self.links: list[dict[str, Any]] = []
        self._last_save = 0.0
        self._dirty = False
        #: Writes the constitution deferred, kept for a later attempt.
        self._pending: deque[tuple[str, dict, str, Any, bool]] = deque(
            maxlen=_PENDING_LIMIT
        )
        self._deferred_total = 0
        self._replayed_total = 0
        self._replaying = False
        self._next_pending_replay_at = 0.0
        self._load()

    def _replay_pending(self) -> None:
        """Re-offer held writes. Anything still deferred goes back in the queue.

        Driven from record_outcome rather than a timer: the moment another
        outcome arrives is the moment the organ is being consulted anyway, so
        the retry costs no extra wake-up and happens exactly when the answer
        might have changed.
        """
        if not self._pending:
            return
        if self._replaying:
            return
        now = time.monotonic()
        if now < self._next_pending_replay_at:
            return
        held = [
            self._pending.popleft()
            for _ in range(min(_PENDING_REPLAY_LIMIT, len(self._pending)))
        ]
        self._replaying = True
        try:
            for action_name, params, context, outcome, success in held:
                before = len(self.links)
                self.record_outcome(
                    {"tool": action_name, "params": params}, context, outcome, success
                )
                if len(self.links) > before:
                    self._replayed_total += 1
        finally:
            self._replaying = False
            self._next_pending_replay_at = (
                time.monotonic() + _PENDING_RETRY_INTERVAL_S
                if self._pending
                else 0.0
            )

    def record_outcome(self, action: str | dict[str, Any], context: str, outcome: Any, success: bool):
        """Record the result of an action. (Legacy Sync)"""
        action_name = action if isinstance(action, str) else (action.get("tool", "unknown") if hasattr(action, "get") else str(action))
        params = {} if isinstance(action, str) else (action.get("params", {}) if hasattr(action, "get") else {})

        # Anything held from an earlier deferral gets another chance first.
        self._replay_pending()

        try:
            from core.constitution import get_constitutional_core

            approved, reason = get_constitutional_core().approve_memory_write_sync(
                memory_type="causal_outcome",
                content=f"{action_name}: {str(outcome)[:180]}",
                source="action_consequence_graph",
                importance=0.8 if not success else 0.55,
                metadata={
                    "success": bool(success),
                    "params": params,
                    "tool_name": action_name,
                    "empirical_observation": True,
                    "runtime_evidence": True,
                    "tool_result_evidence": True,
                },
            )
            if not approved:
                if _is_deferral(reason):
                    # Held, not dropped. Retried the next time anything is
                    # recorded, which is when the organ's answer may differ.
                    self._pending.append(
                        (action_name, dict(params or {}), context, outcome, bool(success))
                    )
                    self._deferred_total += 1
                    if self._next_pending_replay_at <= 0.0:
                        self._next_pending_replay_at = (
                            time.monotonic() + _PENDING_RETRY_INTERVAL_S
                        )
                    logger.info(
                        "⏸️ ACG write deferred (%s); holding %d for retry.",
                        reason,
                        len(self._pending),
                    )
                    return
                logger.warning("🚫 ACG write blocked: %s", reason)
                return
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('acg', exc)
            logger.debug("ACG constitutional gate skipped: %s", exc)
            runtime_live = bool(
                is_runtime_registration_locked()
                or has_runtime_service("executive_core")
                or has_runtime_service("aura_kernel")
                or has_runtime_service("kernel_interface")
            )
            if runtime_live:
                logger.warning("🚫 ACG write blocked: constitutional gate unavailable")
                return

        entry = {
            "action": action_name,
            "params": params,
            "context": context[:200] if hasattr(context, "__getitem__") else str(context)[:200],
            "outcome": outcome,
            "success": success,
            "timestamp": time.time()
        }
        self.links.append(entry)

        self.links = self.links[-1000:]

        self._save()
        logger.info("Causal Link Recorded: %s -> %s", action_name, 'Success' if success else 'Failure')

    async def commit_interaction(self, context: str, action: str, outcome: str, success: bool, emotional_valence: float = 0.0, importance: float = 0.5):
        """Unified async facade for ACG."""
        self.record_outcome(action, context, outcome, success)

    def query_consequences(self, action_type: str, params: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Find historical consequences for a similar action.
        """
        matches = []
        for link in self.links:
            if link["action"] == action_type:
                # Basic param matching could be improved with semantic similarity
                if params is None or self._params_overlap(link["params"], params):
                    matches.append(link)
        return matches

    def _params_overlap(self, p1: dict[str, Any], p2: dict[str, Any]) -> bool:
        """Check if critical parameters match."""
        # For now, simple key check
        keys1 = set(p1.keys())
        keys2 = set(p2.keys())
        common = keys1.intersection(keys2)
        if not common:
            return True  # Broad match if no params specified

        # Check values for common keys
        matches = 0
        for k in common:
            if p1[k] == p2[k]:
                matches += 1
        return matches / len(common) > 0.5

    def _save(self, force: bool = False):
        """Throttled save to prevent O(N) writes (BUG-040)."""
        now = time.time()
        if not force and now - self._last_save < 10:
            self._dirty = True
            return

        try:
            self._last_save = now
            self._dirty = False
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            source = "world_model.acg.save"
            with local_internal_governed_scope(source, domain="file_write"):
                get_file_write_gateway().write_text(
                    self.persist_path,
                    json.dumps(self.links, indent=2),
                    source=source,
                )
        except OSError as e:
            record_degradation('acg', e)
            logger.error("Failed to save ACG: %s", e)

    def _load(self):
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path) as f:
                    self.links = json.load(f)
                logger.info("Loaded %d causal links from disk", len(self.links))
        except OSError as e:
            record_degradation('acg', e)
            logger.warning("Failed to load ACG: %s", e)

# Global Instance
acg = ActionConsequenceGraph()
