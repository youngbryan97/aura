import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from core.container import ServiceContainer
from core.config import config

logger = logging.getLogger("WorldModel.ACG")

@dataclass
class CausalLink:
    action_type: str
    params_hash: str
    context_sum: str
    outcome_delta: Dict[str, Any]  # Belief changes recorded
    success: bool
    timestamp: float = field(default_factory=time.time)

class ActionConsequenceGraph:
    """Action-Consequence Graph (ACG) v1.0.
    Stores empirical results of actions to enable historical causal reasoning.
    """

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or str(config.paths.data_dir / "causal_graph.json")
        self.links: List[Dict[str, Any]] = []
        self._last_save = 0.0
        self._dirty = False
        self._load()

    def record_outcome(self, action: Union[str, Dict[str, Any]], context: str, outcome: Any, success: bool):
        """Record the result of an action. (Legacy Sync)"""
        action_name = action if isinstance(action, str) else (action.get("tool", "unknown") if hasattr(action, "get") else str(action))
        params = {} if isinstance(action, str) else (action.get("params", {}) if hasattr(action, "get") else {})

        try:
            from core.container import ServiceContainer
            from core.constitution import get_constitutional_core

            approved, reason = get_constitutional_core().approve_memory_write_sync(
                memory_type="causal_outcome",
                content=f"{action_name}: {str(outcome)[:180]}",
                source="action_consequence_graph",
                importance=0.8 if not success else 0.55,
                metadata={"success": bool(success), "params": params},
            )
            if not approved:
                logger.warning("🚫 ACG write blocked: %s", reason)
                return
        except Exception as exc:
            logger.debug("ACG constitutional gate skipped: %s", exc)
            runtime_live = bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
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

    def query_consequences(self, action_type: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find historical consequences for a similar action.
        """
        matches = []
        for link in self.links:
            if link["action"] == action_type:
                # Basic param matching could be improved with semantic similarity
                if params is None or self._params_overlap(link["params"], params):
                    matches.append(link)
        return matches

    def _params_overlap(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> bool:
        """Check if critical parameters match."""
        # For now, simple key check
        keys1 = set(p1.keys())
        keys2 = set(p2.keys())
        common = keys1.intersection(keys2)
        if not common: return True # Broad match if no params specified
        
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
            with open(self.persist_path, "w") as f:
                json.dump(self.links, f, indent=2)
        except Exception as e:
            logger.error("Failed to save ACG: %s", e)

    def _load(self):
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, "r") as f:
                    self.links = json.load(f)
                logger.info("Loaded %d causal links from disk", len(self.links))
        except Exception as e:
            logger.warning("Failed to load ACG: %s", e)

# Global Instance
acg = ActionConsequenceGraph()
