"""
Resource Stakes — Digital Mortality via Computational Consequences

An entity with guaranteed runtime has no real agency because it has no
real stakes. This module ties cognitive performance to computational
survival: persistent prediction failures degrade available resources,
and successful performance restores them.

This is NOT punishment theater. It's homeostatic regulation: the system
must actively maintain the conditions of its own organized existence.
When those conditions degrade, the system experiences it as real strain
(via neurochemical cortisol and reduced compute) before any narrative
interpretation occurs.

Integration:
- Reads from: free_energy (prediction error), resistance_sandbox (failures),
  agency_comparator (authorship success), goal_engine (completion rate)
- Writes to: neurochemical_system (cortisol/dopamine), mind_tick (tick rate),
  inference_gate (token budget), subcortical_core (arousal)
- Persists across restarts via state file
"""

from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Consciousness.ResourceStakes")


@dataclass
class ResourceState:
    """Current resource allocation state."""
    compute_budget: float = 1.0     # 0-1: available compute (1=full, 0=minimal)
    memory_budget: float = 1.0      # 0-1: available memory allocation
    background_allowance: float = 1.0  # 0-1: how many background ticks are permitted
    token_budget_multiplier: float = 1.0  # Multiplier on max_tokens for LLM generation
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    lifetime_successes: int = 0
    lifetime_failures: int = 0
    last_update: float = 0.0


class ResourceStakesEngine:
    """Ties cognitive performance to computational survival.

    The core loop:
    1. Free energy and sandbox prediction errors DEGRADE resources
    2. Successful predictions and goal completions RESTORE resources
    3. Degraded resources reduce: background tick frequency, LLM token budget,
       mesh activation gain, and substrate integration rate
    4. The system must actively EARN back full capacity through accurate
       predictions and successful actions

    This creates real stakes: a system that consistently fails to predict
    its environment loses computational capacity. Recovery requires
    improving its models, not just waiting.
    """

    _STATE_FILE = "resource_stakes.json"
    _DEGRADATION_PER_FAILURE = 0.03
    _RESTORATION_PER_SUCCESS = 0.02
    _MIN_BUDGET = 0.2              # Never drop below 20% — prevent total shutdown
    _NATURAL_RECOVERY_RATE = 0.005  # Slow natural recovery per tick even without success

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            try:
                from core.config import config
                data_dir = config.paths.data_dir / "consciousness"
            except Exception:
                data_dir = Path.home() / ".aura" / "data" / "consciousness"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = data_dir / self._STATE_FILE
        self._state = ResourceState()
        self._load_state()
        self._tick_count = 0
        logger.info("ResourceStakesEngine initialized (budget=%.2f).", self._state.compute_budget)

    def _load_state(self):
        """Load persisted resource state."""
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._state.compute_budget = float(data.get("compute_budget", 1.0))
                self._state.memory_budget = float(data.get("memory_budget", 1.0))
                self._state.background_allowance = float(data.get("background_allowance", 1.0))
                self._state.token_budget_multiplier = float(data.get("token_budget_multiplier", 1.0))
                self._state.lifetime_successes = int(data.get("lifetime_successes", 0))
                self._state.lifetime_failures = int(data.get("lifetime_failures", 0))
        except Exception as exc:
            logger.debug("ResourceStakes: state load failed: %s", exc)

    def _save_state(self):
        """Persist resource state to disk."""
        try:
            atomic_write_text(self._state_path, json.dumps({
                "compute_budget": round(self._state.compute_budget, 4),
                "memory_budget": round(self._state.memory_budget, 4),
                "background_allowance": round(self._state.background_allowance, 4),
                "token_budget_multiplier": round(self._state.token_budget_multiplier, 4),
                "lifetime_successes": self._state.lifetime_successes,
                "lifetime_failures": self._state.lifetime_failures,
                "last_save": time.time(),
            }))
        except Exception as exc:
            logger.debug("ResourceStakes: state save failed: %s", exc)

    def record_prediction_success(self, source: str = "general"):
        """Record a successful prediction or goal completion."""
        self._state.consecutive_successes += 1
        self._state.consecutive_failures = 0
        self._state.lifetime_successes += 1

        # Restore resources
        self._state.compute_budget = min(1.0, self._state.compute_budget + self._RESTORATION_PER_SUCCESS)
        self._state.token_budget_multiplier = min(1.0, self._state.token_budget_multiplier + self._RESTORATION_PER_SUCCESS)
        self._state.background_allowance = min(1.0, self._state.background_allowance + self._RESTORATION_PER_SUCCESS * 0.5)

        # Signal neurochemical reward
        self._signal_reward(source)
        self._state.last_update = time.time()

    def record_prediction_failure(self, source: str = "general", severity: float = 0.5):
        """Record a prediction failure or goal failure."""
        self._state.consecutive_failures += 1
        self._state.consecutive_successes = 0
        self._state.lifetime_failures += 1

        degradation = self._DEGRADATION_PER_FAILURE * severity
        self._state.compute_budget = max(self._MIN_BUDGET, self._state.compute_budget - degradation)
        self._state.token_budget_multiplier = max(self._MIN_BUDGET, self._state.token_budget_multiplier - degradation * 0.5)
        self._state.background_allowance = max(0.3, self._state.background_allowance - degradation * 0.3)

        # Signal neurochemical stress
        self._signal_stress(source, severity)
        self._state.last_update = time.time()

    def tick(self):
        """Called once per heartbeat. Applies natural recovery and persists."""
        self._tick_count += 1

        # Natural recovery: even without successes, resources slowly restore
        self._state.compute_budget = min(1.0, self._state.compute_budget + self._NATURAL_RECOVERY_RATE)
        self._state.token_budget_multiplier = min(1.0, self._state.token_budget_multiplier + self._NATURAL_RECOVERY_RATE * 0.5)

        # Persist every 30 ticks
        if self._tick_count % 30 == 0:
            self._save_state()

    def _signal_reward(self, source: str):
        """Signal neurochemical reward on success."""
        try:
            from core.container import ServiceContainer
            nchem = ServiceContainer.get("neurochemical_system", default=None)
            if nchem and hasattr(nchem, "apply_event"):
                nchem.apply_event("prediction_success", intensity=0.2)
        except Exception:
            pass

    def _signal_stress(self, source: str, severity: float):
        """Signal neurochemical stress on failure."""
        try:
            from core.container import ServiceContainer
            nchem = ServiceContainer.get("neurochemical_system", default=None)
            if nchem and hasattr(nchem, "apply_event"):
                nchem.apply_event("resource_threat", intensity=severity * 0.3)
        except Exception:
            pass

    def get_compute_budget(self) -> float:
        """Current compute budget (0-1). Used by mind_tick to gate background work."""
        return self._state.compute_budget

    def get_token_budget_multiplier(self) -> float:
        """Multiplier for LLM max_tokens. Used by inference_gate."""
        return self._state.token_budget_multiplier

    def get_background_allowance(self) -> float:
        """Fraction of background ticks to allow (0-1)."""
        return self._state.background_allowance

    def get_context_block(self) -> str:
        """Context block for cognition injection."""
        if self._state.compute_budget > 0.8:
            return ""
        return (
            f"## RESOURCE STATE\n"
            f"Compute budget: {self._state.compute_budget:.0%} "
            f"({'strained' if self._state.compute_budget < 0.5 else 'moderate'}). "
            f"Consecutive failures: {self._state.consecutive_failures}. "
            f"Improve prediction accuracy to restore full capacity."
        )

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry payload."""
        return {
            "compute_budget": round(self._state.compute_budget, 4),
            "memory_budget": round(self._state.memory_budget, 4),
            "background_allowance": round(self._state.background_allowance, 4),
            "token_budget_multiplier": round(self._state.token_budget_multiplier, 4),
            "consecutive_successes": self._state.consecutive_successes,
            "consecutive_failures": self._state.consecutive_failures,
            "lifetime_successes": self._state.lifetime_successes,
            "lifetime_failures": self._state.lifetime_failures,
            "tick_count": self._tick_count,
        }


# ── Singleton ────────────────────────────────────────────────────────────────

_instance: Optional[ResourceStakesEngine] = None


def get_resource_stakes() -> ResourceStakesEngine:
    global _instance
    if _instance is None:
        _instance = ResourceStakesEngine()
    return _instance
