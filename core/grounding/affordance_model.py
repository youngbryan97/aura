"""core/grounding/affordance_model.py
==================================
Online learning engine tracking action affordances, success rates, risks, and selection.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Grounding.AffordanceModel")

_SINGLETON_LOCK = threading.Lock()
_affordance_model_instance: AffordanceModel | None = None


@dataclass
class AffordanceInfo:
    """Dynamic representation of a learned action affordance within a specific context."""

    action_id: str
    context: str
    execution_count: int = 0
    success_count: int = 0
    average_latency: float = 0.0
    risk_score: float = 0.1  # Initial low risk
    reversibility: float = 1.0  # 1.0 = completely reversible, 0.0 = completely irreversible

    @property
    def success_rate(self) -> float:
        """Helper to get the current empirical success rate."""
        if self.execution_count == 0:
            return 0.5  # Neutral prior
        return self.success_count / self.execution_count

    def compute_utility(self) -> float:
        """Computes action utility as a function of success, risk, and reversibility."""
        # Avoid division by zero with small epsilon
        eps = 0.05
        # Higher success + high reversibility + lower risk = higher utility
        utility = (self.success_rate / (self.risk_score + eps)) * (0.5 + 0.5 * self.reversibility)
        return float(utility)


class AffordanceModel:
    """Online affordance learning engine updating action mappings based on execution feedback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.affordances: dict[tuple[str, str], AffordanceInfo] = {}
        self._seed_default_affordances()
        logger.info("AffordanceModel initialized.")

    def _seed_default_affordances(self) -> None:
        """Populates baseline affordances for standard virtual environments."""
        defaults = [
            # Filesystem Context
            ("file_read", "filesystem", 0.95, 0.01, 1.0),
            ("file_write", "filesystem", 0.85, 0.15, 0.8),
            ("file_delete", "filesystem", 0.80, 0.40, 0.1),
            # Terminal Context
            ("command_execute", "terminal", 0.90, 0.10, 0.7),
            # Pytest Context
            ("run_test", "pytest", 0.98, 0.02, 1.0),
            ("patch_code", "pytest", 0.70, 0.30, 0.6),
            ("commit_code", "pytest", 0.95, 0.10, 0.3),
        ]
        for act, ctx, success_rate, risk, reversibility in defaults:
            key = (act, ctx)
            self.affordances[key] = AffordanceInfo(
                action_id=act,
                context=ctx,
                execution_count=10,  # Seed with prior observation count
                success_count=int(10 * success_rate),
                average_latency=15.0,
                risk_score=risk,
                reversibility=reversibility,
            )

    def get_available_affordances(self, context: str) -> list[AffordanceInfo]:
        """Thread-safe retrieval of available affordances for a given context."""
        with self._lock:
            results = [aff for aff in self.affordances.values() if aff.context == context]
            if not results:
                # Provide a generic fallback action if context has nothing registered
                fallback = AffordanceInfo(action_id="reflect", context=context, reversibility=1.0)
                self.affordances[("reflect", context)] = fallback
                results = [fallback]
            return results

    def update_affordance(
        self, action_id: str, context: str, success: bool, latency_ms: float, risk_factor: float
    ) -> None:
        """Online updating of affordance metrics using Exponential Moving Averages (EMA)."""
        key = (action_id, context)
        with self._lock:
            if key not in self.affordances:
                self.affordances[key] = AffordanceInfo(action_id=action_id, context=context)

            aff = self.affordances[key]
            aff.execution_count += 1
            if success:
                aff.success_count += 1

            # EMA coefficient
            alpha = 0.15

            # Update latency EMA
            if aff.average_latency == 0.0:
                aff.average_latency = latency_ms
            else:
                aff.average_latency = (1 - alpha) * aff.average_latency + alpha * latency_ms

            # Update risk score EMA
            aff.risk_score = (1 - alpha) * aff.risk_score + alpha * risk_factor

            logger.debug(
                "Updated affordance %s/%s: count=%d, success_rate=%.2f, risk=%.2f",
                action_id,
                context,
                aff.execution_count,
                aff.success_rate,
                aff.risk_score,
            )

    def select_action(self, context: str, temperature: float = 0.5) -> str:
        """Selects an action using a Boltzmann/Softmax exploration policy over utilities."""
        affordances = self.get_available_affordances(context)
        if not affordances:
            return "reflect"

        utilities = np.array([aff.compute_utility() for aff in affordances], dtype=np.float32)

        # Guard temperature against extreme low values to avoid divide-by-zero
        t = max(0.01, temperature)

        try:
            # Softmax calculation with numerical stability subtraction
            shifted_utilities = utilities - np.max(utilities)
            exp_utilities = np.exp(shifted_utilities / t)
            probabilities = exp_utilities / np.sum(exp_utilities)

            # Execute random draw based on learned probabilities
            chosen_index = np.random.choice(len(affordances), p=probabilities)
            chosen_action = affordances[chosen_index].action_id
            logger.info(
                "Affordance selected action '%s' in context '%s' (utilities=%s, probs=%s)",
                chosen_action,
                context,
                np.round(utilities, 2),
                np.round(probabilities, 2),
            )
            return chosen_action
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "affordance_selection",
                exc,
                severity="warning",
                action="selected deterministic affordance fallback after softmax selection failed",
            )
            logger.error("Failed executing softmax action selection: %s", exc)
            # Default fallback to first available or standard reflect
            return affordances[0].action_id if affordances else "reflect"


def get_affordance_model() -> AffordanceModel:
    """Thread-safe accessor for the AffordanceModel singleton."""
    global _affordance_model_instance
    if _affordance_model_instance is None:
        with _SINGLETON_LOCK:
            if _affordance_model_instance is None:
                _affordance_model_instance = AffordanceModel()
                # Auto register in ServiceContainer
                ServiceContainer.register_instance("affordance_model", _affordance_model_instance)
    return _affordance_model_instance
