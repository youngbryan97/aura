"""core/consciousness/ganglion_node.py

Decentralized Ganglion Node — Autonomous Processing Cluster.

Inspired by octopus ganglia: each node owns a stimulus domain (e.g., "memory",
"affect", "motor") and processes local stimuli independently. Actions are
published to a shared queue for executive review before execution.

Safety features:
- Refractory period prevents rapid re-firing (configurable, default 2s)
- Actions require ExecutiveInhibitor approval before execution
- Local activation tracking for telemetry
"""

from core.runtime.errors import record_degradation
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("Consciousness.Ganglion")


@dataclass
class GanglionAction:
    """An action proposed by a ganglion node for executive review."""

    source_domain: str          # Which ganglion proposed this
    action_type: str            # e.g. "recall", "affect_shift", "motor_plan"
    payload: Dict[str, Any]     # Action-specific data
    priority: float = 0.5       # 0.0-1.0
    is_critical: bool = False   # Critical actions bypass executive inhibition
    timestamp: float = field(default_factory=time.time)


class GanglionNode:
    """A decentralized processing node that handles stimuli for a specific domain.

    Each node:
    - Registers handlers for stimulus types in its domain
    - Processes stimuli independently (no central orchestrator needed)
    - Proposes actions to a shared queue for executive review
    - Enforces a refractory period between firings

    Usage:
        node = GanglionNode("memory", action_queue)
        node.register_handler("recall_request", handle_recall)
        await node.process_stimulus("recall_request", {"query": "last meeting"})
    """

    def __init__(
        self,
        domain: str,
        action_queue: asyncio.Queue,
        refractory_seconds: float = 2.0,
    ):
        """
        Args:
            domain: The stimulus domain this node owns (e.g., "memory", "affect").
            action_queue: Shared queue where proposed actions are sent for review.
            refractory_seconds: Minimum time between consecutive firings.
        """
        self.domain = domain
        self._action_queue = action_queue
        self._refractory_period = refractory_seconds

        # Handler registry: stimulus_type -> async handler function
        self._handlers: Dict[str, Callable[..., Coroutine]] = {}

        # State
        self._last_fire_time: float = 0.0
        self._activation: float = 0.0      # Current activation level (0.0-1.0)
        self._fire_count: int = 0
        self._suppressed_count: int = 0     # Firings blocked by refractory
        self._last_stimulus: Optional[str] = None

        logger.info("Ganglion node [%s] initialized (refractory=%.1fs)", domain, refractory_seconds)

    def register_handler(
        self,
        stimulus_type: str,
        handler: Callable[..., Coroutine],
    ) -> None:
        """Register a handler for a specific stimulus type.

        Args:
            stimulus_type: Type of stimulus this handler responds to.
            handler: Async function(payload) -> Optional[GanglionAction].
        """
        self._handlers[stimulus_type] = handler
        logger.debug("Ganglion [%s]: registered handler for '%s'", self.domain, stimulus_type)

    async def process_stimulus(
        self,
        stimulus_type: str,
        payload: Dict[str, Any],
    ) -> Optional[GanglionAction]:
        """Process an incoming stimulus.

        If a handler matches and the refractory period has elapsed,
        the handler is called and any resulting action is queued.

        Returns:
            The proposed action, or None if suppressed/no handler.
        """
        self._last_stimulus = stimulus_type

        # Check refractory period
        now = time.time()
        if now - self._last_fire_time < self._refractory_period:
            self._suppressed_count += 1
            logger.debug(
                "Ganglion [%s]: suppressed '%s' (refractory, %.1fs remaining)",
                self.domain, stimulus_type,
                self._refractory_period - (now - self._last_fire_time),
            )
            return None

        # Check for handler
        handler = self._handlers.get(stimulus_type)
        if handler is None:
            logger.debug("Ganglion [%s]: no handler for '%s'", self.domain, stimulus_type)
            return None

        # Fire the handler
        try:
            self._last_fire_time = now
            self._fire_count += 1

            # Update activation (spikes on fire, decays naturally)
            self._activation = min(1.0, self._activation + 0.3)

            action = await handler(payload)

            if action is not None and isinstance(action, GanglionAction):
                # Tag with our domain
                action.source_domain = self.domain
                # Submit for executive review
                await self._action_queue.put(action)
                logger.debug(
                    "Ganglion [%s]: fired '%s' -> action '%s' (pri=%.2f)",
                    self.domain, stimulus_type, action.action_type, action.priority,
                )
                return action

        except Exception as e:
            record_degradation('ganglion_node', e)
            logger.error("Ganglion [%s]: handler error for '%s': %s", self.domain, stimulus_type, e)

        return None

    def decay_activation(self, dt: float = 1.0, rate: float = 0.1) -> None:
        """Natural activation decay, called each tick."""
        self._activation = max(0.0, self._activation - rate * dt)

    def get_snapshot(self) -> Dict[str, Any]:
        """Telemetry snapshot."""
        return {
            "domain": self.domain,
            "activation": round(self._activation, 3),
            "fire_count": self._fire_count,
            "suppressed_count": self._suppressed_count,
            "last_stimulus": self._last_stimulus,
            "refractory_remaining": max(
                0.0,
                self._refractory_period - (time.time() - self._last_fire_time),
            ),
            "handler_count": len(self._handlers),
        }
