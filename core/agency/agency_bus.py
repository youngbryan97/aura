"""core/agency_bus.py
Unified AgencyBus — singleton cooldown enforcer for ALL autonomous outputs.

Prevents triple-fire from VolitionEngine + AgencyCore + orchestrator _process_cycle
by enforcing a single global cooldown gate across all autonomous output pathways.
"""
from __future__ import annotations


import logging
import threading
import time
from collections import deque

logger = logging.getLogger("Aura.AgencyBus")

_instance_lock = threading.Lock()


class AgencyBus:
    """Singleton cooldown enforcer for all autonomous outputs.

    All autonomous message pathways (VolitionEngine, AgencyCore, orchestrator
    boredom/reflection) must call submit() before emitting. Only one message
    per cooldown window is allowed through.

    Priority classes control minimum cooldown (seconds):
        duty:     3s  — system obligations
        drive:    5s  — curiosity/exploration
        impulse:  8s  — spontaneous thoughts
        boredom: 10s  — idle chatter

    Note: COOLDOWNS values are the source of truth. The earlier draft of this
    docstring listed 30/60/90/120s as the intended values; production tuning
    moved them to 3/5/8/10s and the docstring is kept in sync with the code.
    """
    _instance: AgencyBus | None = None

    COOLDOWNS = {
        'duty': 3,
        'drive': 5,
        'impulse': 8,
        'boredom': 10,
    }

    DEFAULT_COOLDOWN = 8

    def __init__(self) -> None:
        self._last_output: float = 0.0
        self._audit: deque[dict[str, object]] = deque(maxlen=50)
        self._suppressed_count: int = 0

    @classmethod
    def get(cls) -> AgencyBus:
        """Get or create the singleton instance (thread-safe)."""
        if cls._instance is None:
            with _instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("🚌 AgencyBus singleton initialized")
        return cls._instance

    def submit(self, proposal: dict[str, object]) -> bool:
        """Returns True if proposal passes the cooldown gate AND has valid Will receipt.

        Args:
            proposal: dict with keys:
                origin (str): 'volition', 'agency_core', 'orchestrator', etc.
                text (str): the proposed message
                priority_class (str): 'duty', 'drive', 'impulse', 'boredom'
                will_receipt (str): required receipt ID from UnifiedWill
        """
        now = time.time()
        priority_class = str(proposal.get("priority_class", "impulse"))
        origin = str(proposal.get("origin") or "").strip()
        if not origin or origin.casefold() == "unknown":
            logger.warning("🚌 AgencyBus REJECTED: autonomous proposal has no attributable origin")
            return False

        # Will receipt enforcement: autonomous output must be causally gated.
        receipt_id = proposal.get("will_receipt")
        issuing_will = None
        if not receipt_id:
            try:
                from core.governance.will import ActionDomain
                from core.runtime.action_executor import ActionExecutor

                admission = ActionExecutor.authorize_action(
                    action_name="agency_bus.autonomous_output",
                    params={
                        "origin": origin,
                        "priority_class": priority_class,
                        "proposal_text": str(proposal.get("text") or "")[:500],
                    },
                    source=origin,
                    domain=ActionDomain.INITIATIVE,
                    priority=0.4,
                    context={
                        "source": origin,
                        "autonomous": True,
                        "agency_bus": True,
                        "priority_class": priority_class,
                    },
                )
                if not admission.approved:
                    logger.debug(
                        "🚌 AgencyBus: Will denied auto-receipt for %s: %s",
                        origin,
                        admission.reason,
                    )
                    return False
                receipt_id = admission.receipt_id
                issuing_will = admission.authority
                proposal["will_receipt"] = receipt_id
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _will_err:
                logger.warning("🚌 AgencyBus REJECTED: Will unavailable for receipt acquisition: %s", _will_err)
                return False

        try:
            if issuing_will is None:
                from core.governance.will import get_will

                issuing_will = get_will()

            if not issuing_will.verify_receipt(str(receipt_id)):
                logger.warning("🚌 AgencyBus REJECTED: invalid will_receipt %s", receipt_id)
                return False
        except (ImportError, AttributeError, RuntimeError) as _will_err:
            logger.warning("🚌 AgencyBus REJECTED: Will unavailable for receipt verification: %s", _will_err)
            return False

        min_cooldown = self.COOLDOWNS.get(priority_class, self.DEFAULT_COOLDOWN)

        elapsed = now - self._last_output
        if elapsed < min_cooldown:
            self._suppressed_count += 1
            if self._suppressed_count % 5 == 0:
                logger.warning(
                    "🚌 AgencyBus GATE CLOSED: %s from %s (%.0fs < %ds cooldown, %d suppressed total)",
                    priority_class, proposal.get('origin', '?'),
                    elapsed, min_cooldown, self._suppressed_count
                )
            return False

        self._last_output = now
        self._audit.append({'ts': now, **proposal})
        logger.debug(
            "🚌 AgencyBus GATE OPEN: %s from %s (receipt=%s)",
            priority_class, proposal.get('origin', '?'), receipt_id
        )
        return True

    def on_user_interaction(self) -> None:
        """Open the gate after a user interaction.

        Right after the user talks to Aura, an autonomous turn is allowed to
        fire as soon as the minimum cooldown has elapsed — we don't want a
        stale autonomous-emission cooldown to suppress the next legitimate
        spontaneous turn. Concretely: pretend the last autonomous emission
        happened DEFAULT_COOLDOWN seconds ago, so the gate re-opens on the
        next tick.
        """
        self._last_output = time.time() - self.DEFAULT_COOLDOWN

    @property
    def stats(self) -> dict[str, object]:
        return {
            'suppressed_total': self._suppressed_count,
            'last_output_ago': time.time() - self._last_output if self._last_output else None,
            'recent_audit': list(self._audit)[-5:],
        }
