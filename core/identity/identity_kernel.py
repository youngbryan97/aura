"""core/identity/identity_kernel.py
Core Identity Kernel maintaining Aura's stable self-continuity over time.
"""
import logging
from typing import Any

from core.identity.continuity_guard import ContinuityGuard
from core.identity.identity_history import IdentityHistoryTracker
from core.identity.self_contract import SelfContract
from core.identity.self_revision_protocol import SelfRevisionProtocol

logger = logging.getLogger("Identity.IdentityKernel")


class IdentityKernel:
    """Canonical manager of Aura's persistent identity kernel."""

    def __init__(self):
        self.contract = SelfContract()
        self.guard = ContinuityGuard()
        self.history = IdentityHistoryTracker()
        self.protocol = SelfRevisionProtocol()

        # Initialize baseline state
        self._identity_state = {
            "name": self.contract.name,
            "origin": self.contract.origin,
            "core_values": self.contract.core_values,
            "primary_operator": self.contract.get_relationship_constraints().get("primary_operator", "Bryan")
        }

    def get_current_identity(self) -> dict[str, Any]:
        return self._identity_state

    async def guard_identity_continuity(self, state: Any) -> None:
        """Executed during the life loop tick to assert self-coherence."""
        # Ensure LifeState has active identity initialized
        if not state.identity:
            state.identity = self.get_current_identity()

        # Verify against our baseline
        is_coherent = self.guard.verify_continuity(state.identity, self._identity_state)
        if not is_coherent:
            logger.warning("Identity misalignment resolved: restoring baseline constraints.")
            state.identity = self.get_current_identity()

    async def propose_value_modification(self, key: str, new_value: Any, reason: str) -> bool:
        """Gated route allowing safe value shifts under the revision protocol."""
        if not self.protocol.is_revision_allowed(key, new_value):
            return False

        old_val = self._identity_state.get(key)
        self._identity_state[key] = new_value
        self.history.record_revision(key, old_val, new_value, reason)
        logger.info("Successfully updated identity parameter '%s' via protocol approval.", key)
        return True
