from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.state.aura_state import AuraState

from .shadow_kernel import ShadowExecutionPhase, ShadowValidationReceipt

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.Arena")

@dataclass
class SpeculativeBranch:
    """A single hypothesis branch in the Arena."""
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: AuraState | None = None
    score: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

class SpeculativeArena:
    """
    [ZENITH-v2] The Subconscious Parallelism Engine.
    Allows for high-throughput branching of the state tree to test multiple future-trajectories.
    """
    def __init__(self, kernel: AuraKernel):
        self.kernel = kernel
        self.branches: dict[str, SpeculativeBranch] = {}
        self._sandbox = ShadowExecutionPhase(kernel)

    async def open_arena(self, base_state: AuraState, count: int = 3) -> list[str]:
        """Creates N parallel branches from the base state."""
        branch_ids = []
        for i in range(count):
            # Lineage derivation for the branch
            branch_state = await base_state.derive_async(f"speculative_branch_{i}", origin="arena")
            branch = SpeculativeBranch(state=branch_state)
            self.branches[branch.branch_id] = branch
            branch_ids.append(branch.branch_id)
        
        logger.info("Arena: Opened with %s speculative branches.", count)
        return branch_ids

    async def execute_branch(self, branch_id: str, mutated_code: str, validator_code: str) -> bool:
        """
        Executes a mutation in the context of a specific branch.
        Uses the Shadow Kernel infrastructure for process isolation.
        """
        if branch_id not in self.branches:
            return False
            
        branch = self.branches[branch_id]
        
        receipt = await self._sandbox.evaluate_mutation_safely(mutated_code, validator_code)
        branch.info["last_shadow_receipt"] = receipt.to_dict()
        
        if receipt.success:
            score_delta = self._score_receipt(receipt)
            branch.score += score_delta
            branch.info["last_score_delta"] = score_delta
            
        return receipt.success

    @staticmethod
    def _score_receipt(receipt: ShadowValidationReceipt) -> float:
        """Convert validation evidence into a bounded branch score."""

        score = 0.0
        if receipt.behavioral_ok:
            score += 0.45
        if receipt.structural_ok:
            score += 0.35

        info_score = SpeculativeArena._extract_validator_score(receipt.validator_info)
        if info_score is not None:
            score += 0.20 * info_score
        elif receipt.success:
            score += 0.10

        elapsed_penalty = min(max(receipt.elapsed_ms, 0.0) / 10_000.0, 0.10)
        return max(0.0, min(1.0, score - elapsed_penalty))

    @staticmethod
    def _extract_validator_score(info: object) -> float | None:
        if isinstance(info, dict):
            for key in ("score", "fitness", "reward", "confidence"):
                value = info.get(key)
                if isinstance(value, (int, float)):
                    return max(0.0, min(1.0, float(value)))
        if isinstance(info, (int, float)):
            return max(0.0, min(1.0, float(info)))
        return None

    async def promote_branch(self, branch_id: str) -> AuraState:
        """Promotes a branch to become the canonical kernel state."""
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found in Arena")
            
        winner = self.branches[branch_id]
        logger.info("Arena: Promoting branch %s (Score: %s) to Canonical.", branch_id, winner.score)
        
        if winner.state is None:
            raise ValueError(f"Branch {branch_id} has no state")

        winner.state.transition_cause = f"arena_promotion: {branch_id}"
        winner.state.response_modifiers["arena_promotion"] = {
            "branch_id": branch_id,
            "score": winner.score,
            "receipt": winner.info.get("last_shadow_receipt", {}),
        }
        return winner.state

    def close_arena(self):
        """Purges all speculative branches."""
        self.branches.clear()
        logger.info("Arena: Closed and purged.")
