"""The approvals the executive grants, one door each.

Lifted whole out of `executive_core`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .executive_core import (
        Dict,
        Tuple,
    )


class _ApprovesWhatItIsAsked:
    """Lifted whole out of ExecutiveCore; see executive_core.py."""

    async def approve_tool(self, tool_name: str, args: Dict[str, Any],
                           source: str = "unknown") -> Tuple[bool, str, Dict]:
        """Quick check: should this tool execution proceed?

        Returns (approved, reason, constraints).
        """
        from .executive_core import (
            DecisionOutcome,
        )

        intent, record = await self.prepare_tool_intent(tool_name, args, source=source)
        approved = record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED)
        if approved:
            self.complete_intent(intent.intent_id, success=True)
        return (approved, record.reason, record.constraints)

    async def approve_emission(self, content: str, source: str = "unknown",
                               urgency: float = 0.5) -> Tuple[bool, str]:
        """Quick check: should this spontaneous message be emitted?"""
        from .executive_core import (
            ActionType,
            DecisionOutcome,
            Intent,
            IntentSource,
        )

        intent = Intent(
            source=IntentSource.SOCIAL if source == "proactive_presence" else IntentSource.AUTONOMOUS,
            goal=f"emit_message:{content[:40]}",
            action_type=ActionType.EMIT_MESSAGE,
            payload={"content": content, "source": source},
            priority=urgency,
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_memory_write(self, memory_type: str, content: str,
                                    importance: float = 0.5,
                                    source: str = "unknown") -> Tuple[bool, str]:
        """Quick check: should this memory be committed?"""
        from .executive_core import (
            ActionType,
            DecisionOutcome,
            Intent,
            IntentSource,
        )

        intent = Intent(
            source=IntentSource.SYSTEM,
            goal=f"write_memory:{memory_type}",
            action_type=ActionType.WRITE_MEMORY,
            payload={"type": memory_type, "content": content[:200], "importance": importance},
            priority=importance,
            requires_memory_commit=True,
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_state_mutation(self, origin: str, cause: str) -> Tuple[bool, str]:
        """Quick check: should this state mutation proceed?"""
        from .executive_core import (
            ActionType,
            DecisionOutcome,
            Intent,
            IntentSource,
        )

        intent = Intent(
            source=IntentSource.SYSTEM,
            goal=f"mutate_state:{origin}",
            action_type=ActionType.MUTATE_STATE,
            payload={"origin": origin, "cause": cause},
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

    async def approve_background_task(self, task_name: str,
                                       source: str = "unknown") -> Tuple[bool, str]:
        """Quick check: should this background task be spawned?"""
        from .executive_core import (
            ActionType,
            DecisionOutcome,
            Intent,
            IntentSource,
        )

        intent = Intent(
            source=IntentSource.BACKGROUND,
            goal=f"spawn_task:{task_name}",
            action_type=ActionType.SPAWN_TASK,
            payload={"task_name": task_name, "source": source},
        )
        record = await self.request_approval(intent)
        if record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED):
            self.complete_intent(intent.intent_id, success=True)
        return (
            record.outcome in (DecisionOutcome.APPROVED, DecisionOutcome.DEGRADED),
            record.reason,
        )

