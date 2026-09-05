"""core/workspace/global_workspace.py
Global Workspace coordinator aggregating scratchpad, attention, and inner dialogue.
"""
from typing import Any

from core.workspace.attention_field import AttentionField
from core.workspace.inner_dialogue import InnerDialogueGenerator
from core.workspace.metacognitive_monitor import MetacognitiveMonitor
from core.workspace.private_scratchpad import PrivateScratchpad
from core.workspace.thought_lifecycle import ThoughtLifecycle


class GlobalWorkspace:
    """Canonical aggregator of Aura's private working memory workspace."""

    def __init__(self):
        self.scratchpad = PrivateScratchpad()
        self.attention = AttentionField()
        self.dialogue = InnerDialogueGenerator()
        self.monitor = MetacognitiveMonitor()
        self.lifecycle = ThoughtLifecycle()
        
        self._monologue_history: list[str] = []

    def process_workspace_tick(self, state: Any) -> None:
        """Runs evaluation updates on private working memory workspace."""
        # 1. Update active attention slots
        self.attention.assign_attention_slot("active_attention", state.cognition.active_attention)
        
        # 2. Generate monologue
        monologue = self.dialogue.generate_monologue(state)
        state.cognition.inner_monologue = monologue
        self._monologue_history.append(monologue)

        # 3. Metacognitive audits
        stall_detected = self.monitor.audit_thought_traces(self._monologue_history)
        if stall_detected:
            # Inject correction goal to state
            state.cognition.current_goals.append({
                "id": "clear_thought_stall",
                "status": "pending"
            })

        # 4. Manage lifecycle of thought nodes
        self.lifecycle.spawn_thought(f"tick_{state.tick_count}", monologue)
        self.lifecycle.evict_stale_thoughts()
