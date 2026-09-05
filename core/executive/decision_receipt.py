"""core/executive/decision_receipt.py
Decision receipt compiler logging structural action choices.
"""
from typing import Dict, Any
import time


class DecisionReceiptCompiler:
    """Formats choices and constraints into audit receipts."""

    def compile_receipt(self, intent: Dict[str, Any], state: Any, inhibited: bool) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "tick": state.tick_count,
            "intent": intent,
            "inhibited": inhibited,
            "attention_focus": state.cognition.active_attention,
            "welfare_index": state.welfare.welfare_index
        }
