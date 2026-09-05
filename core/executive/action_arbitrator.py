"""core/executive/action_arbitrator.py
Arbitrates between conflicting pending actions in the queue.
"""
from typing import Dict, List, Any


class ActionArbitrator:
    """Selects the highest priority action when multiple actions compete."""

    def arbitrate(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sorts pending actions: critical safety actions take precedence."""
        # Simple sorting where 'gesture' or 'cool_down' have precedence
        priority_orders = {"gesture": 0, "desktop": 1, "file": 2, "terminal": 3}
        return sorted(actions, key=lambda a: priority_orders.get(a.get("channel", "terminal"), 99))
