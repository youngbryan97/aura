"""core/executive/conflict_resolver.py
Resolves conflicting goal parameters and execution conflicts.
"""
from typing import List, Dict, Any


class ExecutiveConflictResolver:
    """Detects and clears duplicate goals inside the active queues."""

    def resolve_goal_clashes(self, goals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for g in goals:
            gid = g.get("id")
            if gid not in seen:
                seen.add(gid)
                deduped.append(g)
        return deduped
