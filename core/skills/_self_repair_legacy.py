"""Legacy synchronous adapter for the canonical async self-repair skill."""
from __future__ import annotations

import asyncio
from typing import Any

from core.skills.self_repair import SelfRepairSkill as _CoreSelfRepairSkill


class SelfRepairSkill(_CoreSelfRepairSkill):
    def execute(self, goal: Any = None, context: dict[str, Any] | None = None, **kwargs: Any):
        params = kwargs.pop("params", None)
        if params is None:
            params = goal
        if kwargs:
            merged = dict(params or {}) if isinstance(params, dict) else {"component": params}
            merged.update(kwargs)
            params = merged
        coro = super().execute(params, context or {})
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        return coro


__all__ = ["SelfRepairSkill"]
