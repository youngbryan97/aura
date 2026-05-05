"""Canonical client for UnifiedWill decisions.

This wrapper keeps live systems from depending on historical call shapes.
It speaks the current ``UnifiedWill.decide(content, source, domain, ...)``
signature, but tolerates older test doubles that expose async or legacy
keyword signatures.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class WillRequest:
    content: str
    source: str
    domain: Any
    priority: float = 0.5
    is_critical: bool = False
    context: dict[str, Any] | None = None


class WillClient:
    def __init__(self, will: Any | None = None) -> None:
        self._will = will

    def _resolve_will(self) -> Any:
        if self._will is not None:
            return self._will
        from core.will import get_will

        return get_will()

    @staticmethod
    def _coerce_domain(domain: Any) -> Any:
        try:
            from core.will import ActionDomain

            if isinstance(domain, ActionDomain):
                return domain
            value = getattr(domain, "value", domain)
            if isinstance(value, str):
                aliases = {
                    "memory": "memory_write",
                    "tool": "tool_execution",
                    "tools": "tool_execution",
                    "state": "state_mutation",
                    "mutation": "state_mutation",
                    "external_communication": "expression",
                    "self_modification": "state_mutation",
                }
                value = aliases.get(value, value)
                for candidate in ActionDomain:
                    if candidate.value == value or candidate.name == value:
                        return candidate
        except Exception:
            pass
        return domain

    def decide(self, req: WillRequest) -> Any:
        will = self._resolve_will()
        if will is None or not hasattr(will, "decide"):
            return None
        domain = self._coerce_domain(req.domain)
        context = req.context or {}
        decide = will.decide
        try:
            return decide(
                content=req.content,
                source=req.source,
                domain=domain,
                priority=req.priority,
                is_critical=req.is_critical,
                context=context,
            )
        except TypeError:
            # Legacy/test-double shape.
            try:
                return decide(
                    action=req.content,
                    domain=domain,
                    context={**context, "source": req.source},
                    priority=req.priority,
                )
            except TypeError:
                return decide(req.content, req.source, domain)

    async def decide_async(self, req: WillRequest) -> Any:
        decision = self.decide(req)
        if asyncio.iscoroutine(decision):
            return await decision
        return decision

    @staticmethod
    def is_approved(decision: Any) -> bool:
        if decision is None:
            return False
        checker = getattr(decision, "is_approved", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        if isinstance(decision, dict):
            return bool(decision.get("approved", False))
        approved = getattr(decision, "approved", None)
        if approved is not None:
            return bool(approved)
        outcome = getattr(decision, "outcome", None)
        if outcome is not None:
            value = getattr(outcome, "value", str(outcome)).lower()
            return value in {"proceed", "constrain", "critical", "approved", "allow", "allowed"}
        return bool(decision)


def decide(req: WillRequest) -> Any:
    return WillClient().decide(req)


__all__ = ["WillRequest", "WillClient", "decide"]
