"""Reading up for a goal names whose reading it is.

The lookups named no origin, so the capability engine filed them under
itself — neither a person's request nor her own initiative — and no standing
grant covers that. LIVE 2026-09-23, mid-game, every lookup: "Tool execution
'local_reference_search' blocked by Constitution ...
(signed_standing_authority_lease_missing)".
"""

from __future__ import annotations

import asyncio

from core.agency import task_knowledge
from core.executive.standing_authority import (
    AUTONOMOUS_AUTHORITY_ORIGINS,
    INTROSPECTION_TOOLS,
    PUBLIC_RESEARCH_TOOLS,
)


class _Engine:
    def __init__(self) -> None:
        self.ran: list[tuple[str, dict, dict]] = []

    async def execute(self, name, params, context):
        self.ran.append((name, dict(params), dict(context)))
        return {"results": []}


def test_her_own_shelf_is_read_as_her_own_research():
    engine = _Engine()
    asyncio.run(task_knowledge._from_her_own_shelf("what is 2048", engine=engine))
    name, _params, context = engine.ran[0]
    assert name in INTROSPECTION_TOOLS
    assert context["origin"] in AUTONOMOUS_AUTHORITY_ORIGINS


def test_the_web_is_read_as_her_own_research():
    engine = _Engine()
    asyncio.run(task_knowledge._from_search("how is 2048 played", engine=engine))
    name, _params, context = engine.ran[0]
    assert name in PUBLIC_RESEARCH_TOOLS
    assert context["origin"] in AUTONOMOUS_AUTHORITY_ORIGINS


def test_the_engine_files_it_under_that_origin():
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine.__new__(CapabilityEngine)
    source = engine._resolve_execution_source({"origin": task_knowledge.READING_UP})
    assert source == task_knowledge.READING_UP
