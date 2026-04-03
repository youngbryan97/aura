import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.consciousness.executive_authority import ExecutiveAuthority
from core.state.aura_state import AuraState


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_no_direct_pending_initiative_appenders_remain():
    targets = [
        PROJECT_ROOT / "core" / "phases" / "initiative_generation.py",
        PROJECT_ROOT / "core" / "phases" / "motivation_update.py",
        PROJECT_ROOT / "core" / "phases" / "memory_consolidation.py",
        PROJECT_ROOT / "core" / "mind_tick.py",
        PROJECT_ROOT / "core" / "consciousness" / "executive_closure.py",
        PROJECT_ROOT / "core" / "autonomy" / "research_cycle.py",
        PROJECT_ROOT / "core" / "kernel" / "upgrades_10x.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "pending_initiatives.append(" not in text, f"direct initiative append still present in {path.name}"
        if path.name == "research_cycle.py":
            assert "state.cognition.pending_initiatives =" not in text, "research cycle still mutates initiative queue directly"


def test_background_seeders_do_not_bind_objectives_directly():
    text = (PROJECT_ROOT / "core" / "kernel" / "upgrades_10x.py").read_text(encoding="utf-8")
    assert 'state.cognition.current_objective = (' not in text
    assert "state.cognition.current_objective = transcript" not in text


def test_kernel_pending_initiative_consumer_is_retired():
    text = (PROJECT_ROOT / "core" / "kernel" / "aura_kernel.py").read_text(encoding="utf-8")
    assert "await self._dispatch_pending_initiatives()" not in text
    assert "Retired compatibility hook." in text


@pytest.mark.asyncio
async def test_executive_authority_promotes_and_completes_initiatives():
    state = AuraState()
    authority = ExecutiveAuthority(SimpleNamespace())

    state, _ = await authority.propose_initiative_to_state(
        state,
        "Investigate runtime drift",
        source="initiative_generation",
        urgency=0.7,
        triggered_by="curiosity",
    )
    state, _ = await authority.propose_initiative_to_state(
        state,
        "Stabilize thermal load",
        source="executive_closure",
        urgency=0.9,
        triggered_by="stability",
    )

    promoted_state, initiative, decision = await authority.promote_next_initiative(state, source="mind_tick")

    assert decision["action"] == "promoted"
    assert initiative is not None
    assert promoted_state.cognition.current_objective == "Stabilize thermal load"
    assert promoted_state.cognition.current_origin in {"executive_closure", "mind_tick"}
    assert len(promoted_state.cognition.pending_initiatives) == 1

    completed_state, completion = await authority.complete_current_objective(
        promoted_state,
        reason="tick_cycle_complete",
        source="mind_tick",
    )

    assert completion["action"] == "completed"
    assert completed_state.cognition.current_objective is None
    assert all(item.get("goal") != "Stabilize thermal load" for item in completed_state.cognition.pending_initiatives)
