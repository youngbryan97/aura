import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.consciousness.executive_authority import ExecutiveAuthority
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_executive_authority_queues_and_refreshes_initiatives(service_container):
    state = AuraState()

    repo = SimpleNamespace()
    repo.current = state
    repo.get_current = AsyncMock(side_effect=lambda: repo.current)

    async def _commit(new_state, _cause):
        repo.current = new_state
        return new_state

    repo.commit = AsyncMock(side_effect=_commit)
    service_container.register_instance("state_repository", repo)

    authority = ExecutiveAuthority()

    first = await authority.queue_initiative(
        "Investigate thermal surprise",
        source="personhood_engine",
        urgency=0.55,
    )
    second = await authority.queue_initiative(
        "Investigate thermal surprise",
        source="personhood_engine",
        urgency=0.9,
    )

    assert first["ok"] is True
    assert first["action"] == "queued"
    assert second["reason"] == "initiative_refreshed"
    assert repo.commit.await_count == 2
    assert repo.current.cognition.pending_initiatives[0]["goal"] == "Investigate thermal surprise"
    assert repo.current.cognition.pending_initiatives[0]["urgency"] == pytest.approx(0.9, abs=1e-6)


@pytest.mark.asyncio
async def test_executive_authority_reroutes_low_urgency_output_to_secondary(service_container):
    gate = SimpleNamespace(emit=AsyncMock())
    orch = SimpleNamespace(
        output_gate=gate,
        _last_user_interaction_time=time.time() - 5.0,
        status=SimpleNamespace(is_processing=False),
        is_busy=False,
    )

    service_container.register_instance("orchestrator", orch)
    service_container.register_instance(
        "executive_closure",
        SimpleNamespace(
            get_status=lambda: {
                "dominant_need": "stability",
                "need_pressure": 0.91,
                "closure_score": 0.24,
                "vitality": 0.78,
            }
        ),
    )

    authority = ExecutiveAuthority(orch)
    result = await authority.release_expression(
        "I noticed a pattern worth holding internally for now.",
        source="personhood_engine",
        urgency=0.45,
    )

    assert result["ok"] is True
    assert result["target"] == "secondary"
    gate.emit.assert_awaited_once()
    _, kwargs = gate.emit.await_args
    assert kwargs["target"] == "secondary"
    assert kwargs["metadata"]["autonomous"] is True
    assert kwargs["metadata"]["authority_reason"] in {"runtime_guard", "closure_low", "user_recently_active"}


@pytest.mark.asyncio
async def test_executive_authority_allows_high_urgency_primary_release(service_container):
    gate = SimpleNamespace(emit=AsyncMock())
    orch = SimpleNamespace(
        output_gate=gate,
        _last_user_interaction_time=time.time() - 300.0,
        status=SimpleNamespace(is_processing=False),
        is_busy=False,
    )

    service_container.register_instance("orchestrator", orch)
    service_container.register_instance(
        "executive_closure",
        SimpleNamespace(
            get_status=lambda: {
                "dominant_need": "curiosity",
                "need_pressure": 0.21,
                "closure_score": 0.82,
                "vitality": 0.91,
            }
        ),
    )

    authority = ExecutiveAuthority(orch)
    result = await authority.release_expression(
        "I found something important and I want to share it with you now.",
        source="initiative_engine",
        urgency=0.95,
        metadata={"voice": True},
    )

    assert result["ok"] is True
    assert result["target"] == "primary"
    gate.emit.assert_awaited_once()
    _, kwargs = gate.emit.await_args
    assert kwargs["target"] == "primary"
    assert kwargs["metadata"]["spontaneous"] is True
    assert kwargs["metadata"]["force_user"] is True
    assert kwargs["metadata"]["voice"] is True
