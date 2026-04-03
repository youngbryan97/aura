import asyncio
from types import SimpleNamespace

import pytest

from core.phases.sensory_ingestion import SensoryIngestionPhase
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_sensory_ingestion_preserves_background_origin_from_five_tuple_queue_item():
    orchestrator = SimpleNamespace(message_queue=asyncio.Queue(), _last_thought_time=0.0)
    await orchestrator.message_queue.put(
        (15, 1234.5, 7, "Impulse: reflect on the previous exchange.", "autonomous_thought")
    )
    container = SimpleNamespace(get=lambda name, default=None: orchestrator if name == "orchestrator" else default)
    phase = SensoryIngestionPhase(container)

    state = AuraState.default()
    new_state = await phase.execute(state)

    assert len(new_state.cognition.working_memory) == 1
    entry = new_state.cognition.working_memory[-1]
    assert entry["content"] == "Impulse: reflect on the previous exchange."
    assert entry["origin"] == "autonomous_thought"
    assert entry["role"] == "system"


@pytest.mark.asyncio
async def test_sensory_ingestion_keeps_user_role_for_user_facing_queue_origin():
    orchestrator = SimpleNamespace(message_queue=asyncio.Queue(), _last_thought_time=0.0)
    await orchestrator.message_queue.put((5, 1234.5, 8, "Hello Aura", "api"))
    container = SimpleNamespace(get=lambda name, default=None: orchestrator if name == "orchestrator" else default)
    phase = SensoryIngestionPhase(container)

    state = AuraState.default()
    new_state = await phase.execute(state)

    entry = new_state.cognition.working_memory[-1]
    assert entry["content"] == "Hello Aura"
    assert entry["origin"] == "api"
    assert entry["role"] == "user"
