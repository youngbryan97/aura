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



def _interactions(state):
    return [
        percept
        for percept in list(getattr(state.world, "recent_percepts", []) or [])
        if isinstance(percept, dict) and percept.get("type") == "interaction"
    ]


async def _ingest(message, origin, state=None, orchestrator=None):
    orchestrator = orchestrator or SimpleNamespace(message_queue=asyncio.Queue(), _last_thought_time=0.0)
    await orchestrator.message_queue.put((5, 1234.5, 8, message, origin))
    container = SimpleNamespace(get=lambda name, default=None: orchestrator if name == "orchestrator" else default)
    return await SensoryIngestionPhase(container).execute(state or AuraState.default()), orchestrator


@pytest.mark.asyncio
async def test_a_persons_message_is_perceived_as_an_interaction():
    """The affect phase maps `interaction` to trust and warmth, and until this
    nothing in production emitted one."""
    new_state, _ = await _ingest("Hello Aura", "api")
    seen = _interactions(new_state)
    assert len(seen) == 1
    assert seen[0]["content"] == "Hello Aura"
    assert seen[0]["source"] == "api"


@pytest.mark.asyncio
async def test_an_impulse_of_her_own_is_not_an_interaction():
    new_state, _ = await _ingest("Impulse: reflect on the previous exchange.", "autonomous_thought")
    assert _interactions(new_state) == []


@pytest.mark.asyncio
async def test_a_motor_reflex_is_not_an_interaction_though_it_gets_a_user_role():
    """The embodied feeds are user-facing and are not a person."""
    new_state, _ = await _ingest("[EMBODIED CONTROL CONTRACT] press left", "embodied_motor_reflex")
    assert new_state.cognition.working_memory[-1]["role"] == "user"
    assert _interactions(new_state) == []


@pytest.mark.asyncio
async def test_a_named_user_is_an_interaction_though_it_gets_a_system_role():
    new_state, _ = await _ingest("it is me again", "user:bryan")
    assert new_state.cognition.working_memory[-1]["role"] == "system"
    assert len(_interactions(new_state)) == 1


@pytest.mark.asyncio
async def test_a_message_ignored_as_a_duplicate_is_not_perceived_again():
    once, orchestrator = await _ingest("Hello Aura", "api")
    twice, _ = await _ingest("Hello Aura", "api", state=once, orchestrator=orchestrator)
    assert len(_interactions(twice)) == 1
