import asyncio
import time
from types import SimpleNamespace

import pytest

from core.coordinators.metabolic_coordinator import MetabolicCoordinator


class _Tracker:
    def __init__(self):
        self.tasks = []

    def create_task(self, coro, name=None):
        task = asyncio.create_task(coro, name=name)
        self.tasks.append((name, task))
        return task

    track = create_task


def _orch_for_cycle():
    async def _trigger(_event, _payload):
        return None

    async def _save_state(_reason):
        return None

    async def _acquire_next_message():
        return None

    return SimpleNamespace(
        status=SimpleNamespace(
            cycle_count=5,
            is_processing=False,
            singularity_threshold=False,
            state="idle",
            last_user_interaction_time=0.0,
            acceleration_factor=1.0,
        ),
        hooks=SimpleNamespace(trigger=_trigger),
        _save_state_async=_save_state,
        drive_controller=None,
        drives=None,
        is_busy=False,
        latent_core=None,
        predictive_model=None,
        kernel=None,
        state=None,
        message_queue=SimpleNamespace(_queue=[]),
        _acquire_next_message=_acquire_next_message,
        _dispatch_message=lambda _message: None,
        memory_manager=None,
        swarm=None,
        _last_thought_time=time.time(),
        _last_pulse=time.time(),
    )


@pytest.mark.asyncio
async def test_bci_subscription_is_shutdown_bounded(monkeypatch):
    import core.coordinators.metabolic_coordinator as metabolic_module
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown

    clear_shutdown_request()
    tracker = _Tracker()
    queue = asyncio.Queue()

    class _EventBus:
        async def subscribe(self, _topic):
            return queue

    monkeypatch.setattr(metabolic_module, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(metabolic_module, "_BCI_EVENT_POLL_SECONDS", 0.01)
    monkeypatch.setattr("core.event_bus.get_event_bus", lambda: _EventBus())

    coord = MetabolicCoordinator(orch=_orch_for_cycle())
    coord._consume_energy = lambda _cost: False
    coord.update_liquid_pacing = lambda: None
    coord.manage_memory_hygiene = lambda: None
    coord.process_world_decay = lambda: None
    coord.trigger_autonomous_thought = lambda _has_message: None
    coord.run_terminal_self_heal = lambda: None

    try:
        await coord._process_metabolic_tasks()
        subscription = next(
            task for name, task in tracker.tasks if name == "metabolic.bci_event_subscription"
        )

        await queue.put(("topic", "id", {"data": {"command": "blink", "confidence": 0.75}}))
        await asyncio.sleep(0.03)

        assert coord._neural_events[-1] == {"command": "blink", "confidence": 0.75}

        request_shutdown("unit_test")
        await asyncio.wait_for(subscription, timeout=0.5)
        assert subscription.done()
    finally:
        clear_shutdown_request()


def test_bci_event_extraction_accepts_tuple_and_mapping_payloads():
    coord = MetabolicCoordinator(orch=SimpleNamespace())

    assert coord._extract_bci_event_data(
        ("topic", "id", {"data": {"command": "left", "confidence": 0.9}})
    ) == {"command": "left", "confidence": 0.9}
    assert coord._extract_bci_event_data({"command": "right"}) == {"command": "right"}
    assert coord._extract_bci_event_data(("bad",)) is None
    assert coord._extract_bci_event_data("bad") is None


@pytest.mark.asyncio
async def test_track_metabolic_task_creates_active_set_and_cleans_finished_task(monkeypatch):
    import core.coordinators.metabolic_coordinator as metabolic_module

    tracker = _Tracker()
    monkeypatch.setattr(metabolic_module, "get_task_tracker", lambda: tracker)
    orch = SimpleNamespace()
    coord = MetabolicCoordinator(orch=orch)

    async def _work():
        return "ok"

    task = coord.track_metabolic_task("metabolic.contract", _work())

    assert "metabolic.contract" in orch._active_metabolic_tasks
    assert task is tracker.tasks[0][1]
    assert await task == "ok"
    await asyncio.sleep(0)
    assert "metabolic.contract" not in orch._active_metabolic_tasks


@pytest.mark.asyncio
async def test_autonomous_thought_uses_safe_interval_when_runtime_value_is_invalid(monkeypatch):
    import core.coordinators.metabolic_coordinator as metabolic_module

    tracker = _Tracker()
    ran = []

    async def _perform():
        ran.append(True)

    orch = SimpleNamespace(
        cognitive_engine=SimpleNamespace(singularity_factor="invalid"),
        _current_thought_task=None,
        _last_thought_time=time.time() - 120.0,
        singularity_monitor=None,
        kernel=SimpleNamespace(volition_level=2),
        boredom=0,
        _perform_autonomous_thought=_perform,
    )
    coord = MetabolicCoordinator(orch=orch)

    monkeypatch.setattr(metabolic_module, "get_task_tracker", lambda: tracker)
    monkeypatch.setattr(metabolic_module, "runtime_mode_value", lambda *_args, **_kwargs: "invalid")

    await coord.trigger_autonomous_thought(False)
    await orch._current_thought_task

    assert ran == [True]
    assert tracker.tasks[0][0] == "metabolic.autonomous_thought"
