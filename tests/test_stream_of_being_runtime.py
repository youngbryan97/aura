import asyncio
from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker


def test_stream_quarantines_corrupt_persisted_state(tmp_path, service_container):
    state_path = tmp_path / "stream_state.json"
    state_path.write_text("{not valid json", encoding="utf-8")

    from core.consciousness.stream_of_being import StreamOfBeing

    stream = StreamOfBeing(save_dir=tmp_path)

    assert not state_path.exists()
    assert list(tmp_path.glob("stream_state.corrupt.*.json"))
    assert stream.get_status()["thread_length"] == 0


def test_experience_integrator_sanitizes_malformed_live_inputs(service_container):
    class BadSubstrate:
        _current_phi = float("nan")
        em_field_magnitude = "not-a-number"
        start_time = "not-a-time"

        def get_substrate_affect(self):
            return {
                "valence": float("nan"),
                "arousal": "hot",
                "energy": float("inf"),
                "volatility": "-3",
            }

    class BadMarkers:
        heart_rate = "fast"
        gsr = float("inf")

        def get_wheel(self):
            return {"primary": {"joy": 0.9, "bad": float("nan")}}

    class BadAffect:
        markers = BadMarkers()

    class BadDrives:
        urgency = float("nan")

        def get_dominant_motivation(self):
            return None

    service_container.register_instance("conscious_substrate", BadSubstrate(), required=False)
    service_container.register_instance("affect_engine_v2", BadAffect(), required=False)
    service_container.register_instance("drives", BadDrives(), required=False)

    from core.consciousness.stream_of_being import ExperienceIntegrator

    moment = ExperienceIntegrator().synthesize()

    assert moment.substrate.valence == 0.0
    assert moment.substrate.arousal == 0.3
    assert moment.substrate.energy == 0.7
    assert moment.substrate.volatility == 0.0
    assert moment.substrate.phi == 0.0
    assert moment.drive.dominant_drive == "at_rest"
    assert 0.0 <= moment.synthesis_depth <= 1.0
    assert moment.interior_text


def test_deep_narrative_accepts_thought_objects_and_scrubs_leakage(tmp_path, service_container):
    class Router:
        async def think(self, **_kwargs):
            return SimpleNamespace(
                content=(
                    "Thinking: A bright pressure gathers into a coherent thread. "
                    "I am present with the work, steady enough to keep moving."
                )
            )

    service_container.register_instance("llm_router", Router(), required=False)

    from core.consciousness.stream_of_being import NowMoment, StreamOfBeing

    stream = StreamOfBeing(save_dir=tmp_path)
    moment = NowMoment()
    stream._thread.add(moment)

    asyncio.run(stream._run_deep_narrative(moment))

    assert stream._deep_narrative.startswith("A bright pressure")
    assert "Thinking:" not in stream._deep_narrative
    assert stream._deep_narrative_timestamp > 0
    assert stream._thread.current_moment.is_llm_generated is True


def test_rejected_deep_narrative_backs_off_instead_of_hammering_llm(
    tmp_path,
    service_container,
):
    class Router:
        async def think(self, **_kwargs):
            return "Still pond, swirling leaves, thought is a leaf without urgency."

    service_container.register_instance("llm_router", Router(), required=False)

    from core.consciousness.stream_of_being import NowMoment, StreamOfBeing

    stream = StreamOfBeing(save_dir=tmp_path)
    stream._deep_narrative = "old narrative that should be purged"
    stream._deep_narrative_timestamp = 0.0
    stream._thread.add(NowMoment())

    asyncio.run(stream._run_deep_narrative(stream._thread.current_moment))

    assert stream._deep_narrative == ""
    assert stream._deep_narrative_timestamp > 0.0


def test_stream_start_fails_closed_when_task_tracker_cannot_supervise(
    tmp_path,
    monkeypatch,
    service_container,
):
    class BrokenTracker:
        def __init__(self):
            self.called = False

        def create_task(self, *_args, **_kwargs):
            self.called = True
            raise RuntimeError("scheduler offline")

    import core.consciousness.stream_of_being as stream_module

    monkeypatch.setattr(stream_module, "get_task_tracker", lambda: BrokenTracker())

    stream = stream_module.StreamOfBeing(save_dir=tmp_path)
    get_degradation_tracker().reset()

    try:
        with pytest.raises(RuntimeError, match="scheduler offline"):
            asyncio.run(stream.start())

        assert stream.get_status()["running"] is False
        assert get_degradation_tracker().count("stream_of_being", "critical") >= 1
    finally:
        get_degradation_tracker().reset()


def test_boot_stream_wires_orchestrator_only_once(tmp_path, monkeypatch, service_container):
    import core.consciousness.stream_of_being as stream_module

    stream = stream_module.StreamOfBeing(save_dir=tmp_path)
    monkeypatch.setattr(stream_module, "get_stream", lambda: stream)

    calls = []

    class Orchestrator:
        async def handle_input(self, message):
            calls.append(message)
            return "ok"

    orchestrator = Orchestrator()

    async def scenario():
        await stream_module.boot_stream_of_being(orchestrator)
        await stream_module.boot_stream_of_being(orchestrator)
        result = await orchestrator.handle_input("hello")
        await stream.stop()
        return result

    result = asyncio.run(scenario())

    assert result == "ok"
    assert calls == ["hello"]
