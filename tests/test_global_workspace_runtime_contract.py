from pathlib import Path

import pytest


def test_global_workspace_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/consciousness/global_workspace.py")) == []


def _install_services(monkeypatch, *, mycelium=None, inhibition=None):
    import core.consciousness.global_workspace as workspace_module

    def _get(name, default=None):
        if name == "mycelial_network":
            return mycelium
        if name == "inhibition_manager":
            return inhibition
        return default

    monkeypatch.setattr(workspace_module.ServiceContainer, "get", staticmethod(_get))


def _install_auxiliary_feeds(monkeypatch, *, broken=False):
    import core.consciousness.peripheral_awareness as peripheral_awareness
    import core.consciousness.theory_arbitration as theory_arbitration
    import core.thought_stream as thought_stream
    import core.unity as unity

    class Peripheral:
        def process_workspace_results(self, *, winner_source, all_candidates):
            if broken:
                reason = f"{winner_source}:{len(all_candidates)}:peripheral offline"
                raise RuntimeError(reason)
            return {"winner_source": winner_source, "count": len(all_candidates)}

    class UnityRuntime:
        def record_workspace_competition(self, winner, losers):
            if broken:
                reason = f"{winner.source}:{len(losers)}:unity offline"
                raise RuntimeError(reason)
            return True

    class TheoryArbitration:
        def log_prediction(self, *, theory, event_id, prediction, confidence):
            if broken:
                reason = f"{theory}:{event_id}:theory feed offline"
                raise RuntimeError(reason)
            return prediction

    class Emitter:
        def emit(self, *, title, content, level, metadata):
            if broken:
                reason = f"{title}:{level}:thought stream offline"
                raise RuntimeError(reason)
            return metadata

    monkeypatch.setattr(peripheral_awareness, "get_peripheral_awareness_engine", lambda: Peripheral())
    monkeypatch.setattr(unity, "get_unity_runtime", lambda: UnityRuntime())
    monkeypatch.setattr(theory_arbitration, "get_theory_arbitration", lambda: TheoryArbitration())
    monkeypatch.setattr(thought_stream, "get_emitter", lambda: Emitter())


@pytest.mark.asyncio
async def test_workspace_records_auxiliary_failures_without_losing_winner(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class BrokenMycelium:
        def get_hypha(self, source, target):
            reason = f"{source}:{target}:mycelium offline"
            raise RuntimeError(reason)

    class BrokenAttention:
        async def set_focus(self, *, content, source, priority):
            reason = f"{source}:{priority}:focus unavailable"
            raise RuntimeError(reason)

    _install_services(monkeypatch, mycelium=BrokenMycelium())
    _install_auxiliary_feeds(monkeypatch, broken=True)
    workspace = GlobalWorkspace(attention_schema=BrokenAttention())

    assert await workspace.submit(CognitiveCandidate("urgent content", "drive", 1.0))
    winner = await workspace.run_competition()

    degraded = workspace.get_snapshot()["degraded_channels"]
    assert winner is not None
    assert winner.source == "drive"
    assert set(degraded) >= {
        "workspace_pulse",
        "peripheral_awareness",
        "unity_runtime",
        "theory_arbitration",
        "thought_stream",
        "attention_schema",
    }


@pytest.mark.asyncio
async def test_workspace_isolates_processor_failure_and_continues_broadcast(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    _install_services(monkeypatch)
    _install_auxiliary_feeds(monkeypatch, broken=False)
    workspace = GlobalWorkspace()
    received = []

    async def broken_processor(event):
        reason = f"{len(event.winners)}:processor unavailable"
        raise RuntimeError(reason)

    async def working_processor(event):
        received.append(event.winners[0].source)

    workspace.register_processor(broken_processor)
    workspace.register_processor(working_processor)

    assert await workspace.submit(CognitiveCandidate("processor test", "memory", 0.9))
    winner = await workspace.run_competition()

    snapshot = workspace.get_snapshot()
    assert winner is not None
    assert received == ["memory"]
    assert "processor_broadcast" in snapshot["degraded_channels"]
    assert any(name.endswith("broken_processor") for name in snapshot["processor_failures"])


@pytest.mark.asyncio
async def test_workspace_flood_guard_drops_bid_and_records_reflex_failure(monkeypatch):
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    class BrokenMycelium:
        def get_hypha(self, source, target):
            reason = f"{source}:{target}:flood reflex offline"
            raise RuntimeError(reason)

    _install_services(monkeypatch, mycelium=BrokenMycelium())
    workspace = GlobalWorkspace()
    workspace._candidates = [
        CognitiveCandidate(f"content-{index}", f"source-{index}", 0.1)
        for index in range(workspace._MAX_CANDIDATES)
    ]

    accepted = await workspace.submit(CognitiveCandidate("extra", "overflow", 0.8))

    snapshot = workspace.get_snapshot()
    assert accepted is False
    assert "seizure_guard_reflex" in snapshot["degraded_channels"]
    assert len(workspace._candidates) == workspace._MAX_CANDIDATES
