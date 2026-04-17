import core.event_bus as event_bus

from core.security import cheat_codes as cheat_code_module
from core.security import trust_engine as trust_engine_module
from core.security.trust_engine import TrustEngine, TrustLevel


class _Bus:
    def __init__(self, published):
        self._published = published

    def publish_threadsafe(self, topic, payload):
        self._published.append((topic, payload))


def test_establish_sovereign_session_can_be_silent(monkeypatch):
    published = []
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: _Bus(published))

    engine = TrustEngine()
    level = engine.establish_sovereign_session(reason="test", announce=False)

    assert level == TrustLevel.SOVEREIGN
    assert published == []


def test_sovereign_cheat_code_emits_single_message(monkeypatch):
    published = []
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: _Bus(published))
    monkeypatch.setattr(cheat_code_module, "_matches_sovereign_code", lambda _code: True)

    trust_engine_module._engine = None
    try:
        result = cheat_code_module.activate_cheat_code("owner", silent=False, source="test")
    finally:
        trust_engine_module._engine = None

    assert result["ok"] is True
    assert len(published) == 1
    assert published[0][0] == "telemetry"
    assert published[0][1]["message"] == result["message"]
