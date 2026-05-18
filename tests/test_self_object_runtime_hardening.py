from core.identity import self_object
from core.identity.self_object import SelfObject


class _BrokenContainer:
    @staticmethod
    def get(*_args, **_kwargs):
        raise RuntimeError("container unavailable")


def test_self_object_readers_degrade_to_safe_identity_facets(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        self_object,
        "record_degradation",
        lambda module, exc: recorded.append((module, type(exc).__name__)),
    )

    assert SelfObject._read_drives(_BrokenContainer) == {}
    assert SelfObject._read_affect(_BrokenContainer) == {}
    assert SelfObject._read_active_goals(_BrokenContainer) == []
    assert SelfObject._read_recent_belief_revisions(_BrokenContainer) == []
    assert SelfObject._read_recent_memory_consolidations(_BrokenContainer) == []
    assert SelfObject._read_recent_self_mods(_BrokenContainer) == []

    assert recorded
    assert all(module == "self_object" for module, _ in recorded)
