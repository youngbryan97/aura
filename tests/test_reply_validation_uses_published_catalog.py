import asyncio
from types import SimpleNamespace

from core.conversation import response_reliability as reliability


def test_snapshot_does_not_load_and_tracks_registry_publication(monkeypatch):
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine()
    monkeypatch.setattr(engine, "_ensure_catalog_loaded", lambda: (_ for _ in ()).throw(
        AssertionError("snapshot cannot own catalog initialization")
    ))
    assert engine.registered_skill_names_snapshot() is None
    with engine._catalog_guard():
        engine._skills = {"web_search": object()}
        engine._catalog_loaded = True
    assert engine.registered_skill_names_snapshot() == ("web_search",)
    with engine._catalog_guard():
        engine._skills = {"local_math": object()}
    assert engine.registered_skill_names_snapshot() == ("local_math",)


def test_ordinary_answers_do_not_read_the_capability_catalog(monkeypatch):
    def unexpected():
        raise AssertionError("an answer without a tool claim must not scan capabilities")

    monkeypatch.setattr(reliability, "_registered_capability_names", unexpected)
    assert not reliability._claims_a_capability_it_does_not_have(
        "why redo?", "Redo restores committed changes not yet written to disk."
    )


def test_live_claim_validation_reads_published_names_without_discovery(monkeypatch):
    import core.capability_engine as capabilities
    import core.skills.discovery as discovery

    names = ["web_search"]
    engine = SimpleNamespace(registered_skill_names_snapshot=lambda: tuple(names))
    monkeypatch.setattr(capabilities, "live_capability_engine", lambda: engine)
    monkeypatch.setattr(discovery, "build_skill_catalog", lambda: (_ for _ in ()).throw(
        AssertionError("live reply validation must not rebuild source catalog")
    ))
    assert not reliability._claims_a_capability_it_does_not_have("what did you do?", "I used my web_search tool.")
    assert reliability._claims_a_capability_it_does_not_have("what did you do?", "I used my invented_tool tool.")
    names.append("invented_tool")
    assert not reliability._claims_a_capability_it_does_not_have("what did you do?", "I used my invented_tool tool.")


def test_unavailable_catalog_on_loop_does_not_start_discovery(monkeypatch):
    import core.capability_engine as capabilities
    import core.skills.discovery as discovery

    monkeypatch.setattr(capabilities, "live_capability_engine", lambda: None)
    calls = []
    monkeypatch.setattr(discovery, "build_skill_catalog", lambda: calls.append(True))

    async def inspect():
        assert reliability._registered_capability_names() == frozenset()

    asyncio.run(inspect())
    assert calls == []
