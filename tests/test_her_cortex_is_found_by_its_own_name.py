"""Her cortex, asked for by its own directory name, is found where her pointer says it is.

Her worker asks the serving resolver for the cortex by its directory name, which
has no folder under the repository's models directory, so the resolver recorded
her own cortex as missing and answered a path that does not exist (her desktop
log, 23 October).
"""

from __future__ import annotations

from core.brain.llm import model_registry as registry


def test_the_active_cortex_by_its_name_is_the_active_cortex(tmp_path, monkeypatch):
    cortex = tmp_path / "elsewhere" / "Aura-Test-27B-persona"
    cortex.mkdir(parents=True)
    recorded = []
    monkeypatch.setattr(registry, "_current_cortex_path", lambda: cortex)
    monkeypatch.setattr(registry, "get_models_dir", lambda: tmp_path / "repository-models")
    monkeypatch.setattr(registry, "record_degradation", lambda *a, **k: recorded.append(a))
    assert registry.get_model_path("Aura-Test-27B-persona") == str(cortex.resolve())
    assert recorded == []


def test_a_name_that_is_not_the_cortex_is_still_said_to_be_missing(tmp_path, monkeypatch):
    cortex = tmp_path / "elsewhere" / "Aura-Test-27B-persona"
    cortex.mkdir(parents=True)
    recorded = []
    monkeypatch.setattr(registry, "_current_cortex_path", lambda: cortex)
    monkeypatch.setattr(registry, "get_models_dir", lambda: tmp_path / "repository-models")
    monkeypatch.setattr(registry, "record_degradation", lambda *a, **k: recorded.append(a))
    monkeypatch.setattr("core.brain.llm.a_model_in_the_local_cache.model_in_local_cache", lambda name: None)
    registry.get_model_path("Some-Other-Model-4bit")
    assert recorded, "a model that is nowhere is still reported"
