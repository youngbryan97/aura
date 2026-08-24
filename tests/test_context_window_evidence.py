"""A context window that says where its number came from.

The registry returned the same 32768 from a measured config and from three
separate dead ends, and no caller could tell them apart. Sizing the whole
prompt budget from an unmeasured number is the shape of the
indefinite-coherence defect; these tests pin the distinction.
"""
from __future__ import annotations

import json

import pytest

from core.brain.llm import context_window_evidence as cwe
from core.brain.llm.context_window_evidence import WindowSource


@pytest.fixture(autouse=True)
def _clean():
    cwe.reset_for_test()
    yield
    cwe.reset_for_test()


def test_measured_evidence_is_marked_measured():
    ev = cwe.measured(32768, model="m")
    assert ev.is_measured
    assert ev.source is WindowSource.MEASURED
    assert int(ev) == 32768


def test_assumed_evidence_is_not_measured():
    ev = cwe.assumed(32768, model="m", detail="nothing readable")
    assert not ev.is_measured
    assert ev.source is WindowSource.ASSUMED
    # Same NUMBER as a measured one — which is exactly why the label matters.
    assert int(ev) == 32768


def test_derived_evidence_is_measured_but_distinguishable():
    ev = cwe.derived(131072, model="m", detail="tokenizer maximum")
    assert ev.is_measured
    assert ev.source is WindowSource.DERIVED
    assert ev.to_dict()["source"] == "derived"


def test_an_assumption_is_reported_once_per_model():
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    before = sum(
        1 for r in tracker._records if r.subsystem == "context_window_evidence"
    )
    for _ in range(5):
        cwe.note_assumption(cwe.assumed(32768, model="repeat-model"))
    after = sum(
        1 for r in tracker._records if r.subsystem == "context_window_evidence"
    )
    assert after - before == 1, "an unreadable artifact should degrade once, not per prompt"


def test_a_measured_window_is_never_reported():
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    before = sum(
        1 for r in tracker._records if r.subsystem == "context_window_evidence"
    )
    cwe.note_assumption(cwe.measured(32768, model="fine-model"))
    after = sum(
        1 for r in tracker._records if r.subsystem == "context_window_evidence"
    )
    assert after == before


def test_registry_reports_measured_for_a_real_config(tmp_path, monkeypatch):
    from core.brain.llm import model_registry as mr

    model_dir = tmp_path / "FakeModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
    monkeypatch.setitem(mr.MODEL_PATHS, "FakeModel", model_dir)
    mr.get_model_context_window.cache_clear()

    ev = mr.get_context_window_evidence("FakeModel")
    assert ev.source is WindowSource.MEASURED
    assert ev.tokens == 8192
    # The int-returning entry point still behaves exactly as before.
    assert mr.get_model_context_window("FakeModel") == 8192


def test_registry_reads_nested_text_model_context(tmp_path, monkeypatch):
    from core.brain.llm import model_registry as mr

    model_dir = tmp_path / "NestedTextModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "multimodal_wrapper",
                "text_config": {"max_position_embeddings": 262144},
            }
        )
    )
    monkeypatch.setitem(mr.MODEL_PATHS, "NestedTextModel", model_dir)
    mr.get_model_context_window.cache_clear()

    ev = mr.get_context_window_evidence("NestedTextModel")
    assert ev.source is WindowSource.MEASURED
    assert ev.tokens == 262144


def test_registry_marks_an_unreadable_artifact_assumed(tmp_path, monkeypatch):
    from core.brain.llm import model_registry as mr

    model_dir = tmp_path / "EmptyModel"
    model_dir.mkdir()
    monkeypatch.setitem(mr.MODEL_PATHS, "EmptyModel", model_dir)
    mr.get_model_context_window.cache_clear()

    ev = mr.get_context_window_evidence("EmptyModel")
    assert ev.source is WindowSource.ASSUMED
    assert not ev.is_measured
    assert "no readable" in ev.detail


def test_tokenizer_only_window_is_derived_not_measured(tmp_path, monkeypatch):
    """A tokenizer maximum may need rope scaling actually switched on."""
    from core.brain.llm import model_registry as mr

    model_dir = tmp_path / "TokOnly"
    model_dir.mkdir()
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 131072})
    )
    monkeypatch.setitem(mr.MODEL_PATHS, "TokOnly", model_dir)
    mr.get_model_context_window.cache_clear()

    ev = mr.get_context_window_evidence("TokOnly")
    assert ev.source is WindowSource.DERIVED
    assert ev.tokens == 131072


def test_live_active_model_window_is_measured_not_guessed():
    """The number sizing every live prompt must come from the artifact.

    Skips when the model artifact is absent (worktrees carry no models/),
    because an artifact that is not there is not evidence of a defect.
    """
    from pathlib import Path

    from core.brain.llm import model_registry as mr

    path = Path(mr.get_runtime_model_path(mr.ACTIVE_MODEL))
    if not (path / "config.json").exists():
        pytest.skip("active model artifact is not present in this checkout")
    ev = mr.get_context_window_evidence()
    assert ev.is_measured, f"live context budget is a guess: {ev.detail}"
