"""A certificate must describe the checkpoint actually loaded by its runner."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools import measure_fusion_channel as runner


@pytest.mark.parametrize("explicit", [True, False])
def test_probe_load_and_certificate_share_resolved_checkpoint(tmp_path, monkeypatch, explicit):
    monkeypatch.setattr("core.brain.llm.model_registry.resolve_cortex_bound_artifact",
                        lambda path: SimpleNamespace(matched=False, reason="non_cortex_model"))
    checkpoint = tmp_path / "checkpoint"
    model, tokenizer = object(), object()
    loaded, described, measured = [], [], []
    monkeypatch.setattr("mlx_lm.load", lambda path: (loaded.append(path) or model, tokenizer))
    args = SimpleNamespace(model="some/default-model", model_path=str(checkpoint) if explicit else "",
                           sweep="0.05", steps=2, seed=5, no_write=True)
    def descriptor(path, **kwargs):
        described.append((path, kwargs))
        return {"descriptor_sha256": "actual-basis"}
    class Engine:
        _hooks = [SimpleNamespace(_layer_idx=3)]
        def attach(self, actual, tok, **kwargs):
            assert actual is model and tok is tokenizer
            assert kwargs["model_path"] == checkpoint
            return True
        def set_alpha(self, value):
            pass
        def controlled_measurement(self):
            return nullcontext()
    def measure(actual, tok, hooks, set_alpha, **kwargs):
        assert actual is model and tok is tokenizer
        measured.append(kwargs)
        return []
    assert runner._measure(args, checkpoint, descriptor, Engine, measure, None) == 1
    assert loaded == [str(checkpoint)]
    assert described == [(checkpoint, {"repository_id": "" if explicit else args.model})]
    assert measured[0]["model_name"] == str(checkpoint)
    assert measured[0]["model_identity"] == "actual-basis"


def test_registered_repository_and_revision_are_part_of_the_probe_identity(tmp_path, monkeypatch):
    registered = {"descriptor_sha256": "registered", "repository_id": "local/resident", "revision": "fusion-v2"}
    monkeypatch.setattr("core.brain.llm.model_registry.resolve_cortex_bound_artifact",
                        lambda path: SimpleNamespace(matched=True, descriptor=registered))
    observed = []
    def describe(path, **kwargs):
        observed.append((path, kwargs))
        return registered.copy()
    args = SimpleNamespace(model="ignored/default", model_path=str(tmp_path), require_active_cortex=True)
    assert runner._probe_descriptor(args, tmp_path, describe) == registered
    assert observed == [(tmp_path, {"repository_id": "local/resident", "revision": "fusion-v2"})]


@pytest.mark.parametrize("matched", [False, True])
def test_active_identity_failure_precedes_model_load(tmp_path, monkeypatch, matched):
    monkeypatch.setattr("core.brain.llm.model_registry.resolve_cortex_bound_artifact",
        lambda path: SimpleNamespace(matched=matched, reason="authority_unavailable",
            descriptor={"descriptor_sha256": "registered"}))
    loaded = []
    monkeypatch.setattr("mlx_lm.load", lambda path: loaded.append(path))
    args = SimpleNamespace(model="ignored/default", model_path=str(tmp_path), require_active_cortex=True)
    with pytest.raises(ValueError, match="fusion_probe_active"):
        runner._measure(args, tmp_path, lambda *a, **k: {"descriptor_sha256": "changed"}, None, None, None)
    assert not loaded
