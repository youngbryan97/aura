"""A certificate must describe the checkpoint actually loaded by its runner."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tools import measure_fusion_channel as runner


@pytest.mark.parametrize("explicit", [True, False])
def test_probe_load_and_certificate_share_resolved_checkpoint(tmp_path, monkeypatch, explicit):
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
