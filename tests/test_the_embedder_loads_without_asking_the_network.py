"""The embedder loads from the copy on disk before it asks the network.

Measured 2026-09-19: building the retrieval embedder printed "You are sending
unauthenticated requests to the HF Hub" and took nine seconds, because the
constructor was not told the model was already cached. With no network the
same call waits out the hub's retries first. She is local; the one download
there is happens only when the model is not on the machine.
"""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from core.memory import embedding_model


class _Model:
    max_seq_length = embedding_model.MAX_INPUT_TOKENS


def _patch(monkeypatch: pytest.MonkeyPatch, cached: bool) -> list[dict]:
    calls: list[dict] = []

    def construct(_repo, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("local_files_only") and not cached:
            raise FileNotFoundError("not in the local cache")
        return _Model()

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = construct
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(
        "core.runtime.model_lane_control.require_active_synchronous_in_process_model_lane",
        lambda _lease: None,
    )
    return calls


def test_a_cached_model_never_touches_the_network(monkeypatch):
    calls = _patch(monkeypatch, cached=True)
    embedding_model.load_encoder(model_lane_lease=object())
    assert [call.get("local_files_only") for call in calls] == [True]


def test_a_model_not_on_the_machine_is_downloaded_once(monkeypatch):
    calls = _patch(monkeypatch, cached=False)
    embedding_model.load_encoder(model_lane_lease=object())
    assert [call.get("local_files_only") for call in calls] == [True, None]
