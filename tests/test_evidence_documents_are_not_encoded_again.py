"""Unchanged history reuses measurements without crossing encoder identities."""
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from core.cognition import evidence_relevance as module


@pytest.fixture
def encoder(monkeypatch):
    calls = []
    def encode(texts):
        calls.append(list(texts))
        return [[float(len(text)), 1.] for text in texts]
    engine = SimpleNamespace(embed_batch=encode)
    monkeypatch.setattr(module, '_embedder', lambda: engine)
    monkeypatch.setattr(module, '_DOCUMENT_CACHE', OrderedDict())
    monkeypatch.setattr(module, '_CACHE_ENGINE_TOKEN', None)
    for name in ('_ANCHOR_CACHE', '_REQUEST_CACHE', '_ALIGNMENT_QUERY_CACHE'):
        monkeypatch.setattr(module, name, {})
    monkeypatch.setattr(module, '_WARMING', {'asked': False})
    return engine, calls


def test_unchanged_passages_keep_the_same_vectors_without_another_forward(encoder):
    _engine, calls = encoder
    first = module._embed_documents(['a previous exchange', 'another exchange', 'a previous exchange'])
    second = module._embed_documents(['another exchange', 'the new exchange', 'a previous exchange'])
    assert calls == [['a previous exchange', 'another exchange'], ['the new exchange']]
    assert first[0] == first[2] == second[2]
    assert first[1] == second[0]


def test_model_replacement_reencodes_even_identical_text(encoder, monkeypatch):
    _engine, _calls = encoder
    old = module._embed_documents(['same history'])
    replacement = SimpleNamespace(embed_batch=lambda texts: [[7., 8.] for _ in texts])
    monkeypatch.setattr(module, '_embedder', lambda: replacement)
    assert module._embed_documents(['same history']) == [[7., 8.]]
    assert old != [[7., 8.]]


def test_a_failed_measurement_is_retried_instead_of_cached(encoder):
    engine, calls = encoder
    encode = engine.embed_batch
    engine.embed_batch = lambda texts: [None for _ in texts]
    assert module._embed_documents(['history']) == [None]
    engine.embed_batch = encode
    assert module._embed_documents(['history']) == [[7., 1.]]
    assert calls == [['history']]


def test_switch_during_encoding_cannot_cache_the_retired_vector(encoder):
    engine, _calls = encoder
    def switch(texts):
        module._bind_cache_to_engine(SimpleNamespace(identity='replacement'))
        return [[1., 0.] for _ in texts]
    engine.embed_batch = switch
    assert module._embed_documents(['history']) == [None]
    assert not module._DOCUMENT_CACHE


def test_one_batch_keeps_its_encoder_even_if_the_container_replaces_it(encoder, monkeypatch):
    engine, calls = encoder
    next_engine = SimpleNamespace(embed_batch=lambda texts: [[99., 99.] for _ in texts])
    engines = iter([engine, next_engine])
    monkeypatch.setattr(module, '_embedder', lambda: next(engines))
    assert module._embed_documents(['history']) == [[7., 1.]]
    assert calls == [['history']]
    assert module._embed_documents(['history']) == [[99., 99.]]


def test_in_place_model_change_during_encoding_cannot_keep_the_old_vector(encoder):
    engine, _calls = encoder
    def switch(texts):
        engine.identity = 'a different model'
        return [[1., 0.] for _ in texts]
    engine.embed_batch = switch
    assert module._embed_documents(['history']) == [None]
    assert not module._DOCUMENT_CACHE


def test_an_unsized_measurement_cannot_raise_from_cache_retention(encoder):
    engine, _calls = encoder
    engine.embed_batch = lambda texts: [3. for _ in texts]
    assert module._embed_documents(['history']) == [3.]
    assert not module._DOCUMENT_CACHE


def test_document_vector_retention_is_bounded(encoder, monkeypatch):
    _engine, calls = encoder
    monkeypatch.setattr(module, '_REQUEST_CACHE_MAX', 2)
    module._embed_documents(['one', 'two'])
    module._embed_documents(['three'])
    assert len(module._DOCUMENT_CACHE) == 2
    assert list(module.document_cache_is_bounded()) == []
    module._embed_documents(['one'])
    assert calls[-1] == ['one']
    module._DOCUMENT_CACHE['unadmitted'] = [1., 0.]
    assert list(module.document_cache_is_bounded())
