"""Her one-shot memory's recall made on the GPU recalls what the recall made on the CPU recalls.

MEASURED 20 October 2026: her 27B decoded 17.4 tokens a second plainly and 10.2 with the CPU recall
(tools/her_decode_with_each_channel.py). Under a vocabulary of 64 the CPU recall's blend covers every token, so the
two can be held to the same distribution.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.nonparametric_memory import EntryProvenance, NonParametricMemory  # noqa: E402
from core.brain.recall_on_the_gpu import RecallOnTheGpu  # noqa: E402

DIM, VOCABULARY = 32, 48


class _Tap:
    last_hidden = None

    @property
    def last_key(self):
        h = np.asarray(self.last_hidden, dtype=np.float32)
        return h / np.linalg.norm(h)


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(7)
    memory = NonParametricMemory(DIM, tmp_path / "store", max_entries=1024)
    keys = rng.normal(size=(300, DIM)).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    for i, key in enumerate(keys):
        memory.add(key, i % VOCABULARY, f"t{i}", provenance=EntryProvenance(source_id="test", principal="bryan"))
    # Someone else's entry, exactly where the first query will be.
    memory.add(keys[3], 40, "theirs", provenance=EntryProvenance(source_id="test", principal="someone-else"))
    for one in rng.normal(size=(300, DIM)).astype(np.float32):  # enough queries for the centred mean
        memory.query(one / np.linalg.norm(one), k=1)
    return memory, keys


def _cpu(memory):
    from core.brain.nonparametric_binding import MemoryBinding
    from core.brain.nonparametric_worker import make_tapped_nonparametric_processor

    tap = _Tap()
    return tap, make_tapped_nonparametric_processor(tap, memory, binding=MemoryBinding(principal="bryan",
                                                                                         source_id="test"))


def _softmax(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    e = np.exp(values - values.max())
    return e / e.sum()


def test_far_from_every_entry_the_logits_are_the_models(store):
    memory, _keys = store
    logits = mx.array(np.random.default_rng(1).normal(size=(1, VOCABULARY)).astype(np.float32))
    far = mx.array(np.ones(DIM, dtype=np.float32))  # the common direction: nothing near it once centred
    recall = RecallOnTheGpu(memory, principal="bryan")
    out = recall(None, logits, hidden=far)
    assert np.array_equal(np.asarray(out), np.asarray(logits))


def test_at_an_entry_both_recalls_give_the_same_distribution(store):
    memory, keys = store
    logits = mx.array(np.random.default_rng(2).normal(size=(1, VOCABULARY)).astype(np.float32))
    tap, cpu = _cpu(memory)
    tap.last_hidden = keys[3]
    on_the_cpu = _softmax(np.asarray(cpu(None, logits)))
    on_the_gpu = _softmax(np.asarray(RecallOnTheGpu(memory, principal="bryan")(None, logits, hidden=mx.array(keys[3]))))
    model = _softmax(np.asarray(logits))
    assert not np.allclose(on_the_gpu, model, atol=1e-3), "recall fired"
    assert np.allclose(on_the_gpu, on_the_cpu, atol=2e-3)
    # Her entry's next token, not the other person's at the same place.
    assert on_the_gpu[3 % VOCABULARY] > model[3 % VOCABULARY]
    assert np.isclose(on_the_gpu[40], on_the_cpu[40], atol=2e-3)


def test_the_same_nearest_entry_twice_running_does_not_fire_twice(store):
    memory, keys = store
    logits = mx.array(np.zeros((1, VOCABULARY), dtype=np.float32))
    recall = RecallOnTheGpu(memory, principal="bryan")
    first = np.asarray(recall(None, logits, hidden=mx.array(keys[3])))
    second = np.asarray(recall(None, logits, hidden=mx.array(keys[3])))
    assert not np.array_equal(first, np.asarray(logits))
    assert np.array_equal(second, np.asarray(logits))


def test_the_replys_queries_are_given_back_to_the_store(store):
    memory, keys = store
    queried = memory._stats["queried"]
    recall = RecallOnTheGpu(memory, principal="bryan")
    logits = mx.array(np.zeros((1, VOCABULARY), dtype=np.float32))
    for i in (3, 4, 5):
        out = recall(None, logits, hidden=mx.array(keys[i]))
        mx.eval(out)
    recall.given_back()
    assert memory._stats["queried"] == queried + 3
    assert memory._stats["interpolated"] >= 3
