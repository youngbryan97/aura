"""A memory stored for the very context a reply is in decides the next token; any other memory only informs it.

MEASURED 10 November 2026 (docs/evidence/F03_ONE_SHOT_MEMORY_PAIRED_TRIALS_2026-11-10.md): an exact match weighed 0.29
against her model and lost to any model more than about 40% sure of something else, so her 27B recalled 9 of 40
invented facts and 8 of 80. Exactness is the store's own measure: the widest gap between a key as ingest stores it
and the same position as the decode computes it.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.brain.nonparametric_memory import EntryProvenance, NonParametricMemory

DIM, VOCABULARY = 32, 48


def _unit(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


def _store(tmp_path, *, measured=True):
    rng = np.random.default_rng(17)
    memory = NonParametricMemory(DIM, tmp_path / "store", max_entries=1024)
    keys = _unit(rng.normal(size=(40, DIM)))
    for i, key in enumerate(keys):
        memory.add(key, i, f"t{i}", provenance=EntryProvenance(source_id="test", principal="bryan"))
    memory.took_queries(_unit(rng.normal(size=(64, DIM))))
    if measured:
        decoded = _unit(keys + 1e-4 * rng.normal(size=keys.shape).astype(np.float32))
        memory.observe_exact_repeat(keys, decoded)
    return memory, keys


def test_the_tolerance_is_measured_kept_and_needs_the_mean(tmp_path):
    memory = NonParametricMemory(DIM, tmp_path / "cold", max_entries=256)
    keys = _unit(np.random.default_rng(1).normal(size=(4, DIM)))
    assert memory.observe_exact_repeat(keys, keys) is None, "no centred similarity before the mean is ready"
    assert memory.exact_tolerance() is None and not memory.holds_this_context(1.0)

    warm, _keys = _store(tmp_path)
    tolerance = warm.exact_tolerance()
    assert tolerance is not None and 0.0 < tolerance < 1e-3
    assert warm.holds_this_context(1.0) and not warm.holds_this_context(1.0 - 2 * tolerance)
    assert warm.persist()
    again = NonParametricMemory(DIM, tmp_path / "store", max_entries=1024)
    assert again._exact_tolerance == pytest.approx(tolerance)


def test_on_the_gpu_an_exact_match_overrules_a_sure_model_and_a_near_one_does_not(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from core.brain.recall_on_the_gpu import RecallOnTheGpu

    memory, keys = _store(tmp_path)
    sure = np.full((1, VOCABULARY), -10.0, dtype=np.float32)
    sure[0, 47] = 10.0  # the model is all but certain of token 47
    logits = mx.array(sure)
    exact = np.asarray(RecallOnTheGpu(memory, principal="bryan")(None, logits, hidden=mx.array(keys[5])))
    assert int(exact.argmax()) == 5, "the entry stored for this context decides"
    near = _unit(keys[5] + 0.08 * np.random.default_rng(2).normal(size=DIM).astype(np.float32))
    assert memory.query(near, k=1, update_mean=False)[0].similarity < 1.0 - memory.exact_tolerance()
    informed = np.asarray(RecallOnTheGpu(memory, principal="bryan")(None, logits, hidden=mx.array(near)))
    assert int(informed.argmax()) == 47, "a memory of another context informs; it does not overrule"


def test_without_a_measured_tolerance_nothing_chooses(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from core.brain.recall_on_the_gpu import RecallOnTheGpu

    memory, keys = _store(tmp_path, measured=False)
    sure = np.full((1, VOCABULARY), -10.0, dtype=np.float32)
    sure[0, 47] = 10.0
    out = np.asarray(RecallOnTheGpu(memory, principal="bryan")(None, mx.array(sure), hidden=mx.array(keys[5])))
    assert int(out.argmax()) == 47


def test_the_cpu_recall_chooses_the_same_way(tmp_path):
    mx = pytest.importorskip("mlx.core")
    from core.brain.nonparametric_binding import MemoryBinding
    from core.brain.nonparametric_worker import make_tapped_nonparametric_processor

    class _Tap:
        last_key = None

    memory, keys = _store(tmp_path)
    tap = _Tap()
    processor = make_tapped_nonparametric_processor(tap, memory, binding=MemoryBinding(principal="bryan",
                                                                                       source_id="test"))
    sure = np.full((1, VOCABULARY), -10.0, dtype=np.float32)
    sure[0, 47] = 10.0
    tap.last_key = keys[9]
    assert int(np.asarray(processor(None, mx.array(sure))).argmax()) == 9


def test_ingest_measures_the_tolerance_through_the_decode_path(tmp_path):
    from core.brain.nonparametric_ingest import NonParametricIngestor, ingest_provenance
    from tests.test_nonparametric_ingest import FakeBatchSeqEncoder

    class _Decoding(FakeBatchSeqEncoder):
        dim = DIM

        def encode_decode_path_sequence_ids(self, ids, start):
            stored = self.encode_hidden_sequence_ids(ids)[start:len(ids) - 1]
            noise = 1e-4 * np.random.default_rng(len(ids)).normal(size=stored.shape).astype(np.float32)
            return _unit(stored + noise)

    memory = NonParametricMemory(DIM, tmp_path / "ingest", max_entries=1024)
    memory.took_queries(_unit(np.random.default_rng(4).normal(size=(64, DIM))))
    ingestor = NonParametricIngestor(memory, provenance=ingest_provenance(principal="bryan", source_id="test"),
                                     dedup_path=tmp_path / "seen.json")
    assert ingestor.ingest_sequence("the valve torque limit is", "71.6 newton metres", _Decoding()) > 0
    tolerance = memory.exact_tolerance()
    assert tolerance is not None and 0.0 < tolerance < 1e-3
