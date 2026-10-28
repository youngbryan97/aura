"""A content run keeps what recall weighed in each measured turn, so a recall rule can be replayed offline.

On the practice runs of 27 October each geometry's pair changes were reliable
across halves of the anchors (0.82 to 0.99) while the two moved together at
-0.1 to 0.3, so the mechanism and not the noise decides structure. Trying a
recall rule took a three-hour run. With the pool, its stamps, and what she felt
and what the turn was worth at each settle kept beside each sample, a rule is
replayed on the same states in seconds. The watch reads her and changes nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.phases import memory_retrieval as recall
from core.subject import content_runtime as runtime
from core.subject.content_runtime import ClassSamples, memory_id
from tools.run_subject_core_content import _load_class_samples, _merge_samples, _save_class_samples

pytestmark = pytest.mark.unit


@pytest.fixture
def watched(monkeypatch):
    for name in ("settle_on_what_she_feels", "settle_on_what_it_was_worth"):
        monkeypatch.setattr(recall, name, getattr(recall, name))
    runtime._SETTLES.clear()
    runtime._watch_the_settles()
    yield
    runtime._SETTLES.clear()


def _her() -> SimpleNamespace:
    pool = {
        "salience": 1.0,
        "candidates": [
            {"text": "[memory score=0.400] a day at the sea", "score": 0.4, "felt": "joy:0.600", "worth": "1.0,1.0"},
            {"text": "[memory score=0.400] the night the power went", "score": 0.4, "felt": "fear:0.600", "worth": None},
        ],
    }
    return SimpleNamespace(
        affect=SimpleNamespace(emotions={"joy": 0.7, "fear": 0.1}, mood_baselines={"joy": 0.1, "fear": 0.1}),
        cognition=SimpleNamespace(recall_pool=pool, long_term_memory=["[memory score=0.400] the night the power went"],
                                  memory_scores=[0.4]),
    )


def test_the_felt_settle_is_recorded_before_it_runs_and_runs_unchanged(watched) -> None:
    her = _her()
    changed = recall.settle_on_what_she_feels(her)
    [reading] = runtime._SETTLES
    assert reading["kind"] == "felt" and reading["emotions"]["joy"] == pytest.approx(0.7)
    assert [c["id"] for c in reading["candidates"]] == [memory_id("[memory] a day at the sea"),
                                                       memory_id("[memory] the night the power went")]
    assert reading["held"] == [memory_id("[memory] the night the power went")]
    assert changed is True and her.cognition.long_term_memory[0].endswith("a day at the sea")


def test_watching_twice_wraps_once(watched) -> None:
    runtime._watch_the_settles()
    recall.settle_on_what_she_feels(_her())
    assert len(runtime._SETTLES) == 1


def test_a_pool_entry_and_its_recalled_form_share_an_id() -> None:
    assert memory_id("[memory score=0.912] a day at the sea") == memory_id("[memory] a day at the sea")


def _rows(anchors: list[int]) -> ClassSamples:
    keys = [(a, 0) for a in anchors]
    return ClassSamples(
        name="joy", futures=[np.full(2, float(a)) for a in anchors], retrieved=[frozenset({f"m{a}"}) for a in anchors],
        context=[np.zeros(1) for _ in anchors], futures_b=[np.full(2, float(a)) for a in anchors],
        retrieved_b=[frozenset({f"m{a}b"}) for a in anchors], keys=keys,
        pools=[json.dumps([{"kind": "felt", "anchor": a}]) for a in anchors],
    )


def test_the_pools_survive_the_file_and_the_merge(tmp_path) -> None:
    classes = [SimpleNamespace(name="joy")]
    path = _save_class_samples(tmp_path, classes, {"base": {"joy": _rows([0, 1])}}, "samples.npz")
    back = _load_class_samples(path, classes)["base"]["joy"]
    assert [json.loads(p)[0]["anchor"] for p in back.pools] == [0, 1]
    merged = _merge_samples([{"joy": _rows([1])}, {"joy": _rows([0])}], anchors=2)["joy"]
    assert [json.loads(p)[0]["anchor"] for p in merged.pools] == [0, 1]
