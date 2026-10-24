"""What she feels is a cue as well as the words.

Every recall source was searched with the turn's words, so her mood could only
reorder what they found: on 24 October, feeling better changed the distance
between what a good and a bad percept brought back by 0.0002 on average, where
her internal state drew them 0.94 closer. Memories made in what she feels now
come to mind (Bower, 1981) and compete with the rest.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.felt_at_encoding import encode
from core.memory.memory_facade import MemoryFacade

pytestmark = pytest.mark.unit


class _Collection:
    def __init__(self, rows):
        self._rows = rows

    def get(self, include=None):
        return {
            "ids": [row[0] for row in self._rows],
            "documents": [row[1] for row in self._rows],
            "metadatas": [row[2] for row in self._rows],
        }


def _facade(rows) -> MemoryFacade:
    facade = MemoryFacade.__new__(MemoryFacade)
    facade._vector = SimpleNamespace(_collection=_Collection(rows))
    return facade


ROWS = [
    ("a", "the afternoon the build finally passed", {"felt": encode({"joy": 0.3, "pride": 0.2})}),
    ("b", "the evening the run was lost", {"felt": encode({"sadness": 0.3, "frustration": 0.2})}),
    ("c", "a note about the weather", {}),
    ("d", "a quiet morning", {"felt": encode({"joy": 0.1, "sadness": 0.1})}),
]


def test_memories_made_in_what_she_feels_now_come_first():
    found = _facade(ROWS).search_by_feeling({"joy": 0.4, "pride": 0.2}, limit=3)
    assert [row["content"] for row in found][:1] == ["the afternoon the build finally passed"]
    assert "a note about the weather" not in [row["content"] for row in found]
    assert all(row["score"] > 0.0 for row in found)
    low = _facade(ROWS).search_by_feeling({"sadness": 0.4, "frustration": 0.3}, limit=1)
    assert [row["content"] for row in low] == ["the evening the run was lost"]


def test_feeling_nothing_brings_nothing_to_mind():
    assert _facade(ROWS).search_by_feeling({}, limit=3) == []
    without_a_store = MemoryFacade.__new__(MemoryFacade)
    without_a_store._vector = None
    assert without_a_store.search_by_feeling({"joy": 0.3}) == []


def test_recall_takes_her_feeling_as_a_source():
    source = (__import__("pathlib").Path(__file__).resolve().parents[1] / "core/phases/memory_retrieval.py").read_text()
    assert '_timed("feeling", _get_felt())' in source
    assert "memory.search_by_feeling" in source
