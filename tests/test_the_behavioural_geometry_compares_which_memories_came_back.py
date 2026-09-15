"""The content run's behavioural geometry compares which memories came back.

Recall writes each recollection with the score it was kept at, as
"[memory score=0.683] ...". The score follows her mood through the valence
alignment, so the same memory kept at two moods was two different strings, and
a displacement of affect would have moved the behavioural geometry without
changing a single recollection. These pin that the score is taken off and the
memory itself is what is compared.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.subject.content_runtime import _jaccard_distance, _retrieved

pytestmark = pytest.mark.unit


def _runtime(*items: str) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(cognition=SimpleNamespace(long_term_memory=list(items))))


def test_the_same_memory_kept_at_two_scores_is_one_memory() -> None:
    calm = _retrieved(_runtime("[memory score=0.700] Interaction: the concert", "[fact] the venue"))
    moved = _retrieved(_runtime("[memory score=0.683] Interaction: the concert", "[fact] the venue"))
    assert calm == moved
    assert _jaccard_distance(calm, moved) == pytest.approx(0.0)


def test_a_different_memory_is_still_a_different_memory() -> None:
    first = _retrieved(_runtime("[memory score=0.700] Interaction: the concert"))
    second = _retrieved(_runtime("[memory score=0.700] Interaction: the argument"))
    assert _jaccard_distance(first, second) == pytest.approx(1.0)
