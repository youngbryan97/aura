"""A roll under a read plan reads only the frames its measurement uses, and an unread frame cannot be read.

Reading K costs about two fifths of a turn, and a carrier roll or a content
presentation uses one or two of its thirty-three frames a turn. Seed 79 in one
process (27 October): content, cut and grain readings bit-identical with the
plan on and off in 25 comparisons, 15 to 24 percent less time.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.clamp import compose
from core.subject.read_plan import lag_frames, reading_only
from core.subject.state import DOMAINS, CoreState, UnreadFrame

pytestmark = pytest.mark.unit


def _frame(value: float) -> CoreState:
    return CoreState(values={key: np.full(2, value) for key in DOMAINS}, t=0.0)


def test_the_plan_names_the_lag_frames_and_the_last() -> None:
    assert lag_frames([33, 66], 33, 2) == frozenset({32, 65})
    assert lag_frames([33], 33, 2) == frozenset({32, 65})


def test_an_unread_frame_refuses_every_reading() -> None:
    frame = UnreadFrame(condition="idle", tag="after", index=7)
    with pytest.raises(LookupError, match="frame 7"):
        frame.vector()
    with pytest.raises(LookupError):
        frame.values  # noqa: B018
    with pytest.raises(LookupError):
        frame.domain("A")


def test_the_plan_is_undone_on_the_way_out() -> None:
    runtime = SimpleNamespace(read_plan=None)
    with reading_only(runtime, [32]):
        assert runtime.read_plan == frozenset({32})
        with reading_only(runtime, None):
            assert runtime.read_plan is None
        assert runtime.read_plan == frozenset({32})
    assert runtime.read_plan is None


def test_a_cut_keeps_a_frame_unread_when_either_side_left_it_unread() -> None:
    left = [UnreadFrame(index=0), _frame(1.0)]
    right = [UnreadFrame(index=0), _frame(2.0)]
    cut = compose(left, right, ["A"])
    assert isinstance(cut[0], UnreadFrame)
    assert float(cut[1].values["A"][0]) == 1.0
    assert float(cut[1].values["M"][0]) == 2.0
