"""A source that never reads should cost eight turns, not an hour.

The run already refuses a recording where a declared reader failed for most of
it. That check runs after the whole recording: on 25 September a campaign spent
fifty-two minutes recording 79,200 frames and was then refused for one reader
that had never worked. One pass through the eight conditions is the smallest
complete exercise of the organism — the conditions are the declared set of
ordinary situations and between them they run every phase — so a source that
has not read once by the end of it has no writer in this build.
"""

from __future__ import annotations

import pytest

from tools.run_subject_core import _never_read


class _Frame:
    def __init__(self, misses: dict[str, str]) -> None:
        self.misses = misses


def test_nothing_recorded_names_nothing():
    assert _never_read([]) == []


def test_a_source_that_read_every_frame_is_not_named():
    frames = [_Frame({}) for _ in range(8)]
    assert _never_read(frames) == []


def test_a_source_that_failed_every_frame_is_named():
    frames = [_Frame({"organ:experience.ownership_confidence": "organ absent"}) for _ in range(8)]
    assert _never_read(frames) == ["organ:experience.ownership_confidence"]


def test_a_source_that_read_once_is_not_named():
    frames = [_Frame({"organ:field.W_field": "organ absent"}) for _ in range(7)]
    frames.append(_Frame({}))
    assert _never_read(frames) == []


def test_two_dead_sources_are_both_named():
    always = {"organ:a.x": "absent", "organ:b.y": "absent"}
    frames = [_Frame(dict(always)) for _ in range(4)]
    frames[2].misses["organ:c.z"] = "absent"
    assert _never_read(frames) == ["organ:a.x", "organ:b.y"]


def test_a_frame_without_misses_at_all_clears_everyone():
    frames = [_Frame({"organ:a.x": "absent"}), _Frame({})]
    assert _never_read(frames) == []


@pytest.mark.asyncio
async def test_the_recording_stops_after_the_first_pass():
    from tools.run_subject_core import DeadReader, _record

    class _Runtime:
        kernel = None
        turns = 0

        async def turn_once(self, condition):
            self.turns += 1
            return [_Frame({"organ:gone.x": "organ absent"})]

    class _Conditions(list):
        pass

    runtime = _Runtime()
    conditions = _Conditions(range(8))
    with pytest.raises(DeadReader) as raised:
        await _record(runtime, conditions, rounds=300)
    assert "organ:gone.x" in str(raised.value)
    assert runtime.turns == 8


@pytest.mark.asyncio
async def test_allow_degraded_records_it_anyway():
    from tools.run_subject_core import _record

    class _Runtime:
        kernel = None
        turns = 0

        async def turn_once(self, condition):
            self.turns += 1
            return [_Frame({"organ:gone.x": "organ absent"})]

    runtime = _Runtime()
    frames, _ = await _record(runtime, list(range(8)), rounds=2, strict=False)
    assert runtime.turns == 16
    assert len(frames) == 16
