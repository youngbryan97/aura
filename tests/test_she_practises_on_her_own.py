"""She sets herself tasks in a world, tries them, grades them by the world's answer and keeps them.

The shape of SIMA 2's self-improvement loop, with the setter reading the
screen and her record, and the judge being what the screen did.
"""

from __future__ import annotations

import pytest

from core.agency import what_she_tried
from core.agency.setting_herself_a_task import things_in_view, what_to_practise
from core.agency.what_she_tried import HERSELF, Episode, how_it_goes, keep


def _layout(*names: str) -> list[dict]:
    return [{"text": name, "center_x": 0.5, "width": 0.1, "height": 0.1} for name in names]


def test_things_in_view_leave_out_prompts_titles_and_sentences():
    layout = _layout("Door", "Press E to open", "A room to walk in", "The door is open", "Chest")
    assert things_in_view(layout, title="A room to walk in") == ["door", "chest"]


def test_what_she_has_never_tried_comes_first():
    record = {"go to the door": (3, 2)}
    assert what_to_practise(_layout("Door", "Chest"), record) == "go to the chest"


def test_what_she_might_or_might_not_manage_beats_what_she_always_does():
    record = {"go to the door": (4, 4), "go to the chest": (4, 2), "go to the well": (4, 0)}
    assert what_to_practise(_layout("Door", "Chest", "Well"), record) == "go to the chest"


def test_nothing_in_view_that_is_a_thing_sets_no_task():
    assert what_to_practise(_layout("Press E to open"), {}) == ""


def test_what_she_tried_is_kept_and_read_back(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_tried, "_kept_in", lambda: tmp_path)
    for worked in (True, False, True):
        assert keep(Episode(world="A Room", task="go to the door", set_by=HERSELF, succeeded=worked))
    assert how_it_goes("A Room") == {"go to the door": (3, 2)}
    assert [episode.set_by for episode in what_she_tried.what_she_tried("A Room")] == [HERSELF] * 3


@pytest.mark.asyncio
async def test_she_practises_and_the_world_grades_it():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_acting_in_a_world_through_a_camera import Simulated, _world

    from core.skills.practising_in_a_world import practise

    sim = Simulated()
    kept: list[Episode] = []
    lived = await practise(
        _world(sim), "a simulated room", keys=("w", "s"), slot_s=0.2, attempts=2,
        most_chunks=80, keeping=kept.append,
    )
    # She opened the door. Once managed, it had nothing left to teach from
    # there, so the second turn went to looking elsewhere, not to the door again.
    assert [(episode.task, episode.succeeded) for episode in lived] == [("go to the door", True)]
    assert kept == lived and all(episode.set_by == HERSELF for episode in kept)
    assert lived[0].played and lived[0].answered == "the screen now says 'The door is open'"


def test_a_sentence_on_screen_is_not_a_second_thing_of_the_same_name():
    """Live: her own "The Chest is open" made a second chest to choose between."""
    from core.agency.going_to_what_she_sees import seen_named

    layout = [
        {"text": "Chest", "center_x": 0.5, "width": 0.1, "height": 0.05},
        {"text": "The Chest is open", "center_x": 0.3, "width": 0.3, "height": 0.03},
    ]
    assert [sight.text for sight in seen_named(layout, "chest")] == ["Chest"]


def test_a_word_cut_by_the_edge_of_the_view_is_not_read_as_a_name():
    """Live: a chest half out of view was read as "hest"."""
    layout = [{"text": "hest", "x": 0.0, "center_x": 0.03, "width": 0.06, "height": 0.04}]
    assert things_in_view(layout) == []
