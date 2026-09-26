"""A request to walk to something in a world is read as a walk, and only then.

The camera loop could go to a door and open it, and nothing in her request
path could reach it: "walk over to the chest in Python and open it" ended in
the board loop. These hold the reading of such a request, and the places it
must not reach: an address, a folder, an idiom, and a sentence said in
conversation.
"""

from __future__ import annotations

import pytest

from core.runtime.watched_goal import a_trip_asked_for


@pytest.mark.parametrize(
    ("said", "thing"),
    [
        ("Walk to the red door and open it", "red door"),
        ("go over to the chest in Minecraft", "chest"),
        ("head to the tower, then climb it", "tower"),
        ("approach the campfire", "campfire"),
        ("make your way to that bridge", "bridge"),
    ],
)
def test_a_walk_names_the_thing_walked_to(said, thing):
    assert a_trip_asked_for(said) == thing


@pytest.mark.parametrize(
    "said",
    [
        "go to google.com",
        "go to the downloads folder",
        "go to the settings page",
        "go to sleep",
        "go to work tomorrow",
        "play 2048 until 512",
    ],
)
def test_places_on_a_computer_and_idioms_are_not_walks(said):
    assert a_trip_asked_for(said) == ""


def test_a_walk_with_nowhere_to_take_it_is_not_a_goal(monkeypatch):
    from core.runtime import watched_goal

    # What is installed is the machine's to say, and a test does not ask it.
    monkeypatch.setattr(watched_goal, "_an_application_here", lambda *a, **k: "")
    assert watched_goal.read_watched_goal("I want to go to the store later") is None


@pytest.mark.asyncio
async def test_the_pursuit_hands_a_walk_to_the_camera_loop(monkeypatch):
    from core.skills import in_a_world_through_a_camera, screen_pursuit

    asked: list[tuple] = []

    async def a_trip(named, app, move_keys, *, tell, within_s):
        asked.append((named, app))
        return {"completed": True, "outcome": "reached", "moves": [], "attempts": []}

    monkeypatch.setattr(in_a_world_through_a_camera, "a_trip_for_the_pursuit", a_trip)
    result = await screen_pursuit.pursue_on_screen(
        goal="Walk over to the chest in Python and open it", success_when="",
        target_app="Python", narrate=False, max_seconds=5,
    )
    assert asked == [("chest", "Python")] and result["completed"]
