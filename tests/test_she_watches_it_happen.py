"""She watches a surface as a stream: what goes on by itself is told apart from what answers her.

LIVE-like 2026-10-05, an archived game's opening scene: speech bubbles came one
after another, read one still at a time each looked like a button, and each
click on one seemed to work because the scene went on by itself.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

from core.perception import watching_it_happen
from core.perception.watching_it_happen import Watching
from core.skills.screen_pursuit_bearings import things_to_click, what_it_says
from core.skills.screen_pursuit_on_a_page import it_was_answered


def _picture(*boxes: tuple[float, float, float, float], spin: int = 0) -> np.ndarray:
    """A grey picture with white boxes (shares: left, top, right, bottom) and a small thing that spins in a corner."""
    pixels = np.full((120, 160), 60, dtype=np.uint8)
    for left, top, right, bottom in boxes:
        pixels[int(top * 120):int(bottom * 120), int(left * 160):int(right * 160)] = 240
    pixels[2:8, 2:8] = 240 if spin % 2 else 60
    return pixels


async def _never() -> None:
    return None


def _stream(frames: list[tuple[float, np.ndarray]], acts: list[tuple[float, str]] = ()) -> Watching:
    watching = Watching(take=_never)
    for at, act in acts:
        watching.acted(act, at)
    for at, picture in frames:
        watching.saw(picture, at)
    return watching


_BUBBLE_ONE, _BUBBLE_TWO, _BUTTON = (0.1, 0.1, 0.5, 0.3), (0.5, 0.1, 0.9, 0.3), (0.4, 0.8, 0.6, 0.9)


def _a_scene(now: float) -> list[tuple[float, np.ndarray]]:
    """Twelve seconds: bubbles coming and going by themselves, a corner spinning throughout."""
    frames = []
    for n in range(96):
        at = now - 12.0 + n / 8
        second = int(at - (now - 12.0))
        shown = [_BUTTON] + ([_BUBBLE_ONE] if second % 4 in (1, 2) else []) + ([_BUBBLE_TWO] if second % 4 == 3 else [])
        frames.append((at, _picture(*shown, spin=n)))
    return frames


def test_what_goes_on_by_itself_is_seen_as_going_on():
    now = time.monotonic()
    watching = _stream(_a_scene(now))
    assert watching.playing_on_its_own(now)
    assert watching.keeps_changing_on_its_own((0.1, 0.1, 0.4, 0.2))  # where the bubbles come and go
    assert not watching.keeps_changing_on_its_own((0.4, 0.8, 0.2, 0.1))  # the button, which stays


def test_a_corner_that_always_spins_is_part_of_the_picture():
    now = time.monotonic()
    watching = _stream([(now - 10.0 + n / 8, _picture(_BUTTON, spin=n)) for n in range(80)])
    assert not watching.playing_on_its_own(now)


def test_an_act_is_answered_only_by_what_was_not_changing_anyway():
    now = time.monotonic()
    frames = _a_scene(now)
    # A click while the bubbles were changing, and the only change after it is the bubbles.
    act_at = now - 2.9
    watching = _stream(frames, [(act_at, "click on a bubble")])
    assert watching.answered(act_at) is False
    # A press on a still picture that brings up something new.
    still = [(now - 5.0 + n / 8, _picture(_BUTTON)) for n in range(24)]
    after = [(now - 2.0 + n / 8, _picture(_BUTTON, (0.1, 0.5, 0.3, 0.7))) for n in range(8)]
    pressed = _stream(still + after, [(now - 2.05, "space")])
    assert pressed.answered(now - 2.05) is True


def test_her_own_acts_are_not_things_going_on():
    now = time.monotonic()
    frames, acts = [], []
    for n in range(80):
        at = now - 10.0 + n / 8
        frames.append((at, _picture(_BUTTON, (0.1, 0.5, 0.3, 0.7)) if (n // 16) % 2 else _picture(_BUTTON)))
        if n % 16 == 15:
            acts.append((at - 0.05, "space"))  # every change follows a press of hers
    watching = _stream(frames, acts)
    assert not watching.playing_on_its_own(now)


def test_she_waits_while_it_plays_and_not_after(monkeypatch):
    monkeypatch.setattr(watching_it_happen, "SCENE_ENDS_AFTER_S", 0.5)
    # A scene: one line, a pause, the next line, a pause, then still.
    lines = [(0.1, 0.1, 0.4, 0.2), (0.5, 0.1, 0.9, 0.2), (0.1, 0.3, 0.4, 0.4)]
    pictures = iter([_picture(_BUTTON, *lines[: 1 + n // 8]) for n in range(24)] + [_picture(_BUTTON, *lines)] * 1000)

    async def take():
        return next(pictures)

    async def scene() -> float:
        watching = Watching(take=take, per_second=20.0)
        watching.start()
        await asyncio.sleep(0.55)  # past the first line of the scene
        said = []
        watched = await watching.watch(at_most_s=5.0, tell=said.append)
        later = await watching.watch(at_most_s=5.0)
        await watching.stop()
        assert said and later == 0.0
        return watched

    watched = asyncio.run(scene())
    assert 0.2 < watched < 3.0


def test_a_scenes_lines_are_read_not_pressed_and_its_answer_is_believed():
    seen = {"layout": [
        {"text": "THE PINS ARE COMING TO LIFE!!", "x": 0.1, "y": 0.1, "width": 0.4, "height": 0.08, "of_its_own": True},
        {"text": "SKIP", "x": 0.85, "y": 0.9, "width": 0.1, "height": 0.05},
    ]}
    assert things_to_click(seen, drawn_where=(0, 0, 1, 1)) == ('click "SKIP"',)
    assert "COMING TO LIFE" in what_it_says(seen, drawn_where=(0, 0, 1, 1))
    assert it_was_answered(True, {"answered": True}) and it_was_answered(True, {})
    assert not it_was_answered(True, {"answered": False}) and not it_was_answered(False, {"answered": True})


def test_a_world_in_motion_is_played_not_waited_for():
    """A ball moving every frame is a game to play; waiting it out would give the game its points."""
    now = time.monotonic()
    frames = [(now - 3.0 + n / 8, _picture(_BUTTON, (0.05 * (n % 18), 0.5, 0.05 * (n % 18) + 0.04, 0.55))) for n in range(24)]
    watching = _stream(frames)
    assert watching.in_motion(now) and not watching.playing_on_its_own(now)


def test_a_file_changed_is_another_game_to_her_memory(tmp_path):
    """LIVE 2026-10-06 a mended game was played from what she had kept of it broken."""
    from types import SimpleNamespace

    from core.skills.screen_pursuit_as_it_happens import _this_game

    game = tmp_path / "pong.html"
    game.write_text("<canvas></canvas><script>broken()</script>")
    broken = _this_game(SimpleNamespace(url=game.as_uri()))
    game.write_text("<canvas></canvas><script>mended()</script>")
    assert _this_game(SimpleNamespace(url=game.as_uri())) != broken
    assert _this_game(SimpleNamespace(url="https://example.org/game")) == "played as it happens at https://example.org/game"
