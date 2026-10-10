"""A thing sent by a press pulled and let go is aimed by where the last ones went, and what pays is found by trying.

Nothing here is a golf course: a ball that a pull sends the other way, slowing as it goes, a place it scores in, and a
count that goes up when it stops there. Which way a pull sends it, and how far, she is not told.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from core.agency.how_hard_and_which_way import PULL, Setting, Shot, Shots
from core.agency.playing_by_shots import play_by_shots, sends_by_letting_go

pytestmark = pytest.mark.unit

WIDE, TALL = 240, 180


class _World:
    """A ball at rest until a press on it is pulled and let go; it sets off the other way, slows, and stops."""

    def __init__(self, ball=(0.3, 0.7), hole=(0.75, 0.3), gain=2.2, sling=-1.0) -> None:
        self.home = ball
        self.ball = list(ball)
        self.velocity = [0.0, 0.0]
        self.hole = hole
        self.gain, self.sling = gain, sling
        self.score = 0
        self.now = 0.0
        self.pressed_at: tuple[float, float] | None = None
        self.pointer = (0.0, 0.0)

    # -- eyes
    async def look(self):
        self.now += 1 / 30
        speed = math.hypot(*self.velocity)
        if speed > 0:
            self.ball[0] += self.velocity[0] / 30
            self.ball[1] += self.velocity[1] / 30
            slower = max(0.0, speed - 0.6 / 30) / speed
            self.velocity = [v * slower for v in self.velocity]
            if math.hypot(*self.velocity) < 0.01:
                self.velocity = [0.0, 0.0]
                if math.dist(self.ball, self.hole) < 0.04:
                    self.score += 1
                self.ball = list(self.home)
        picture = np.full((TALL, WIDE, 3), (40, 140, 60), np.uint8)
        hx, hy = int(self.hole[0] * WIDE), int(self.hole[1] * TALL)
        picture[hy - 5:hy + 5, hx - 5:hx + 5] = (20, 20, 20)
        bx, by = int(self.ball[0] * WIDE), int(self.ball[1] * TALL)
        picture[max(0, by - 4):by + 4, max(0, bx - 4):bx + 4] = (250, 250, 250)
        return picture, self.now

    def words(self, _picture):
        return [{"text": f"SCORE {self.score}", "center_x": 0.1, "center_y": 0.05, "height": 0.05}]

    # -- hands
    async def press(self, x, y):
        self.pressed_at = (x, y)
        self.pointer = (x, y)

    async def point(self, x, y):
        self.pointer = (x, y)

    async def release(self):
        if self.pressed_at is not None and math.dist(self.pressed_at, self.home) < 0.06 and not any(self.velocity):
            pull = (self.pointer[0] - self.pressed_at[0], self.pointer[1] - self.pressed_at[1])
            self.velocity = [self.sling * self.gain * pull[0], self.sling * self.gain * pull[1]]
        self.pressed_at = None


def test_two_shots_and_a_straight_line_aim_the_third():
    shots = Shots(way=PULL)
    # A pull of (a, b) sends the thing to (0.3 - a, 0.7 - b): the other way, as far.
    for pull in ((0.15, 0.0), (0.0, 0.15)):
        setting = Setting.of(PULL, pull)
        shots.took(Shot(setting, (0.3 - pull[0], 0.7 - pull[1]), aimed_at=(0.5, 0.5)))
    aimed = shots.next_setting((0.5, 0.5), start=(0.3, 0.7))
    assert aimed.across == pytest.approx(-0.2, abs=0.02) and aimed.down == pytest.approx(0.2, abs=0.02), aimed


def test_a_setting_that_got_there_is_kept_while_the_aim_stays():
    shots = Shots(way=PULL)
    good = Setting.of(PULL, (-0.1, 0.12))
    shots.took(Shot(good, (0.5, 0.5), aimed_at=(0.5, 0.5)))
    assert shots.next_setting((0.5, 0.5)) == good
    assert shots.next_setting((0.9, 0.1)) != good


def test_she_finds_the_pull_that_sends_the_ball_into_the_place_it_scores():
    world = _World()
    keep: dict = {}
    result = asyncio.run(play_by_shots(world.look, world, seconds=12.0, keep=keep, read_words=world.words,
                                       aim_at=lambda _picture: world.hole, ways=(PULL,)))
    assert keep.get("sends_from") is not None and math.dist(keep["sends_from"], world.home) < 0.06, keep.get("sends_from")
    assert world.score >= 1, result


def test_words_that_say_to_pull_and_let_go_are_heard():
    assert sends_by_letting_go("Click anywhere in the yard. Drag the mouse in the direction you would like. Release the mouse button")
    assert sends_by_letting_go("Use the mouse to aim and toss Bloo")
    assert not sends_by_letting_go("Use the arrow keys to move")


def test_where_it_sends_from_and_its_measure_are_kept_for_the_next_time():
    from core.agency.what_she_keeps_of_a_game import kept_from, to_keep

    shots = Shots(way=PULL)
    good = Setting.of(PULL, (-0.1, 0.12))
    shots.took(Shot(good, (0.5, 0.5), gained=1))
    held = to_keep({"by_shots": {"sends_from": (0.3, 0.7), "shots": {PULL: shots}}})
    again = kept_from(held)["by_shots"]
    assert again["sends_from"] == (0.3, 0.7)
    assert again["shots"][PULL].next_setting(None) == good


def test_a_world_whose_words_say_to_pull_and_let_go_is_played_by_shots_on_her_page(monkeypatch):
    import core.skills.screen_pursuit_as_it_happens as reflexes

    world = _World()

    class _OnHerPage(reflexes.PlayingAsItHappens):
        async def look(self):
            return await world.look()

        async def press(self, x, y):
            await world.press(x, y)

        async def point(self, x, y):
            await world.point(x, y)

        async def release(self):
            await world.release()

    monkeypatch.setattr(reflexes, "_keep_what_she_learned", lambda page, keep: None)
    monkeypatch.setattr(reflexes, "_said_while_playing", lambda line: None)
    monkeypatch.setattr("core.perception.what_the_pixels_show.recognize_text", world.words)
    import time

    playing = _OnHerPage(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 8)
    playing.words = ["Drag back from the ball and let go to putt it into the hole"]
    asyncio.run(playing._by_shots(time.monotonic()))
    assert playing.keep["by_shots"].get("sends_from") is not None
    assert playing.stretches and playing.stretches[-1]["by_shots"]["shots"] >= 2


def test_shots_end_when_the_screen_offers_a_way_on():
    # LIVE 2026-10-09 a game's clock ran out three shots in, and forty more were let go at "Game Over, Play Again".
    world = _World()
    over = {"after": 3}

    def words(picture):
        over["after"] -= 1
        said = [{"text": f"SCORE {world.score}", "center_x": 0.1, "center_y": 0.05, "height": 0.05}]
        return said + ([{"text": "Play Again", "center_x": 0.5, "center_y": 0.6, "height": 0.05}] if over["after"] < 0 else [])

    result = asyncio.run(play_by_shots(world.look, world, seconds=12.0, keep={}, read_words=words, ways=(PULL,)))
    assert result["ended"] == "the screen offers a way on (Play Again)", result
    assert result["shots"] <= 4


def test_a_place_that_sent_once_long_ago_and_sends_nothing_now_is_given_up_for_the_one_that_sends():
    """LIVE 2026-10-09 in a putt game one early shot sent something; four hundred after it, held where she stood, sent
    nothing, and she went on holding there. What sends now is what is found."""
    world = _World()
    keep = {"sends_from": (0.8, 0.2)}                    # where something was sent from once, long ago
    keep["shots"] = {"hold": Shots(way="hold")}
    keep["shots"]["hold"].took(Shot(Setting(0.0, 0.0, 1.0), (0.81, 0.21)))

    async def go():
        return await play_by_shots(world.look, world, seconds=40.0, keep=keep, read_words=world.words)

    played = asyncio.run(go())
    assert keep.get("sends_from") is not None and math.dist(keep["sends_from"], world.home) < 0.06, keep.get("sends_from")
    assert played["shots"] > 0


def test_a_press_that_changes_the_whole_screen_is_a_button_not_a_shot_and_is_not_pressed_as_one_again():
    """LIVE 2026-10-09 a press on a game's rules screen went back to its title, the title moving into place was taken
    for a thing sent, and every shot after it pressed START."""
    from core.agency.playing_by_shots import _places_to_send_from

    world = _World()
    pressed = {"button": False}
    looked = world.look

    async def look():
        picture, at = await looked()
        if pressed["button"]:
            picture = picture.copy()
            picture[:] = (200, 40, 160)                     # another screen altogether
        return picture, at

    async def press(x, y):
        if math.dist((x, y), (0.75, 0.3)) < 0.06:            # the dark square is a button here
            pressed["button"] = True
        await world.press(x, y)

    world.press_first = world.press
    keep = {"sends_from": (0.75, 0.3)}

    class _Hands:
        async def press(self, x, y):
            await press(x, y)

        async def point(self, x, y):
            await world.point(x, y)

        async def release(self):
            await world.release()

    played = asyncio.run(play_by_shots(look, _Hands(), seconds=20.0, keep=keep))
    assert played["ended"] == "the screen changed" and keep.get("sends_from") is None
    pressed["button"] = False
    picture, _ = asyncio.run(look())
    assert all(math.dist(place, (0.75, 0.3)) > 0.05 for place in _places_to_send_from(picture, keep))


def test_a_small_round_thing_is_among_the_places_to_send_from():
    """LIVE 2026-10-09 a putt game's ball, sat on its tee, was never among the places she pressed."""
    from core.agency.playing_by_shots import _places_to_send_from

    picture = np.full((200, 300, 3), (200, 180, 120), np.uint8)
    picture[20:40, 100:200] = (60, 60, 160)                     # a heading box
    yy, xx = np.mgrid[0:200, 0:300]
    picture[(yy - 140) ** 2 + (xx - 40) ** 2 <= 16] = (120, 40, 160)   # a ball on its tee
    places = _places_to_send_from(picture, {})
    assert any(math.dist(place, (40 / 300, 140 / 200)) < 0.03 for place in places), places
