"""The camera-world loop, end to end, against a simulated world.

The world renders a panorama for her eyes, as pixels, and reports the words on
it with their boxes, as her OCR would. The mouse turns the camera and one key
walks; which key, which way and how far are hers to find out. Then she is
sent to a door she can read, and the trip is told as it goes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.agency.what_hands_do import Chunk
from core.skills.in_a_world_through_a_camera import ACameraWorld, go_to, learn_the_body

VIEW = (240, 640)       # pixels of the window
FIELD = 90.0            # degrees across the view
PX_PER_DEGREE = VIEW[1] / FIELD
TURN = 0.25             # degrees of turn per point of mouse travel
PACE = 3.0              # distance walked per second held


class Simulated:
    def __init__(self, *, walks: str = "w") -> None:
        roll = np.random.default_rng(3)
        # All the way round: 360 degrees of wall, so a full turn is the view she began with.
        self.panorama = np.kron(roll.random((30, 160)), np.ones((12, 16))) * 255.0
        assert self.panorama.shape[1] == round(360 * PX_PER_DEGREE)
        self.facing, self.x, self.y = 0.0, 0.0, 0.0
        self.walks = walks
        self.backs = "s" if walks == "w" else "down"
        self.door = (5.0, 12.0)
        self.said: list[str] = []
        self.pressed: list[str] = []

    #: The room is round, this far from its middle to its wall.
    RADIUS = 15.0

    def _zoom(self) -> float:
        # The wall ahead looks bigger the nearer it is: how far it is along the
        # way she is facing, from where she stands.
        ux, uy = math.sin(math.radians(self.facing)), math.cos(math.radians(self.facing))
        along = self.x * ux + self.y * uy
        ahead = -along + math.sqrt(max(0.0, along * along - (self.x ** 2 + self.y ** 2 - self.RADIUS ** 2)))
        return min(20.0, max(0.5, self.RADIUS / max(ahead, 0.5)))

    def frame(self) -> np.ndarray:
        high, wide = VIEW
        zoom = self._zoom()
        crop_h, crop_w = int(high / zoom), int(wide / zoom)
        left = int(round(self.facing * PX_PER_DEGREE)) + (wide - crop_w) // 2
        top = (self.panorama.shape[0] - crop_h) // 2
        rows = top + np.linspace(0, crop_h - 1, high).round().astype(int)
        cols = (left + np.linspace(0, crop_w - 1, wide).round().astype(int)) % self.panorama.shape[1]
        return self.panorama[np.ix_(rows, cols)]

    def layout(self) -> list[dict]:
        dx, dy = self.door[0] - self.x, self.door[1] - self.y
        off = (math.degrees(math.atan2(dx, dy)) - self.facing + 180.0) % 360.0 - 180.0
        far = math.hypot(dx, dy)
        seen = []
        if abs(off) < FIELD / 2:
            high = min(0.9, 1.0 / far)
            seen.append({"text": "Door", "center_x": 0.5 + off / FIELD, "center_y": 0.5,
                         "width": high * 0.5, "height": high})
            if far < 2.0:
                seen.append({"text": "Press E to enter", "center_x": 0.5, "center_y": 0.85,
                             "width": 0.2, "height": 0.03})
        return seen

    def look(self):
        return self.frame(), self.layout()

    async def play(self, chunk: Chunk) -> None:
        for slot in chunk.slots:
            self.facing += TURN * slot.moved[0]
            ahead = 1.0 if self.walks in slot.held else -1.0 if self.backs in slot.held else 0.0
            if ahead:
                step = ahead * PACE * chunk.slot_s
                self.x += step * math.sin(math.radians(self.facing))
                self.y += step * math.cos(math.radians(self.facing))
            self.pressed.extend(sorted(slot.held - {self.walks, self.backs}))


def _world(sim: Simulated) -> ACameraWorld:
    return ACameraWorld(look=sim.look, play=sim.play)


@pytest.mark.asyncio
async def test_she_finds_out_which_key_walks_and_how_the_mouse_turns():
    sim = Simulated(walks="up")
    body = await learn_the_body(_world(sim), keys=("w", "up", "space"), slot_s=0.2)
    assert body.walks_forward() == "up"
    gain, _ = body.mouse_gain()
    assert gain < 0  # the picture slides against the mouse when the camera turns toward it
    assert sim.facing == pytest.approx(0.0)  # every test turn was taken back


@pytest.mark.asyncio
async def test_she_goes_to_the_door_and_does_what_it_says_telling_it_as_she_goes():
    sim = Simulated()
    world = _world(sim)
    body = await learn_the_body(world, keys=("w", "s"), slot_s=0.2)
    trying = len(sim.pressed)  # what she pressed finding out what the keys do
    heard: list[str] = []
    trip = await go_to(world, "door", body, slot_s=0.2, most_chunks=80, tell=heard.append)
    assert trip.done and trip.ended == "the screen said to press e"
    assert sim.pressed[trying:] == ["e"]
    assert heard[0] == "turning right toward the door"
    assert "walking toward the door" in heard and heard[-1] == "the screen said to press e"


@pytest.mark.asyncio
async def test_where_no_key_walks_she_says_so_instead_of_wandering():
    sim = Simulated(walks="nothing-she-tries")
    world = _world(sim)
    body = await learn_the_body(world, keys=("w", "s"), slot_s=0.2)
    trip = await go_to(world, "door", body, slot_s=0.2, most_chunks=10)
    assert not trip.done and trip.chunks == 0
    assert "nothing she pressed moved her forward" in trip.ended


@pytest.mark.asyncio
async def test_a_refused_chunk_is_not_read_as_an_act_that_did_nothing():
    """Live, the first time: the chat window came in front, every test move was
    refused, and she concluded that no key walked here."""
    from types import SimpleNamespace

    from core.skills.in_a_world_through_a_camera import NotInFront

    sim = Simulated()

    async def refused(_chunk):
        return SimpleNamespace(stopped="the window she was acting in is no longer in front")

    with pytest.raises(NotInFront, match="no longer in front"):
        await learn_the_body(ACameraWorld(look=sim.look, play=refused), keys=("w",), slot_s=0.2)


@pytest.mark.asyncio
async def test_a_window_that_will_not_come_forward_is_said_so_before_anything_is_pressed():
    from core.skills.in_a_world_through_a_camera import NotInFront

    sim = Simulated()

    async def stays_behind() -> bool:
        return False

    world = ACameraWorld(look=sim.look, play=sim.play, bring_forward=stays_behind)
    with pytest.raises(NotInFront, match="would not come to the front"):
        await learn_the_body(world, keys=("w",), slot_s=0.2)
    assert sim.pressed == [] and sim.facing == 0.0


class _AWindow:
    bounds = (0, 0, 640, 240)


def _a_machine(monkeypatch, *, touched: list[float], front: list[bool]):
    """The real world's plumbing with the machine replaced: who is typing, and who is in front."""
    from types import SimpleNamespace

    from core.capabilities import host_automation, window_server
    from core.perception import what_the_pixels_show

    focused: list[str] = []
    monkeypatch.setattr(window_server, "window_of", lambda app, on_screen_only=False: _AWindow())
    monkeypatch.setattr(window_server, "capture", lambda window: np.zeros((24, 64, 3)))
    monkeypatch.setattr(what_the_pixels_show, "recognize_text", lambda frame: [])
    monkeypatch.setattr(
        window_server, "seconds_since_someone_touched_it",
        lambda: touched.pop(0) if len(touched) > 1 else touched[0],
    )
    monkeypatch.setattr(window_server, "owns_the_front", lambda app: bool(focused) and front[0])

    async def focus_app(app):
        focused.append(app)
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(host_automation, "get_host_automation", lambda: SimpleNamespace(focus_app=focus_app))
    return focused


@pytest.mark.asyncio
async def test_she_does_not_take_the_front_while_someone_is_typing(monkeypatch):
    from core.skills.in_a_world_through_a_camera import live_camera_world

    focused = _a_machine(monkeypatch, touched=[0.0], front=[True])
    world = live_camera_world("A room", wait_s=0.2)
    assert await world.bring_forward() is False
    assert focused == []


@pytest.mark.asyncio
async def test_and_takes_it_once_they_pause(monkeypatch):
    from core.skills.in_a_world_through_a_camera import live_camera_world

    focused = _a_machine(monkeypatch, touched=[0.0, 0.0, 30.0], front=[True])
    world = live_camera_world("A room", wait_s=30.0)
    assert await world.bring_forward() is True
    assert focused == ["A room"]


@pytest.mark.asyncio
async def test_what_is_behind_her_is_found_by_looking_around():
    sim = Simulated()
    sim.door = (-6.0, -8.0)  # behind and to the left: out of view at the start
    world = _world(sim)
    body = await learn_the_body(world, keys=("w", "s"), slot_s=0.2)
    heard: list[str] = []
    trip = await go_to(world, "door", body, slot_s=0.2, most_chunks=120, tell=heard.append)
    assert heard[0] == "looking around for the door"
    assert trip.done and trip.ended == "the screen said to press e"


@pytest.mark.asyncio
async def test_what_is_nowhere_is_given_up_on_after_one_full_turn():
    sim = Simulated()
    sim.door = (0.0, 500.0)  # far past the horizon: never read
    sim.layout = lambda: []
    world = _world(sim)
    body = await learn_the_body(world, keys=("w", "s"), slot_s=0.2)
    trip = await go_to(world, "door", body, slot_s=0.2, most_chunks=200)
    assert trip.ended == "looked all the way round and nothing answers to 'door'"
    # One full turn and no more.
    assert abs(sim.facing) == pytest.approx(360.0, abs=FIELD)


@pytest.mark.asyncio
async def test_up_against_a_wall_she_backs_off_and_finds_the_key_that_walks():
    """Live: standing at the door she had just opened, walking changed nothing,
    and she concluded that no key walked."""
    sim = Simulated()
    sim.y = Simulated.RADIUS - 2.0  # nose to something just ahead
    real_play = sim.play

    async def blocked_ahead(chunk):
        for slot in chunk.slots:
            if "w" in slot.held and sim.y >= Simulated.RADIUS - 2.0:
                continue  # it does not move
            await real_play(Chunk((slot,), chunk.slot_s))

    body = await learn_the_body(ACameraWorld(look=sim.look, play=blocked_ahead), keys=("w", "s"), slot_s=0.2)
    assert body.walks_forward() == "w"
