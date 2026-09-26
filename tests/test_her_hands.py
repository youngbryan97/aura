"""Her hands: one grammar for keys, holds, the mouse and clicks, played slot by slot.

She could tap one key per look, which plays a board and nothing like a world:
walking is a key held while the camera turns. SIMA 2 writes keys, clicks and
relative mouse moves as text and parses them deterministically (arXiv
2512.04797, §3.2). These tests hold the grammar and the player to that.
"""

from __future__ import annotations

import pytest

from core.agency.what_hands_do import NotAHand, Slot, chunk_of, hands_read
from core.capabilities.hands import play

WINDOW = (100.0, 50.0, 800.0, 600.0)


class Recorder:
    """A sink that writes down what it was asked to do."""

    def __init__(self, fail_on: str = "") -> None:
        self.events: list[tuple] = []
        self.fail_on = fail_on

    def key(self, name, down):
        if self.fail_on and name == self.fail_on and down:
            raise RuntimeError("the key jammed")
        self.events.append(("key", name, down))
        return True

    def move_by(self, dx, dy):
        self.events.append(("move", dx, dy))
        return True

    def click(self, button, x, y):
        self.events.append(("click", button, round(x, 1), round(y, 1)))
        return True

    def scroll(self, dx, dy):
        self.events.append(("scroll", dx, dy))
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


async def _played(text: str, *, sink=None, still_ours=lambda: True, slot_s=0.1):
    clock = Clock()
    sink = sink or Recorder()
    done = await play(
        hands_read(text, slot_s=slot_s), sink=sink, window=WINDOW,
        still_ours=still_ours, clock=clock, sleep=clock.sleep,
    )
    return done, sink, clock


# ── the grammar ─────────────────────────────────────────────────────────

def test_a_chunk_reads_back_as_it_was_written():
    text = "w | shift w | mouse(40,-3) w | . | click(0.41,0.62) | scroll(0,-3) | done"
    chunk = hands_read(text, slot_s=0.1)
    assert len(chunk.slots) == 6 and chunk.done and not chunk.think
    assert hands_read(chunk.as_text(), slot_s=0.1) == chunk


def test_other_names_for_the_same_key_are_the_same_key():
    assert hands_read("control option esc", slot_s=0.1).slots[0].held == {"ctrl", "alt", "escape"}


@pytest.mark.parametrize("text", ["cmd q", "command+w", "fn", "super"])
def test_command_is_never_a_key(text):
    with pytest.raises(NotAHand, match="never pressed"):
        hands_read(text, slot_s=0.1)


@pytest.mark.parametrize(
    "text",
    ["jump", "click(1.5,0.2)", "w | done | w", "click(0.1,0.1) click(0.2,0.2)", "mouse(3)"],
)
def test_anything_it_cannot_play_is_refused_with_a_reason(text):
    with pytest.raises(NotAHand):
        hands_read(text, slot_s=0.1)


def test_think_hands_the_next_decision_back():
    chunk = hands_read("w | w | think", slot_s=0.1)
    assert chunk.think and not chunk.done and len(chunk.slots) == 2


def test_an_empty_slot_is_a_hand_waiting():
    assert hands_read(".", slot_s=0.1).slots[0].empty()


# ── the player ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_key_held_across_slots_is_one_press_held():
    done, sink, clock = await _played("w | w | w | .")
    keys = [event for event in sink.events if event[0] == "key"]
    assert keys == [("key", "w", True), ("key", "w", False)]
    assert done.slots == 4 and clock.now == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_a_key_comes_up_when_a_slot_stops_naming_it():
    _done, sink, _clock = await _played("w shift | w | .")
    assert sink.events == [
        ("key", "shift", True), ("key", "w", True), ("key", "shift", False), ("key", "w", False),
    ]


@pytest.mark.asyncio
async def test_the_mouse_moves_by_its_delta_and_clicks_land_in_the_window():
    _done, sink, _clock = await _played("mouse(12,-4) | click(right,0.5,0.25) | scroll(0,-2)")
    assert ("move", 12, -4) in sink.events
    assert ("click", "right", 500.0, 200.0) in sink.events
    assert ("scroll", 0, -2) in sink.events


@pytest.mark.asyncio
async def test_she_stops_when_the_window_is_no_longer_hers():
    asked = iter([True, True, False, True])
    done, sink, _clock = await _played("w | w | w | w", still_ours=lambda: next(asked))
    assert done.slots == 2 and "no longer in front" in done.stopped
    assert sink.events[-1] == ("key", "w", False)


@pytest.mark.asyncio
async def test_every_key_comes_up_even_when_something_fails():
    sink = Recorder(fail_on="d")
    with pytest.raises(RuntimeError):
        await _played("w | w d", sink=sink)
    assert sink.events[-1] == ("key", "w", False)


@pytest.mark.asyncio
async def test_a_chunk_made_in_code_plays_the_same():
    chunk = chunk_of(Slot(frozenset({"a"})), Slot(), slot_s=0.05)
    assert chunk.as_text() == "a | ."
    clock = Clock()
    done = await play(chunk, sink=Recorder(), window=WINDOW, still_ours=lambda: True,
                      clock=clock, sleep=clock.sleep)
    assert done.slots == 2 and done.pressed == {"a"} and done.late_s == 0.0


def test_the_machine_side_knows_every_key_a_hand_may_hold():
    from core.agency.what_hands_do import KEYS
    from core.capabilities import window_server
    from core.capabilities.hands import _POSITIONS

    unplayable = {key for key in KEYS if key not in _POSITIONS and window_server.key_code(key) is None}
    assert not unplayable, sorted(unplayable)


def test_her_own_last_press_is_not_a_person_at_the_keyboard(monkeypatch):
    """The hardware counts her posted events as input, so right after she
    pressed something it read nought and she would have waited on herself."""
    import time
    from types import SimpleNamespace

    from core.capabilities import window_server

    idle = {"s": 0.01}
    fake = SimpleNamespace(
        kCGEventSourceStateHIDSystemState=1, kCGAnyInputEventType=2,
        CGEventSourceSecondsSinceLastEventType=lambda state, kind: idle["s"],
    )
    monkeypatch.setattr(window_server, "_quartz", lambda: fake)
    monkeypatch.setattr(window_server, "_HER_OWN_LAST", [0.0])
    assert window_server.seconds_since_someone_touched_it() == 0.01  # a person, nothing of hers
    window_server.her_hands_posted()
    assert window_server.seconds_since_someone_touched_it() == float("inf")  # the latest was hers
    window_server._HER_OWN_LAST[0] = time.monotonic() - 30.0
    idle["s"] = 2.0
    assert window_server.seconds_since_someone_touched_it() == 2.0  # a person, after her
