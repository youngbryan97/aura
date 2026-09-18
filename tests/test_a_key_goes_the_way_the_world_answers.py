"""A key goes to an application the way that application answers.

Live, 2026-09-18: the 2048 window that had answered every arrow that morning
answered none, from her or from a trusted shell, while it was in front. Keys
addressed to its process were posted and dropped. Every move read as a board
that refused everything, so she decided it had ended and began again forty
times.

Addressing a process is the better way when it lands: the key reaches the
thing it is for whatever the person is doing with the rest of the screen. It
is not the only way. Typed into the stream a keyboard types into, a key
reaches whatever is in front. Which of them this application takes is found
out, the same way everything else about a world is.
"""

from __future__ import annotations

import asyncio

import pytest

from core.skills import screen_pursuit_surface as surface


@pytest.fixture(autouse=True)
def _nothing_found_out_yet():
    surface._HOW_KEYS_LAND.clear()
    surface._UNANSWERED.clear()
    yield
    surface._HOW_KEYS_LAND.clear()
    surface._UNANSWERED.clear()


def test_keys_go_to_the_window_until_they_are_shown_not_to_land():
    assert surface.how_keys_land("2048 Game") == surface.TO_THE_WINDOW
    for key in ("up", "left", "right"):
        surface.it_answered("2048 Game", key, changed=False)
    assert surface.how_keys_land("2048 Game") == surface.TO_THE_WINDOW
    surface.it_answered("2048 Game", "down", changed=False)
    assert surface.how_keys_land("2048 Game") == surface.AS_TYPED


def test_one_key_refused_again_and_again_is_the_board_not_the_delivery():
    """A board can refuse one direction for a while; that says nothing about the keys."""
    for _ in range(8):
        surface.it_answered("2048 Game", "left", changed=False)
    assert surface.how_keys_land("2048 Game") == surface.TO_THE_WINDOW


def test_any_answer_clears_the_count():
    for key in ("up", "left", "right"):
        surface.it_answered("2048 Game", key, changed=False)
    surface.it_answered("2048 Game", "down", changed=True)
    for key in ("up", "left", "right"):
        surface.it_answered("2048 Game", key, changed=False)
    assert surface.how_keys_land("2048 Game") == surface.TO_THE_WINDOW


def test_the_way_that_answers_is_the_way_that_stays():
    for key in ("up", "left", "right", "down"):
        surface.it_answered("2048 Game", key, changed=False)
    assert surface.how_keys_land("2048 Game") == surface.AS_TYPED
    surface.it_answered("2048 Game", "up", changed=True)
    for key in ("up", "left", "right"):
        surface.it_answered("2048 Game", key, changed=False)
    assert surface.how_keys_land("2048 Game") == surface.AS_TYPED


def test_typed_keys_go_only_when_the_application_is_in_front(monkeypatch):
    from core.capabilities import window_server

    surface._HOW_KEYS_LAND["2048 game"] = surface.AS_TYPED
    typed: list[list[str]] = []
    monkeypatch.setattr(window_server, "type_keys", lambda keys, **_k: typed.append(list(keys)) or len(keys))
    monkeypatch.setattr(window_server, "owns_the_front", lambda app: False)
    assert asyncio.run(surface._send("2048 Game", ["up"])) == 0
    assert typed == []
    monkeypatch.setattr(window_server, "owns_the_front", lambda app: True)
    assert asyncio.run(surface._send("2048 Game", ["up"])) == 1
    assert typed == [["up"]]


def test_an_arrow_carries_what_a_keyboard_arrow_carries():
    from core.capabilities import window_server

    class _Quartz:
        def __init__(self):
            self.flags = []

        def CGEventCreateKeyboardEvent(self, _source, code, down):  # noqa: N802 - Quartz's name
            return {"code": code, "down": down}

        def CGEventSetFlags(self, event, flags):  # noqa: N802
            self.flags.append((event["code"], flags))

    quartz = _Quartz()
    window_server._key_event(quartz, "up", 126, True)
    window_server._key_event(quartz, "return", 36, True)
    assert quartz.flags == [(126, window_server._ARROW_FLAGS)]


def test_the_loop_tells_the_delivery_whether_the_world_answered():
    from screen_pursuit_support import pursuit_source
    from source_contract import in_order

    in_order(
        pursuit_source(),
        "can_do.tried(previous.chosen.name, attempt.verdict.observed_change)",
        "it_answered(",
    )
