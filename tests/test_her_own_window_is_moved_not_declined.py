"""When the thing in front of her work is her own window, she moves it.

Declining is right for a dialog somebody else put there. It does nothing to
the companion bubble, which floats above everything by design and has no
decline key — so in companion mode she found something in front, pressed
Escape at it, reported it would not close, and stopped, with the board fully
visible and her own window the only thing on it.
"""

from __future__ import annotations

import pytest

from core.perception.ambient_presence import PresenceMode
from core.skills.screen_pursuit import _move_her_own_surface_aside


class _Parked:
    """A presence with her window at a known place."""

    def __init__(self, where, *, mode=PresenceMode.BUBBLE, attached=True):
        self._where = where
        self.mode = mode
        self._attached = attached
        self.asked_for = None

    def drawing_surface_attached(self) -> bool:
        return self._attached

    def bubble_position(self):
        return self._where

    def request_bubble_move(self, x, y):
        self.asked_for = (x, y)
        return 1


@pytest.fixture
def _presence(monkeypatch):
    made = {}

    def install(parked):
        made["it"] = parked
        monkeypatch.setattr(
            "core.perception.ambient_presence.get_ambient_presence",
            lambda: parked,
        )
        return parked

    return install


#: A window at the origin, a thousand across and eight hundred down.
MINE = (0, 0, 1000, 800)
#: The board occupies the middle of it.
OVER = (0.3, 0.3, 0.7, 0.7)


@pytest.mark.asyncio
async def test_her_window_over_the_board_is_asked_to_move(_presence):
    parked = _presence(_Parked((500.0, 400.0)))
    assert await _move_her_own_surface_aside(OVER, MINE) is True
    assert parked.asked_for is not None


@pytest.mark.asyncio
async def test_it_moves_to_the_side_with_more_room(_presence):
    # The work sits against the right edge, so the room is all to the left.
    parked = _presence(_Parked((700.0, 400.0)))
    await _move_her_own_surface_aside((0.6, 0.3, 0.95, 0.7), MINE)
    x, _y = parked.asked_for
    assert x == 0.0


@pytest.mark.asyncio
async def test_a_window_above_but_not_over_the_work_is_left_alone(_presence):
    """Moving it would be fussing at somebody's screen for no reason."""
    parked = _presence(_Parked((950.0, 60.0)))
    assert await _move_her_own_surface_aside(OVER, MINE) is False
    assert parked.asked_for is None


@pytest.mark.asyncio
async def test_nothing_is_asked_when_the_surface_is_not_attached(_presence):
    parked = _presence(_Parked((500.0, 400.0), attached=False))
    assert await _move_her_own_surface_aside(OVER, MINE) is False
    assert parked.asked_for is None


@pytest.mark.asyncio
async def test_nothing_is_asked_when_she_is_not_a_bubble(_presence):
    parked = _presence(_Parked((500.0, 400.0), mode=PresenceMode.HIDDEN))
    assert await _move_her_own_surface_aside(OVER, MINE) is False
    assert parked.asked_for is None


@pytest.mark.asyncio
async def test_it_is_never_closed(_presence):
    """It is how the person is talking to her."""
    parked = _presence(_Parked((500.0, 400.0)))
    await _move_her_own_surface_aside(OVER, MINE)
    assert not hasattr(parked, "closed")


@pytest.mark.asyncio
@pytest.mark.parametrize("over,mine", [(None, MINE), (OVER, None), (None, None)])
async def test_nothing_happens_without_a_place_to_compare(_presence, over, mine):
    parked = _presence(_Parked((500.0, 400.0)))
    assert await _move_her_own_surface_aside(over, mine) is False
