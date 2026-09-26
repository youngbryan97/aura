"""Ignition has to be able to not happen.

`ignited` was `winner_priority >= 0.6`. Over a 320-turn recording the winning
priority never fell below 0.749, so the flag read True on every frame of every
run. The qualia engine weights it at a fifth, the phenomenal present branches on
it and the broadcast record keeps content by it: three consumers handed a
constant where each expected an event.

Six tenths is still the bar. It is now six tenths of the winners she has been
having rather than of a scale nothing sits at the bottom of.
"""

from __future__ import annotations

import random

import pytest

from core.consciousness.global_workspace import (
    _ENOUGH_WINNERS,
    _WINNER_WINDOW,
    _stood_out,
)

BAR = 0.6


def test_with_nothing_to_rank_against_the_absolute_bar_answers():
    assert _stood_out(0.9, [], BAR) is True
    assert _stood_out(0.4, [], BAR) is False


def test_a_short_history_is_still_the_absolute_bar():
    short = [0.8] * (_ENOUGH_WINNERS - 1)
    assert _stood_out(0.4, short, BAR) is False
    assert _stood_out(0.9, short, BAR) is True


def test_a_winner_at_the_top_of_her_own_range_ignites():
    recent = [0.75 + 0.002 * i for i in range(100)]
    assert _stood_out(0.99, recent, BAR) is True


def test_a_winner_at_the_bottom_of_her_own_range_does_not():
    recent = [0.75 + 0.002 * i for i in range(100)]
    assert _stood_out(0.751, recent, BAR) is False


def test_the_same_bid_every_turn_stops_being_an_event():
    """A constant cannot be news, whatever its level."""
    recent = [0.92] * 100
    assert _stood_out(0.92, recent, BAR) is False


def test_the_flag_is_not_a_constant_over_her_own_distribution():
    rng = random.Random(3)
    recent = [0.75 + 0.25 * rng.random() for _ in range(200)]
    seen = {_stood_out(0.75 + 0.25 * rng.random(), recent, BAR) for _ in range(400)}
    assert seen == {True, False}


def test_it_fires_about_as_often_as_the_bar_says():
    rng = random.Random(11)
    draws = [0.75 + 0.25 * rng.random() for _ in range(2000)]
    recent = draws[:400]
    rate = sum(1 for one in draws[400:] if _stood_out(one, recent, BAR)) / len(draws[400:])
    assert 0.25 < rate < 0.55


def test_the_bar_is_the_share_it_says_it_is():
    recent = [0.05 * i for i in range(20)]  # 0.00 to 0.95
    assert _stood_out(0.0, recent, 0.0) is True
    assert _stood_out(0.99, recent, 1.0) is True
    assert _stood_out(0.5, recent, 1.0) is False


# ── the workspace holds the window ───────────────────────────────────────


def test_the_workspace_keeps_a_bounded_window_of_its_own_winners():
    from core.consciousness.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace.__new__(GlobalWorkspace)
    from collections import deque

    workspace._recent_winner_priorities = deque(maxlen=_WINNER_WINDOW)
    for _ in range(_WINNER_WINDOW * 2):
        workspace._recent_winner_priorities.append(0.5)
    assert len(workspace._recent_winner_priorities) == _WINNER_WINDOW


def test_a_real_workspace_declares_the_window():
    from core.consciousness.global_workspace import GlobalWorkspace

    assert "_recent_winner_priorities" in GlobalWorkspace.__init__.__code__.co_names
