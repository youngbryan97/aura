"""A line that is wholly a way on is a button, however close it is set under a paragraph.

LIVE 2026-10-08 a game's "How To Play" page had START just under its last
line of rules; it was read as part of them and never pressed.
"""
from __future__ import annotations

import pytest

from core.skills.screen_pursuit_bearings import things_to_click


@pytest.mark.unit
def test_start_under_the_rules_is_clickable():
    layout = [
        {"text": "Use the mouse to move and grab", "x": 0.2, "y": 0.60, "width": 0.6, "height": 0.05},
        {"text": "as much food as you can on each jump.", "x": 0.18, "y": 0.66, "width": 0.64, "height": 0.05},
        {"text": "START", "x": 0.4, "y": 0.72, "width": 0.2, "height": 0.05},
    ]
    clickable = things_to_click({"layout": layout}, (0.0, 0.0, 1.0, 1.0))
    assert 'click "START"' in clickable
    assert not any("as much food" in c for c in clickable)
