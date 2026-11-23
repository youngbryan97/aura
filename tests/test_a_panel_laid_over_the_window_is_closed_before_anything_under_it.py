"""A panel laid over the window, with the rest darkened behind it, is seen whatever it says, and closed first.

LIVE 23 November 2026 she pressed a heading in En Croissant, and it laid "FIDE Player Info: Player not found" over the
board and darkened the rest of the window. Her reader of dialogs knew one only by words such as "cookies" or
"welcome", so she pressed the up arrow at the dimmed board under it until the attempt was over. Seen from her
pictures, everything outside the panel kept one common share of its contrast; that is what a layer of shade does.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.perception.something_laid_over_it as laid
from core.perception.blocking_overlay import LAID_OVER, assess_overlay


def _a_window(seed: int = 3) -> np.ndarray:
    """A window of things: blocks of many tones on a dark ground, as an application draws them."""
    rng = np.random.default_rng(seed)
    picture = np.full((600, 800, 3), 28, dtype=np.float32)
    for _ in range(60):
        top, left = rng.integers(0, 560), rng.integers(0, 760)
        tall, wide = rng.integers(20, 120), rng.integers(20, 160)
        picture[top:top + tall, left:left + wide] = rng.integers(40, 250, size=3)
    return picture


def _shaded(picture: np.ndarray, share: float, colour: float = 0.0) -> np.ndarray:
    return picture * (1 - share) + colour * share


def _with_a_panel(picture: np.ndarray, tone: float = 32.0) -> np.ndarray:
    out = picture.copy()
    out[60:150, 240:560] = tone
    out[90:110, 270:500] = 235.0  # its words
    return out


def _tones(picture: np.ndarray) -> list[list[float]]:
    return laid.how_light_each_part_is(np.clip(picture, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("share,colour", [(0.6, 0.0), (0.35, 0.0), (0.5, 255.0)])
def test_a_panel_over_a_shaded_window_is_seen_where_it_is(share: float, colour: float) -> None:
    clear = _a_window()
    found = laid.laid_over(_tones(clear), _tones(_with_a_panel(_shaded(clear, share, colour))))
    assert found is not None
    # The panel spans 0.30-0.70 across and 0.10-0.25 down; the grid is coarse, so its place is to a part or two.
    assert found.left <= 0.31 and found.right >= 0.69 and found.top <= 0.11 and found.bottom >= 0.24
    assert found.right - found.left < 0.5 and found.bottom - found.top < 0.3
    assert found.kept == pytest.approx(1 - share, abs=0.05)


def test_what_changes_a_window_otherwise_is_no_panel() -> None:
    clear = _a_window()
    moved_on = clear.copy()
    moved_on[300:360, 100:160] = 200.0  # a piece moved
    moved_on[420:480, 600:700] = clear[100:160, 50:150]  # and another
    for now in (clear, moved_on, np.roll(clear, 30, axis=0), _with_a_panel(clear), _shaded(clear, 0.5),
                _a_window(seed=11)):
        assert laid.laid_over(_tones(clear), _tones(now)) is None


def test_the_last_clear_picture_is_what_a_panel_is_told_against(monkeypatch) -> None:
    monkeypatch.setattr(laid, "_CLEAR", {})
    clear = _a_window()
    covered = _with_a_panel(_shaded(clear, 0.6))
    assert laid.laid_over_what_she_saw("app:1", _tones(clear)) is None
    assert laid.laid_over_what_she_saw("app:1", _tones(covered)) is not None
    # While the panel stays, it is still told against the picture from before it.
    assert laid.laid_over_what_she_saw("app:1", _tones(covered)) is not None
    assert laid.laid_over_what_she_saw("app:1", _tones(clear)) is None
    # Another window is told against its own pictures, and its first is nothing to go by.
    assert laid.laid_over_what_she_saw("app:2", _tones(covered)) is None


PANEL = {"left": 0.28, "top": 0.08, "right": 0.72, "bottom": 0.25, "kept": 0.4, "fitted": 0.93}


def _run(text: str, x: float, y: float) -> dict:
    return {"text": text, "center_x": x, "center_y": y}


def test_a_panel_laid_over_is_closed_with_escape_whatever_it_says() -> None:
    seen = {"text": "FIDE Player Info Player not found New Game Analyze", "laid_over": PANEL,
            "layout": [_run("FIDE Player Info", 0.37, 0.14), _run("Player not found", 0.37, 0.19),
                       _run("New Game", 0.6, 0.47), _run("Analyze", 0.74, 0.47)]}
    verdict = assess_overlay(seen)
    assert verdict.present and verdict.suggested_key == "escape" and LAID_OVER in verdict.reasons
    # With words like these and no panel seen, nothing was in the way: the panel is what she sees now.
    assert not assess_overlay({k: v for k, v in seen.items() if k != "laid_over"}).present


def test_a_control_on_the_panel_that_closes_it_is_pressed_and_one_under_it_is_not() -> None:
    on_it = {"text": "Saved Close Cancel", "laid_over": PANEL,
             "layout": [_run("Saved", 0.4, 0.12), _run("Close", 0.6, 0.22), _run("Cancel", 0.6, 0.8)]}
    verdict = assess_overlay(on_it)
    assert verdict.label == "Close" and (verdict.click_x, verdict.click_y) == (0.6, 0.22)
    under_it = {"text": "Saved Cancel", "laid_over": PANEL,
                "layout": [_run("Saved", 0.4, 0.12), _run("Cancel", 0.6, 0.8)]}
    assert assess_overlay(under_it).suggested_key == "escape"


def test_a_panel_that_warns_of_a_loss_is_still_the_persons() -> None:
    seen = {"text": "Your progress will be lost. Start New Game", "laid_over": PANEL,
            "layout": [_run("Your progress will be lost.", 0.4, 0.12), _run("Start New Game", 0.5, 0.2)]}
    verdict = assess_overlay(seen)
    assert verdict.needs_person and not verdict.suggested_key and verdict.click_x is None
