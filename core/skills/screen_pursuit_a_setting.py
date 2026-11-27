"""A setting the goal names, set to the number it gives, on a window of labelled controls.

A goal may say how something is to be set as well as what to do: "play a game with the other player's strength set to
8". Where the window shows numbers beside labels, the label nearest in meaning to the setting the goal names is the one,
as a person reading the form would take it; where none stands out, her own model is asked which, the labels the only
answers. A number alone beside its label is a field: it is pressed, everything in it chosen, the goal's number typed
over it, and the window read again to see the number there. Several numbers beside one label are choices: the one that
is the goal's number is pressed. Where the setting is not in view, the form is scrolled a step and read again, as a
person looks further down a form. A setting set is not set again in the same pursuit.

What a game of her own practice changes is put back for the next game she is asked to play there: an application keeps
a setting from one game to the next, and the person's games go on at what they set. The value a field held before her
practice first changed it is kept, and the next goal in that application that does not set it itself sets it back.

Nothing here knows an application or what its settings are: a setting is a label and a number.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
from typing import Any

logger = logging.getLogger("Aura.ScreenPursuit.ASetting")

__all__ = ["FOR_HER_PRACTICE", "numbers_beside_labels", "set_what_the_goal_sets", "settings_the_goal_sets",
           "with_what_she_puts_back", "without_its_settings"]

#: Set while a game of her own practice is pursued: what it changes is to be put back for the person's next game.
FOR_HER_PRACTICE: contextvars.ContextVar[bool] = contextvars.ContextVar("for_her_practice", default=False)
#: Where what she is to put back is kept: per application, each setting with the value it held before.
PUT_BACK = "settings-her-practice-changed"

SETTING = re.compile(r",?\s*\bwith (?P<name>[^,.;]+?) set to (?P<value>\d+(?:\.\d+)?)", re.I)
#: How far apart, as a share of the window's height, a label and its number may stand on one row; and how far above a
#: number, and across from it, its label may stand where none is on its row.
ONE_ROW = 0.02
ABOVE_WITHIN = 0.06
ACROSS_WITHIN = 0.15
#: Steps a form is scrolled to find a setting not in view, at most; and how far a step goes, in lines. Short steps, so
#: a row passes wholly into view: LIVE-like 27 November 2026 a row cut in half by the top of a form read "Depul" "44"
#: for "Depth" "24", and wholly in view, right.
SCROLLS_AT_MOST = 8
SCROLL_LINES = 2
#: The longest a window is given to show what a press or a typing did.
SHOWS_IN_S = 0.6


def with_what_she_puts_back(goal: str, target_app: str) -> str:
    """The goal, with each setting her practice changed in ``target_app`` set back to what it held, where the goal
    does not set it itself."""
    from core.runtime.what_she_learned import recall

    pending = dict((recall(PUT_BACK) or {}).get(target_app) or {}) if target_app else {}
    named = {name for name, _value in settings_the_goal_sets(goal)}
    extra = "".join(f", with {name} set to {value}" for name, value in pending.items() if name not in named)
    if not extra:
        return goal
    return goal.rstrip().rstrip(".") + extra + "."


def _noted(target_app: str, name: str, before: str | None, value: str) -> None:
    """A setting set and read back: put back if it was waiting to be, or, in a game of her practice, its value before
    kept to be put back where nothing is kept for it yet."""
    from core.runtime.what_she_learned import recall, remember

    held = dict(recall(PUT_BACK) or {})
    pending = dict(held.get(target_app) or {})
    if name in pending and float(pending[name]) == float(value):
        del pending[name]
    elif FOR_HER_PRACTICE.get() and name not in pending and before is not None and float(before) != float(value):
        pending[name] = before
    else:
        return
    held[target_app] = pending
    remember(PUT_BACK, held or {"nothing": {}})


def settings_the_goal_sets(goal: str) -> list[tuple[str, str]]:
    """Each setting the goal names with the number it is to be set to."""
    return [(m.group("name").strip(), m.group("value")) for m in SETTING.finditer(goal or "")]


def without_its_settings(goal: str) -> str:
    """The goal with what it sets taken out: what it asks to be done."""
    return SETTING.sub("", goal or "").strip()


def _a_number(text: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?", text.strip()) is not None


def _read(reading: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for region in reading.get("layout") or []:
        if not isinstance(region, dict):
            continue
        text = " ".join(str(region.get("text") or "").split())
        if not text:
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        found.append({"text": text, "x": x, "y": y})
    return found


def numbers_beside_labels(reading: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Each label with the numbers shown beside it: on its row to its right, or under it where nothing on the number's
    row is a label."""
    regions = _read(reading)
    labels = [r for r in regions if any(ch.isalpha() for ch in r["text"])]
    found: dict[str, list[dict[str, Any]]] = {}
    for number in (r for r in regions if _a_number(r["text"])):
        row = [lab for lab in labels if abs(lab["y"] - number["y"]) <= ONE_ROW and lab["x"] < number["x"]]
        if row:
            label = max(row, key=lambda lab: lab["x"])
        else:
            above = [lab for lab in labels if 0 < number["y"] - lab["y"] <= ABOVE_WITHIN
                     and abs(lab["x"] - number["x"]) <= ACROSS_WITHIN]
            if not above:
                continue
            label = max(above, key=lambda lab: lab["y"])
        found.setdefault(label["text"], []).append(number)
    return found


async def _the_label_for(target_app: str, name: str, labels: list[str]) -> int | None:
    """Which label sets ``name``: the one nearest in meaning where one stands out, else her own model's answer."""
    import numpy as np

    from core.language.learned_matcher import embed_sentences

    from .screen_pursuit_a_way_on import _her_understanding, _stands_out

    vectors = await asyncio.to_thread(embed_sentences, [name, *labels])
    if len(vectors) == len(labels) + 1:
        matrix = np.asarray(vectors, dtype=np.float64)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        scores = [float(v) for v in matrix[1:] @ matrix[0]]
        chosen = _stands_out(scores)
        if chosen is not None:
            return chosen
    return await _her_understanding(target_app, f"set {name}", labels)


async def set_what_the_goal_sets(target_app: str, goal: str, keep: dict[str, Any], reading: dict[str, Any],
                                 read_screen: Any, from_a_window: bool) -> bool:
    """The first setting the goal names that is not yet set, set on this window; whether the window changed for it."""
    from core.capabilities.host_automation import get_host_automation

    from .screen_pursuit import _tell
    from .screen_pursuit_a_way_on import _a_reading
    from .screen_pursuit_surface import click_normalized

    # Every act here is bound to the window that was read: with no application named, nothing is pressed or typed,
    # for an act bound to nothing lands in whatever is in front.
    window_app = str(target_app or "")
    if not window_app:
        return False
    done: dict[str, str] = keep.setdefault("settings_set", {})
    left = [(name, value) for name, value in settings_the_goal_sets(goal) if name not in done]
    if not left:
        return False
    name, value = left[0]
    tried: dict[str, int] = keep.setdefault("settings_tried", {})
    if tried.get(name, 0) >= 2:
        return False
    host = get_host_automation()
    bounds = list(reading.get("bounds") or [])
    # A screen is looked down once for a setting: a page with none on it is scrolled through once, and back.
    screen = " | ".join(sorted(r["text"] for r in _read(reading))[:30])
    looked: list[str] = keep.setdefault("settings_looked_down", [])
    may_scroll = screen not in looked
    looked.append(screen)
    steps, under = 0, None
    for scrolled in range(SCROLLS_AT_MOST + 1):
        beside = numbers_beside_labels(reading)
        labels = list(beside)
        chosen = await _the_label_for(target_app, name, labels) if labels else None
        if chosen is not None:
            label = labels[chosen]
            numbers = beside[label]
            break
        if scrolled == SCROLLS_AT_MOST or len(bounds) < 4 or not may_scroll:
            if steps:
                await host.scroll(dy=SCROLL_LINES * steps, at=under)
            logger.info("the setting %r is not to be found in %r", name, target_app)
            return False
        # Further down the form, where its words are: the pointer over them, then a step down.
        regions = _read(reading)
        if not regions:
            return False
        middle_x = sorted(r["x"] for r in regions)[len(regions) // 2]
        middle_y = sorted(r["y"] for r in regions)[len(regions) // 2]
        under = (int(bounds[0] + middle_x * bounds[2]), int(bounds[1] + middle_y * bounds[3]))
        await host.scroll(dy=-SCROLL_LINES, at=under)
        steps += 1
        await asyncio.sleep(SHOWS_IN_S)
        reading = await _a_reading(read_screen, target_app, from_a_window=from_a_window) or {}
    tried[name] = tried.get(name, 0) + 1
    before = numbers[0]["text"] if len(numbers) == 1 else None
    if len(numbers) > 1:
        # Choices beside one label: the one that is the goal's number is pressed.
        same = [n for n in numbers if float(n["text"]) == float(value)]
        if not same:
            logger.info("%r offers %s, not %s", label, [n["text"] for n in numbers], value)
            return False
        await click_normalized(same[0]["x"], same[0]["y"], expect_app=window_app, bounds=bounds)
    elif float(numbers[0]["text"]) != float(value):
        field = numbers[0]
        await click_normalized(field["x"], field["y"], expect_app=window_app, bounds=bounds)
        await host.hotkey("command", "a", expect_app=window_app)
        await host.type_text(value, use_clipboard=False, expect_app=window_app)
        await host.hotkey("tab", expect_app=window_app)
    await asyncio.sleep(SHOWS_IN_S)
    after = await _a_reading(read_screen, target_app, from_a_window=from_a_window) or {}
    shown = numbers_beside_labels(after).get(label, [])
    if any(float(n["text"]) == float(value) for n in shown):
        done[name] = value
        _noted(target_app, name, before, value)
        _tell(f"I set \"{label}\" to {value}, for {name}.")
        logger.info("set %r to %s in %r, for %s", label, value, target_app, name)
        return True
    logger.info("set %r to %s in %r, and the window shows %s", label, value, target_app, [n["text"] for n in shown])
    return False
