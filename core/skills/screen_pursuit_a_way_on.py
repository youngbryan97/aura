"""A way on through an application, taken by what its labels mean: the card or button nearest what she was asked to do.

Asked to play a game in an application, a person who opens it and finds a page of cards ("Play Chess: play against an
engine or a friend", "Analysis Board", "Puzzles") does not press the arrow keys at it; they read the cards and press the
one that means what they came to do. LIVE 20 November 2026 she opened En Croissant on its start page, took the grid of
five cards for a board, pressed arrows at it for a minute, and gave up: "Nothing I do is changing anything here".

So where the window holds words and nothing she has done there has changed it yet, the words are gathered into the
controls they belong to (a card's heading, its line under it, its button: words stacked close in one column), each
control is read by her encoder against the first act she was asked that is still to do (acts are said in the order they
are done, and "open it" is done once she is in it), and the nearest control is pressed where it is further ahead of the
next than the next is of the one after. Within it the words nearest the act are pressed first; where the screen does not change, the next. A control
that names the application she is in is where she already is, and is left. A control is taken once. Nothing here knows
an application or a word on a button.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.ScreenPursuit.AWayOn")

__all__ = ["took_a_way_on"]

#: How often the window is read for a way on, in seconds.
LOOKED_EVERY_S = 3.0
#: How long a press is given to change the screen.
SHOWS_IN_S = 1.0
#: Presses a way on is given before it is taken not to answer: a second, for a press that did not land.
PRESSES_OF_A_WAY_ON = 2
#: Words this close, as shares of the window, belong to one control: across, the width of a column of text; down, a
#: line and a half of a card.
SAME_COLUMN = 0.06
SAME_CARD = 0.09
#: The most runs of words a thing drawn to be pressed says what it does in: a card's heading, its line and its button.
#: A box of more is something to read (a list of moves, a page of notes).
SAYS_AT_MOST = 3


def _regions(reading: dict[str, Any]) -> list[dict[str, Any]]:
    found = []
    for region in reading.get("layout") or []:
        text = " ".join(str((region or {}).get("text") or "").split())
        if not any(ch.isalpha() for ch in text):
            continue
        try:
            x = float(region.get("center_x", region.get("x", 0.0)))
            y = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        one = {"text": text, "x": x, "y": y}
        if isinstance(region.get("within"), (list, tuple)) and len(region["within"]) == 4:
            one["within"] = [float(v) for v in region["within"]]
        found.append(one)
    return found


def _a_box_of_its_own(control: list[dict[str, Any]], controls: list[list[dict[str, Any]]]) -> bool:
    """Whether a control's words are drawn in a box that holds no other control's words, and say no more than a card
    does: a button, or a card, as against words set on a page among others. The box is the smallest drawn around any of its words (read with them,
    core/perception/what_the_pixels_show.py)."""
    boxes = [r["within"] for r in control if "within" in r]
    if not boxes or len(control) > SAYS_AT_MOST:
        return False
    box = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return not any(_holds(box, r) for other in controls if other is not control for r in other)


def _holds(box: list[float] | None, region: dict[str, Any]) -> bool:
    return box is not None and box[0] <= region["x"] <= box[2] and box[1] <= region["y"] <= box[3]


def _drawn_together(one: dict[str, Any], other: dict[str, Any]) -> bool:
    """Whether two runs of words may be one control by what is drawn around them: where the box drawn around either
    holds the other (a card's heading and its button), or nothing is seen drawn around them. A button's words and the
    words under it in another box are two things."""
    mine, theirs = one.get("within"), other.get("within")
    if mine is None or theirs is None:
        return True
    return _holds(mine, other) or _holds(theirs, one)


def _controls(regions: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Words stacked close in one column, top to bottom, and drawn in one box, as one control."""
    joined = list(range(len(regions)))

    def root(i: int) -> int:
        while joined[i] != i:
            joined[i] = joined[joined[i]]
            i = joined[i]
        return i

    for i, one in enumerate(regions):
        for j in range(i + 1, len(regions)):
            other = regions[j]
            if (abs(one["x"] - other["x"]) <= SAME_COLUMN and abs(one["y"] - other["y"]) <= SAME_CARD
                    and _drawn_together(one, other)):
                joined[root(i)] = root(j)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, region in enumerate(regions):
        groups.setdefault(root(i), []).append(region)
    return [sorted(group, key=lambda r: r["y"]) for group in groups.values()]


def _stands_out(scores: list[float]) -> int | None:
    """The one that stands out: further ahead of the next than the next is of the one after it."""
    if len(scores) < 2:
        return 0 if scores else None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    lead = scores[order[0]] - scores[order[1]]
    behind = scores[order[1]] - scores[order[2]] if len(order) > 2 else 0.0
    return order[0] if lead > behind else None


def _names(text: str, app: str) -> bool:
    """Whether words name the application she is in: where she is already is no way further into it."""
    import re

    letters = re.sub(r"[^a-z0-9]+", "", str(app or "").lower())
    return bool(letters) and letters in re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _the_words_differ(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    one = {r["text"] for r in before}
    other = {r["text"] for r in after}
    if not one and not other:
        return False
    return len(one ^ other) > 0.3 * max(len(one), len(other))


#: The longest a reading is waited on before it is taken to show nothing.
READ_AT_MOST_S = 5.0


async def _a_reading(read_screen: Any, target_app: str, *, from_a_window: bool = True) -> dict[str, Any] | None:
    """A careful reading of the window: its own pixels at full size, read for words; where those cannot be had, the
    pursuit's reading. None where none comes back in time: a look that does not come back is not a look. LIVE 20
    November 2026 the pursuit's quick reading of En Croissant's start page gave "Plav Chess", "Analwcie Dnard Open" and
    "Train uane abaee alilll", and nothing on it meant anything; the window's own pixels read "Play Chess", "Play
    against an engine or a friend", "Play" in a quarter of a second."""
    import asyncio

    try:
        careful = (await asyncio.wait_for(_its_own_pixels_read(target_app), timeout=READ_AT_MOST_S)
                   if from_a_window else None)
        if careful:
            return careful
        return await asyncio.wait_for(read_screen(target_app), timeout=READ_AT_MOST_S)
    except (TimeoutError, OSError, RuntimeError) as why:
        logger.info("the window was not read for a way on (%s)", type(why).__name__)
        return None


async def _its_own_pixels_read(target_app: str) -> dict[str, Any] | None:
    import asyncio

    from core.capabilities import window_server
    from core.perception.what_the_pixels_show import _the_pixels_of, recognize_text

    window = await asyncio.to_thread(window_server.window_of, target_app)
    if window is None:
        return None
    picture = await _the_pixels_of(window)
    if picture is None:
        return None
    pixels = np.asarray(picture)
    words = await asyncio.to_thread(recognize_text, np.ascontiguousarray(pixels[:, :, 2::-1]))
    tall, wide = pixels.shape[:2]
    layout = []
    for region in words or []:
        x = float(region.get("center_x", region.get("x", 0.0)) or 0.0)
        y = float(region.get("center_y", region.get("y", 0.0)) or 0.0)
        # Fractions of the picture, top-left, as every reading gives them.
        layout.append({"text": region.get("text", ""), "center_x": x / wide if x > 1.5 else x,
                       "center_y": y / tall if y > 1.5 else y})
    return {"ok": True, "layout": layout, "bounds": list(window.bounds)} if layout else None


async def _her_understanding(target_app: str, act: str, texts: list[str]) -> int | None:
    """Which of the controls a person using this application would press for the next step of ``act``, as her own model
    reads it, the controls the only answers it may give; None where it says none of them, or cannot be asked."""
    from typing import Literal

    from pydantic import create_model

    from core.cognition.asking_in_turn import THE_PLACE
    from core.cognition.what_things_are import asked_patiently
    from core.rebuilding.her_model import ask_her_model

    named = [" ".join(text.split())[:80] for text in texts]
    choices = list(dict.fromkeys(named))
    if not choices:
        return None
    none = "none of these"
    schema = create_model("_Press", press=(Literal[(*choices, none)], ...))  # type: ignore[valid-type]
    prompt = (f"Someone asked her to {act}. She is in the application “{target_app}”, and what it shows now that can "
              "be pressed is this:\n" + "\n".join(f"- {choice}" for choice in choices)
              + "\nAs a person using this application would, which one does the next step of what she was asked? "
              f"Where none of them does, say “{none}”.")
    try:
        got = await asked_patiently(ask_her_model, prompt, schema, 120, matters=THE_PLACE)
    except (RuntimeError, OSError, ValueError, TypeError, TimeoutError) as why:
        logger.info("which control to press was not asked: %s", str(why)[:160])
        return None
    pressed = str(getattr(got, "press", "") or "")
    if not pressed or pressed == none or pressed not in named:
        return None
    logger.info("a way on in %r by her understanding of it: %r for %r", target_app, pressed, act[:60])
    return named.index(pressed)


async def _pressed(region: dict[str, Any], before: list[dict[str, Any]], target_app: str, reading: dict[str, Any],
                   keep: dict[str, Any], read_screen: Any, from_a_window: bool) -> bool:
    """Press one label and see whether the screen changed for it."""
    import asyncio

    from .screen_pursuit import _tell
    from .screen_pursuit_surface import click_normalized

    keep.setdefault("ways_on_taken", []).append(region["text"])
    await click_normalized(region["x"], region["y"], expect_app=target_app, bounds=list(reading.get("bounds") or []))
    await asyncio.sleep(SHOWS_IN_S)
    after = _regions(await _a_reading(read_screen, target_app, from_a_window=from_a_window) or {})
    if _the_words_differ(before, after):
        _tell(f"I pressed \"{region['text']}\": it's the next step here toward what you asked.")
        return True
    return False


def _a_surface_of_controls(layout: list[dict[str, Any]]) -> bool:
    """Whether what a window shows is mostly labels said in words: an application's own interface (a start page, a
    setup form, a dialog), not a world of tiles, numbers or things that move. Every run of text read counts, numbers
    as well: a board of numbered tiles with a "New Game" button under it is a world, not a form."""
    import re

    texts = [" ".join(str((region or {}).get("text") or "").split()) for region in layout or []]
    texts = [text for text in texts if text]
    if len(texts) < 3:
        return False
    in_words = sum(1 for text in texts if re.search(r"[^\W\d_]{2,}", text))
    return in_words >= 0.6 * len(texts)


async def took_a_way_on(target_app: str, goal: str, keep: dict[str, Any], *, nothing_answered: bool,
                        seen: dict[str, Any] | None = None) -> bool:
    """Press the control in ``target_app``'s window whose words mean most nearly ``goal``, where the window is a
    surface of controls, or nothing she has done there has changed the screen yet; whether the screen changed for it.

    Sets ``keep["surface_is_controls"]`` from each look, and ``keep["way_on_found_nothing"]`` where it read the window
    and found nothing to press."""
    from .screen_pursuit import read_screen
    from .screen_pursuit_on_a_page import on_her_page

    now = time.monotonic()
    if not target_app or not goal or on_her_page() is not None:
        return False
    # Not where the look just taken failed: a capture that wedged once is not asked again this cycle.
    if seen is not None and seen.get("ok") is False:
        return False
    # A window's own pixels only where the look just taken was of a window: a reading given any other way (a test's,
    # a page's) is the whole of what there is to read, and the desktop is not looked at behind it.
    from_a_window = seen is None or bool(seen.get("window_number"))
    # What kind of surface this is comes first, from the look just taken. One of labelled controls is pressed by what
    # its controls mean, whatever her keys did on another surface of the same application: LIVE 20 November 2026 her
    # record of En Croissant said her keys had done something there, and she pressed arrows at its start page.
    if seen is not None:
        keep["surface_is_controls"] = bool(from_a_window and _a_surface_of_controls(seen.get("layout") or []))
    if not (nothing_answered or keep.get("surface_is_controls")):
        return False
    if now - float(keep.get("way_on_looked_at", -1e9)) < LOOKED_EVERY_S:
        return False
    keep["way_on_looked_at"] = now
    keep["way_on_found_nothing"] = False
    reading = await _a_reading(read_screen, target_app, from_a_window=from_a_window)
    if not reading or reading.get("ok") is False:
        return False
    keep["surface_is_controls"] = bool(from_a_window and _a_surface_of_controls(reading.get("layout") or []))
    if not (nothing_answered or keep["surface_is_controls"]):
        return False
    pressed = await _a_way_on_pressed(target_app, goal, keep, reading, read_screen, from_a_window)
    keep["way_on_found_nothing"] = not pressed
    return pressed


async def _a_way_on_pressed(target_app: str, goal: str, keep: dict[str, Any], reading: dict[str, Any],
                            read_screen: Any, from_a_window: bool) -> bool:
    """The control on a reading whose words mean most nearly ``goal``, or her model's choice where none stands out,
    pressed; whether the screen changed for it."""
    import asyncio

    from core.language.learned_matcher import embed_sentences

    from .screen_pursuit_surface import click_normalized

    regions = _regions(reading)
    if len(regions) < 3:
        return False
    taken: list[str] = keep.setdefault("ways_on_taken", [])
    # Every control is weighed, those already pressed too: where the one that means what was asked has not answered,
    # the next best is a different act, not a way on. LIVE 21 November 2026 "Play Chess" did not answer a press made
    # as the screen was unlocked, and with it set aside "Puzzles: Train your chess skills" stood out and was pressed.
    controls = [c for c in _controls(regions) if not _names(" ".join(r["text"] for r in c), target_app)]
    if len(controls) < 2:
        return False
    texts = [" ".join(r["text"] for r in control) for control in controls]
    # Each act she was asked, apart: "open it and play a game against X" is two, and the first is done once she is in.
    from core.language.what_she_is_asked_to_do import the_parts_that_ask_her

    acts = [act for act in (the_parts_that_ask_her(goal) or [goal]) if not _names(act, target_app)] or [goal]
    vectors = await asyncio.to_thread(embed_sentences, [*acts, *texts])
    if len(vectors) != len(acts) + len(texts):
        return False
    matrix = np.asarray(vectors, dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    by_act = matrix[len(acts):] @ matrix[:len(acts)].T
    # The act to find a way on for is the first one still to do: a person says what to do in the order it is done,
    # and "open it" is done once she is in it. "Tell me each move" and "close it when you're done" come after.
    act = 0
    scores = [float(v) for v in by_act[:, act]]
    chosen = _stands_out(scores)
    # What is drawn to be pressed first. A person looking for the way on looks at the buttons and the cards, and reads
    # the words set among other words (a heading, a name, a date) as things to read. Among the controls in a box of
    # their own, the one that stands out is taken, where it means more than the middle of what the screen says. LIVE
    # 23 November 2026 at a finished game "BLACK Stockfish", a heading over the players' names, was nearer "play a game
    # against Stockfish" (0.77) than the "New Game" button (0.60); pressing it laid "Player not found" over the board.
    boxed = [i for i, control in enumerate(controls) if _a_box_of_its_own(control, controls)]
    if len(boxed) >= 2:
        among = _stands_out([scores[i] for i in boxed])
        if among is not None and scores[boxed[among]] > float(np.median(scores)):
            if boxed[among] != chosen and chosen is not None:
                logger.info("of the controls drawn to be pressed in %r, %r stands out (%.2f) over the words %r (%.2f)",
                            target_app, texts[boxed[among]][:60], scores[boxed[among]], texts[chosen][:60],
                            scores[chosen])
            chosen = boxed[among]
    # The control that means most nearly what was asked, pressed as often as a way on is given, without what was asked
    # coming of it, has been answered: it is not the way. Which other control does the next step is the question put
    # where nothing stands out. LIVE 25 November 2026 at a finished game "BLACK Stockfish" was pressed twice, the board
    # still took no move, and the run ended with "New Game" on the screen, never pressed.
    exhausted = chosen is not None and taken.count(" / ".join(r["text"] for r in controls[chosen])) >= \
        PRESSES_OF_A_WAY_ON
    if exhausted:
        logger.info("the way on in %r, %r, has not answered %d presses; which other control does the next step is put "
                    "to her", target_app, texts[chosen][:80], PRESSES_OF_A_WAY_ON)
    if chosen is None or exhausted:
        # Nothing stands out by what the words are about ("Start game" is not about chess against an engine, and is
        # what starts one): which control does the next step is a question of understanding the place, put to her own
        # model, with the controls as the only answers. A still screen gives her the time to think.
        pressed = Counter(t for c in taken for t in c.split(" / "))
        labels = [r for r in regions if not _names(r["text"], target_app) and pressed[r["text"]] < PRESSES_OF_A_WAY_ON]
        picked = await _her_understanding(target_app, acts[act], [r["text"] for r in labels])
        if picked is None:
            logger.info("no way on stands out in %r for %r, nor seen by her", target_app, acts[act][:60])
            return False
        return await _pressed(labels[picked], regions, target_app, reading, keep, read_screen, from_a_window)
    control = controls[chosen]
    taken.append(" / ".join(r["text"] for r in control))
    bounds = list(reading.get("bounds") or [])
    logger.info("a way on in %r by what it means: %r for %r (%.2f, next %.2f)", target_app, texts[chosen][:80],
                acts[act][:60], scores[chosen], sorted(scores)[-2])
    from .screen_pursuit import _tell

    # Within it, the words nearest the act first: on a card its button or its heading, in a column of settings the
    # one that starts what was asked rather than one that changes how.
    own = await asyncio.to_thread(embed_sentences, [acts[act], *(r["text"] for r in control)])
    order = list(control)
    if len(own) == len(control) + 1:
        near = np.asarray(own, dtype=np.float64)
        near /= np.linalg.norm(near, axis=1, keepdims=True) + 1e-12
        closeness = near[1:] @ near[0]
        order = [control[i] for i in np.argsort(-closeness)]
    for region in order:
        await click_normalized(region["x"], region["y"], expect_app=target_app, bounds=bounds)
        await asyncio.sleep(SHOWS_IN_S)
        after = _regions(await _a_reading(read_screen, target_app, from_a_window=from_a_window) or {})
        if _the_words_differ(regions, after):
            _tell(f"I pressed \"{region['text']}\": it's the nearest thing here to what you asked.")
            return True
    return False
