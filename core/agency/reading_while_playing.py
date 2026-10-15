"""What her fast way of playing reads as it plays: the counters, the bars and the words on the screen.

Moved out of core/agency/playing_as_it_happens.py, which re-exports every name here. A counter's rise or a bar's fall
is a gain or a loss as its words say; a loss with nothing touching her is put down to what was near her then; and the
things that are writing are for reading, not for touching.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from core.agency.navigation_before_body_experiments import (
    capture_before_reading,
    keep_navigation_reading,
    words_bound_to_capture,
)
from core.verify import invariant

if TYPE_CHECKING:
    from core.agency.playing_as_it_happens import _Run
    from core.agency.what_meeting_things_does import WhatMeetingDoes
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.what_moves_in_the_picture import WhatMoves


async def _keep_reading(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves,
                        picture: Any, at: float, read_words: Callable[[Any], list[dict[str, Any]]] | None,
                        *, say: Any = None) -> None:
    """Read the counters off to one side, and the bars every picture; take in the last reading when it is done."""
    from core.agency.playing_as_it_happens import READ_EVERY_S
    from core.perception.the_drawing_as_objects import words_in

    _read_the_bars(run, meeting, hers, moves, picture, at, say)
    rendered = words_in(picture)
    if rendered is not None:
        if run.reading is not None:
            run.reading.cancel()
            run.reading = None
        run.read_at = at
        _read_the_words(run, meeting, hers, moves,
                        words_bound_to_capture(rendered, capture_before_reading(picture, at)), at, say)
        return
    if read_words is None:
        return
    if run.reading is not None and run.reading.done():
        regions, when = run.reading.result()
        run.reading = None
        _read_the_words(run, meeting, hers, moves, regions, when, say)
    if run.reading is None and at - run.read_at >= READ_EVERY_S:
        run.read_at = at
        bgr = picture[:, :, ::-1].copy()
        captured = capture_before_reading(picture, at)

        async def _read() -> tuple[list[dict[str, Any]], float]:
            regions = await asyncio.to_thread(read_words, bgr)
            return words_bound_to_capture(regions, captured), at

        run.reading = asyncio.ensure_future(_read())


#: The most readings a stretch keeps, and how many of its first it always keeps: what was there as it began.
READINGS_KEPT = 400
FIRST_READINGS_KEPT = 100


def kept_reading(run: _Run, when: float, said: str) -> None:
    """A reading kept with when its picture was taken: the first ones always, then the latest. Offline 2026-10-11 a
    stretch stopped keeping what it read at its four hundredth reading, about a minute in, and a GAME OVER after that
    was never read."""
    if not said:
        return
    if len(run.words_read) >= READINGS_KEPT:
        del run.words_read[FIRST_READINGS_KEPT]
    run.words_read.append((when, said))


#: The most lines of writing read in play that are remembered as already taken in for what they tell her to do.
LINES_HEARD_KEPT = 600


def rules_as_told_since(run: _Run, rules: Any) -> Any:
    """The rules, with what the screen has told her since play began: a lesson in play says one thing at a time ("USE
    WASD TO MOVE", then "NOW USE YOUR MOUSE TO AIM AND SHOOT THE EVIL PIN!"), and each is taken in as it is read.
    Offline 2026-10-11 a bowling game's second lesson came up mid-stretch, was never taken in, and she never aimed."""
    from core.agency.what_the_rules_said import WhatTheRulesSaid

    if not run.words_read or run.words_read[-1][0] <= run.rules_read_at:
        return rules
    # A reading's lines are one screen's words: a lesson set out over two lines ("NOW USE YOUR MOUSE TO" / "AND SHOOT
    # THE EVIL PIN!") says what neither line says alone.
    readings: dict[float, list[str]] = {}
    for when, said in run.words_read:
        if when > run.rules_read_at:
            readings.setdefault(when, []).append(said)
    fresh = list(dict.fromkeys(" ".join(lines) for lines in readings.values()))
    fresh = [said for said in fresh if said not in run.lines_heard]
    run.rules_read_at = run.words_read[-1][0]
    if len(run.lines_heard) > LINES_HEARD_KEPT:
        run.lines_heard.clear()
    run.lines_heard.update(fresh)
    told = [said for said in fresh if WhatTheRulesSaid.read(said).instructions]
    if not told:
        return rules
    rules = WhatTheRulesSaid.read(" ".join([getattr(rules, "words", "") or "", *told]).strip())
    run.pointer_trigger = rules.a_click_is_a_shot
    return rules


def _read_the_words(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers,
                    moves: WhatMoves, regions: list[dict[str, Any]], when: float, say: Any = None) -> None:
    """The same counter semantics for renderer text and text read from pixels."""
    from core.agency.playing_as_it_happens import _say
    from core.agency.words_read_with_their_picture import kept_word_picture
    from core.cognition.a_guide_to_a_place import heard_in_play

    kept_word_picture(run, when)
    keep_navigation_reading(run, regions)
    run.regions_read = list(regions)
    for region in regions:
        kept_reading(run, when, " ".join(str(region.get("text") or "").lower().split()))
    # What play says as it goes may change how it is played (a new power, a new stage): the guide hears it.
    for line in heard_in_play([str(region.get("text") or "") for region in regions])[:1]:
        _say(run, say, line, when, once=line)
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    for verdict in meeting.read(regions, when, her_x):
        if verdict["what"] == "gain":
            run.gains += 1
            run.blows.gained(when)
            run.charging.gained(when)
        else:
            run.losses += 1
            hers.lost_a_life(when)
            _lost_from_a_distance(run, meeting, mine, moves, verdict)
            # A loss sets her back and changes the place: where she seemed not to get to before it is not known to be
            # out of her reach (offline 2026-10-09, set back to the start each time, the way to a coin was marked out
            # of her reach and she waited in a corner while it stood there).
            run.barred.clear()
    run.contest.heard(" ".join(str(region.get("text") or "") for region in regions))
    run.contest.counted(meeting.readouts.current, meeting.readouts.where, her_x, when, meeting.readouts.out_of)
    meeting.writing = _what_is_writing(moves, regions)


#: The words that name a count of points; and those that name a control a bar is for ("SPACE TO STOMP", "HOLD X").
_A_SCORE = re.compile(r"\b(?:score|points?|pts|hi-?score|high score)\b", re.I)
_A_CONTROLS = re.compile(r"\b(?:space|spacebar|press|hold|click|tap)\b.*\bto\b|\b(?:press|hold)\s+\w+\b", re.I)


def _read_the_bars(run: _Run, meeting: WhatMeetingDoes, hers: WhichIsHers, moves: WhatMoves, picture: Any, at: float,
                   say: Any) -> None:
    """Every bar's rise or fall, as a gain or a loss as its words say; and how much she has left, for how careful to be.

    A bar that empties as she is hit is the cost of what hit her, learned as a counter's fall is: LIVE 2026-10-09 three
    heroes shared a health bar, and she played as if nothing could hurt her.
    """
    from core.agency.naming_what_she_sees import where_on_screen
    from core.agency.playing_as_it_happens import _say
    from core.agency.what_she_has_left import bars_read, what_is_left
    from core.perception.how_full_a_bar_is import BarsOnTheScreen

    if run.bars is None:
        run.bars = BarsOnTheScreen()

    def place(x: float, y: float) -> str:
        return where_on_screen(moves, x * moves.shape[1], y * moves.shape[0]) if moves.shape[0] else "middle"

    # A bar with no name going down before anything on the screen answers her may be a story playing out or a thing
    # loading. Keep its measurements without crediting reward or resources until there is an interactive reference.
    answering = (getattr(hers, "kind", None) is not None or getattr(hers, "follows_pointer", False)
                 or (run.presses is not None and run.presses.moves_anything(4.0)))
    for verdict in bars_read(run.bars, picture, at, run.regions_read, place):
        # A bar named as a score is a count, not what she has left, and a score read as falling was misread: LIVE
        # 2026-10-11 "The SCORE bar goes down: it's what I have left." A bar named by a control ("SPACE TO STOMP") fills
        # for that control and empties as it is used: it is no loss either.
        if _A_SCORE.search(str(verdict.get("counter") or "")) or _A_CONTROLS.search(str(verdict.get("counter") or "")):
            continue
        if not _bar_change_can_teach(str(verdict.get('counter') or ''), answering):
            continue
        meeting._verdict(verdict)
        if verdict["what"] == "gain":
            run.gains += 1
            continue
        run.losses += 1
        _lost_from_a_distance(run, meeting, hers.thing(moves), moves, verdict)
        hit = any(0.0 <= at - when <= 1.0 for when in meeting._met.values())
        named = verdict["counter"]
        unnamed = named.startswith("the bar")
        # Of the bars with no words, one is said to be what she has left: offline 2026-10-12 five were, in one game.
        if unnamed and run.bars.said_unnamed and run.bars.said_unnamed != named:
            continue
        spoken = named if unnamed else _a_bar_in_words(named)
        if spoken and (answering or not unnamed):
            run.bars.said_unnamed = run.bars.said_unnamed or (named if unnamed else "")
            _say(run, say, f"{spoken[0].upper()}{spoken[1:]} goes down{' as I am hit' if hit else ''}: it's what I have left.",
                 at, once=f"bar {named}")
    remaining = what_is_left(run.bars) if answering else what_is_left(run.bars, allow_unnamed=False)
    run.vitals = meeting.vitals = remaining


def _bar_change_can_teach(counter: str, answering: bool) -> bool:
    """A nameless changing strip needs an interactive reference before it can teach reward."""
    return answering or bool(counter) and not counter.startswith('the bar')


@invariant('agency.unowned_strips_cannot_teach_reward', scope='agency',
           owner='core/agency/reading_while_playing.py', observational=False)
def _unowned_bar_reward_invariant() -> tuple:
    assert not _bar_change_can_teach('the bar at the middle', False)
    assert not _bar_change_can_teach('', False)
    assert _bar_change_can_teach('the HEALTH bar', False)
    assert _bar_change_can_teach('the bar at the middle', True)
    return ()


def _a_bar_in_words(named: str) -> str:
    """A bar named for the words beside it, as those words would be said ("the energy bar"), or '' where they are not
    words of the language: a misread or lettered label is not read out (LIVE 2026-10-13 "The no InVentoRY bar goes
    down" was said twice to an audience). The bar is still counted; only its name is not said."""
    from core.language.words_of_the_language import words_of

    label = named.removeprefix("the ").removesuffix(" bar")
    words = words_of(label, most=3)
    return f"the {words} bar" if words else ""


def _lost_from_a_distance(run: _Run, meeting: WhatMeetingDoes, mine: Any, moves: WhatMoves, verdict: dict[str, Any]) -> None:
    """A loss with nothing having touched her lately is put down to the things near her when it came, as they stood then."""
    from core.agency.playing_as_it_happens import TOUCHED_WITHIN_S

    if run.reach is None or mine is None:
        return
    then = (float(verdict.get("from", verdict["at"])) + float(verdict["at"])) / 2
    if any(abs(when - then) <= TOUCHED_WITHIN_S for when in meeting._met.values()):
        return
    run.reach.lost_untouched(_as_it_stood(mine, then), [_as_it_stood(thing, then) for thing in moves.things.values()
                                                         if thing.number != mine.number and thing.number not in meeting.writing])


def _as_it_stood(thing: Any, then: float) -> Any:
    """A thing as it stood at ``then``, from where its way had it nearest that time."""
    from types import SimpleNamespace

    path = list(getattr(thing, "path", ()) or ())
    at, x, y = min(path, key=lambda step: abs(step[0] - then)) if path else (then, thing.x, thing.y)
    return SimpleNamespace(number=thing.number, kind=thing.kind, x=x, y=y, vx=thing.vx, vy=thing.vy, w=thing.w, h=thing.h)


def _what_is_writing(moves: WhatMoves, regions: list[dict[str, Any]]) -> set[int]:
    """The things that are writing on the screen: words are for reading, not for touching."""
    from core.agency.playing_as_it_happens import _boxes_meet, _moved_box

    tall, wide = moves.shape
    boxes = [
        (float(r["x"]) * wide, float(r["y"]) * tall, (float(r["x"]) + float(r["width"])) * wide, (float(r["y"]) + float(r["height"])) * tall)
        for r in regions
    ]
    return {t.number for t in moves.things.values() if not t.moved and any(_boxes_meet(t.box(), _moved_box(b, 0, 0, 1.0)) for b in boxes)}
