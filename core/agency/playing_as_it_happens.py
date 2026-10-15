"""Playing something that does not wait for her, at the speed it goes.

Her other ways of playing look, think, act once and look again. That is right
for a board, which waits, and hopeless for a ball, which is past her paddle in
the time one look takes. Here the looking and the acting are one loop that
runs as fast as pictures arrive, twenty to fifty times a second, and the
judgement in it is arithmetic over what she has measured in this run:

- which thing is hers (core/agency/which_one_answers_to_her.py), and what each
  key does to it;
- where every other thing is going (core/perception/what_moves_in_the_picture.py);
- what comes of meeting each kind of thing, or letting it by
  (core/agency/what_meeting_things_does.py).

From those she picks a place to be: where the next thing worth meeting will
cross her path, in line with the next thing worth shooting, clear of the path
of anything that costs her. Each picture she holds the key that brings her
nearest to that place without walking into anything, and lets it go when she
is there. Nothing in it names a game, a key or a colour.

A stretch of play ends when the picture stops moving: a new screen that holds
still, a game over, a menu. What to do about that screen is a question for
her slower way of playing, which reads it.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import statistics
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.agency.a_body_still_expected_when_it_is_unseen import press_in_time as _press_in_time
from core.agency.a_choice_and_its_reasons import chose_in_play
from core.agency.acting_on_a_prepared_motion import act_on_prepared_motion
from core.agency.building_up import builds_up
from core.agency.choosing_where_to_be import (  # noqa: F401 (the existing decision API)
    CARRIED_AHEAD_S,
    CLEARER,
    DANGER_AHEAD_S,
    GENTLE_SHARE,
    IMAGINES,
    POINTER_SPEED,
    _ahead,
    _boxes_meet,
    _Choosing,
    _contact_line,
    _moved_box,
    _range_of,
    _times_to,
    goes_with,
)
from core.agency.controls_ready_for_the_motion_model import has_a_measured_response
from core.agency.countering_motion_as_it_appears import CounteringMotion
from core.agency.facing_seen_under_its_input import observe_facing
from core.agency.ground_her_code_names import mark_the_ground_under_her
from core.agency.holding_to_charge import Charging
from core.agency.how_far_her_blow_reaches import BLOW_EVERY_S as BLOW_EVERY_S
from core.agency.how_far_her_blow_reaches import HerBlows
from core.agency.how_the_contest_stands import ContestStands
from core.agency.inputs_while_measuring_control import (
    body_probe_is_active,
    hold_control,
)
from core.agency.inputs_while_measuring_control import (
    release_after_play as release_after_play,
)
from core.agency.keeping_a_watcher_company import keeping_company
from core.agency.naming_what_she_sees import (
    SeeingInPlay,
    a_stance_said,
    colour_name,
    describe,
    i_go_where_the_mouse_goes,
    seen_in_play,
    still_looking_at,
    where_on_screen,
)
from core.agency.naming_what_she_sees import named_for_its_part as _named
from core.agency.naming_what_she_sees import shape_name as shape_name
from core.agency.noticing_in_play import a_key_that_pays, noticed
from core.agency.pressing_what_is_shown import PRESSED_FOR_S as PRESSED_FOR_S
from core.agency.reading_while_playing import (  # noqa: F401  (re-exported: they were defined here)
    _as_it_stood,
    _keep_reading,
    _lost_from_a_distance,
    _read_the_bars,
    _read_the_words,
    _what_is_writing,
    rules_as_told_since,
)
from core.agency.responding_before_slower_planning import reflex_record, respond_before_planning
from core.agency.signs_that_come_along import told_to_follow_signs
from core.agency.the_controls_a_game_names import controls_named_in
from core.agency.time_from_picture_to_control import ControlTiming, allow_for_picture_age
from core.agency.time_from_picture_to_control import measure_velocity_response as _measure_response
from core.agency.using_a_control_on_a_thing import (
    UsingOnThings,
    a_place_to_go,
    use_controls_on_things,
)
from core.agency.what_meeting_things_does import AVOID, CLICK, IGNORE, MEET, SHOOT, WhatMeetingDoes
from core.agency.what_play_measures_of_a_place import measured_in_play
from core.agency.what_she_cannot_reach import out_of_her_reach as _out_of_her_reach
from core.agency.what_she_has_left import CAREFUL_BELOW as CAREFUL_BELOW
from core.agency.what_the_rules_said import WhatTheRulesSaid
from core.agency.whether_it_goes_by_itself import HELD_TO_SEE_S as HELD_TO_SEE_S
from core.agency.whether_it_goes_by_itself import THE_PRESS_S as THE_PRESS_S
from core.agency.whether_it_goes_by_itself import goes_somewhere as goes_somewhere
from core.agency.whether_it_goes_by_itself import it_goes_while_held as it_goes_while_held
from core.agency.whether_it_goes_by_itself import (
    the_world_moves_on_its_own as the_world_moves_on_its_own,
)
from core.agency.which_one_answers_to_her import RESPONSE_S, WAYS, WhichIsHers, last_moved
from core.agency.which_one_answers_to_her import ways_carried as ways_carried
from core.agency.who_is_who import FRIENDLY, IN_HER_WAY
from core.agency.who_is_who import NEVER_SHOT as NEVER_SHOT
from core.perception.how_things_move_here import HowThingsMoveHere
from core.perception.play_observation_without_holding_the_loop import open_play_observation
from core.perception.seeing_play_elsewhere import eyes_for_play, short_turns_while_playing
from core.perception.what_changed_and_stayed import WhatChangedAndStayed
from core.perception.what_moves_in_the_picture import TOLD_FROM_STANDING_PX, WhatMoves
from core.perception.where_in_a_world import WhereInTheWorld

logger = logging.getLogger("Aura.PlayingAsItHappens")

__all__ = ["controls_named_in", "it_goes_while_held", "play_as_it_happens", "the_world_moves_on_its_own"]

#: How long each key is held while she finds out what it does.
TRY_A_KEY_S = 0.45

#: How near in time to a loss a thing must have met her for the loss to be its doing, in seconds: a counter is read
#: every half second and the reading takes a moment.
TOUCHED_WITHIN_S = 1.0

#: How often the counters on the screen are read. Reading takes a tenth of a
#: second off to one side and is never waited for.
READ_EVERY_S = 0.5

#: Nothing moving for this long ends a stretch of play.
STILL_FOR_S = 3.0

#: After a new screen, nothing moving for this long ends it sooner.
NEW_SCREEN_STILL_S = 1.5

#: Where the world waits for her, nothing moving for this long ends a stretch: she has tried her ways, and none goes.
WAITING_STILL_FOR_S = 8.0
#: How much longer than the longest quiet a place has kept while played a stillness must last to end a stretch.
QUIET_KEPT = 1.5

#: How long a thing must have been on the screen to be taken for her by how her presses move it: offline 2026-10-11 in a
#: flying shooter things on the screen a second or two (an enemy, a shot) were taken for her and lost a second later,
#: a dozen times in a minute.
THERE_BEFORE_S = 3.0

#: Spoken lines are at least this far apart, so each can be read; and what kinds of things are to her, one at a time.
SAY_EVERY_S = 5.0
STANCE_EVERY_S = 2.5

#: The least time between two clicks, and how far ahead of a moving thing a
#: click is aimed: the click's trip through the browser and the next frame.
CLICK_EVERY_S = 0.25
CLICK_LANDS_S = 0.08

#: How far the pointer goes in one picture while she sweeps it, as a share of the picture.
POINTER_STEP = 0.04

#: Revisit inconclusive control experiments while the world keeps moving.
RECHECK_CONTROLS_S = 5.0

#: Where the pointer is taken while she finds out whether anything follows it.
_POINTER_TRIAL = ((0.2, 0.5), (0.8, 0.5), (0.5, 0.2), (0.5, 0.8), (0.3, 0.3), (0.7, 0.7))


def _against_someone() -> bool:
    """Whether she understands the place is played against someone on the other side (core/agency/how_a_game_works.py)."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None or not hasattr(guide, "controls"):
        return False
    from core.agency.how_a_game_works import understood

    return understood(guide).against_someone()


def _no_body_to_find() -> bool:
    """Whether she understands the place before her puts no body of hers in the picture (core/agency/how_a_game_works.py)."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    if guide is None or not hasattr(guide, "controls"):
        return False
    from core.agency.how_a_game_works import understood

    return understood(guide).has_a_body() is False


#: How often what she understands of the place (someone against her, a body of hers) is looked at again in play.
UNDERSTOOD_EVERY_S = 1.0


def _how_it_stands(run: Any, at: float) -> tuple[bool, bool]:
    """Whether the place is played against someone, and whether it puts no body of hers in the picture, as she
    understands it: looked at again once a second, not every picture."""
    kept = getattr(run, "how_it_stands", None)
    if kept is None or at - kept[0] >= UNDERSTOOD_EVERY_S:
        kept = (at, _against_someone(), _no_body_to_find())
        try:
            run.how_it_stands = kept
        except AttributeError:
            return kept[1], kept[2]
    return kept[1], kept[2]


@dataclass
class _Run:
    keys: list[str]
    began: float
    held: str = ""
    held_since: float = 0.0
    held_trying: bool = False
    trying: int = 0
    tried_at: float = 0.0
    last_payoff_trial_at: float = -math.inf
    #: Each delivered body trial keeps its deadline; the largest duration used also bounds the search allowance.
    body_trial: Any = None
    longest_control_trial_s: float = 0.0
    #: The finite experiment that waits for measured motion before trying to counter it.
    counter_trial: CounteringMotion = field(default_factory=CounteringMotion)
    #: When she last found the thing she took for hers was not.
    lost_at: float = -math.inf
    #: What she understands of the place, looked at again now and then: when, someone against her, no body of hers.
    how_it_stands: tuple[float, bool, bool] | None = None
    #: The one she last threw at, against someone: where it was, when, and how big.
    thrown_at: tuple[float, float, float, float] | None = None
    #: When her thing was last seen to go, and how many presses of the keys that move it she had made by then.
    answered: tuple[float, int] = (-math.inf, 0)
    #: When anything of hers last answered her: what is read after it, on a stretch ended by nothing answering, is not
    #: the game's.
    answered_at: float = -math.inf
    last_moving: float = 0.0
    #: Whether the place's words put her in a situation played in time, and the longest it has gone still between
    #: movements while played: its own rhythm, against which a stillness is or is not an ending.
    in_time: bool = False
    longest_still: float = 0.0
    new_screen_at: float = -math.inf
    #: Her acts (keys pressed, the pointer tried or moved) when the screen last changed, and the pointer's moves.
    acts_at_new_screen: int = 0
    pointer_moves: int = 0
    clicked: dict[int, float] = field(default_factory=dict)
    last_click: float = -math.inf
    #: Keys the words say to press in turn, fast (core/agency/playing_as_it_happens.py `keys_pressed_fast`), and when last.
    burst_keys: list[str] = field(default_factory=list)
    burst_at: float = -math.inf
    last_trigger_began: float = -math.inf
    #: Where in the picture a click at nothing in particular has paid (core/agency/where_clicks_pay.py).
    clicks_pay: Any = None
    pointer: tuple[float, float] = (0.5, 0.5)
    pointed: int = 0
    pointer_first: bool = False
    #: Whether the visible instructions give the pointer a separate trigger.
    pointer_trigger: bool = False
    taps: int = 0
    clicks: int = 0
    #: How long her throws take to land, as seen (core/agency/thrown_at_what_shows.py), and since when a throw at someone
    #: running has been held for them to stop.
    throw_takes: Any = None
    held_since: float | None = None
    #: When her last few shots were made, by the pictures' clock: what comes into sight just after one is that shot.
    shots_at: list[float] = field(default_factory=list)
    #: What a legend on the place's screens drew and said of each thing (core/perception/what_a_legend_shows.py).
    legend: tuple[Any, ...] = ()
    said_at: float = -math.inf
    #: When what a kind of thing is to her was last said: one at a time, the rest when their turn comes.
    stance_said_at: float = -math.inf
    said: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)
    reading: asyncio.Task | None = None
    read_at: float = -math.inf
    counted: dict[str, int] = field(default_factory=dict)
    reported_at: float = 0.0
    stayed: WhatChangedAndStayed = field(default_factory=WhatChangedAndStayed)
    #: Where on her thing she meets things, as a share of its length from the
    #: middle, and for each: how many meetings, and how many were followed by a
    #: gain for her.
    meeting_with: dict[float, list[int]] = field(default_factory=lambda: {-0.7: [0, 0], 0.0: [0, 0], 0.7: [0, 0]})
    meeting_now: float = 0.0
    told_checked: dict[int, bool] = field(default_factory=dict)
    #: Every line of writing read on the screen during the stretch.
    #: When something was first seen moving in this stretch.
    first_moving: float = math.inf
    #: When writing that says the round is over was first read in play.
    over_read_at: float = math.inf
    #: Every reading's words, with when its picture was taken; the latest reading taken in for what it tells her to do,
    #: and the lines already taken in.
    words_read: list[tuple[float, str]] = field(default_factory=list)
    #: The first picture whose words were read, even if it had no writing.
    word_picture_read_at: float | None = None
    rules_read_at: float = -math.inf
    lines_heard: set[str] = field(default_factory=set)
    #: The bars on the screen that fill and empty (core/perception/how_full_a_bar_is.py), the last words read with where
    #: each stood, and how much she has left: the lowest of her bars that are worse lower, 1 where none is known.
    bars: Any = None
    regions_read: list[dict[str, Any]] = field(default_factory=list)
    vitals: float = 1.0
    situation: str = ""
    #: What she has said of the game, each part without where things stood.
    situation_known: set[str] = field(default_factory=set)
    hers_description: str = ""
    hers_descriptions: Counter[str] = field(default_factory=Counter)
    situation_at: float = -math.inf
    met: list[tuple[float, float]] = field(default_factory=list)
    credited_up_to: int = 0
    touching: set[int] = field(default_factory=set)
    pictures: int = 0
    observation_sources: Counter[str] = field(default_factory=Counter)
    input_key_downs: Counter[str] = field(default_factory=Counter)
    responsive_pictures: int = 0
    #: Each line said in this game, and when: carried from one stretch to the next.
    lately: dict = field(default_factory=dict)
    #: Where the rules ask for a place to be gone over: the cells her thing has been over, and their size (set once).
    covered: set = field(default_factory=set)
    cell: float = 0.0
    #: The ground its code names, laid on her picture with what she has marked of it (core/agency/ground_her_code_names.py).
    ground: Any = None
    #: How far ahead of a moving thing a click is aimed (core/agency/leading_a_click.py).
    leading: Any = None
    #: Cells she held a way toward and did not go: out of her reach, as far as she knows.
    barred: set = field(default_factory=set)
    #: Whether the world stood still until she moved in it: then stillness is hers to break.
    waits_for_her: bool = False
    #: The way held, where her thing was, and since when, while it stays put under it.
    stuck: tuple[str, float, float, float] | None = None
    #: The place she is making for, the nearest she has got to it, and since when.
    making_for: tuple[tuple[int, int], float, float, int] | None = None
    #: The walls she has been stopped by on this screen, and the way round them (core/agency/a_way_through.py).
    way_through: Any = None
    control_probe_retries: int = 0
    control_attribution: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=128))
    attribution_changes: int = 0
    last_attribution: tuple[Any, ...] | None = None
    picture_at: float | None = None
    intervals: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    responses: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    motion_responses: deque[float] = field(default_factory=lambda: deque(maxlen=12))
    control_timing: ControlTiming = field(default_factory=ControlTiming)
    control_delivery: Any = None
    previous_control: tuple[Any, ...] | None = None
    gains: int = 0
    losses: int = 0
    company: dict = field(default_factory=dict)
    #: How the contest stands (core/agency/how_the_contest_stands.py), and the last of it she said.
    contest: ContestStands = field(default_factory=ContestStands)
    #: Where the view going by has taken her, in a world bigger than it.
    world: WhereInTheWorld = field(default_factory=WhereInTheWorld)
    contest_said: str = ""
    #: How it stood, and the counters, when she came in: what they were then is not news (LIVE 2026-10-07 "I have 0.").
    contest_first: str | None = None
    counters_first: dict[str, Any] | None = None
    #: Keys the screen shows as pictures in play, and whether it asks for them now (core/agency/pressing_what_is_shown.py).
    shown: Any = None
    #: Signs coming along a line, each a key's, the front one's key held (core/agency/signs_that_come_along.py).
    coming: Any = None
    #: What each press does over the moment after it (core/agency/what_a_press_does.py), the things in the last
    #: picture, and presses made beside the key held, each with when to let it go.
    presses: Any = None
    things: dict = field(default_factory=dict)
    #: Where no body answers her keys: what pressing has paid by where the moving things were (when_a_press_pays.py).
    timing: Any = None
    how: Any = None
    #: What she notices as she plays (core/agency/noticing_in_play.py), and what she sees things as (naming_what_she_sees.py).
    noticing: Any = None
    seeing: Any = None
    using: Any = None
    #: The kinds she kept clear of when the stretch began: what she comes to keep clear of in it is what it taught her.
    avoided_before: set[int] = field(default_factory=set)
    #: Whether the place is one where pieces are dropped to build up (core/agency/building_up.py).
    building_up: bool = False
    #: How far each kind reaches: the distance and side at which it has cost her untouched (how_far_a_thing_reaches.py).
    reach: Any = None
    #: The tells of each kind of thing, learned from what followed each change in it (core/agency/warning_signs.py).
    signs: Any = None
    settings: Any = None
    #: Each choice she makes, with its reason, craft, what it counts on and what it came to (a_choice_and_its_reasons.py).
    choices: Any = None
    #: What the place does to her whatever she does (core/agency/what_the_place_does_to_her.py), and the thing she is
    #: going for, kept to until there is a reason to let it go (core/agency/a_thing_gone_for.py).
    pressures: Any = None
    gone_for: Any = None
    #: Where she faces, where her keys turn her and take her the way she faces (core/agency/which_way_she_faces.py),
    #: and how many trials it has asked for.
    faces: Any = None
    prepared_motion_binding: Any = None
    facing_trials: int = 0
    facing_trial: Any = None
    #: The things that come for her, told from those that go their own way (core/agency/what_comes_for_her.py).
    comes: Any = None
    #: What she plays for and what each pull is worth in its terms (core/agency/what_she_plays_for.py), what each thing
    #: is to her (core/agency/who_is_who.py), and how many settled touches of each kind have been told to them.
    plays_for: Any = None
    who: Any = None
    touches_told: dict[int, int] = field(default_factory=dict)
    #: What each sound the page plays comes with (core/agency/what_a_sound_comes_with.py), and how many presses of hers
    #: it has been told of.
    sounds: Any = None
    presses_told: int = 0
    #: What comes for her allies (core/agency/keeping_allies.py), and which key spends which counter and on what
    #: (core/agency/spending_a_supply.py).
    allies: Any = None
    supply: Any = None
    #: What she pushes and where each goes (core/agency/pushing_a_thing.py).
    pushing: Any = None
    #: The latest picture, and when the walls drawn in it were last looked for (core/agency/a_way_through.py).
    picture: Any = None
    walls_seen_at: float = -math.inf
    #: Each press of a key that strikes without sending anything out, and how far it has paid (how_far_her_blow_reaches.py).
    blows: HerBlows = field(default_factory=HerBlows)
    #: A key held to build something up and let go to use it, and how long a hold has paid (holding_to_charge.py).
    charging: Charging = field(default_factory=Charging)
    letting_go: dict[str, float] = field(default_factory=dict)


# -- what she was told ---------------------------------------------------------

def _record_control_attribution(run: _Run, moves: Any, hers: WhichIsHers, at: float) -> None:
    """Retain bounded ownership changes from visible geometry and measured controls."""
    from core.perception.recording_the_frames_and_inputs import record_control_observation
    record_control_observation(run, moves, hers, at)
    mine = hers.thing(moves)
    keys = tuple(sorted(hers.keys_that_move_her(run.keys))) if mine is not None else ()
    source = "pointer" if mine is not None and hers.follows_pointer else "keys" if keys else "unknown"
    state = (mine.number if mine is not None else None, source, keys)
    if state == run.last_attribution:
        return
    run.last_attribution = state
    run.attribution_changes += 1
    shape = getattr(moves, "shape", None)
    position = ([round(mine.x / shape[1], 4), round(mine.y / shape[0], 4)]
                if mine is not None and shape and shape[0] > 0 and shape[1] > 0 else None)
    run.control_attribution.append({"seconds": round(at - run.began, 3), "thing": state[0],
                                    "source": source, "keys": list(keys), "position": position})

#: Words that say keys are pressed in turn and fast: "press left and right rapidly", "tap space repeatedly".
_FAST = re.compile(r"\b(?:rapidly|repeatedly|quickly|as fast as|mash\w*|alternat\w*|in turn|over and over|again and again)\b",
                   re.I)

#: How long a burst of presses lasts, how often one may come, and the time between presses in it, in seconds.
BURST_S = 1.5
BURST_EVERY_S = 3.0
BURST_TAP_S = 0.06


def keys_pressed_fast(text: str) -> list[str]:
    """The keys a sentence says to press in turn and fast, one or two of them (LIVE 2026-10-09 a game wanted left and
    right pressed in turn, fast, when she was caught, and she pressed each a second at a time)."""
    keys: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        if not _FAST.search(sentence):
            continue
        named, _pointer = controls_named_in(sentence, keys_without_words=(), during_play=True)
        if 1 <= len(named) <= 2:
            keys += [key for key in named if key not in keys]
    return keys[:2]


async def _burst(hands: Any, run: _Run, at: float, *, say: Any = None) -> None:
    """The keys pressed in turn, as fast as a person can, for a moment."""
    if run.held:
        await hands.up(run.held)
        run.held = ""
    _say(run, say, f"Pressing {' and '.join(run.burst_keys)} in turn, fast, as I read I should.", at, once="burst")
    began, n = time.monotonic(), 0
    while time.monotonic() - began < BURST_S:
        await hands.tap(run.burst_keys[n % len(run.burst_keys)])
        n += 1
        await asyncio.sleep(BURST_TAP_S)
    run.burst_at = at


# -- the loop -----------------------------------------------------------------

#: How long a line once said is not said again, word for word, in the same game: a watcher heard it.
REPEAT_AFTER_S = 90.0


def _say(run: _Run, say: Callable[[str], Any] | None, line: str, at: float, *, once: str = "") -> None:
    if not line.strip(" ."):
        return
    if once and once in run.said:
        return
    # The same words a moment ago, about another thing of the same look or in the stretch before: LIVE 2026-10-07
    # "The black things are worth shooting." three times in two seconds, and "Space fires." over and over.
    if at - run.lately.get(line, -REPEAT_AFTER_S) < REPEAT_AFTER_S:
        return
    if at - run.said_at < SAY_EVERY_S and not once:
        return
    if once:
        run.said.add(once)
    run.said_at = at
    run.lately[line] = at
    run.lines.append(line)
    logger.info("saying while playing: %s", line)
    if say is not None:
        try:
            say(line)
        except (RuntimeError, ValueError, TypeError, OSError) as why:
            logger.debug("a line said while playing did not reach her chat: %s", why)


async def _hold(hands: Any, run: _Run, hers: WhichIsHers, key: str, at: float, *, trying: bool = False) -> None:
    await hold_control(hands, run, hers, key, at, trying=trying, now=time.monotonic)


async def _try_the_keys(hands: Any, run: _Run, hers: WhichIsHers, at: float, moves: Any = None) -> None:
    """Hold each key in turn, with a rest between, to see what answers. The key that points against the way the
    biggest moving thing is going is held first, and again after one rest (core/agency/the_key_against_its_going.py):
    LIVE 2026-10-11 a lander down in five seconds was never found to be hers in ten goes of four seconds a round."""
    from core.agency.trying_controls_with_a_view_of_their_response import try_controls

    await try_controls(hands, run, hers, at, moves, hold=_hold, seconds=TRY_A_KEY_S)


#: The most trials a game is given to learn what turns her (core/agency/which_way_she_faces.py).
FACING_TRIALS = 10


async def _a_facing_trial(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing, at: float) -> bool:
    from core.agency.a_source_motion_model_checked_in_view import source_motion_checked
    from core.agency.giving_a_trial_time import try_a_facing_control

    if source_motion_checked(run, hers, choosing, at):
        return False
    return await try_a_facing_control(hands, run, hers, choosing, at, hold=_hold,
                                     seconds=TRY_A_KEY_S, limit=FACING_TRIALS)


async def _click_things(hands: Any, run: _Run, moves: WhatMoves, meeting: WhatMeetingDoes, at: float,
                        *, only_named: bool = False) -> None:
    """With no thing of her own, a click is how she meets things.

    She clicks the thing that is about to leave first, among the kinds she
    has not learned to leave alone, where it will be when the click lands.
    """
    if at - run.last_click < CLICK_EVERY_S:
        return
    tall, wide = moves.shape
    candidates = [
        t for t in moves.things.values()
        if (t.moved or meeting.stance(t.kind) == CLICK)
        and t.number not in meeting.writing and meeting.stance(t.kind) in (MEET, CLICK)
        and (not only_named or meeting.stance(t.kind) == CLICK)
        and not meeting.clicked_lately(t.number, at)
    ]
    if not candidates:
        return

    def leaving(thing: Any) -> tuple[int, float]:
        exits = [
            (edge - position) / speed
            for position, speed, edge in ((thing.x, thing.vx, wide if thing.vx > 0 else 0.0), (thing.y, thing.vy, tall if thing.vy > 0 else 0.0))
            if abs(speed) > 1.0
        ]
        return (0 if meeting.known(thing.kind) else 1, min(exits, default=math.inf))

    thing = min(candidates, key=leaving)
    # Led by what has taken things before, where the thing moves (core/agency/leading_a_click.py).
    lead = run.leading.lead() if run.leading is not None and math.hypot(thing.vx, thing.vy) > 8.0 else CLICK_LANDS_S
    x, y = thing.where_at(lead)
    sx, sy = moves.share(x, y)
    if 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
        await hands.click(sx, sy)
        meeting.clicked(thing, at)
        run.last_click, run.clicks = at, getattr(run, "clicks", 0) + 1
        if run.leading is not None and lead != CLICK_LANDS_S:
            run.leading.aimed(thing.number, at, lead)


#: How often the picture itself is clicked, where the game is played with the pointer and nothing in it is to be clicked.
CLICK_THE_PICTURE_S = 1.5


async def _click_the_picture(hands: Any, run: _Run, moves: WhatMoves, at: float) -> None:
    """Where the game's words name the pointer and nothing she could click has come of it, a click on the picture itself.

    "Click your mouse button to start the hamster in motion... click again to
    launch": a click aimed at nothing is how such a game is played (LIVE
    2026-10-07, four minutes in one with nothing clicked). Paced, so what one
    did can be seen before the next. Motion holds back an unspecified click;
    a trigger named by the visible instructions still works during motion.
    Where in it is learned as she goes: the places a click
    paid are clicked more (core/agency/where_clicks_pay.py), and where such
    clicks have cost more than they paid, she stops making them.
    """
    from core.agency.where_clicks_pay import WhereClicksPay

    if not run.pointer_first or at - run.last_click < CLICK_THE_PICTURE_S:
        return
    if not run.pointer_trigger and any(thing.moved for thing in moves.things.values()):
        return
    run.clicks_pay = run.clicks_pay or WhereClicksPay()
    if run.clicks_pay.costing():
        return
    x, y = run.clicks_pay.where(run.gains, run.losses)
    await hands.click(x, y)
    run.last_click, run.clicks = at, getattr(run, "clicks", 0) + 1


async def _try_the_pointer(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves, at: float) -> None:
    """Sweep the pointer through a few places, to see whether anything follows it.

    In small steps, a picture at a time: a thing that jumps with a jumping
    pointer cannot be followed from one picture to the next.
    """
    goal = _POINTER_TRIAL[(run.pointed // 2) % len(_POINTER_TRIAL)]
    x, y = run.pointer
    dx, dy = goal[0] - x, goal[1] - y
    far = math.hypot(dx, dy)
    if far <= POINTER_STEP:
        x, y = goal
        run.pointed += 1
    else:
        x, y = x + dx * POINTER_STEP / far, y + dy * POINTER_STEP / far
    run.pointer = (x, y)
    await hands.point(x, y)
    tall, wide = moves.shape
    hers.pointed(x * wide, y * tall, time.monotonic())


async def _point_at(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves, choosing: _Choosing, at: float) -> None:
    """Where a thing follows the pointer, the pointer goes where she wants it to be."""
    (gx, gy), _why, _aim = choosing.target()
    mine = choosing.mine
    tall, wide = moves.shape
    x = gx if gx is not None else mine.x
    y = gy if gy is not None else mine.y
    sx, sy = min(1.0, max(0.0, x / max(1, wide))), min(1.0, max(0.0, y / max(1, tall)))
    # A move of the pointer costs a trip to the browser; a hundredth of the
    # picture is closer than her thing can be put anyway.
    if math.dist((sx, sy), run.pointer) > 0.02:
        run.pointer = (sx, sy)
        run.pointer_moves += 1
        await hands.point(sx, sy)
        hers.pointed(x, y, time.monotonic())


def _what_the_place_does(run: _Run, choosing: _Choosing, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                         at: float) -> None:
    """What the place does to her whatever she does, taken in and given to the choosing: how she goes with no key held,
    which is what doing nothing is; how pressing time is; how much faster things come than at first, which widens the
    berth she gives what costs her (core/agency/what_the_place_does_to_her.py)."""
    mine = choosing.mine
    run.pressures.saw(at=at, mine=mine, held=bool(run.held or run.letting_go),
                      others=[t for t in moves.things.values() if mine is None or t.number != mine.number],
                      counters=meeting.readouts.current, vitals=meeting.vitals)
    choosing.gone_for, choosing.urgency = run.gone_for, run.pressures.urgency()
    choosing.faces, choosing.comes, choosing.way_through = run.faces, run.comes, run.way_through
    observe_facing(run.faces, mine, hers, at)
    from core.agency.steering_from_a_model import observed_steering_model
    observed_steering_model(run.faces, mine, hers, moves, at, run=run)
    if mine is not None:
        run.comes.saw(moves.things.values(), mine)
    # Carried on by her own going, doing nothing is already what her going does; under the pointer, it is staying put.
    if mine is not None and not getattr(hers, "carried", False) and not choosing.pointing:
        choosing.rest = run.pressures.pull_at(float(mine.x), float(mine.y))
    choosing.caution *= math.sqrt(run.pressures.harder(at))


#: How often the walls drawn in the picture are looked for again, once their colour is known.
WALLS_LOOKED_FOR_S = 2.0


def _what_meeting_did(meeting: WhatMeetingDoes, kind: int, kept: Any) -> str | None:
    """What meeting a kind has settled into: costly, worth getting, nothing, or not yet clear. By the meeting record's
    own rule, which weighs meeting it against letting it by; and by what touching it did on its own, since a thing that
    costs her whether she meets it or lets it by (an invader that lands) weighs nothing either way and is no less a
    threat for that."""
    touch = kept.touch_sum / kept.touches_settled if kept.touches_settled else None
    passing = kept.pass_sum / kept.passes_settled if kept.passes_settled else 0.0
    if (meeting.known(kind) and kept.meet <= -0.6) or (touch is not None and kept.touches_settled >= 2 and touch <= -0.5):
        return "loss"
    if (meeting.known(kind) and kept.meet >= 0.5) or (touch is not None and kept.touches_settled >= 2 and touch >= 0.5):
        return "gain"
    if touch is not None and kept.touches_settled >= 2 and abs(touch) < 0.2 and abs(passing) < 0.2 and kept.shoot < 0.5:
        return "nothing"
    return None


def _what_she_plays_for(run: _Run, choosing: _Choosing, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                        at: float, say: Callable[[str], Any] | None) -> None:
    """What she plays for and who is who, taken in and given to the choosing: her care from the stakes, the threats and
    what taking each out is worth, how far out of her way to go to find out what a thing is, and what she says of them
    (core/agency/what_she_plays_for.py, core/agency/who_is_who.py)."""
    goes, who, mine = run.plays_for, run.who, choosing.mine
    things = list(moves.things.values())
    # What she sends out is hers as much as what goes with her.
    sent = {shot.kind for shot in hers.makes.values()}
    size = max(float(mine.w), float(mine.h)) if mine is not None else 16.0
    # What is taken from her is a prize shown to be one, not a kind that once came with a count going up.
    prizes = {kind for kind, kept in who.kinds.items() if kept.gains and who.prize_shown(kind)}
    who.saw(at, things, mine, prizes=prizes, comes_for_her=lambda thing: run.comes.comes_for_her(thing, size),
            with_her=lambda thing: goes_with(thing, mine) or thing.kind in sent, shape=moves.shape)
    for kind, kept in meeting.evidence.items():
        if kept.touches_settled <= run.touches_told.get(kind, 0):
            continue
        run.touches_told[kind] = kept.touches_settled
        what = _what_meeting_did(meeting, kind, kept)
        if what is not None:
            who.met(kind, what)
            goes.met(kind, what)
    for kind, kept in meeting.evidence.items():
        if kept.shoot >= 0.5 and kind not in goes.pays_to_shoot:
            goes.shot_paid(kind)
    has_bar = run.bars is not None and any(bar.meaning == "down is bad" for bar in run.bars.bars())
    blows, gains = len(goes.blows), len(goes.gains)
    goes.saw(at, meeting.readouts.current, out_of=meeting.readouts.out_of, vitals=run.vitals if has_bar else None)
    if len(goes.blows) > blows and mine is not None:
        kind = who.blow(at, mine, things)
        if kind is not None and who.kinds[kind].blows:
            goes.blow_from(kind)
    # What she hears is learned against what happens (core/agency/what_a_sound_comes_with.py).
    sounds = run.sounds
    if len(goes.blows) > blows:
        sounds.blow(at, mine)
    if len(goes.gains) > gains:
        sounds.gained(at)
    presses = sum(run.input_key_downs.values()) + run.taps
    if presses > run.presses_told:
        sounds.pressed(at)
        run.presses_told = presses
    sounds.judge_clear(at, mine)
    choosing.warned_spots = tuple(sounds.spots_to_keep_clear_of(at))
    for line in sounds.to_say(at):
        _say(run, say, line, at, once=line)
    threats = [thing for thing, _found in who.threats(things)]
    goes.saw_threats(at, [(thing.number, thing.kind) for thing in threats])
    run.supply.counted(at, meeting.readouts.current)
    run.pushing.saw(mine, things, choosing.ways.get(run.held) if run.held else None)
    # The walls drawn in a colour she has been stopped by are walls wherever they are drawn (core/agency/a_way_through.py).
    if run.way_through.wall_colours and run.way_through.cell and at - run.walls_seen_at >= WALLS_LOOKED_FOR_S:
        tall, wide = moves.shape
        run.way_through.walls_seen(run.picture, moves.scale, wide, tall)
        run.walls_seen_at = at
    if len(goes.gains) > gains:
        run.pushing.counted(things)
    choosing.pushing = run.pushing
    allies = tuple(thing for thing in things if (mine is None or thing.number != mine.number)
                   and who.of(thing).name == FRIENDLY)
    run.allies.saw(threats, allies)
    choosing.allies, choosing.allies_here = run.allies, allies
    # What she has found is in her way stands in the way of every plan to go round it (core/agency/a_way_through.py):
    # each thing of a kind that stopped her is a wall over all it covers, before she has bumped into the rest of it.
    for thing in things:
        if run.way_through.cell and who.of(thing).name == IN_HER_WAY:
            run.way_through.a_wall_at(thing.box())
            if run.way_through.drawn_like(run.picture, moves.scale, thing):
                run.walls_seen_at = -math.inf
    choosing.seconds_left = run.pressures.seconds_left()
    choosing.caution *= goes.care(at, choosing.seconds_left)
    choosing.plays_for, choosing.who = goes, who
    spent = goes.what_to_say(at)
    if spent:
        _say(run, say, spent, at, once="spent")
    # What each kind is to her, one at a time: the rest are said in their turn (core/agency/who_is_who.py).
    if at - run.stance_said_at >= STANCE_EVERY_S:
        for line in who.to_say([t for t in things if mine is None or t.number != mine.number],
                               lambda thing: _named(moves, thing.kind, thing)):
            _say(run, say, line, at, once=line)
            run.stance_said_at = at


def _over(run: _Run, moves: WhatMoves, happened: list[dict[str, Any]], at: float) -> str:
    from core.agency.an_observed_scene_transition import keep_scene_phase

    keep_scene_phase(run, happened, at)
    acts = sum(run.input_key_downs.values()) + run.pointed + run.pointer_moves
    # Motion seen in this picture, not a thing carried on along its last speed
    # because it was not seen.
    activity = getattr(moves, 'activity', None)
    if (activity is not None and activity.at == at and activity.changed is True
            or any(thing.seen == at for thing in moves.moving(faster_than=8.0))):
        if run.first_moving < at:
            run.longest_still = max(run.longest_still, at - run.last_moving)
        run.last_moving = at
        run.first_moving = min(run.first_moving, at)
    still = at - run.last_moving
    if activity is not None:
        quiet = activity.quiet_for(at)
        still = min(still, quiet) if quiet is not None else 0.
    # A world that waits for her is still until she moves, and one played in time goes still between what comes: its
    # stillness ends a stretch only once her ways have had time, and never sooner than half again the longest quiet it
    # has kept while played. Offline 2026-10-11 a food fight whose opponents rose from behind the tables now and then
    # was handed back three seconds into its first quiet, and the round ran out with nothing thrown.
    still_for = WAITING_STILL_FOR_S if run.waits_for_her or run.in_time else STILL_FOR_S
    still_for = max(still_for, QUIET_KEPT * run.longest_still)
    # A new screen that stands still is handed back once she has tried it: a level often waits for the player's first
    # move (LIVE 2026-10-10 a boat game's level was handed back 2 seconds in, and played itself to GAME OVER).
    if (at - run.new_screen_at < still_for and still >= max(NEW_SCREEN_STILL_S, still_for - STILL_FOR_S)
            and at - run.began > 2.0 and (acts > run.acts_at_new_screen or at - run.new_screen_at >= STILL_FOR_S)):
        return "the screen changed and nothing on it moves"
    if still >= still_for and at - run.began > still_for:
        return "nothing on the screen has moved for a while"
    return ""


#: How long she plays a moving picture in which nothing answers to her before
#: handing it back: a title screen that animates is not a game.
NOTHING_ANSWERS_S = 10.0


#: How long things she touches may go on gaining and losing nothing before the picture is handed back.
TOUCHING_TO_NO_END_S = 20.0


#: How many turns at each key, a press and a rest, her trials have in all once her thing is found.
KNOWN_BY = 4


def _still_trying_keys(run: _Run, hers: WhichIsHers, choosing: Any, trying_the_pointer: bool, *,
                       at: float | None = None) -> bool:
    """Whether her keys are still being tried: two turns at each until her thing is found; then until each is known,
    and no more than so many turns at each, known or not: a key held into a floor tells nothing however often it is
    pressed. Offline 2026-10-12 a brawler on the ground was tried with "down" every half second for the rest of the
    round, and never once punched the robots walking up to her."""
    from core.agency.a_source_motion_model_checked_in_view import source_motion_checked

    return bool(not source_motion_checked(run, hers, choosing, at)
                and not trying_the_pointer and run.keys and not hers.follows_pointer
                and (choosing.mine is None or hers.kind is not None)
                and body_probe_is_active(run, hers, pointer_steps=2 * len(_POINTER_TRIAL), at=at))


#: How many presses of the keys that moved her thing may go by with it not going, and for how long at least, before
#: it is taken to answer to nothing now.
UNANSWERED_PRESSES = 6
UNANSWERED_FOR_S = 8.0


def _answers_now(run: _Run, hers: WhichIsHers, moving: dict[str, Any], at: float) -> bool:
    """Whether the thing her keys moved still goes under them. What answered once and stands still under press after
    press answers to nothing now: offline 2026-10-11 a lander's crash screen kept the lander drawn on it, the keys
    that had flown it were known, and she held them at the picture for three minutes."""
    moved = last_moved(hers)
    downs = sum(run.input_key_downs[key] for key in moving)
    # Her thing found afresh is counted from then: what the one before it last did says nothing of it.
    if moved != run.answered[0]:
        run.answered = (moved, downs)
    since, downs_then = run.answered
    return at - max(since, run.began) <= UNANSWERED_FOR_S or downs - downs_then < UNANSWERED_PRESSES


def _nothing_answers(run: _Run, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> str:
    # Allow two complete experiments and their intervening wait before
    # declaring that no control answers. The run's own deadline still bounds it.
    experiment_time = (8 * len(run.keys) * max(TRY_A_KEY_S, run.longest_control_trial_s)
                       + RECHECK_CONTROLS_S) if run.keys else 0.0
    # A thing her eyes took for hers that nothing of hers has ever moved answers to nothing: LIVE 2026-10-11 the boy on
    # a game's animated title was taken for her, no key moved him, and she played the title for a minute before START.
    moving = hers.keys_that_move_her(run.keys) if hers.kind is not None else {}
    answers = hers.kind is not None and (hers.follows_pointer or (bool(moving) and _answers_now(run, hers, moving, at)))
    run.answered_at = (at if hers.follows_pointer else max(run.answered_at, run.answered[0])) if answers else run.answered_at
    # Where she understands no body of hers is in the picture, nothing is there to answer her: what she does is aimed
    # from outside it. Offline 2026-10-12 a food fight she was holding her own in was handed back mid-round, every round,
    # at the moment a title screen would have been.
    if at - run.began < NOTHING_ANSWERS_S + experiment_time or answers or _how_it_stands(run, at)[1]:
        return ""
    if meeting.verdicts:
        # What is counted keeps it going while it is not all going against her: LIVE 2026-10-09 a game's bar went down
        # twice while nothing answered to her, and she played on without anything of her own for three minutes.
        if run.losses <= run.gains or at - run.began < NOTHING_ANSWERS_S + experiment_time + TOUCHING_TO_NO_END_S:
            return ""
        return "nothing here answers to me, and it is going against me"
    # Things she touched keep it going for a while, and only a while, where
    # touching them has gained and lost nothing: LIVE 2026-10-06 a game's
    # title screen, its picture moving, its START! drawn as words: she clicked
    # the moving figure for the whole five minutes and never left it for START.
    if any(kept.touched for kept in meeting.evidence.values()):
        if at - run.began < NOTHING_ANSWERS_S + experiment_time + TOUCHING_TO_NO_END_S:
            return ""
        return "nothing I touch here gains or loses anything"
    return "nothing here answers to me while it moves"


def _where_she_has_come(run: _Run, moves: WhatMoves, hers: WhichIsHers, say: Any, at: float) -> None:
    """Where the view going by has taken her in a world bigger than it (core/perception/where_in_a_world.py): a view
    she had not been in is getting somewhere, and how far she has come is said each new view along."""
    if not moves.view_over or moves.view_moved_at != at:
        return
    tall, wide = moves.shape
    if run.world.saw(moves.view_moved, moves.view_over, tall, wide, hers.thing(moves)):
        line = run.world.said()
        if line:
            _say(run, say, line, at, once=f"come {round(run.world.views_along())}")


def _report(run: _Run, getting_somewhere: Callable[[str], Any] | None, at: float) -> None:
    """Tell whoever holds a deadline over this that it is still getting somewhere."""
    if getting_somewhere is None or at - run.reported_at < 5.0:
        return
    run.reported_at = at
    try:
        getting_somewhere(f"playing as it happens: {run.pictures} pictures, {run.gains} gains, {run.losses} losses")
    except (RuntimeError, ValueError, TypeError, OSError) as why:
        logger.debug("progress while playing was not taken: %s", why)


def _what_the_rules_said_of(rules: WhatTheRulesSaid | None, moves: WhatMoves, meeting: WhatMeetingDoes,
                            run: _Run, say: Any, at: float, *, pointer_steers: bool = False) -> None:
    """Tie the rules' words to the kinds on screen by colour, and a legend's drawings by look, as each kind first appears."""
    if rules is None and not run.legend:
        return
    for kind in moves.kinds:
        if run.told_checked.get(kind.number) == pointer_steers:
            continue
        run.told_checked[kind.number] = pointer_steers
        stance = rules.stance_for_colour(colour_name(kind.colour), pointer_steers=pointer_steers) if rules is not None else None
        if stance is None and _as_a_legend_drew(run, kind, meeting, say, at):
            continue
        if stance is not None:
            meeting.told[kind.number] = stance
            line = {
                MEET: "the {c} ones are to be got", AVOID: "the {c} ones are to be kept clear of",
                SHOOT: "the {c} ones are to be shot", CLICK: "the {c} ones are to be clicked",
            }.get(stance, "")
            if line:
                _say(run, say, "The rules say " + line.format(c=colour_name(kind.colour)) + ".", at, once=f"told {kind.number} {stance}")


def _as_a_legend_drew(run: _Run, kind: Any, meeting: WhatMeetingDoes, say: Any, at: float) -> bool:
    """A kind that looks like a thing a legend drew is taken for what the legend said of it, until play says otherwise."""
    from core.perception.what_a_legend_shows import most_like

    drawn = most_like(kind.look, run.legend, colour=kind.colour)
    stance = {"meet": MEET, "avoid": AVOID, "shoot": SHOOT}.get(drawn.stance if drawn is not None else "")
    if drawn is None or stance is None:
        return False
    meeting.told[kind.number] = stance
    _say(run, say, f"The {colour_name(kind.colour)} thing is what the rules showed: \u201c{drawn.words}\u201d", at,
         once=f"legend {drawn.words}")
    return True


def _counters_without_reading(run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    her_x = moves.share(mine.x, mine.y)[0] if mine is not None else None
    never = {hers.number} if hers.number is not None else set()
    for when, x, y in run.stayed.see(getattr(moves, "_last", None), moves.things, at, never=never):
        meeting.changed_in_place(when, x, y, her_x)


#: How long between two accounts of what kind of game this is.
SITUATION_EVERY_S = 30.0
ACCOUNTS_A_GAME = 2

#: Where a thing stands, as where_on_screen names it, inside an account of the game.
_WHERE_IT_STANDS = re.compile(r" at the (?:(?:top|bottom) )?(?:left|right)\b| at the (?:top|bottom|middle)\b")


def _what_kind_of_game(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                       physics: HowThingsMoveHere, at: float) -> None:
    """Say what kind of game this is, once there is something to say, and again when it changes."""
    from core.agency.what_kind_of_game_this_is import in_a_sentence

    if "me" not in run.said or at - run.situation_at < SITUATION_EVERY_S:
        return
    # An account of the game a few times a game, not whenever a kind does something new: LIVE 2026-10-11 a flying
    # shooter had one every thirty seconds, each of things that had bounced or turned back off her.
    if sum(1 for once in run.said if once.startswith("situation ")) >= ACCOUNTS_A_GAME:
        return
    sentence = in_a_sentence(moves, hers, meeting, physics, run.keys, me=run.hers_description)
    if not sentence or ";" not in sentence:
        return
    # Said again only for something learned: her paddle having moved to the
    # top, or the other side being still a moment, is not news. LIVE
    # 2026-10-06 the same account came every thirty seconds of a game.
    parts = {_WHERE_IT_STANDS.sub("", part).strip(" .") for part in sentence.split(";")}
    if parts <= run.situation_known:
        return
    run.situation_known |= parts
    run.situation, run.situation_at = sentence, at
    _say(run, say, sentence[0].upper() + sentence[1:], at, once=f"situation {len(run.lines)}")


def _what_she_says(run: _Run, say: Any, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes, at: float) -> None:
    mine = hers.thing(moves)
    keys = sorted(hers.keys_that_move_her(run.keys))
    if mine is not None and hers.kind is not None and (keys or hers.follows_pointer):
        run.hers_descriptions[describe(moves, hers.kind, mine)] += 1
        # A brief overlap changes a connected region's outline. Describe the
        # control by its usual observed shape rather than that one picture.
        run.hers_description = run.hers_descriptions.most_common(1)[0][0]
    settled = all(hers.tried(k) >= 4 for k in run.keys) or at - run.began > 8.0
    if mine is not None and hers.kind is not None and hers.follows_pointer:
        if not still_looking_at(run, hers.kind, at, hers.number):
            _say(run, say, i_go_where_the_mouse_goes(moves, hers.kind, mine), at, once="me")
    elif mine is not None and hers.kind is not None and keys and settled:
        how = " and ".join(keys)
        verb = "moves" if len(keys) == 1 else "move"
        _say(run, say, f"That's me: the {_named(moves, hers.kind, mine, 'me')} at the {where_on_screen(moves, mine.x, mine.y)}. {how.capitalize()} {verb} it.",
             at, once="me")
    for key in hers.makes:
        _say(run, say, f"{key.capitalize()} fires.", at, once=f"fires {key}")
    for kind in list(meeting.evidence):
        # What things are to her is said one at a time: LIVE 2026-10-11 one loss had seven kinds said to be threats in
        # a fifth of a second. The others are said in their turn, if what she makes of them still holds by then.
        if at - run.stance_said_at < STANCE_EVERY_S:
            break
        if not meeting.known(kind) or not any(t.kind == kind for t in moves.things.values()):
            continue
        stance = meeting.stance(kind)
        if f"{kind} {stance}" not in run.said:
            kept = meeting.evidence.get(kind)
            line = a_stance_said(run, moves, kind, stance, at, met=kept is not None and (kept.touched > 0 or kept.shot > 0))
            if line:
                _say(run, say, line, at, once=f"{kind} {stance}")
                if f"{kind} {stance}" in run.said:
                    run.stance_said_at = at
    # How it stands, said when it changes: the score, what is left to win, what is left to lose.
    standing = run.contest.says()
    if standing and run.contest_first is None:
        run.contest_first = standing
    if standing and standing != run.contest_said and standing != run.contest_first:
        if at - run.said_at < SAY_EVERY_S:
            return
        run.contest_said = standing
        _say(run, say, standing[:1].upper() + standing[1:], at)
        return
    # The counters as read are said only where no standing could be made of
    # them: LIVE 2026-10-06 an end screen's "You 5 Computer 4" was read as one
    # counter, and "You computer 4." followed "4-4, level".
    # Said by the names a person reading the screen would say: words of the language, a misread letter mended
    # ("Leuel" is level); a name that is not one ("Ini", "iin", "x") is not said (core/language/words_of_the_language.py).
    from core.language.words_of_the_language import words_of

    counters = {said: value for name, value in meeting.readouts.values.items()
                if not name.startswith("number") and (said := words_of(name)) is not None}
    # Each counter as it first read is the baseline: read a few at a time, a set that grew was not a change.
    first = run.counters_first if run.counters_first is not None else {}
    for name, value in counters.items():
        first.setdefault(name, value)
    run.counters_first = first
    moved = any(first.get(name) != value for name, value in counters.items())
    if counters and moved and not run.contest_said and counters != run.counted and at - run.said_at > 25.0:
        run.counted = dict(counters)
        _say(run, say, ", ".join(f"{name} {value}" for name, value in list(counters.items())[:3]).capitalize() + ".", at)
    _say(run, say, keeping_company(run, at), at)  # a watcher is kept company (core/agency/keeping_a_watcher_company.py)


async def play_as_it_happens(
    look: Callable[[], Awaitable[Any]],
    hands: Any,
    *,
    keys: Sequence[str],
    seconds: float,
    say: Callable[[str], Any] | None = None,
    read_words: Callable[[Any], list[dict[str, Any]]] | None = None,
    keep: dict[str, Any] | None = None,
    pointer_first: bool = False,
    getting_somewhere: Callable[[str], Any] | None = None,
    told: str = "",
    waits_for_her: bool = False,
) -> dict[str, Any]:
    from core.perception.recording_the_frames_and_inputs import recorded_play_ports
    from core.runtime.what_stops_it import current, interruptible

    look, hands = recorded_play_ports(look, hands)
    with interruptible(current(whose="play_as_it_happens")), short_turns_while_playing():
        return await _play_as_it_happens(
            look, hands, keys=keys, seconds=seconds, say=say, read_words=read_words,
            keep=keep, pointer_first=pointer_first, getting_somewhere=getting_somewhere,
            told=told, waits_for_her=waits_for_her,
        )


async def _play_as_it_happens(
    look: Callable[[], Awaitable[Any]],
    hands: Any,
    *,
    keys: Sequence[str],
    seconds: float,
    say: Callable[[str], Any] | None = None,
    read_words: Callable[[Any], list[dict[str, Any]]] | None = None,
    keep: dict[str, Any] | None = None,
    pointer_first: bool = False,
    getting_somewhere: Callable[[str], Any] | None = None,
    told: str = "",
    waits_for_her: bool = False,
) -> dict[str, Any]:
    """Play what ``look`` shows through ``hands`` until it stops moving or ``seconds`` pass.

    ``look`` returns an RGB picture and when it was taken, or None. ``hands``
    has ``down(key)``, ``up(key)``, ``tap(key)``, ``click(x, y)`` and
    ``point(x, y)``, positions as shares of the picture. ``keep`` carries what
    she learned from one stretch of play into the next one in the same game.
    ``waits_for_her`` says the world stood still until she moved in it
    (``it_goes_while_held``): with nothing to go to, she goes where she has not been.
    """
    from core.runtime.what_stops_it import current

    context = current(whose="play_as_it_happens.frames")
    began = time.monotonic()
    keep = keep if keep is not None else {}
    from core.agency.when_motion_breaks_a_rule import MotionChecks

    motion_checks = MotionChecks(frozenset(keep.get("required_edges") or []), str(keep.get("edge_provenance") or ""))
    moves = await open_play_observation(kinds=keep.get("kinds"), factory=eyes_for_play)
    physics: HowThingsMoveHere = keep.get("physics") or HowThingsMoveHere()
    rules = WhatTheRulesSaid.read(told) if told else None
    from core.agency.situations_she_knows import caution_of, situations_in

    situations = situations_in(told) if told else ()
    from core.agency.situations_she_knows import played_in_time
    hers: WhichIsHers = keep.get("hers") or WhichIsHers()
    meeting: WhatMeetingDoes = keep.get("meeting") or WhatMeetingDoes()
    from core.agency.how_a_game_works import understood
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    how = understood(THE_GUIDE.get(), also=told)
    pointer_first = how.try_pointer_first(pointer_first)
    keys = how.keys_for_body_trials(keys)
    for kept in (physics, hers, meeting):
        kept.numbered_afresh()
    run = _Run(keys=list(keys), began=began, last_moving=began, pointer_first=pointer_first, legend=tuple(keep.get("legend") or ()),
               contest=keep.get("contest") or ContestStands(), waits_for_her=waits_for_her,
               in_time=played_in_time(situations), longest_still=float(keep.get("longest_still") or 0.0))
    run.pointer_trigger = rules is not None and rules.a_click_is_a_shot
    run.how = how
    if _no_body_to_find():
        # Where she understands there is no body of hers in the picture, the pointer is not swept to see what follows
        # it: offline 2026-10-11 a food fight's throws went out between sweeps of the pointer across the room.
        run.pointed = 2 * len(_POINTER_TRIAL)
    run.burst_keys = keys_pressed_fast(told)
    run.building_up = builds_up(told)
    from core.agency.a_requested_relation_in_play import take_motion_directions

    take_motion_directions(meeting, told)
    run.contest.heard(told)
    run.situation_known = set(keep.get("situation_known") or ())
    run.clicks_pay = keep.get("where_clicks_pay")
    if run.clicks_pay is not None:
        run.clicks_pay.begin_stretch(run.gains, run.losses)
    run.lately = dict(keep.get("said_lately") or {})
    from core.agency.leading_a_click import LeadingAClick

    run.leading = keep.get("leading") or LeadingAClick()
    # What is said once ("That's me", what a kind of thing is worth) is said once a game, not once a stretch.
    run.said = set(keep.get("said_once") or ())
    run.avoided_before = {kind for kind in meeting.evidence if meeting.stance(kind) == AVOID}
    from core.agency.pressing_what_is_shown import KeysShown
    from core.agency.signs_that_come_along import SignsComing

    run.shown, run.coming = KeysShown(keep.get("keys_never_absent")), SignsComing(keep.get("signs_going"))
    from core.agency.what_a_press_does import WhatAPressDoes

    run.presses = keep.get("presses") or WhatAPressDoes()
    from core.agency.searching_before_a_body_is_known import (
        pixel_search_record,
        release_search_after_play,
        search_after_unresolved_body,
        search_inputs,
    )
    hands = search_inputs(hands, run, hers)
    from core.agency.when_a_press_pays import WhenAPressPays

    run.timing = keep.get("timing") or WhenAPressPays()
    from core.agency.noticing_in_play import NoticingInPlay

    run.noticing = keep.get("noticing") or NoticingInPlay()
    run.seeing = keep.get("seeing") or SeeingInPlay()
    run.using = keep.get("using") or UsingOnThings()
    from core.agency.how_far_a_thing_reaches import HowFarThingsReach

    run.reach = keep.get("reach") or HowFarThingsReach()
    from core.agency.warning_signs import WarningSigns, read_the_signs

    run.signs, run.bars = keep.get("signs") or WarningSigns(), keep.get("bars")  # a person knows her bar again
    from core.agency.settings_that_paid import after_the_stretch, for_this_stretch

    run.settings = for_this_stretch(keep)  # how often to throw, to duck: by what has paid (settings_that_paid.py)
    from core.agency.a_choice_and_its_reasons import ChoicesMade

    run.choices = keep.get("choices") or ChoicesMade()
    from core.agency.a_thing_gone_for import GoneFor
    from core.agency.what_the_place_does_to_her import Pressures

    run.pressures = keep.get("pressures") or Pressures()
    run.gone_for = keep.get("gone_for") or GoneFor()
    from core.agency.which_way_she_faces import WhichWaySheFaces

    run.faces = keep.get("faces") or WhichWaySheFaces()
    run.prepared_motion_binding = keep.get("observed_model_binding")
    from core.agency.what_comes_for_her import WhatComesForHer

    run.comes = keep.get("comes") or WhatComesForHer()
    from core.agency.a_way_through import AWayThrough

    run.way_through = keep.get("way_through") or AWayThrough()
    from core.agency.what_she_plays_for import WhatSheGoesFor
    from core.agency.who_is_who import WhoIsWho

    run.plays_for = keep.get("plays_for") or WhatSheGoesFor()
    run.plays_for.told(asked=str(keep.get("asked") or ""), words=told)
    run.who = keep.get("who") or WhoIsWho()
    run.touches_told = dict(keep.get("touches_told") or {})
    from core.agency.what_a_sound_comes_with import WhatSoundsComeWith

    run.sounds = keep.get("sounds_heard") or WhatSoundsComeWith()
    from core.agency.keeping_allies import KeepingAllies

    run.allies = keep.get("allies") or KeepingAllies()
    from core.agency.spending_a_supply import SpendingASupply

    run.supply = keep.get("supply") or SpendingASupply()
    from core.agency.pushing_a_thing import Pushing

    run.pushing, run.harm_lands = keep.get("pushing") or Pushing(), keep.get("harm_lands")
    run.presses_told = sum(run.input_key_downs.values()) + run.taps
    if keep.get("meeting_with"):
        run.meeting_with = {float(part): list(counts) for part, counts in keep["meeting_with"].items()}
    ended = ""
    try:
        while not ended:
            context.check()
            if time.monotonic() - began >= seconds:
                ended = "out of time"
                break
            seen = await look()
            context.check()
            if seen is None:
                ended = "the picture could not be taken"
                break
            picture, at = seen
            run.observation_sources["canvas-paint-v1" if getattr(picture, "drawing_scene", None) is not None else "pixels"] += 1
            if run.picture_at is not None and 0.0 < at - run.picture_at <= 0.5:
                run.intervals.append(at - run.picture_at)
            run.picture_at = at
            run.pictures += 1
            happened = await moves.see(picture, at)
            if happened is None:
                await asyncio.sleep(0)
                continue
            run.picture = picture
            for sound in getattr(picture, "sounds", ()):
                run.sounds.heard_sound(sound.at, sound.sig, hers.thing(moves))
            run.leading.saw(happened, at)
            for gone in (h.get("thing") for h in happened if h.get("what") == "gone"):
                if gone is not None and gone == run.gone_for.number:
                    run.gone_for.got(gone, at)
            run.gone_for.judge_changes(at)
            _where_she_has_come(run, moves, hers, say, at)
            hers.saw(moves, happened, at)
            # A key named for a way moves her; it is taken to fire only where the game's words say it does: LIVE
            # 2026-10-07 in a shooter whose words said "SPACE press to fire", left, up and right were each said to fire,
            # shots turning up beside her whatever she pressed.
            for key in [k for k in hers.makes if k in WAYS and not (rules is not None and k in rules.fire_keys)]:
                del hers.makes[key]
            _record_control_attribution(run, moves, hers, at)
            _measure_response(run, hers.thing(moves), at)
            physics.saw(moves, hers, happened, at)
            _what_the_rules_said_of(rules, moves, meeting, run, say, at, pointer_steers=hers.follows_pointer)
            # The ground is the point where the rules say so, and where the world waits for her: with nothing coming and
            # nothing to go to, a person looks round the place for what is there (a way on, a thing to take).
            goes_over = (rules is not None and rules.covers) or waits_for_her
            from core.agency.an_observed_scene_transition import keep_scene_transition

            keep_scene_transition(run, happened, at)
            ground = mark_the_ground_under_her(run, moves, hers.thing(moves), happened)
            goes_over = goes_over or ground is not None
            choosing = _Choosing(moves, hers, meeting, run.keys, physics,
                                 next_picture_s=statistics.median(run.intervals) if run.intervals else 1 / 30,
                                 response_s=run.control_timing.ahead(at, time.monotonic()) or 0.,
                                 covered=(run.covered | run.barred, run.cell) if goes_over and run.cell else None,
                                 barred=(run.barred, run.cell) if run.cell else None, presses=run.presses,
                                 reach=run.reach)
            choosing.signs, choosing.at, choosing.situations, choosing.ground = run.signs, at, situations, ground
            choosing.watched = keep.get("watched")
            choosing.caution *= caution_of(situations)
            _what_the_place_does(run, choosing, moves, hers, meeting, at)
            _what_she_plays_for(run, choosing, moves, hers, meeting, at, say)
            if rules is not None:
                choosing.strikes = run.blows.stands_ground(run.blows.keys(rules.fire_keys, hers.makes))
            choosing.going_to = a_place_to_go(run, moves, hers, at, lambda line, once, at=at: _say(run, say, line, at, once=once))
            if choosing.mine is not None and (choosing.ways or choosing.lifts or choosing.pointing):
                run.responsive_pictures += 1
            run.things = moves.things
            run.presses.saw(moves.things, at)
            run.timing.saw(moves.things.values())
            _found_by_her_presses(run, moves, hers, at)
            await _let_go_of_presses(hands, run, at)
            meeting.saw(moves, hers, happened, at, choosing.line() if choosing.mine is not None else None)
            read_the_signs(run.signs, moves, choosing.mine, meeting.verdicts, at,
                           lambda line, at=at: _say(run, say, line, at, once=line),
                           out_of_the_way=bool(rules is not None and rules.dodge_keys) and _how_it_stands(run, at)[1])
            run.choices.settle(at, met=meeting._met, gains=run.gains, losses=run.losses,
                               mine_at=(choosing.mine.x, choosing.mine.y) if choosing.mine is not None else None,
                               say=lambda line, at=at: _say(run, say, line, at, once=line))
            noticed(run, moves, meeting, at, say, _named)
            seen_in_play(run, moves, hers, meeting, picture, at, lambda line, once, at=at: _say(run, say, line, at, once=once))
            measured_in_play(run, moves, hers, meeting, at)
            if getattr(picture, "drawing_scene", None) is None:
                _counters_without_reading(run, moves, hers, meeting, at)
            violations = motion_checks.see(moves, happened, at, hers.number)
            if violations:
                ended = "runtime contract violated"
                _say(run, say, "This still behaves incorrectly: " + violations[0]["finding"] + ". I need to check the repair.", at, once="runtime_fault")
                break
            context.check()
            run.shown.look(picture, at)
            run.coming.see(moves, picture, at)
            reflexive = await respond_before_planning(hands, run, hers, choosing, at)
            if not reflexive:
                await _keep_reading(run, meeting, hers, moves, picture, at, read_words, say=say)
                from core.agency.navigation_before_body_experiments import (
                    navigation_before_body_experiments,
                )
                ended = navigation_before_body_experiments(run, hers, picture, at, keep, now=time.monotonic())
                if ended:
                    break
                rules = rules_as_told_since(run, rules, meeting=meeting)
                if await search_after_unresolved_body(hands, run, hers, moves, look=look, context=context,
                                                     picture=picture, at=at, deadline=began + seconds):
                    continue
                await _act(hands, run, moves, hers, meeting, choosing, at, say=say, rules=rules)
            context.check()
            if not reflexive:
                await use_controls_on_things(hands, run, moves, hers, happened, at,
                                             lambda line, once, at=at: _say(run, say, line, at, once=once), picture)
            for line in _out_of_her_reach(run, choosing, at):
                _say(run, say, line, at, once=line)
            if goes_over:
                if waits_for_her and choosing.mine is not None and choosing.ways and choosing.target()[1] == "cover":
                    _say(run, say, "Nothing's coming at me, so I'm looking around the place.", at, once="looking around")
            _what_she_says(run, say, moves, hers, meeting, at)
            _what_kind_of_game(run, say, moves, hers, meeting, physics, at)
            _report(run, getting_somewhere, at)
            ended = _over(run, moves, happened, at) or _nothing_answers(run, hers, meeting, at) or _said_it_is_over(run)
    finally:
        await moves.close()
        await release_search_after_play(hands, run, hers)
        await run.coming.let_go(hands)
        keep["signs_going"] = run.coming.known() or keep.get("signs_going")
        keep["keys_never_absent"] = run.shown.done()
    keep.update({"hers": hers, "meeting": meeting, "kinds": moves.kinds, "physics": physics, "meeting_with": run.meeting_with,
                 "contest": run.contest, "situation_known": run.situation_known, "said_lately": run.lately,
                 "where_clicks_pay": run.clicks_pay, "presses": run.presses, "timing": run.timing, "reach": run.reach, "signs": run.signs, "bars": run.bars, "settings": run.settings, "noticing": run.noticing, "seeing": run.seeing, "using": run.using,
                 "longest_still": run.longest_still,
                 "said_once": run.said, "leading": run.leading, "choices": run.choices, "pressures": run.pressures, "gone_for": run.gone_for,
                 "faces": run.faces, "comes": run.comes, "way_through": run.way_through,
                 "plays_for": run.plays_for, "who": run.who, "touches_told": run.touches_told,
                 "sounds_heard": run.sounds, "allies": run.allies,
                 "supply": run.supply, "pushing": run.pushing, "harm_lands": getattr(run, "harm_lands", None)})
    result = _what_it_came_to(run, moves, hers, meeting, ended, began)
    result['reflexes'] = reflex_record(run)
    result['pixel_search'] = pixel_search_record(run)
    after_the_stretch(run.settings, run.gains, run.losses)
    result["runtime_checks"] = {"required_edges": sorted(motion_checks.required_edges),
                                "provenance": motion_checks.provenance, "violations": motion_checks.violations,
                                "unconfirmed": motion_checks.unconfirmed}
    return result


async def _act(hands: Any, run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
               choosing: _Choosing, at: float, *, say: Any = None, rules: WhatTheRulesSaid | None = None) -> None:
    # Signs coming along a line ask for the front one's key, held, and nothing else pressed (signs_that_come_along.py).
    # Not where the place is played with the pointer: a row of targets crossing a gallery is no line of keys' signs.
    # And only where the place's words ask for keys shown to be followed (signs_that_come_along.py told_to_follow_signs).
    if (run.coming is not None and not run.pointer_first and not run.pointer_trigger and rules is not None
            and told_to_follow_signs(rules.words)
            and await run.coming.answered_now(hands, run, at, lambda line: _say(run, say, line, at, once="signs coming"))):
        return
    # Keys the screen puts up mid-play are asked for now, before anything she had in mind (pressing_what_is_shown.py).
    if run.shown is not None and run.shown.asks(at):
        if run.held:
            await hands.up(run.held)
            run.held = ""
        await run.shown.press(hands, say=lambda line: _say(run, say, line, at, once="keys shown"))
        return
    if await act_on_prepared_motion(hands, run, moves, hers, at):
        return
    # A key pressed before she can see what it does tells her nothing: the first look at a screen takes its measure.
    if not getattr(moves, "seeing", True):
        return
    # A thing taken for hers that did not answer her keys: try them again.
    if hers.lost_at > run.lost_at:
        run.lost_at, run.trying = hers.lost_at, 0
    # An instruction to click a visible target already names the interaction.
    # It does not require discovering an avatar that follows the pointer.
    aimed_trigger = (choosing.mine is not None and hers.follows_pointer
                     and rules is not None and rules.a_click_is_a_shot)
    if run.pointer_first and not aimed_trigger and any(meeting.stance(t.kind) == CLICK for t in moves.things.values()):
        await _click_things(hands, run, moves, meeting, at, only_named=True)
        return
    # Every key is tried once before any is chosen: a plan that knows one key
    # can only go one way.
    trying_the_pointer = run.pointer_first and run.pointed < 2 * len(_POINTER_TRIAL) and not hers.follows_pointer
    # A game its words say is played with the pointer is clicked between the keys tried, not after all of them:
    # LIVE 2026-10-08 a stretch ended on its ten key trials before the click that would have started the game.
    # Not where a click throws at someone on the other side: there each click is a throw, and one at the picture is a
    # throw wasted (offline 2026-10-12, a third of a food fight's throws went to one spot of the floor).
    throws_at_someone = rules is not None and rules.a_click_is_a_shot and _how_it_stands(run, at)[0]
    if (run.pointer_first and not trying_the_pointer and not hers.follows_pointer and choosing.mine is None
            and not throws_at_someone and hasattr(hands, "click") and at - run.last_click >= CLICK_THE_PICTURE_S):
        await _click_the_picture(hands, run, moves, at)
        if run.last_click == at:
            return
    # Where she understands no body of hers is in the picture, a thing taken for hers in a round before does not slow
    # her throws to the pace of finding out what a click does: offline 2026-10-12 a food fight was thrown at five
    # times a second in its first round and once a second in every round after.
    if (choosing.mine is None or _how_it_stands(run, at)[1]) and not hers.follows_pointer:
        await _fire_while_finding_herself(hands, run, hers, moves, rules, at)
    if _still_trying_keys(run, hers, choosing, trying_the_pointer, at=at):
        await _try_the_keys(hands, run, hers, at, moves)
        return
    from core.agency.steering_from_a_model import steer_with_a_model
    if await steer_with_a_model(hands, run, hers, choosing, at, hold=_hold):
        return
    if await _a_facing_trial(hands, run, hers, choosing, at):
        return
    if choosing.mine is not None and hers.follows_pointer:
        await _hold(hands, run, hers, "", at)
        _goal, why, aim = choosing.target()
        await _trigger(hands, run, hers, choosing, aim, why, at, rules=rules)
        await _point_at(hands, run, hers, moves, choosing, at)
        return
    if choosing.mine is None or not (choosing.ways or choosing.lifts):
        # Held, having moved before: where the words say to press keys in turn, fast, that is how she breaks free.
        if run.burst_keys and hers.kind is not None and at - run.burst_at >= BURST_EVERY_S:
            await _burst(hands, run, at, say=say)
            return
        # No press of her keys has moved anything on the screen, after all her trials of them, and things go on by
        # themselves: a press may count by when it is made, not by what it moves (core/agency/when_a_press_pays.py).
        # A press that moved something the same way each time is a body still to be found, not a clock to beat:
        # offline 2026-10-09 a runner missed by her first trials was timed against and never looked for again.
        # And timing goes on while it pays; where it has not paid once every place has been tried, she looks for a
        # body again.
        if (run.keys and not run.pointer_first and run.timing is not None and run.trying >= 4 * len(run.keys)
                and (run.building_up or not run.presses.moves_anything(TOLD_FROM_STANDING_PX * 2))
                and (run.timing.pays() or at - run.tried_at < max(RECHECK_CONTROLS_S, run.timing.a_round_s()))
                and await _press_in_time(hands, run, moves, at, say)):
            return
        if (run.keys and not body_probe_is_active(run, hers, pointer_steps=2 * len(_POINTER_TRIAL), at=at)
                and at - run.tried_at >= RECHECK_CONTROLS_S):
            run.trying = 0
            run.control_probe_retries += 1
            hers.recheck_controls()
            if run.pointer_first:
                run.pointed = 0
            # Said only where she had found which one was hers and lost it: on a title screen there was nothing to lose,
            # and LIVE 2026-10-07 the line was read out over one.
            if "me" in run.said:
                _say(run, say, "I've lost track of which one is me; trying the keys again to find out.", at, once="recheck_controls")
            else:
                logger.info("trying the keys again: nothing has answered to them yet")
        # A favored tap has its own clock; it cannot postpone the next discovery pass.
        paying = a_key_that_pays(run.keys)
        timing = run.timing is not None and run.timing.has_something_to_time(moves.things.values())
        if (paying and not timing and not body_probe_is_active(run, hers, pointer_steps=2 * len(_POINTER_TRIAL), at=at)
                and at - max(run.tried_at, run.last_payoff_trial_at) >= TRY_A_KEY_S and hasattr(hands, "tap")):
            await hands.tap(paying)
            run.input_key_downs[paying] += 1
            run.last_payoff_trial_at = at
            return
        pointer_trial = run.pointed < 2 * len(_POINTER_TRIAL) and hasattr(hands, "point")
        if pointer_trial and run.pointer_first:
            await _try_the_pointer(hands, run, hers, moves, at)
            return
        # Played with the pointer, nothing following it, and things going on by themselves: a click is a press, timed
        # as one. LIVE 2026-10-09 "when the hamster lines up with the pillow, click again to launch the hamster", and
        # with the hamster moving she held every click back.
        if (run.pointer_first and not run.keys and not pointer_trial and run.timing is not None and hasattr(hands, "click")
                and any(thing.moved for thing in moves.things.values()) and await _press_in_time(hands, run, moves, at, say)):
            return
        # Played with the pointer: a click, paced, before the keys are gone through again.
        if run.pointer_first and hasattr(hands, "click") and at - run.last_click >= CLICK_THE_PICTURE_S:
            await _click_the_picture(hands, run, moves, at)
            if run.last_click == at:
                return
        if body_probe_is_active(run, hers, pointer_steps=2 * len(_POINTER_TRIAL), at=at):
            await _try_the_keys(hands, run, hers, at, moves)
            return
        await _hold(hands, run, hers, "", at)
        if pointer_trial:
            await _try_the_pointer(hands, run, hers, moves, at)
            return
        if hasattr(hands, "click"):
            await _click_things(hands, run, moves, meeting, at)
            await _click_the_picture(hands, run, moves, at)
        return
    allow_for_picture_age(run, choosing, at, time.monotonic())
    untried = [k for k in run.keys if not has_a_measured_response(hers, k) and k not in choosing.ways]
    # Where she means to go and no way she knows takes her nearer, a way she has not learned yet is tried now: LIVE
    # 2026-10-07 her hero stood still beside the thing it meant to go to, left being the one arrow not yet learned.
    if untried and choosing.danger((0.0, 0.0)) == 0.0 and (
            at - run.tried_at > 3.0 or choosing.key(run.held)[:2] in {("", "meet"), ("", "cover")}):
        run.tried_at = at
        await _hold(hands, run, hers, untried[0], at, trying=True)
        return
    if at - run.tried_at < TRY_A_KEY_S and run.held in untried:
        return
    _where_to_meet_things(run, choosing, meeting, at)
    imagined = _the_return_they_cannot_reach(choosing)
    part = imagined if imagined is not None else run.meeting_now
    shift = part * ((choosing.mine.h if choosing.line() == 0 else choosing.mine.w) / 2 if choosing.mine is not None else 0.0)
    offset = (0.0, shift) if choosing.line() == 0 else (shift, 0.0) if choosing.line() == 1 else (0.0, 0.0)
    allow_for_picture_age(run, choosing, at, time.monotonic())
    key, why, aim = choosing.key(run.held, offset=offset)
    chose_in_play(run.choices, choosing, key, why, aim, at, lambda line, once: _say(run, say, line, at, once=once))
    before = choosing.ways.get(run.held, (0.0, 0.0))
    after = choosing.ways.get(key, (0.0, 0.0))
    mine = choosing.mine
    transition = (mine.number, mine.x, mine.y, at, before, after) if key != run.held and not getattr(hers, "carried", False) else None
    await _hold(hands, run, hers, key, at)
    run.previous_control = (*transition, run.control_delivery) if transition is not None else None
    await _press_to_get_clear(hands, run, hers, choosing, key, at, say)
    await _trigger(hands, run, hers, choosing, aim, why, at, rules=rules)


async def _press_to_get_clear(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing, held: str, at: float,
                              say: Any) -> None:
    """A press beside the key held, where the path it takes her on (core/agency/what_a_press_does.py) is clear of what
    is coming and holding her course is not: over a thing running at her, across a gap, out of the way of a fall."""
    if run.letting_go or choosing.mine is None:
        return
    press = choosing.a_press_that_clears(run.presses, held)
    if press is None:
        return
    began = time.monotonic()
    await hands.down(press)
    run.input_key_downs[press] += 1
    run.presses.pressed(press, began, run.things.values(), choosing.mine.number)
    run.letting_go[press] = began + run.presses.held_for(press, THE_PRESS_S)
    hers.pressed_beside(began + max(run.presses.lasts(press, choosing.mine.kind, choosing.mine.number),
                                    run.letting_go[press] - began))
    _say(run, say, f"{press.capitalize()} lifts me clear of what comes at me; I'm using it when something is about to.",
         at, once="a press that clears")


def _found_by_her_presses(run: _Run, moves: WhatMoves, hers: WhichIsHers, at: float) -> None:
    """Where her trials have not found her, the one thing of a kind her presses move alike every time is hers."""
    from core.agency.finding_her_another_way import found_by_press_curves

    found_by_press_curves(run, moves, hers, at, old_for=THERE_BEFORE_S)


async def _let_go_of_presses(hands: Any, run: _Run, at: float) -> None:
    """Let go of each press made beside the held key once it has been held as long as it was when it was watched."""
    for key, when in list(run.letting_go.items()):
        if time.monotonic() >= when:
            await hands.up(key)
            run.presses.released(key, time.monotonic())
            del run.letting_go[key]


async def _trigger(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing,
                   aim: Any, why: str, at: float, *, rules: WhatTheRulesSaid | None = None) -> None:
    """Deliver a trigger alongside steering, using its own physical input.

    Steering and firing are compatible channels. A pointer-controlled object
    used to return before its trigger could run. Named triggers are tried at
    a measured pace until their projectiles can be aimed by learned motion.
    """
    await _charge(hands, run, hers, choosing, rules)
    if await _strike(hands, run, hers, choosing, rules):
        return
    fire = choosing.fire(aim, why)
    discovering = choosing.shot is None
    if fire is None and discovering and choosing.mine is not None and rules is not None and rules.a_click_is_a_shot:
        # Its words say a click shoots: whether the pointer moves her or aims for her, the shot is a click.
        fire = "mouse"
    elif fire is None and discovering and choosing.mine is not None and rules is not None:
        # The key its words say fires is fired at a measured pace until what it sends is seen and can be aimed: a shot
        # too small and quick to be followed (offline 2026-10-11, a two-pixel streak) is never learned by watching for
        # it, and was never fired again.
        if choosing.pointing and rules.a_click_is_a_shot:
            fire = "mouse"
        else:
            # Not a key held to charge: tapped, it never builds up, and charging is held and let go of its own accord.
            fire = next((key for key in rules.fire_keys if key not in WAYS and key not in rules.charge_keys), None)
    window = None if discovering else hers.emission_window(choosing.shot.key, choosing.shot.kind)
    if discovering:
        next_trigger = run.last_click + CLICK_THE_PICTURE_S
    elif window is None:
        next_trigger = run.last_click + max(CLICK_EVERY_S, 0.4 + RESPONSE_S)
    else:
        next_trigger = max(run.last_click + CLICK_EVERY_S, run.last_trigger_began + window + RESPONSE_S)
    if not fire or fire == run.held or at < next_trigger:
        return
    # A key that spends a counter of hers spends it where the shot is worth it (core/agency/spending_a_supply.py).
    if fire != "mouse" and run.supply is not None and not discovering and aim is not None:
        label = run.supply.spends(fire)
        if label is not None and not run.supply.spend(at, _worth_of_a_shot(choosing, aim, at),
                                                      run.supply.last.get(label, 0.0),
                                                      seconds_left=choosing.seconds_left,
                                                      refill=run.supply.refills_a_second(label, at)):
            return
    if fire == "mouse":
        if not hasattr(hands, "click"):
            return
        tall, wide = choosing.moves.shape
        target = None if choosing.pointing else _aimed_with_the_pointer(choosing, aim, run)
        if choosing.pointing:
            run.pointer = (min(1.0, max(0.0, choosing.mine.x / max(1, wide))),
                           min(1.0, max(0.0, choosing.mine.y / max(1, tall))))
        elif target is not None and hasattr(hands, "point"):
            await hands.point(*run.pointer)
        dispatched = time.monotonic()
        await hands.click(*run.pointer)
        if target is not None and run.leading is not None:
            run.leading.aimed(target[0], at, target[1])
    else:
        dispatched = time.monotonic()
        await hands.tap(fire)
    delivered = time.monotonic()
    if fire == "mouse" and choosing.pointing:
        hers.pointed(run.pointer[0] * wide, run.pointer[1] * tall, delivered)
    hers.tapped(fire, delivered, began=dispatched)
    if run.supply is not None and fire != "mouse":
        run.supply.pressed(fire, at)
    run.last_trigger_began = dispatched
    run.last_click = delivered
    run.taps += 1


def _worth_of_a_shot(choosing: _Choosing, aim: Any, at: float) -> float:
    """What a shot at ``aim`` is worth, in seconds of play: what taking it out is worth (core/agency/what_she_plays_for.py),
    one where nothing prices it."""
    goes = choosing.plays_for
    if goes is None or choosing.mine is None:
        return 1.0
    size = max(float(choosing.mine.w), float(choosing.mine.h))
    comes = choosing.comes is not None and choosing.comes.comes_for_her(aim, size)
    return goes.worth_taking_out_s(aim.kind, comes_for_her=comes, at=at, seconds_left=choosing.seconds_left)


def _aimed_with_the_pointer(choosing: _Choosing, aim: Any, run: _Run) -> tuple[int, float] | None:
    """Where the pointer aims what her click sends: on the thing it is for, where it will be when the shot gets there,
    by the lead that has taken things (core/agency/leading_a_click.py). Sets ``run.pointer``; the thing and the lead."""
    from core.agency.leading_a_click import the_thing_to_aim_at

    thing = the_thing_to_aim_at(choosing, aim)
    if thing is None or run.leading is None:
        return None
    lead = run.leading.lead()
    x, y = choosing.moves.share(*thing.where_at(lead))
    run.pointer = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))
    return thing.number, lead


async def _strike(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing, rules: WhatTheRulesSaid | None) -> bool:
    """Strike what is close by a key the words give to hitting that has sent nothing out: a punch, a kick, a swing.

    Read offline 2026-10-09, the 56 games' own words name blows ("PRESS THE 'S' KEY TO PUNCH 'D' TO KICK", "Press the Z
    key to melee attack close enemies"). A fire key that brings nothing out never has a shot to aim, so once she had
    found herself she never pressed it again. Nothing here knows a punch (core/agency/how_far_her_blow_reaches.py).
    """
    if rules is None or choosing.mine is None:
        return False
    things = list(choosing.moves.things.values())
    for key in run.blows.keys(rules.fire_keys, hers.makes):
        if key in WAYS or key == run.held:
            continue
        at = time.monotonic()
        target = run.blows.to_strike(key, choosing.mine, things, lambda thing: _what_is_known_of(choosing.meeting, thing), at)
        if target is None:
            continue
        await hands.tap(key)
        hers.tapped(key, time.monotonic(), began=at)
        run.blows.pressed(key, at, choosing.mine, things)
        run.last_trigger_began, run.taps = at, run.taps + 1
        return True
    return False


async def _charge(hands: Any, run: _Run, hers: WhichIsHers, choosing: _Choosing, rules: WhatTheRulesSaid | None) -> None:
    """Hold a key the words say to hold and let go, alongside the rest of play, and let it go once it has been held as
    long as is meant: "hold down the X key to charge up, then release it to fire" (core/agency/holding_to_charge.py).

    It is held where there is something on the screen to use it on, as a person charges a shot at what is coming and
    not at an empty screen.
    """
    charging = run.charging
    at = time.monotonic()
    if charging.due(at):
        key = charging.key
        await hands.up(key)
        let_go = time.monotonic()
        charging.released(let_go)
        # What a release brings out beside her is learned as a press's is.
        hers.tapped(key, let_go, began=let_go)
        return
    if rules is None or not charging.ready(at) or not hasattr(hands, "down"):
        return
    keys = [key for key in rules.charge_keys if key not in WAYS and key != run.held]
    if not keys or not any(_what_is_known_of(choosing.meeting, thing) not in (MEET, CLICK, IGNORE) for thing in choosing.others()):
        return
    await hands.down(keys[0])
    charging.begin(keys[0], at)


def _what_is_known_of(meeting: WhatMeetingDoes, thing: Any) -> str:
    """What she was told or has learned to do about a thing; "" where her stance toward it is only the guess of meeting
    what is unknown, which a blow does not wait on."""
    if not meeting.known(thing.kind) and thing.kind not in meeting.told:
        return ""
    return meeting.stance(thing.kind, fixture=not thing.moved)


#: How often she fires while still finding which thing is hers, by a key the words named for it or a click that throws.
FIRE_WHILE_FINDING_S = 0.35
THROW_WHILE_FINDING_S = 0.6
THROW_AT_SOMEONE_S = 0.2
#: How often at most the key for getting out of the way is pressed at something coming at her.
GOT_OUT_OF_THE_WAY_EVERY_S = 0.5


async def _fire_while_finding_herself(hands: Any, run: _Run, hers: WhichIsHers, moves: WhatMoves,
                                      rules: WhatTheRulesSaid | None, at: float) -> None:
    """Fire from the first moment, as the words say to, while she is still finding which thing is hers.

    A person dropped into a game that says "SPACE to fire" fires at once and finds their feet as they go; one told
    "aim with your mouse and click to throw" throws at what moves. LIVE 2026-10-08 she spent a hundred seconds of a
    shooter holding one arrow at a time to see what moved, and never fired. A press of the fire key moves nothing, so
    the finding goes on as before, and what each press brings out beside her is how she learns it fires.
    """
    from core.agency import thrown_at_what_shows as thrown
    from core.agency.settings_that_paid import setting

    if rules is None or (not _no_body_to_find() and body_probe_is_active(run, hers, pointer_steps=2 * len(_POINTER_TRIAL), at=at)):
        return
    # A key the words give to getting out of the way is not one to fire with, whatever else a run-on line made of it:
    # LIVE 2026-10-11 "...FIVE TIMES WINS DODGE FOOD BY PRESSING THE SPACE BAR" gave space to hitting as well, she
    # tapped it three times a second as her shot, and in a food fight lost in fourteen seconds she threw nothing.
    away = [key for key in rules.dodge_keys if key not in WAYS]
    fire = next((key for key in rules.fire_keys if key not in WAYS and key != run.held and key not in away), None)
    if fire is not None and at - run.last_trigger_began >= FIRE_WHILE_FINDING_S:
        began = time.monotonic()
        await hands.tap(fire)
        hers.tapped(fire, time.monotonic(), began=began)
        run.last_trigger_began, run.taps = began, run.taps + 1
        return
    # Each picture is taken in for where harm lands: seen only when she might duck, throws landed where they never were.
    coming = bool(away) and thrown.comes_at_her_where_harm_lands(run, moves.things.values(), at, setting(run, "arriving within", thrown.ARRIVES_WITHIN_S), moves.shape)
    if away and at - run.last_trigger_began >= setting(run, "duck every", GOT_OUT_OF_THE_WAY_EVERY_S) and (
            coming or getattr(run, "signs", None) is not None and run.signs.just_shown(at)):
        # With no body of hers in the place, what comes at her grows, or someone shows what comes before harm: the key
        # for getting out of the way is pressed.
        run.last_trigger_began = time.monotonic()
        await hands.tap(away[0])
        run.taps += 1
    against = _how_it_stands(run, at)[0]
    # Against someone, as fast as the place takes throws: offline 2026-10-12 a food fight was lost each round throwing
    # every 0.6 s at the opponent, and won throwing at him five times a second, the place taking what it would.
    every = setting(run, "throw every", THROW_AT_SOMEONE_S) if against else THROW_WHILE_FINDING_S
    if not rules.a_click_is_a_shot or not hasattr(hands, "click") or at - run.last_click < every:
        return
    takes = thrown.how_long_throws_take(run, moves, at)
    aimed = _where_to_throw(run, moves, hers, at, against)
    if aimed is None:
        return
    tall, wide = moves.shape
    await hands.click(min(1.0, max(0.0, aimed[0] / max(1, wide))), min(1.0, max(0.0, aimed[1] / max(1, tall))))
    run.last_click, run.clicks = at, getattr(run, "clicks", 0) + 1
    run.shots_at = [*getattr(run, "shots_at", ())[-5:], at]
    takes.thrown(at, aimed[0], aimed[1])


def _where_to_throw(run: _Run, moves: WhatMoves, hers: WhichIsHers, at: float, against: bool) -> tuple[float, float] | None:
    """Where a throw goes, in the picture's own pixels: at whoever has just shown, before whatever is going across
    (core/agency/thrown_at_what_shows.py), led where it goes; against someone, at the one last thrown at, found again
    where it was though the eye has lost it and found it under another number. Offline 2026-10-12 a food fight's
    opponent, turning and crouching, was found afresh every few looks, taken each time for her own throw just come
    into sight, and nineteen asks in twenty found no one to throw at."""
    from core.agency.thrown_at_what_shows import (
        led,
        what_the_shot_goes_at,
        where_the_one_thrown_at_is,
    )

    target = what_the_shot_goes_at(moves.things.values(), at, not_numbers=(hers.number,),
                                   not_kinds={made.kind for made in hers.makes.values()},
                                   against_someone=against, shots_at=getattr(run, "shots_at", ()))
    if target is not None:
        if against:
            run.thrown_at = (target.x, target.y, at, float(target.size))
        return led(run, target, at)
    if not against:
        return None
    found = where_the_one_thrown_at_is(moves.things.values(), getattr(run, "thrown_at", None), at, moves.shape[1])
    if found is None:
        return None
    x, y, seen = found
    if seen is not None:
        run.thrown_at = (x, y, at, float(seen.size))
    return x, y


#: The parts of her thing she imagines meeting a thing with, from one end
#: (-1) through the middle to the other end (1).
_PARTS = (-0.8, -0.5, -0.25, 0.0, 0.25, 0.5, 0.8)


def _the_return_they_cannot_reach(choosing: _Choosing) -> float | None:
    """Where on her thing to meet what is coming, chosen by running each return forward in her head.

    For each part of her thing: how the thing would leave it (from the
    meetings she has seen), where it would then cross the far side's line, and
    how far the other side's player would have to go in that time against how
    fast it has been seen to go. The part that leaves them furthest short
    wins. None until she has seen enough meetings, or when there is nothing
    coming or nobody on the other side.
    """
    physics, mine, line = choosing.physics, choosing.mine, choosing.line()
    if physics is None or mine is None or line is None:
        return None
    (_gx, _gy), why, coming = choosing.target()
    if why != "meet" or coming is None:
        return None
    free = 1 - line
    contact = _contact_line(mine, coming, line)
    arrival = physics.when_it_reaches(coming, line, contact)
    if arrival is None:
        return None
    them = _the_other_side(choosing)
    if them is None:
        return None
    reach = max(1.0, physics.fastest(them.kind) or 1.0)
    best = None
    for part in _PARTS:
        leaving = physics.after_meeting(coming.kind, part, (coming.vx, coming.vy), across=line == 0)
        if leaving is None:
            return None
        start = [0.0, 0.0]
        start[line] = contact
        start[free] = arrival[1]
        crossing = physics.when_it_reaches(
            coming, line, _contact_line(them, coming, line), start=(start[0], start[1], leaving[0], leaving[1])
        )
        if crossing is None:
            continue
        short = abs(crossing[1] - (them.x, them.y)[free]) - reach * crossing[0]
        if best is None or short > best[0]:
            best = (short, part)
    return best[1] if best is not None else None


def _the_other_side(choosing: _Choosing) -> Any:
    """The thing shaped like hers on the other half of the picture: the other side's player."""
    mine, moves = choosing.mine, choosing.moves
    tall, wide = moves.shape
    middle = (wide, tall)[choosing.line() or 0] / 2
    for thing in moves.things.values():
        if thing.number == mine.number:
            continue
        alike = abs(math.log(max(1.0, thing.w) / max(1.0, mine.w))) < 0.4 and abs(math.log(max(1.0, thing.h) / max(1.0, mine.h))) < 0.4
        across = ((thing.x, thing.y)[choosing.line() or 0] < middle) != ((mine.x, mine.y)[choosing.line() or 0] < middle)
        if alike and across:
            return thing
    return None


#: How long after a meeting a gain for her is put down to it.
GAIN_AFTER_A_MEETING_S = 4.0


def _where_to_meet_things(run: _Run, choosing: _Choosing, meeting: WhatMeetingDoes, at: float) -> None:
    """Learn which part of her thing to meet things with, from the gains that follow.

    Where a thing that meets hers is sent depends on where it met: off the end
    of a paddle a ball goes away steeply, off the middle it comes back the way
    it came. Which of those wins points is a fact about the world she is in,
    so it is measured: each meeting is put down to the part she met with, a
    gain soon after is put down to that meeting, and the next part is drawn
    from what each has earned so far (Thompson sampling over three arms).
    """
    import random

    if choosing.mine is None or choosing.line() is None:
        return
    touching = {t.number for t in choosing.others() if t.moved and _boxes_meet(_moved_box(choosing.mine.box(), 0, 0, 2.0), t.box())}
    if touching - run.touching:
        run.meeting_with[run.meeting_now][0] += 1
        run.met.append((at, run.meeting_now))
        run.meeting_now = max(
            run.meeting_with,
            key=lambda part: random.betavariate(
                1 + run.meeting_with[part][1], 1 + max(0, run.meeting_with[part][0] - run.meeting_with[part][1])
            ),
        )
    run.touching = touching
    for verdict in meeting.verdicts[run.credited_up_to :]:
        if verdict["what"] == "gain":
            earned = [part for when, part in run.met if 0 <= verdict["since"] - when <= GAIN_AFTER_A_MEETING_S]
            if earned and run.meeting_with[earned[-1]][1] < run.meeting_with[earned[-1]][0]:
                run.meeting_with[earned[-1]][1] += 1
    run.credited_up_to = len(meeting.verdicts)


#: How long into a stretch writing that says a round is over must first be read to be its end, not the last one's.
OVER_READ_AFTER_S = 2.0
SAID_IT_IS_OVER = "the screen says the round is over"


def _said_it_is_over(run: _Run) -> str:
    """Writing read in play that says a round is over ("TRY AGAIN", "GAME OVER") and was not there as it began: the
    stretch hands the screen back. LIVE 2026-10-09 an end screen's stars twinkled, and she played on at it for half a
    minute."""
    from core.agency.words_read_with_their_picture import first_word_picture_at
    from core.language.how_a_game_ended import says_a_round_is_over, what_says_the_game_is_over

    # What the first reading of the stretch found was there as it began, however long that reading took.
    first = first_word_picture_at(run)
    began = max(run.began + OVER_READ_AFTER_S, (run.began if first is None else first) + 0.5)
    there = {said for when, said in run.words_read if when < began}
    for when, said in run.words_read[-12:]:
        # A round over, or the game: offline 2026-10-12 a racer's "Your final score" came up mid-stretch, and she played
        # on at it for a minute and a half until her time ran out.
        if when >= began and said not in there and (says_a_round_is_over(said) or what_says_the_game_is_over(said)):
            run.over_read_at = min(run.over_read_at, when)
            return SAID_IT_IS_OVER
    return ""


def _the_words_of_play(run: _Run, ended: str) -> set[str]:
    """The words read while the game was going, not those of the screen it ended on.

    A stretch that ended because nothing moved ended on a still screen, and
    whatever was read from the moment movement stopped (and a little before:
    readings are taken from pictures a step behind) is that screen's. LIVE
    2026-10-04 "to play again" was read in the seconds the end screen stood,
    kept as a word of play, and so the restart that ended a won game was taken
    for the game's furniture and she started another. A stretch ended by
    nothing answering her, after something had, ended on what does not answer
    since: offline 2026-10-12 an
    end screen that kept moving stood for seconds before the stretch gave up on
    it, its "Play Again" was kept as play's, and her third run went uncounted.
    """
    still = ended.startswith(("nothing on the screen", "the screen changed"))
    cutoff = run.last_moving - END_SCREEN_MARGIN_S if still else run.over_read_at - END_SCREEN_MARGIN_S if ended == SAID_IT_IS_OVER else math.inf
    if ended.startswith(("nothing here answers", "nothing I touch")) and run.answered_at > -math.inf:
        cutoff = run.answered_at - END_SCREEN_MARGIN_S
    # Nor the screen it began on, read before anything had moved: a stretch
    # begun as the last game's end screen gave way is not that screen's game.
    return {said for when, said in run.words_read if run.first_moving <= when < cutoff}


#: How long before the last movement a reading may already show the end.
END_SCREEN_MARGIN_S = 0.5


def _what_it_came_to(run: _Run, moves: WhatMoves, hers: WhichIsHers, meeting: WhatMeetingDoes,
                     ended: str, began: float) -> dict[str, Any]:
    took = max(1e-6, time.monotonic() - began)
    learned = {}
    for kind in meeting.evidence:
        if meeting.known(kind) and kind < len(moves.kinds):
            learned[colour_name(moves.kinds[kind].colour)] = meeting.stance(kind)
    return {
        "seconds": round(took, 1),
        "pictures": run.pictures,
        "pictures_a_second": round(run.pictures / took, 1),
        "ended": ended,
        "hers": run.hers_description or (describe(moves, hers.kind, hers.thing(moves)) if hers.kind is not None else ""),
        "keys_that_move_her": sorted(hers.keys_that_move_her(run.keys)),
        "follows_the_pointer": bool(hers.kind is not None and getattr(hers, "follows_pointer", False)),
        "fires": sorted(hers.makes),
        "learned": learned,
        # What cost her in it, by name: kept clear of from now on, and said between rounds.
        "came_to_avoid": [describe(moves, kind) for kind in meeting.evidence if kind < len(moves.kinds)
                          and kind not in run.avoided_before and meeting.stance(kind) == AVOID][:3],
        "gains": run.gains,
        "losses": run.losses,
        # How far the view going by has taken her: views she had not been in, and how far along the way on.
        "new_views": run.world.new_views,
        "views_along": round(run.world.views_along(), 2),
        "counters": dict(meeting.readouts.values),
        "observations": dict(run.observation_sources),
        "input_key_downs": dict(run.input_key_downs),
        # What she did with the pointer and how often she tapped, for whoever watches how she played.
        "clicks": run.clicks,
        "taps": run.taps,
        "responsive_pictures": run.responsive_pictures,
        "control_probe_retries": run.control_probe_retries,
        "control_attribution": list(run.control_attribution),
        "attribution_changes": run.attribution_changes,
        "identification_evidence": list(hers.identification.receipts),
        # How the contest stood when the stretch ended, and what its counters alone settle.
        "standing": run.contest.says(),
        "settled": run.contest.settled(),
        "stalled": run.contest.stalled(time.monotonic()),
        "said": list(run.lines),
        "words_seen": sorted(_the_words_of_play(run, ended)),
        "choices": run.choices.summary() if run.choices is not None else {},
    }
