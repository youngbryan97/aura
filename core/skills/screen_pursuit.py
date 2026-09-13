"""Pursue a goal on screen: look, act, look again, until it is done.

Every part of this already existed and none of them were connected.

``FluidExecutor.pursue`` closes a perceive-decide-act loop with governance and
verification. ``host_automation.get_screen_text`` reads the screen and now
returns WHERE each run of text sat. ``perception_demand`` keeps her eyes open
at task cadence while she acts, instead of the 0.1Hz a foreground generation
used to impose. ``hotkey`` and ``click_at`` press keys and click points.

What was missing was any way to ASK for the combination. Every skill in the
registry performs ONE act and returns; nothing could watch something change
and keep going. So tasks that are trivially describable — "wait for that build
to finish and tell me if it fails", "keep pressing next until the form is
done", "play this until you win" — had no path through the system at all, and
the shape of the failure was always the same: she did one step and stopped.

This is deliberately not about any particular screen. It takes a goal, a way
to recognise success in what is read back, and a bound; it decides each move
from the CURRENT reading rather than from a plan written in advance. A build
log, a wizard, an installer and a game are the same problem.

The decision itself is delegated to a policy callable, so the loop does not
own any judgement about what to press. That keeps the cognition where
cognition lives and leaves this module as what it is: a way to keep looking.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from core.cognition.does_this_world_repeat import DoesItRepeat
from core.cognition.getting_ready_for_what_is_coming import WhatUsuallyComes
from core.cognition.how_far_to_go_before_looking import HowFarToGo
from core.cognition.marks_she_leaves_behind import MarksOnTheGround
from core.cognition.the_furthest_she_has_got import TheFurthestSheHasGot
from core.cognition.the_ones_she_reaches_for import TheOnesSheReachesFor
from core.cognition.what_an_act_costs_beyond_now import WhatEachActHasLeft
from core.cognition.what_happens_while_she_acts import WhatItCostsToBeBusy
from core.cognition.what_having_it_lets_her_do import WhatOpensWhat
from core.cognition.what_she_has_set_in_motion import WhatIsComing
from core.cognition.what_works_against_what import WhatBeatsWhat
from core.cognition.which_way_to_win import which_way_to_win
from core.runtime.errors import record_degradation
from core.runtime.watched_goal import PURSUIT_SECONDS, a_cycle_took
from core.runtime.what_she_learned import TRUST_CARRIED_OVER, named, recall, remember
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.runtime.task_ownership import create_owned_asyncio_task
from .screen_pursuit_surface import (
    DECLINES_AND_NOTHING_ELSE,  # noqa: F401
    _looks_like,  # noqa: F401
    LABEL_REACH,  # noqa: F401
    PRESSABLE_KEYS,
    _a_pass_in_moves,  # noqa: F401
    _bound_to_a_window,  # noqa: F401
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _the_part_of,
    _value_is_on_screen,  # noqa: F401
    _who_the_screen_belongs_to,
    click_normalized,
    goal_reached,
    labelled_by,  # noqa: F401
    press,
    window_bounds,
)
from .screen_pursuit_bearings import (
    RESTART_LABELS,  # noqa: F401
    _is_a_thing_laid_out,  # noqa: F401
    a_way_back_that_was_not_there,  # noqa: F401
    restart_controls,  # noqa: F401
    A_SCREENFUL_AT_LEAST,  # noqa: F401
    DEFAULT_MOVES,
    ENOUGH_TO_BE_A_THING,  # noqa: F401
    LANGUAGE_EVERY,
    MOST_OF_A_SCREEN,  # noqa: F401
    PRESS_ON,  # noqa: F401
    SAY_LESS,  # noqa: F401
    SCREENFULS_TO_LOOK,  # noqa: F401
    SEE_IT_THROUGH,  # noqa: F401
    SETTLE_AFTER_SCROLL_S,  # noqa: F401
    SLOW_DOWN,  # noqa: F401
    START_OVER,
    _AS_IT_USUALLY_IS,  # noqa: F401
    _a_screenful,  # noqa: F401
    _a_step_back,  # noqa: F401
    _ask_again_after,  # noqa: F401
    _both_of_the_thing,  # noqa: F401
    _bring_it_into_view,
    _by_how_much_room,  # noqa: F401
    _how_much_the_tally_moved,  # noqa: F401
    _in_the_same_grid,  # noqa: F401
    _moves_she_will_not_make,  # noqa: F401
    _moves_that_leave_her_nothing,  # noqa: F401
    _she_got_further,  # noqa: F401
    _take_the_run_its_bearings,  # noqa: F401
    _the_biggest_thing_on_it,  # noqa: F401
    _time_left,  # noqa: F401
    _was_of_that_window,  # noqa: F401
    _what_she_is_not_reading,  # noqa: F401
    _what_there_is_to_aim_at,  # noqa: F401
    _within_a_move,  # noqa: F401
    _within_the_run,
    _worth_holding,  # noqa: F401
    a_run_she_can_carry,  # noqa: F401
    am_i_there,  # noqa: F401
    pacing_options,  # noqa: F401
    restart_control,
    screen_options,  # noqa: F401
    ways_out,
)
from .screen_pursuit_looking import (
    _how_long_a_look_takes,  # noqa: F401
    _no_more_than_a_fresh_one_is_worth,  # noqa: F401
    _the_kind_of_world_this_is,  # noqa: F401
    ASKING_TO_CONFIRM,  # noqa: F401
    LONGER_THAN_USUAL,  # noqa: F401
    OBSERVE_TIMEOUT_S,  # noqa: F401
    PASSES_ON_ITS_OWN,
    _ANSWERING_TOOK,  # noqa: F401
    _WHY_SHE_CANNOT_LOOK,
    _answer_own_confirmation,
    _bring_the_thing_back_to_the_front,
    _covers,
    _expected_of,  # noqa: F401
    _her_reasoning,
    _how_full,  # noqa: F401
    _how_long_to_wait,  # noqa: F401
    _move_her_own_surface_aside,  # noqa: F401
    _narrate,  # noqa: F401
    _say_intent,  # noqa: F401
    _say_line,
    _settled_after,  # noqa: F401
    _the_best_reading_available,  # noqa: F401
    _the_thing_she_is_acting_in,  # noqa: F401
    _what_being_refused_a_look_means,
    _what_she_could_not_learn_from,  # noqa: F401
    _where,  # noqa: F401
    _where_clicked,  # noqa: F401
    _where_it_asks,  # noqa: F401
    _why_nothing_answers,  # noqa: F401
    clear_what_is_in_front,  # noqa: F401
    wait_for_a_screen_to_look_at,
)
from types import SimpleNamespace
from .screen_pursuit_blockers import clear_what_blocks_the_run
from .screen_pursuit_decision import decide_the_next_move
from .screen_pursuit_observing import observe_the_screen

logger = logging.getLogger("Aura.ScreenPursuit")


#: How many times a blocker may be attacked before the run reports it.
#:
#: A dismissal Step succeeds when the KEY WAS PRESSED, not when the overlay
#: went away — the same "an action that ran is not an action that worked"
#: distinction the loop exists to enforce, which the blocker path was itself
#: exempt from. Measured live: a page modal that ignores Escape produced forty
#: cycles of Escape and zero moves, each one reported as verified progress.
#:
#: After this many attempts the loop stops trying and says what is in the way.
#: Deciding what ELSE to do about an unknown dialog is cognition's job, not a
#: loop's: the honest end is a named obstacle, never a spin.
MAX_BLOCKER_ATTEMPTS = 3

#: How many graded attempts travel into the next decision. Enough to notice a
#: move that stopped working, short enough that old evidence stops steering.
RECENT_ATTEMPTS = 4
#: How many times one run will go back and look up how the task is done.
#: How many readings in a row have to show the same thing before a run is
#: handed back to the person. A page carries things that go away on their
#: own; a dialog only the person can answer does not.
TWICE_BEFORE_HANDING_BACK = 2

#: Beyond this the problem is not that she is missing a strategy.
MAX_RELEARNS = 2



class ScreenPursuitInput(BaseModel):
    goal: str = Field(..., min_length=1, max_length=400)
    #: Text that appearing on screen means the goal is reached. Matched
    #: case-insensitively against the reading, as a regular expression when it
    #: is one and as plain text otherwise.
    #:
    #: Allowed to be empty, because a request can name a process without
    #: naming an end — "play it and work out how it moves". Then the screen
    #: never says finished, ``goal_reached`` never fires, and the run ends on
    #: the cycle count and the clock it was given, which it always had.
    success_when: str = Field(default="", max_length=200)
    #: Restrict the match to a horizontal band of the screen, top-down, 0..1.
    #:
    #: Needed the moment this meets a real page. On play2048.co the word "2048"
    #: appears in the browser tab, the page heading and a welcome modal — all
    #: above y=0.15 — while the board tiles sit between y=0.25 and y=0.85. A
    #: whole-reading match for "2048" therefore succeeds before a single move
    #: is made, and the loop reports victory on the title.
    #:
    #: Nothing here is about 2048. Any page whose chrome repeats the word being
    #: waited for has the same problem: a build log inside a window titled with
    #: the branch name, a progress dialog in an app whose name contains
    #: "complete". The band is how a caller says "in the content, not the
    #: furniture".
    success_region_top: float = Field(default=0.0, ge=0.0, le=1.0)
    success_region_bottom: float = Field(default=1.0, ge=0.0, le=1.0)
    #: A control this task may click when something is blocking it and nothing
    #: generic is safe to press — "New Game", "Start", "Continue". The loop
    #: never infers this; the caller declares it.
    unblock_with: str = Field(default="", max_length=80)
    #: The moves this task is about, when they are not the arrow keys.
    #:
    #: A vocabulary is part of a task, not part of a screen. Stepping through
    #: a wizard is tab and return; a board is the arrows; a video is space.
    #: Offering every pressable key everywhere would put escape and return in
    #: front of a decision that has no business reaching for them.
    move_keys: list[str] = Field(default_factory=lambda: list(DEFAULT_MOVES), max_length=12)
    #: Where the task happens, when she has to get there first. A URL is
    #: opened; a name is searched for and the destination decided.
    open_page: str = Field(default="", max_length=300)
    #: How much rides on this run, 0..1. Above 0.7 the decision is sharpened
    #: by deep deliberation rather than a single amplified pass.
    stakes: float = Field(default=0.5, ge=0.0, le=1.0)
    #: A URL or title fragment identifying the page this run is about.
    #:
    #: Checked every cycle and restored by tab when it drifts. Without it a
    #: run cannot tell that the browser moved: a stray click sent one to a
    #: different site and it carried on reading and acting there, because
    #: appearance alone cannot answer "is this still the thing I was working
    #: on" — two pages can look alike and one page can change.
    expect_page: str = Field(default="", max_length=200)
    #: The application this run is driving. Its keystrokes are refused unless
    #: this application is frontmost at the moment of sending, so a run cannot
    #: type into whatever the person switched to. Empty means unaimed, which is
    #: only right when nothing is being driven.
    target_app: str = Field(default="", max_length=120)
    max_cycles: int = Field(default=200, ge=1, le=2000)
    max_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)
    #: When the whole thing must be over, on the same monotonic clock. Set by
    #: a caller that started counting before this action did.
    deadline_at: float = Field(default=0.0, ge=0.0)
    narrate: bool = Field(default=True)


ObservationPolicy = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]










async def read_screen(
    app_name: str = "", over: tuple[float, float, float, float] | None = None
) -> dict[str, Any]:
    """One reading: the words, and where they were.

    Scoped to `app_name`'s window when one is given, and this is not an
    optimisation.

    A full-screen reading is a reading of the DESKTOP, and a loop driving one
    application then sees every other application's text as if it belonged to
    the task. Measured live: a run driving a browser detected an "overlay" in
    another app's window, clicked at those screen coordinates, and pulled focus
    away from the thing it was driving — after which every keystroke was
    correctly refused and every reading described the wrong window. The loop
    had no way to notice, because to it the desktop and the task looked the
    same.

    Positions stay normalized 0..1, now against the WINDOW rather than the
    screen, so a caller's region band means "part of the thing I am driving"
    instead of "part of the display it happens to sit on" — which is what makes
    a band portable across window sizes and monitors.
    """
    from core.capabilities.host_automation import get_host_automation

    window = await window_bounds(app_name) if app_name else None
    # Read the thing at the size the thing is.
    #
    # A window is mostly not the task. Reading all of it spends the whole
    # picture on tabs, an address bar, an advertising rail and a footer, and
    # what is left for the part she is acting on is a few pixels a character —
    # which is how a board drawn on a canvas comes back as scattered noise.
    # LIVE 2026-08-29: 67 acts that moved something, and the best hypothesis
    # got 5 of them right, on a board that was drawn perfectly well.
    #
    # Given the part she is using, only that part is read, so every pixel of
    # the picture is spent on it.
    bounds = _the_part_of(window, over) if (window and over) else window
    receipt = await get_host_automation().get_screen_text(
        region=bounds, retain_screenshot=False
    )
    # Who the picture is actually of.
    #
    # A rectangle is not a window. The capture takes whatever is drawn in
    # that part of the display, so a window sitting over hers comes back
    # scoped to her app, the right shape, the right size, and full of
    # somebody else's words — and scoped_to, which exists to catch a reading
    # of the desktop, says it is hers because the bounds were found.
    #
    # LIVE 2026-09-04, driving the 2048 app: arrangements laid into her four
    # by four board reading ". The White House 12h Walker Kessler | .
    # ARCADE.GOV . Show more", learned from as though they were the game, and
    # every move across one of them recorded as having changed nothing.
    #
    # She already refuses to TYPE into a window that is not in front. This is
    # the same rule for looking, and it is the same reason.
    in_front, showing = _who_the_screen_belongs_to(app_name) if app_name else ("", True)
    return {
        "ok": bool(getattr(receipt, "success", False)),
        "text": str(getattr(receipt, "result", "") or ""),
        "layout": list(getattr(receipt, "layout", []) or []),
        "error": str(getattr(receipt, "error", "") or ""),
        "scoped_to": app_name if bounds else "",
        "bounds": list(bounds) if bounds else [],
        # What the positions in the layout are shares OF. A caller that asked
        # for part of a window gets positions within that part, and saying so
        # is what stops a band being applied twice.
        "read_within": "the part" if (window and over) else "the window",
        #: What was in front when the picture was taken. Empty when nobody
        #: asked for an application, which is the unaimed case.
        "in_front_then": in_front,
        #: Whether that application was drawing anything on the screen at
        #: all. A window on another Space keeps its position — the bounds
        #: come back exactly as before — while none of its pixels are on the
        #: display being photographed.
        "her_window_showing": showing,
        "at": time.time(),
    }

















async def current_page_identity() -> dict[str, str]:
    """Where the browser is right now, or empty strings. Never raises."""
    try:
        from core.capabilities.browser_controller import get_browser_controller

        return await get_browser_controller().current_page()
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as why:
        # "unavailable" was true of a browser that is not running, one that
        # refused, and one that is not installed, and told them apart for
        # nobody.
        return {"url": "", "title": "", "error": f"unavailable: {type(why).__name__}: {why}"}
















class ScreenPursuitSkill(BaseSkill):
    """Keep looking at the screen and acting until a goal is reached."""
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "pursue_on_screen"
    description = (
        "Pursue a goal on screen over many steps: read the screen, decide one "
        "action, do it, read again, and repeat until a described condition "
        "appears or a bound is reached. Use for anything that needs watching "
        "rather than a single action — waiting for a long job and reacting to "
        "how it ends, stepping through a wizard, or playing something. Give "
        "the goal and the text that means it is done."
    )
    input_model: type[BaseModel] | None = ScreenPursuitInput
    timeout_seconds: float = 900.0
    metabolic_cost: int = 6
    requires_approval = False
    # Same authority as computer_use: this presses keys on the foreground
    # desktop, and calling it anything gentler would be understating it.
    effect_scope = "foreground_desktop_control"

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = ScreenPursuitInput(**params)
        policy = (context or {}).get("screen_policy")
        return await pursue_on_screen(
            goal=params.goal,
            success_when=params.success_when,
            policy=policy,
            max_cycles=params.max_cycles,
            max_seconds=params.max_seconds,
            deadline_at=params.deadline_at,
            narrate=params.narrate,
            region_top=params.success_region_top,
            region_bottom=params.success_region_bottom,
            target_app=params.target_app,
            expect_page=params.expect_page,
            unblock_with=params.unblock_with,
            move_keys=tuple(params.move_keys or DEFAULT_MOVES),
            stakes=params.stakes,
            open_page=params.open_page,
        )



































#: The one heading every line she takes in a world is filed under.
#:
#: A move is graded against the KIND of position it was made from, because a
#: move that helps in a corner may not help in the middle. A line is not: it
#: is the thing she holds ACROSS positions, and grading it per position would
#: be grading it as if it were a move.
A_LINE_HERE = "the line to take here"

#: What a property she has just invented is worth, until a trial says
#: otherwise. The same weight the authored measures start at, because there is
#: no reason to trust one she wrote less than one she was given before either
#: has been tried.
WORTH_TRYING_AT = 0.4

























































async def _frontmost() -> str:
    """The application in front, for a run that was never told which one."""
    try:
        from core.capabilities.host_automation import get_host_automation

        receipt = await get_host_automation().get_frontmost_app()
        found = str(getattr(receipt, "result", "") or "").strip()
        return found if getattr(receipt, "success", False) else ""
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="acted without knowing the window"
        )
        return ""












#: What she learned in one world carries to another world of the same kind,
#: and what stays behind.
#:
#: A rule about how a thing moves is about the kind of thing, so it carries.
#: Which of her keys do anything, what has worked from a position of a given
#: shape, whether the world adds things of its own, which lines were worth
#: holding — all of those are about the kind too.
#:
#: What does not carry is everything about WHERE: which part of this screen
#: answers to her, the grid it is drawn on, which of its places move rather
#: than report, and how to get to it. Two worlds of a kind move alike; they
#: are not drawn alike, and carrying a band from one to another would put her
#: keystrokes into the wrong part of the second.
CARRIES_TO_A_WORLD_LIKE_IT = ("moves", "acts", "skill", "world", "lines", "lines_held")














#: How many looks are kept to work out what looking costs here. Enough that
#: one slow read does not move the bound, few enough that it follows a machine
#: that has become busy.
LOOKS_REMEMBERED = 12








#: The refusals that mean there IS a screen and she cannot tell what is on it,
#: which is a different situation from having nothing to look at: the thing
#: she wants may simply not be in front, and going to get it is hers to do.
SOMETHING_ELSE_IS_IN_FRONT = frozenset({"foreground_unknown", "browser_title_unknown"})


















async def _whats_on_top(
    mine: str, over: tuple[float, float, float, float] | None = None
) -> str:
    """The first thing above her work, or nothing. See :func:`_everything_on_top`."""
    above = await _everything_on_top(mine, over=over)
    return above[0] if above else ""


async def _everything_on_top(
    mine: str, over: tuple[float, float, float, float] | None = None
) -> tuple[str, ...]:
    """What is above her window AND over the part of it she is using.

    Read from the window server rather than from what claims to be frontmost:
    a dialog can sit above everything while the application underneath is
    still the frontmost one, which is exactly the case that fools every other
    check.

    ``over`` is the part of the screen that answers to her, as she worked it
    out. Without it, anything above her counts — which is the honest reading
    before she knows where the task lives, and the wrong one after.

    Being above her window is not being in her way. LIVE 2026-08-29, on
    play2048.co with the board found and read: a notification banner in the
    corner was reported as in front of the game, she pressed Escape at it
    three times, it stayed where it was, and the run ended having made no
    moves — "nothing on screen offered a move" — with the board untouched and
    entirely visible the whole time.
    """
    try:
        import Quartz  # noqa: PLC0415

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        hers = await window_bounds(mine) if over else None
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="could not see what is on top"
        )
        return ""
    ours = str(mine or "").strip().lower()
    above: list[str] = []
    for window in windows or []:
        try:
            layer = int(window.get("kCGWindowLayer", 0) or 0)
            owner = str(window.get("kCGWindowOwnerName", "") or "")
        except (TypeError, ValueError, AttributeError):
            continue
        # Ordinary windows sit at zero. Anything above that is over everything,
        # including hers, whatever claims to be frontmost.
        if not (0 < layer < ABOVE_EVERYTHING and owner and owner.lower() != ours):
            continue
        if over is not None and hers and not _covers(window, over, hers):
            continue
        above.append(owner)
    return tuple(sorted(set(above), key=above.index))


#: Where the menu bar, the dock and the system's own furniture live. A window
#: above ordinary windows but below these is something put in front of her.
ABOVE_EVERYTHING = 20






async def pursue_on_screen(
    *,
    goal: str,
    success_when: str,
    policy: ObservationPolicy | None = None,
    think: Any = None,
    move_keys: Sequence[str] = DEFAULT_MOVES,
    max_cycles: int = 200,
    max_seconds: float = PURSUIT_SECONDS,
    deadline_at: float = 0.0,
    narrate: bool = True,
    region_top: float = 0.0,
    region_bottom: float = 1.0,
    target_app: str = "",
    expect_page: str = "",
    unblock_with: str = "",
    stakes: float = 0.5,
    open_page: str = "",
    research: bool = True,
    lived: bool = True,
    spine: Any = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Run the loop. Returns the receipt the executor produced.

    With neither ``policy`` nor ``think``, the loop decides through her own
    reasoning: :func:`core.agency.deliberate_action.deliberate` picks from the
    moves that are really available, predicts what should be different once
    the move lands, and the next reading checks that prediction. A broken
    prediction is fed back as evidence, which is what lets a run change its
    mind rather than repeat a move that does nothing.

    ``policy`` still overrides everything, for a caller that has its own
    judgement or a test that needs a fixed one. ``think`` replaces only the
    reasoning, keeping the predict-and-check loop around it.
    """
    from core.agency import what_she_is_doing as doing
    from core.agency.deliberate_action import Attempt
    from core.agency.how_good_is_this import (
        AS_GOOD_A_GUESS_AS_ANY,
        INVENTED,
        promote,
    )
    from core.agency.how_good_is_this import a_trial_is_running as _a_trial_is_running
    from core.agency.inventing_a_measure import measure_named
    from core.agency.standing_strategy import Strategy
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.agency.what_i_cannot_explain import WhatICannotExplain
    from core.agency.what_makes_it_good_here import WhatMakesItGoodHere
    from core.agency.what_worked_before import WhatWorkedBefore
    from core.perception.how_it_moves import HowItMoves
    from core.perception.the_lattice_she_holds import TheLatticeSheHolds
    from core.perception.what_moves_within_itself import MovesWithinItself
    from core.perception.what_the_world_does import WhatTheWorldDoes
    from core.perception.where_it_responds import (
        Responsive,
        describe,
        what_is_there,
    )
    from core.skills.fluid_executor import FluidExecutor, Step
    from core.world_model.unified_world_model import UnifiedWorldModel

    # The clock starts when she takes this on, not when the first key lands.
    #
    # Finding the page and opening it is part of the task and used to be free:
    # the pursuit's own budget began afterwards, so the outer deadline — the
    # one that only has room to report — was reached first and won. Live
    # 2026-08-26: sixty-five narrated moves, nine approaches held, cancelled
    # from outside and reported as "Completed 0/0 steps".
    began = time.monotonic()
    ends_at = began + float(max_seconds)
    if deadline_at > 0.0:
        ends_at = min(ends_at, float(deadline_at))
    if not await wait_for_a_screen_to_look_at(ends_at):
        why = _WHY_SHE_CANNOT_LOOK["value"]
        return {
            "ok": False,
            # Which of them it was, because they do not have the same remedy
            # and the person is the one who can apply it.
            "outcome": (
                "something_else_is_in_front"
                if why in SOMETHING_ELSE_IS_IN_FRONT
                else "not_allowed_to_look"
                if why and why not in PASSES_ON_ITS_OWN
                else "no_screen_to_look_at"
            ),
            "refused_because": why,
            "error": _what_being_refused_a_look_means(why),
            "moves": [],
        }
    moves: list[dict[str, Any]] = []
    history: list[Attempt] = []
    #: Situations she has been in and whether things went well from there,
    #: which is what a property gets weighed against. Watching somebody clear
    #: 2048 in 989 moves and beat a hard checkers engine while three pieces
    #: down, the difference was never how far they searched. It was that they
    #: had decided on something to keep true, and most of what she considers
    #: they never considered at all.
    went: list[tuple[Any, bool]] = []
    #: What she has settled on keeping true, and why. Named apart from the
    #: line she is taking, which `decide` already calls `holding`.
    she_keeps: dict[str, Any] = {"it": None, "why": "", "at": -1}
    #: Situations she was in and how things went from there, so a property her
    #: measure cannot account for has somewhere to come from. See
    #: core/agency/what_i_cannot_explain.py.
    cannot_explain = WhatICannotExplain()
    #: The property she is currently trying out, if any. Nothing replays a
    #: life, so a change to her own judgement is tried IN one.
    #
    # Read from where trials actually live, not started blank. A trial takes
    # sixty observations, which is more than one run, so a handle kept in the
    # run that started it is lost the moment that run ends — and the property
    # went on being used with no verdict ever reached about it.
    trying: dict[str, Any] = {"name": _a_trial_is_running()}
    #: Where she stood when this run began, so what it came to can be reported
    #: as a rate rather than a total. A total says where the window sat.
    began_at: dict[str, Any] = {"worth": None, "seen": 0}
    #: The largest thing she has ever made in this world. A watcher of a run
    #: wants to know when it goes past what she has managed here before, and
    #: she has the fact and has never said it.
    #: ``here`` is what she brought in, ``again`` is the biggest she has seen
    #: THIS sitting. A carried record is provisional until something at least
    #: that big turns up again — see below.
    furthest: dict[str, float] = {"here": 0.0, "again": 0.0}
    #: Where the page says it draws, asked once when the run gets its bearings.
    drawn: dict[str, Any] = {"where": None, "asked": False}
    #: Why the last cycle ended without a move. Eleven different facts used
    #: to arrive at the executor as one silence, and the run then reported
    #: "nothing on screen offered a move" for every one of them — which cost
    #: three wrong diagnoses in a row before this line existed.
    no_move: dict[str, str] = {"because": ""}
    #: The picture of the surface once it had come to rest, kept for the
    #: reading that would otherwise be taken of exactly the same thing.
    #:
    #: Waiting for a move to land means watching until the screen changes and
    #: then stops changing, which is two readings at least. The next cycle
    #: then took a third of the same still surface, and a reading is a
    #: screenshot and an OCR — about a third of the whole cost of a move.
    at_rest: dict[str, Any] = {"reading": None}
    #: How long her last few looks took, so a busy machine is not mistaken
    #: for a wedged one.
    reading_took: list[float] = []
    #: Whether a restart control has APPEARED — turned up where there was
    #: none — which is a thing saying it has finished.
    #:
    #: ``was_there`` is what the first reading of this task showed, and it is
    #: the whole point: a control that has been on screen since before she
    #: made a move is furniture, not an ending. LIVE 2026-09-04: the 2048
    #: desktop app keeps a "New Game" button above the board at all times, so
    #: the presence test fired on cycle one, every cycle, and a run of twelve
    #: cycles pressed no key at all and clicked New Game eleven times. Every
    #: form with a Reset, every wizard with Start Over, every game with a
    #: permanent restart is the same page.
    #:
    #: ``said`` keeps it said once rather than every cycle.
    offered_a_restart: dict[str, Any] = {"was_there": None, "said": False}
    pending: dict[str, Any] = {
        "deliberation": None,
        "before": "",
        "arranged": None,
        # The whole reading, uncropped. The shape of the NEXT reading is held
        # against this one — a board whose top row is empty has no top row to
        # infer, so without a previous reading to place it in, the thing
        # changes shape under her and no rule can survive the comparison. The
        # cropped board beside it is what she learns from; this is what keeps
        # the two readings comparable.
        "whole": None,
        "watched": {},
    }
    # One surface for questions about what a world would do. The rules facet
    # is per-run on purpose: a rule that held on one thing is a guess about
    # the next one, and she should find out rather than assume.
    # What she worked out about this thing last time she was in it.
    #
    # Everything she learns about a world — which part answers to her, how it
    # moves when she pushes it — has been dying with the process, so the
    # fortieth run started as ignorant as the first. Brought back discounted:
    # something she worked out yesterday is evidence about today and not a
    # fact about it, and a few acts that disagree should overturn it.
    this_world = named(target_app, expect_page or open_page)
    knew = recall(this_world)
    # And whether she has been anywhere LIKE it, asked once she has seen it.
    like_it: dict[str, Any] = {"kind": "", "looked": False}
    # Moves she made and could not learn from, by what stopped her.
    # A tally, not a fixed set of keys.
    #
    # It was a plain dict holding the two reasons that had ever fired, and
    # two later reasons were written as `dropped[name] += 1` against keys
    # that were not in it. Both raise KeyError out of the middle of the
    # deciding step — so a branch written to protect the learner took the
    # whole run down the first time it was reached. LIVE 2026-09-04: it had
    # never been reached, because the count it was gated on was never set.
    dropped: Counter[str] = Counter()
    if knew:
        logger.info("she has been in %r before: %s", this_world, sorted(knew))
    # Which of her acts do anything here, found out rather than declared.
    #
    # A solver written for one thing is handed its action set. She was handed
    # hers the same way, and it was the last large thing about an unfamiliar
    # world that somebody else was still establishing for her.
    can_do = WhatWorksHere.from_memory(knew.get("acts") or {}, told=tuple(move_keys))
    knows = UnifiedWorldModel(
        rules=HowItMoves.from_memory(knew.get("moves") or {}, TRUST_CARRIED_OVER)
    )
    # And what she has got GOOD at here, which is a different thing from what
    # she has worked out. Knowing exactly how a world moves and still paying
    # the full price of deciding every time is what she was doing.
    skilled = WhatWorkedBefore.from_memory(knew.get("skill") or {}, TRUST_CARRIED_OVER)
    # And what the world does between her acts, which she was tolerating
    # without ever learning. A future worked out as if the world sits still is
    # a future that cannot happen.
    world = WhatTheWorldDoes.from_memory(knew.get("world") or {}, TRUST_CARRIED_OVER)
    # And which of the lines she has taken here actually left her better off.
    #
    # She wrote down the last approach she happened to be holding when a run
    # ended, and nothing ever read it back — a writer with no reader, storing
    # the most recent line rather than the one that worked. Graded the same
    # way a move is, and resumed as a stance rather than as words, so the
    # first reading of the new run can drop it.
    lines = WhatWorkedBefore.from_memory(knew.get("lines") or {}, TRUST_CARRIED_OVER)
    lines_held: dict[str, Any] = dict(knew.get("lines_held") or {})
    undecided: dict[str, str] = {"reason": ""}
    #: Why the screen could not be read, when that is what stopped her.
    #:
    #: "Nothing on screen offered a move" and "I cannot see the screen" are
    #: different facts and only one of them is about the screen's contents.
    #: LIVE 2026-08-27: nine runs reported the first while the truth was the
    #: second — the machine had no interactive session, so no capture was
    #: possible at all, and every layer above read that as an empty board.
    cannot_see: dict[str, str] = {"reason": ""}
    #: She decided to play this attempt out rather than restart it.
    seen_through: dict[str, Any] = {"value": False, "because": ""}
    #: The finishing condition was already met on the first reading.
    already: dict[str, bool] = {"value": False}
    #: What she has just deliberately decided to do, so a dialog confirming
    #: that decision is not read as an ambush. Named apart from the policy's
    #: own "intent" local, which made this one local to decide() and unbound.
    intending: dict[str, str] = {"value": ""}
    #: Attempts she chose to begin again, and why.
    restarts: dict[str, Any] = {"count": 0, "because": ""}
    foreseen: dict[str, bool] = {"said": False}
    #: The last call about whether a decision was worth thinking over, so a
    #: standing answer is not said again every cycle.
    last_call: dict[str, Any] = {"asked": None, "why": ""}
    #: How she has decided to handle acting faster than she can speak.
    pacing: dict[str, Any] = {"choice": "", "because": "", "brief": False, "waits": 0}
    #: When language was last consulted, so it is asked where it counts.
    asked: dict[str, int] = {"at": -LANGUAGE_EVERY, "after_restarts": 0}
    #: What she knows about doing this, learned once at the start and again
    #: whenever what she is doing stops working.
    knowledge: dict[str, Any] = {"held": None, "relearned": 0, "meant": []}
    # The approach she is taking, and how often she has had to change it.
    plan: dict[str, Any] = {"held": None, "changes": 0, "asked_at": -1}

    async def observe() -> dict[str, Any]:
        """Lifted to screen_pursuit_observing.py; the scope is handed over per call."""
        return await observe_the_screen(
            SimpleNamespace(
                anchor=anchor,
                at_rest=at_rest,
                drawn=drawn,
                expect_page=expect_page,
                lost_page=lost_page,
                open_page=open_page,
                reading_took=reading_took,
                target_app=target_app,
            ),
        )

    def satisfied(observation: dict[str, Any]) -> bool:
        reached = goal_reached(
            observation,
            success_when,
            region_top=region_top,
            region_bottom=region_bottom,
        )
        # True before she did anything is not something she did.
        #
        # A run that reports success off its first reading has not achieved
        # the goal, it has found the condition already met — which usually
        # means the condition is describing something other than the thing
        # being waited for. Measured live: asked to play until a 128 tile,
        # she opened the game, matched the number in the score, and reported
        # the goal reached in 1.2 seconds without a move.
        if (
            reached
            and not moves
            and not restarts["count"]
            and not blocker_attempts["dismissed"]
        ):
            already["value"] = True
        return reached

    async def clear_blocker(observation: dict[str, Any]) -> Step | None:
        """Lifted to screen_pursuit_blockers.py; the scope is handed over per call."""
        return await clear_what_blocks_the_run(
            observation,
            SimpleNamespace(
                anchor=anchor,
                intending=intending,
                needs_person=needs_person,
                target_app=target_app,
                unblock_with=unblock_with,
            ),
        )

    #: ``count`` is the consecutive run, reset the moment the screen is clear,
    #: because that is what decides whether dismissal is working. ``dismissed``
    #: only goes up: clearing a banner IS something she did, and without a
    #: record that survives the reset, a goal met on the very next reading was
    #: reported as "already true before she started" — completed False on a
    #: pursuit that had just succeeded.
    blocker_attempts = {"count": 0, "last": "", "dismissed": 0}
    lost_page = {"value": False}
    needs_person: dict[str, Any] = {"reason": "", "seen": "", "times": 0}
    #: The page this run belongs to, learned on the first cycle when the caller
    #: did not name one.
    anchor: dict[str, str] = {"page": expect_page.strip(), "app": target_app.strip()}
    #: Where her actions have been having their effects.
    responds: dict[str, Any] = {
        "state": Responsive.from_memory(knew.get("responds") or {}, TRUST_CARRIED_OVER),
        # Answering to her is not the same as being the thing. A score, a move
        # counter and a clock all answer to her, and in one frame they look
        # exactly like a board. What a board does that none of them do is show
        # her values that were already in it.
        "moving": MovesWithinItself.from_memory(
            knew.get("moves_within") or {}, TRUST_CARRIED_OVER
        ),
        # The grid itself, held between glances rather than worked out from
        # each one. Where a thing is tends to keep, so it is remembered.
        "lattice": TheLatticeSheHolds.from_memory(knew.get("lattice") or {}),
    }
    # And what she remembers about how the thing moves is dropped where it was
    # read through a grid of a different shape from the one she is holding.
    #
    # Dropping it DURING a run, when the grid corrects itself, was not enough.
    # The counts are written down at the end and read back at the start, so a
    # sitting that began with the wrong grid poisons every sitting after it,
    # and the correction never fires because by then the grid is already
    # right. LIVE 2026-09-02: three sittings running, the grid correct in the
    # last two, the rule stuck at two hundred and five right out of three
    # hundred and one, and not one move looked ahead on in any of them.
    #
    # Counts that cannot say what grid they were read through are dropped too.
    # Evidence that cannot say what it was gathered under cannot be known to
    # still apply, and this particular evidence decides whether she looks
    # ahead at all — a wrong one costs her every move of every sitting, and
    # relearning a right one costs about thirty moves of one.
    if knows.rules is not None and responds["lattice"].held:
        grid = (responds["lattice"].rows, responds["lattice"].columns)
        if knows.rules.read_through != grid:
            knows.rules.learned_through_a_different_reading()
            # And everything else that was read through the same grid. What
            # worked before is looked up by the SHAPE of the situation, and a
            # shape written down under a four by seven grid names nothing
            # under a four by four one. What the world does on its own is
            # learned by comparing two arrangements, so it is the same story:
            # one world file on this machine has it believing that "History"
            # and "Help" turn up on their own, which is the browser's menu bar
            # read as though it were the game.
            skilled.forget_what_was_read_differently()
            world.forget_what_was_read_differently()
    #: The few acts she reaches for, out of all the ones she could. A habit is
    #: worth carrying between sittings, so it is remembered too.
    reaches = TheOnesSheReachesFor.from_memory(
        knew.get("reaches") or {}, TRUST_CARRIED_OVER
    )
    #: How long her acts take, and what the world does while they run. A jump
    #: in Ghosts 'n Goblins cannot be called off, and for that second the
    #: knight cannot answer anything — so the decision to jump is a decision
    #: to be unable to respond, taken while knowing what is coming. Pressing a
    #: key and starting something that runs for five minutes were the same
    #: kind of move to her, and the difference is the whole of it.
    busy = WhatItCostsToBeBusy()
    #: What a language pass costs and what a whole cycle costs, both measured
    #: here, so "is this worth thinking about" can weigh the price.
    costs: dict[str, float] = {
        "pass_s": 0.0, "passes": 0.0, "quiet_s": 0.0, "quiet": 0.0, "at": 0.0,
        # What a whole cycle takes, so a pass has something to be dear
        # against before she has ever gone without one.
        "cycle_s": 0.0, "cycles": 0.0,
    }
    #: How far she has got into this before, and where it stopped. A player on
    #: their sixth go at Ninja Gaiden is not reacting — they are replaying
    #: what they know and thinking only where they died last time.
    got_to = TheFurthestSheHasGot.from_memory(knew.get("got_to") or {})
    # What each thing about a situation is worth HERE, rather than the
    # standing guess. The organ was written, tested and never called by
    # anything: she judged every world by the same five numbers somebody
    # picked, including the worlds she had played for hours.
    matters = WhatMakesItGoodHere.from_memory(
        knew.get("matters") or {}, TRUST_CARRIED_OVER
    )
    # And the properties she invented here and proved worth steering by.
    #
    # A measure she composed lived in the process that composed it. Inventing
    # one takes a run's worth of unexplained pairs and proving one takes sixty
    # observations under trial — more than a run — so every restart threw away
    # the only part of her judgement that was hers rather than authored, and
    # the next run started composing it again from nothing.
    #
    # A measure is four words and a weight; putting it back is looking its
    # name up in the space it came from.
    for name, worth in (knew.get("judging") or {}).items():
        try:
            found = measure_named(str(name))
        except (AttributeError, TypeError, ValueError):
            continue
        if found is not None:
            promote(found, float(worth))
            logger.info("she judges situations here by %r, worth %.2f", name, float(worth))
    try:
        furthest["here"] = float(knew.get("furthest") or 0.0)
    except (TypeError, ValueError):
        # not a failure: a record that is not a number is not a record.
        furthest["here"] = 0.0
    #: Places she has been, marked, so the way back is on the ground rather
    #: than in her head.
    marks = MarksOnTheGround.from_memory(knew.get("marks") or {})
    #: Which of her acts started working once something else was true. An act
    #: that does nothing is written down as an act that does nothing HERE,
    #: which is true and useless — the interesting thing is that it would work
    #: given one thing, and that the thing is gettable. Without it a wall is a
    #: wall for ever and the way round it is never something to go and fetch.
    opens = WhatOpensWhat.from_memory(knew.get("opens") or {})
    #: What has turned up before, how often, and what it wanted — so the work
    #: is done in the light rather than when the sun is already down.
    coming = WhatUsuallyComes.from_memory(knew.get("coming") or {})
    #: How many uses each act has of its own. The strong move runs out while
    #: the weak one is still there, and a fight is lost by somebody who never
    #: lost a turn.
    supply = WhatEachActHasLeft()
    #: What she has started and not seen the end of. Sending a fourth fleet to
    #: a place three are already arriving at is not caution, it is doing the
    #: thing twice.
    in_flight = WhatIsComing()
    #: How her runs here have ended, so she can play for the one she can
    #: bring about rather than the one that sounds best. Kept as the shapes
    #: she passed through and the word it finished on.
    endings: list[tuple[list[str], str]] = [
        (list(one.get("shapes") or []), str(one.get("ended") or ""))
        for one in (knew.get("endings") or [])
        if isinstance(one, dict)
    ]
    #: Which of her acts has gone well against which kind of situation. What
    #: works HERE dies with the place; what works generally averages over
    #: places with nothing in common. Neither can say the thing that is true.
    beats = WhatBeatsWhat.from_memory(knew.get("beats") or {})
    #: Whether this world is the same every time. Memorising a shuffled world
    #: fills her with facts that will not recur; playing a fixed one by policy
    #: throws away the thing that would have made it easy.
    repeats = DoesItRepeat.from_memory(knew.get("repeats") or {})
    #: How the stretch in progress is going. A stretch is over once the score
    #: has moved as many times as there are ways of leaning to compare, which
    #: is how long it takes for the comparison to be worth making and is read
    #: off the size of the choice rather than picked.
    stretch: dict[str, int] = {"rises": 0}
    #: How far she is willing to go on the model between looks, and the board
    #: she expects to find when she next looks.
    far = HowFarToGo()
    expected: dict[str, Any] = {"after": None, "took": 0}
    #: Said once, when the thing she is working in stops answering at all.
    said_it_ended: dict[str, bool] = {"value": False}
    #: Whether she has confirmed being in the thing she was asked to act in.
    confirmed_here: dict[str, bool] = {"value": False}
    not_there: dict[str, str] = {"reason": ""}
    #: What she last tried to move out of her way, so a thing that will not
    #: close is not pressed at once a cycle for the rest of the run.
    in_the_way: dict[str, str] = {"last": ""}

    async def decide(observation: dict[str, Any]) -> Step | None:
        """Lifted to screen_pursuit_decision.py; the scope is handed over per call."""
        return await decide_the_next_move(
            observation,
            SimpleNamespace(
                anchor=anchor,
                asked=asked,
                at_rest=at_rest,
                beats=beats,
                began=began,
                began_at=began_at,
                blocker_attempts=blocker_attempts,
                busy=busy,
                can_do=can_do,
                cannot_explain=cannot_explain,
                clear_blocker=clear_blocker,
                coming=coming,
                confirmed_here=confirmed_here,
                costs=costs,
                drawn=drawn,
                dropped=dropped,
                ends_at=ends_at,
                expect_page=expect_page,
                expected=expected,
                far=far,
                foreseen=foreseen,
                furthest=furthest,
                goal=goal,
                got_to=got_to,
                graph=graph,
                history=history,
                in_flight=in_flight,
                in_the_way=in_the_way,
                intending=intending,
                knowledge=knowledge,
                knows=knows,
                last_call=last_call,
                like_it=like_it,
                lines=lines,
                lines_held=lines_held,
                lived=lived,
                marks=marks,
                matters=matters,
                move_keys=move_keys,
                moves=moves,
                narrate=narrate,
                needs_person=needs_person,
                no_move=no_move,
                not_there=not_there,
                observe=observe,
                offered_a_restart=offered_a_restart,
                open_page=open_page,
                opens=opens,
                pacing=pacing,
                pending=pending,
                plan=plan,
                policy=policy,
                reaches=reaches,
                region_bottom=region_bottom,
                region_top=region_top,
                repeats=repeats,
                research=research,
                responds=responds,
                restarts=restarts,
                said_it_ended=said_it_ended,
                seen_through=seen_through,
                she_keeps=she_keeps,
                skilled=skilled,
                spine=spine,
                stakes=stakes,
                stretch=stretch,
                success_when=success_when,
                target_app=target_app,
                think=think,
                trying=trying,
                undecided=undecided,
                went=went,
                world=world,
            ),
        )

    # Get to where the task happens before looking at anything.
    #
    # A goal that names a place is not doing anything until she is there, and
    # somebody else opening the page first is the part of the task she should
    # be doing herself. Naming a URL makes it an open; naming only the thing
    # makes it a search and then a decision about which result really is it.
    reached = None
    if open_page:
        from core.agency.reach_place import reach

        reached = await reach(open_page, think=think, lived=lived, purpose=goal, graph=graph)
        if not reached.arrived:
            return {
                "goal": goal,
                "completed": False,
                "outcome": "could_not_get_there",
                "could_not_get_there": reached.reason,
                "wanted": open_page,
                "considered": list(reached.considered),
                "moves": [],
                "attempts": [],
                "success_when": success_when,
            }
        if not expect_page:
            # The run belongs to the page she just opened.
            expect_page = reached.url or reached.title
            anchor["page"] = expect_page
        if not target_app and reached.app:
            # And to the application that page is in.
            #
            # Opening a page does not put it in front. Measured live: she
            # found the game, opened it, and then read whatever window
            # happened to be frontmost — reporting that nothing on screen
            # offered a move, half a second in, having never seen the board.
            target_app = reached.app
    elif target_app:
        # An application is a place too, and arriving at one was the only
        # arrival nobody made.
        #
        # `reach` puts a page in front before anything looks at it, and the
        # docstring above says why: a goal that names a place is not doing
        # anything until she is there. An application named instead of a page
        # got none of that. It was raised only reactively — when a reading
        # came back covered, or when the thing changed shape — so the first
        # look, the first decision and the first keystroke all happened
        # against whatever was already frontmost.
        #
        # LIVE 2026-09-01: asked to play 2048, the run anchored to the
        # installed "2048 Game", never launched it, read a job posting that
        # was open in a browser behind everything, reasoned about its "Did you
        # finish applying?" prompt, and made zero moves in 176 seconds.
        if not await _bring_the_thing_back_to_the_front(target_app):
            return {
                "goal": goal,
                "completed": False,
                "outcome": "could_not_get_there",
                "could_not_get_there": (
                    f"{target_app!r} would not come to the front and would not start"
                ),
                "wanted": target_app,
                "considered": [],
                "moves": [],
                "attempts": [],
                "success_when": success_when,
            }

    # A goal already met by what was left behind is a decision, not a finish.
    #
    # Someone else's finished board satisfies "play until a 128 tile" without
    # her having played, and stopping there hands back a result she did not
    # produce. Whether to accept it or begin again is hers, made the same way
    # every other choice is, and it is only offered when there is really a
    # way to begin again.
    # The line that worked here before, resumed as a stance rather than as
    # words. She holds it from the first move, and the ordinary machinery
    # tests it against what is really here: still_holds drops it the moment
    # its condition stops being true, exactly as it would drop a fresh one.
    if plan["held"] is None:
        worked = lines.suggests(A_LINE_HERE)
        if worked:
            again = Strategy.from_memory(lines_held.get(worked) or knew.get("approach") or worked)
            if again is not None:
                plan["held"] = again
                logger.info("taking up the line that worked here before: %r", again.approach)

    # Before anything else, put what she came for on screen.
    #
    # A page is taller than a screen, and what she came for is usually below
    # the writing about it. Without this she reads the heading and the
    # advertising, finds no part of what she can see that answers to her, and
    # says so — truthfully, with the thing itself further down.
    await _bring_it_into_view(
        observe, lambda seen: what_is_there(seen, None, like=None), cannot_see
    )

    if not moves:
        first = await observe()
        if first.get("ok") and satisfied(first):
            already["value"] = False
            fresh = restart_control(first)
            if fresh is not None:
                from core.agency.deliberate_action import deliberate as _decide

                settle = await _decide(
                    goal,
                    f"the finishing condition ({success_when}) is already true, "
                    "and nothing here was done by me",
                    ways_out(first),
                    think=_within_the_run(think or _her_reasoning(stakes), ends_at),
                    control_point="screen_pursuit.pre_met",
                    lived=lived,
                    spine=spine,
                    graph=graph,
                )
                if settle.reached and settle.chosen is not None and settle.chosen.name == START_OVER:
                    label, rx, ry = fresh
                    intending["value"] = START_OVER
                    frame = list(first.get("bounds") or [])
                    if await click_normalized(rx, ry, expect_app=target_app or anchor["app"], bounds=frame):
                        # Look again before judging anything, and before
                        # claiming anything.
                        #
                        # A reset takes a moment to land, and the reading
                        # taken before it did is of the game she just
                        # abandoned. A click that lands on nothing still
                        # reports success — the click happened — so waiting
                        # is not enough on its own. Measured live: "Began
                        # again 1 time(s)" while the score sat unchanged at
                        # 996 the whole time.
                        _after, began = await _answer_own_confirmation(
                            first, target_app, label
                        )
                        if began:
                            restarts["count"] += 1
                            restarts["because"] = (
                                settle.rationale or "the goal was already met by an old game"
                            )
                        else:
                            already["value"] = True
                    else:
                        already["value"] = True
                else:
                    already["value"] = True
            else:
                already["value"] = True

    # Narration runs beside the pursuit, never inside it.
    #
    # Asking for it starts a separate faculty that listens to the global
    # workspace and speaks; the loop below offers its decisions there and
    # carries on regardless. Not asking for it changes nothing else — silent
    # play is the absence of a narrator, not a different code path.
    #
    # Scoped to this run's decisions by default, because a caller asking for
    # a running commentary on a game wants that and not everything she is
    # thinking. A narrator started with no scope narrates whatever reaches
    # her, which is the same faculty doing the more general thing.
    speaker = None
    if narrate:
        try:
            from core.agency.narrator import Narrator

            speaker = Narrator(say=_say_line, about="screen_pursuit.next_move")
            speaker.start()
        except (ImportError, RuntimeError, AttributeError, TypeError) as exc:
            record_degradation(
                "screen_pursuit",
                exc,
                severity="info",
                action="pursued the goal without narrating it",
            )
            speaker = None

    # She has taken something on, and the rest of her should know it.
    doing.taking_on(goal, where=target_app or "")
    executor = FluidExecutor(verifier=None, gateway=None)
    # And it holds the foreground while it runs.
    #
    # A task somebody asked for is foreground for as long as it takes, not for
    # the length of the sentence that started it. Her background thinking
    # already stands aside for a foreground turn — it checks — and a turn ends
    # the moment the reply is composed, so everything she does AFTER that,
    # which is the whole of the task, ran as background beside her own
    # research loops.
    #
    # Live 2026-09-07, playing a game on this machine: thirty-two decisions
    # about what to do next, nine of them refused outright because the
    # inference lanes were exhausted — by her own reimplementation lab and
    # curriculum loop, running against the model she needed to choose a move.
    holding_the_foreground = None
    try:
        from core.runtime.foreground_guard import begin_foreground_turn  # noqa: PLC0415

        holding_the_foreground = begin_foreground_turn(
            owner="screen_pursuit", source="desktop_task"
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info",
            action="pursued a task without holding the foreground",
        )
    try:
        receipt = await executor.pursue(
            goal,
            observe=observe,
            decide=decide,
            is_satisfied=satisfied,
            max_cycles=max_cycles,
            max_seconds=max(1.0, ends_at - time.monotonic()),
            perception_reason=f"pursuing on screen: {goal[:60]}",
        )
    finally:
        if speaker is not None:
            await speaker.stop()
        if holding_the_foreground is not None:
            holding_the_foreground.close()
    result = receipt.to_dict()
    if blocker_attempts["count"] >= MAX_BLOCKER_ATTEMPTS and not receipt.completed:
        # Say what stopped it. "out_of_cycles" describes the budget running
        # out; this describes the reason, which is the part a caller can act
        # on — by dismissing it themselves, or by deciding the dialog is the
        # task.
        result["outcome"] = "blocked_by_overlay"
        result["blocked_by"] = blocker_attempts["last"]
    result["moves"] = moves
    if already["value"]:
        # Said plainly rather than claimed. The person asked her to do
        # something and the condition was true before she started, so what
        # they get is that fact and not a receipt for work nobody did.
        result["already_true_at_the_start"] = success_when
        result["outcome"] = "already_true"
        result["completed"] = False
    result["restarts"] = restarts["count"]
    if pacing["choice"] or pacing["waits"]:
        result["pacing"] = {
            "chose": pacing["choice"],
            "because": pacing["because"],
            "waited": pacing["waits"],
        }
    if restarts["because"]:
        result["restarted_because"] = restarts["because"]
    if seen_through["value"]:
        result["played_out_because"] = seen_through["because"]
    result["attempts"] = [
        {"option": a.option, "expected": a.expected, "held": a.verdict.held, "why": a.verdict.why()}
        for a in history
    ]
    where_it_answers = responds["state"].band()
    if responds["state"].nothing_answers():
        result["stopped_responding"] = True
    if where_it_answers is not None:
        result["responds_within"] = [round(edge, 3) for edge in where_it_answers]
        result["responds_described"] = describe(where_it_answers)
    if plan["changes"]:
        result["changed_approach"] = plan["changes"]
    if plan["held"] is not None:
        result["approach"] = plan["held"].approach
    # A property nobody wrote, if what she could not explain calls for one.
    #
    # At the end, because the outcome of a situation is what came after it and
    # that is only known once the run is over. Put on TRIAL rather than
    # adopted: she cannot replay a life to A/B a change to her own judgement,
    # so the trial happens in the next one and the property keeps its place
    # only if things actually go better with it.
    if not trying["name"]:
        from core.agency.how_good_is_this import on_trial, what_it_was_like_before

        ended_at = pending["arranged"]
        finished_on = sum(ended_at.numbers() or (0.0,)) if ended_at is not None else 0.0
        # The rate this run moved at, which is what the trial will produce.
        # A baseline measured as a total and a trial measured as a rate
        # compares nothing.
        how_it_went = (
            (finished_on - float(began_at["worth"])) / max(1, began_at["seen"])
            if began_at["worth"] is not None
            else 0.0
        )
        for measure, held_back, _pairs in cannot_explain.worth_trying(most=1):
            name = on_trial(measure, WORTH_TRYING_AT)
            if name:
                what_it_was_like_before(name, how_it_went)
                trying["name"] = name
                if narrate:
                    _tell(
                        f"There is something my measure could not account for. "
                        f"I am going to try judging by {name!r} and see whether "
                        f"it does better."
                    )
                logger.info(
                    "put %r on trial — it accounted for %.0f%% of what she could not",
                    name, held_back * 100.0,
                )
            break

    # Which way of finishing she has a route to.
    #
    # Somebody beating a hard checkers engine chose a way of winning that
    # gives pieces away, and said outright it was not the best strategy — it
    # was the one they could reach against that opponent. Most things that
    # play a game never make that decision: they score a position and take the
    # best number, which quietly fixes the ending as whichever the score was
    # built around.
    if len(endings) > 1:
        ways = {
            one: (lambda finish, want=one: finish == want)
            for one in {ended for _shapes, ended in endings if ended}
        }
        if len(ways) > 1:
            for way in which_way_to_win(ways, endings)[:1]:
                logger.info("the ending she has a route to: %s", way.describe())

    # What she worked out about this thing, for the next time she is in it.
    remember(
        this_world,
        {
            "responds": responds["state"].as_memory(),
            # The largest thing she has made here — the carried record only
            # where this sitting saw something at least as big, so a reading
            # that was wrong once cannot stand for ever.
            "furthest": (
                furthest["here"]
                if furthest["again"] >= furthest["here"]
                else furthest["again"]
            ),
            # And what each thing about a situation turned out to be worth.
            "matters": matters.as_memory(),
            # The properties she composed here and kept, by name and weight.
            "judging": {
                str(name): float(AS_GOOD_A_GUESS_AS_ANY.get(str(name), 0.0))
                for name in INVENTED
            },
            # Which of the places that answer to her are the thing itself,
            # rather than a score reporting on it. Kept, because working it
            # out costs her several moves of a fresh game every time.
            "moves_within": responds["moving"].as_memory(),
            "lattice": responds["lattice"].as_memory(),
            "reaches": reaches.as_memory(),
            "got_to": got_to.as_memory(),
            "opens": opens.as_memory(),
            "supply": supply.as_memory() if hasattr(supply, "as_memory") else {},
            "coming": coming.as_memory(),
            "marks": marks.as_memory(),
            # The last several runs, and how each finished. Enough to tell
            # which ending she has a route to, and bounded so the record does
            # not grow for ever.
            "endings": [
                {"shapes": list(shapes)[-12:], "ended": ended}
                for shapes, ended in [
                    *endings,
                    (
                        [str(one) for one, _ in marks.trail][-12:],
                        str(doing.outcome() if hasattr(doing, "outcome") else "")
                        or ("finished" if restarts["count"] else "stopped"),
                    ),
                ][-8:]
            ],
            "beats": beats.as_memory(),
            "repeats": repeats.as_memory(),
            "moves": knows.rules.as_memory() if knows.rules is not None else {},
            "acts": can_do.as_memory(),
            "skill": skilled.as_memory(),
            "world": world.as_memory(),
            "lines": lines.as_memory(),
            # Only the lines she has actually held, so the record does not
            # grow a stance for every phrasing she tried once.
            "lines_held": {
                said: kept
                for said, kept in lines_held.items()
                if said in (lines.known.get(A_LINE_HERE) or {})
            },
            "approach": plan["held"].as_memory() if plan["held"] is not None else {},
        },
    )
    # And the half of it that is about the KIND of world rather than this one,
    # filed under the kind, so the next world that moves like this starts
    # knowing how it moves. Written only when she got far enough to have
    # worked the rule out, because a record of what she failed to establish
    # would be carried into every world of the kind and slow each of them down.
    if like_it["kind"] and knows.rules is not None and knows.rules.rule() is not None:
        kept = remember(
            like_it["kind"],
            {
                part: value
                for part, value in {
                    "moves": knows.rules.as_memory(),
                    "acts": can_do.as_memory(),
                    "skill": skilled.as_memory(),
                    "world": world.as_memory(),
                    "lines": lines.as_memory(),
                }.items()
                if part in CARRIES_TO_A_WORLD_LIKE_IT
            },
        )
        if kept:
            logger.info("what worlds like %r move like, kept", like_it["kind"])
    # What a cycle of this actually cost, so the next watched goal asks for
    # enough time to make the moves it is allowed to make.
    spent = max(0.0, time.monotonic() - began)
    if receipt.cycles:
        a_cycle_took(spent / float(receipt.cycles))
    # How the line she took turned out, written where consequences live, so
    # an approach that keeps failing is harder to reach for next time.
    doing.how_it_went(
        bool(receipt.completed),
        f"{len(moves)} move(s), {plan['changes']} change(s) of approach",
        graph=graph,
    )
    if cannot_see["reason"] and not receipt.completed and not moves:
        # Named apart from every other ending. She was never able to look, so
        # nothing about the screen's contents is being reported and no amount
        # of acting differently would have helped.
        result["outcome"] = "cannot_see"
        result["cannot_see"] = cannot_see["reason"]
    if not_there["reason"] and not receipt.completed:
        # Named apart from every other way a run can end. "Nothing offered a
        # move" would say the screen had nothing on it; this says she was
        # never in the thing she was asked to act in, which is a different
        # failure with a different fix.
        result["outcome"] = "could_not_get_there"
        result["could_not_get_there"] = not_there["reason"]
    if no_move["because"] and not receipt.completed:
        # Whatever else the run managed. A stall after four moves has a cause
        # as much as a stall after none, and gating this on "no moves at all"
        # meant the one run that mattered stayed silent.
        result["why_no_move"] = no_move["because"]
        logger.info(
            "the last cycle made no move: %s (after %d move(s))",
            no_move["because"],
            len(moves),
        )
    if undecided["reason"] and not receipt.completed:
        # Name the judgement, not the budget. "no_move_available" would say
        # the screen offered nothing; this says she could not decide, and why.
        result["outcome"] = "cannot_decide"
        result["cannot_decide"] = undecided["reason"]
    result["success_when"] = success_when
    result["success_region"] = [region_top, region_bottom]
    result["target_app"] = target_app
    result["expect_page"] = expect_page
    result["anchored_to"] = anchor["page"]
    if needs_person["reason"] and not receipt.completed:
        # Name the question rather than the symptom. "no_move_available"
        # describes a loop with nothing to do; this says a person is being
        # waited on, and quotes what they are being asked.
        result["outcome"] = "needs_person"
        result["needs_person"] = needs_person["reason"]
    if lost_page["value"] and not receipt.completed:
        # Name it. "no_move_available" would describe the symptom of reading a
        # page that is not the task's, and hide that the browser had moved.
        result["outcome"] = "navigated_away"
    # How it ended, said once, whatever ended it.
    #
    # A run that stopped left nothing behind saying so. Live 2026-09-07,
    # driving a real game: nineteen moves, a rule worked out, and then the
    # log went quiet — no outcome, no reason, no count. Whether she had
    # finished, run out of budget, lost the window or hit a wall was not
    # recoverable from anything she wrote down, and every question about the
    # run had to start by guessing which of those it was.
    logger.info(
        "the run is over: %s after %d move(s)%s%s",
        result.get("outcome") or ("done" if result.get("completed") else "stopped"),
        len(moves),
        f", {result['restarts']} restart(s)" if result.get("restarts") else "",
        f" — {result['error']}" if result.get("error") else "",
    )
    return result



def _tell(line: str) -> None:
    """Say something that is not a move — the line she is taking, or a change of it.

    A watcher who only ever hears the keystrokes sees a twitch every second
    and no thinking behind it. What she is trying, and the moment she stops
    trying it, is what a watcher came to hear.
    """
    said = " ".join(str(line or "").split())
    if not said:
        return
    logger.info("saying out loud: %r", said[:160])
    try:
        from core.agency.narrator import Narrator

        Narrator.say_everywhere(said)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("screen_pursuit", exc, severity="info", action="held a plan without saying it")








def _publish_decision(said: str, because: str, expected: str, chosen: Any) -> None:
    """Offer one decision to the workspace, whatever kind of line it is."""
    try:
        from core.consciousness.global_workspace import ContentType
        from core.container import ServiceContainer

        workspace = ServiceContainer.get("global_workspace", default=None)
        publish = getattr(workspace, "publish", None) if workspace else None
        if publish is None:
            return
        coroutine = publish(
            priority=0.9,
            source="screen_pursuit.moved",
            payload={
                "schema": "aura.decision.v1",
                "decision": {
                    "chose": said,
                    "because": because,
                    "expected": expected,
                    "spoke": bool(getattr(chosen, "spoke", True)) if chosen is not None else True,
                },
            },
            reason=said,
            content_type=ContentType.SOMATIC,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # not a failure: nothing to publish to when there is no loop
            # running, and closing the coroutine is the tidy way to say so.
            coroutine.close()
            return
        task = create_owned_asyncio_task(coroutine)
        task.add_done_callback(lambda done: done.exception())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("screen_pursuit", exc, severity="info", action="acted without saying so")
























__all__ = [
    "OBSERVE_TIMEOUT_S",
    "PRESSABLE_KEYS",
    "ScreenPursuitInput",
    "ScreenPursuitSkill",
    "goal_reached",
    "press",
    "pursue_on_screen",
    "read_screen",
]
