"""Getting a screen worth looking at, and saying what she saw.

Her own window is in front of the thing she is acting in more often than
anything else, so most of this is about clearing a view and then waiting for
one: what covers what, which window to put away, how long a look takes on this
host. The rest is narration — what she says she is about to do, and what she
says when it did not land, which is the only part of a pursuit a person watching
can actually follow.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from core.runtime.errors import record_degradation

from .screen_pursuit_bearings import (
    _AS_IT_USUALLY_IS,  # noqa: F401
    A_SCREENFUL_AT_LEAST,  # noqa: F401
    ENOUGH_TO_BE_A_THING,  # noqa: F401
    MOST_OF_A_SCREEN,  # noqa: F401
    RESTART_LABELS,
    SCREENFULS_TO_LOOK,  # noqa: F401
    SETTLE_AFTER_SCROLL_S,  # noqa: F401
    _a_screenful,  # noqa: F401
    _a_step_back,  # noqa: F401
    _by_how_much_room,  # noqa: F401
    _moves_that_leave_her_nothing,  # noqa: F401
    _time_left,  # noqa: F401
    _within_a_move,  # noqa: F401
    restart_control,
)
from .screen_pursuit_surface import (
    DECLINES_AND_NOTHING_ELSE,
    LABEL_REACH,  # noqa: F401
    _bound_to_a_window,  # noqa: F401
    _matches,  # noqa: F401
    _screen_size,  # noqa: F401
    _value_is_on_screen,  # noqa: F401
    click_normalized,
    labelled_by,  # noqa: F401
    window_bounds,
)

logger = logging.getLogger("Aura.ScreenPursuit")


#: How long a single observation may take before the cycle is abandoned. A
#: screen read finishes well inside this. A capture still running when the
#: timeout expires is wedged, and waiting on it only makes the loop less
#: responsive.
OBSERVE_TIMEOUT_S = 8.0


async def _move_her_own_surface_aside(
    over: tuple[float, float, float, float] | None,
    mine: tuple[int, int, int, int] | None,
) -> bool:
    """Move her own window off the thing she is working on.

    A window she owns and a window somebody else owns want opposite answers.
    Declining is right for a dialog; it does nothing to her companion bubble,
    which floats above everything by design and has no decline key — so in
    companion mode the loop found something in front, pressed Escape at it,
    reported that it would not close, and stopped, with the board visible the
    whole time and her own window the only thing on it.

    She can place that window. So she places it somewhere else, on the far
    side from the work, and carries on. Nothing here closes it: it is how the
    person is talking to her.
    """
    if over is None or mine is None:
        return False
    try:
        from core.perception.ambient_presence import PresenceMode, get_ambient_presence

        presence = get_ambient_presence()
        placeable = (
            presence.mode is PresenceMode.BUBBLE and presence.drawing_surface_attached()
        )
        where = presence.bubble_position() if placeable else None
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        placeable, where = False, None
    if not placeable:
        # Her whole window, which she cannot place but can put away.
        #
        # This covered only the companion bubble, so in desktop mode her own
        # window sat over the work and nothing here touched it. LIVE
        # 2026-08-31: asked to play a game in a browser, every reading for
        # eighteen moves was of her own panels — LIVE NEURAL FEED, TELEMETRY,
        # MEMORY, SETTINGS — and the board appeared in none of them. She
        # pressed arrow keys into herself and her predictions about what would
        # change were correct, which is why it looked like playing.
        #
        # Asking the thing to the front is not enough on its own: hers is
        # drawn above everything by design and comes straight back. Hiding is
        # what a person does with their own window when it is over their work.
        # Nothing closes, nothing stops, and it returns the moment it is
        # wanted.
        return await _put_her_own_window_away()
    if not where:
        return False
    left, top, right, bottom = (float(edge) for edge in over)
    x, y, wide, tall = (float(edge) for edge in mine)
    if wide <= 0.0 or tall <= 0.0:
        return False
    across = (float(where[0]) - x) / wide
    down = (float(where[1]) - y) / tall
    if not (left <= across <= right and top <= down <= bottom):
        # It is above her window without being over the work, which is not in
        # the way. Moving it would be fussing at the person's screen.
        return False
    # The far side from the work, in whichever direction there is more room.
    room_left, room_right = left, 1.0 - right
    room_up, room_down = top, 1.0 - bottom
    if max(room_left, room_right) >= max(room_up, room_down):
        across = 0.0 if room_left >= room_right else 1.0
    else:
        down = 0.0 if room_up >= room_down else 1.0
    asked = presence.request_bubble_move(x + across * wide, y + down * tall)
    if asked is None:
        return False
    logger.info("her own window was over the work; asked it to move aside")
    return True


def _how_full(reading: Any) -> float:
    """What share of a reading's places hold something."""
    places = int(getattr(reading, "rows", 0) or 0) * int(getattr(reading, "columns", 0) or 0)
    if places <= 0:
        return 0.0
    return float(reading.occupied()) / float(places)


def _what_she_could_not_learn_from(dropped: dict[str, int]) -> str:
    """The moves she made and learned nothing from, said out loud.

    A move whose before and after could not be compared is a move that cost
    her a keystroke and taught her nothing, and it was thrown away in silence.
    The only trace was the rule staying unworked-out, which reads as a hard
    world rather than as evidence going missing on the way to the learner.
    """
    lost = {why: count for why, count in (dropped or {}).items() if count}
    if not lost:
        return ""
    said = ", ".join(f"{count} to {why}" for why, count in sorted(lost.items()))
    return f" | could not learn from {said}"


def _placed_in(lattice: Any, places: Any) -> list[tuple[int, int]]:
    """Places named in shares of the window, said as rows and columns.

    The part of her that works out what moves counts places where they sit on
    the screen; the part that works out how they move counts them by row and
    column in the thing. The lattice is what turns one into the other, and
    without it the two cannot tell each other anything.
    """
    down, across = getattr(lattice, "down_at", ()), getattr(lattice, "across_at", ())
    if not down or not across:
        return []
    said: list[tuple[int, int]] = []
    for place in places or ():
        try:
            x, y = float(place[0]) / 100.0, float(place[1]) / 100.0
        except (IndexError, TypeError, ValueError):
            continue
        row = min(range(len(down)), key=lambda one: abs(down[one] - y))
        column = min(range(len(across)), key=lambda one: abs(across[one] - x))
        said.append((row, column))
    return said


def _how_it_has_been_going(began_at: dict[str, Any], now: Any) -> float:
    """What she has been getting per act so far, positive when it is working.

    The same statistic a trial is judged by — gain per observation — because
    where she stands is not comparable across two stretches of a run and the
    rate she is moving at is.
    """
    was = began_at.get("worth")
    if was is None or now is None:
        return 0.0
    numbers = getattr(now, "numbers", None)
    here = sum(numbers() or (0.0,)) if callable(numbers) else 0.0
    return (here - float(was)) / max(1, int(began_at.get("seen") or 1))


def _the_thing_she_is_acting_in(whole: Any, lattice: Any, like: Any = None) -> Any:
    """The thing inside a reading — unless she is already holding its frame.

    The crop keeps the largest regular block in a reading, and which block
    that is depends on which places happen to be filled. So a reading placed
    into a four-by-four lattice came back four by four and was then cut to
    three by four, or four by three, differently almost every glance. The
    comparison that teaches her how a world moves needs both readings in the
    SAME frame, and it slices the frame's own lines along with the cells, so
    nearly every pair was thrown away.

    LIVE 2026-08-31 on the real game: a correct four-by-four lattice held
    across forty acts, and one comparison out of forty reached the rule. So
    "how this moves is not worked out yet" after fifty-four moves, and a full
    language generation for every move of a world she had already read
    correctly.

    A lattice is what she has instead of a crop. Working the frame out afresh
    from a reading that was just placed into one is asking the question she is
    holding the answer to.
    """
    from core.perception.the_thing_itself import the_thing_itself  # noqa: PLC0415

    if (
        lattice is not None
        and getattr(lattice, "held", False)
        and whole is not None
        and (whole.rows, whole.columns) == (lattice.rows, lattice.columns)
    ):
        return whole
    return the_thing_itself(whole, like=like)


def _the_kind_of_world_this_is(state: Any, acts: Sequence[str], toward: str) -> str:
    """A name this world shares with every world that moves like it."""
    try:
        from core.agency.what_kind_of_problem import recognise  # noqa: PLC0415

        return recognise(
            acts=list(acts), knows_how_it_moves=None, state=state, toward=toward
        ).shape.of_this_kind()
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="acted without naming the kind"
        )
        return ""


def _no_more_than_a_fresh_one_is_worth(held: Any) -> float:
    """How much of another world's evidence to carry: the conclusion, not the
    confidence.

    A rule that survived two hundred acts somewhere else is not two hundred
    acts of evidence about here. Carried whole it would take two hundred
    disagreements to overturn, and she would play a world she had misread for
    an hour rather than a handful of moves. Discounted to exactly what it
    takes to establish a rule here from scratch, it starts her off knowing
    what she knew and loses to the first few things this world does
    differently — which is what evidence from somewhere else is worth.
    """
    from core.perception.how_it_moves import ENOUGH_TO_TRUST  # noqa: PLC0415

    counts = [
        value
        for value in ((held or {}).get("tried") or {}).values()
        if isinstance(value, (int, float))
    ]
    most = max(counts, default=0)
    return min(1.0, ENOUGH_TO_TRUST / most) if most > ENOUGH_TO_TRUST else 1.0


#: How much longer than usual a look may take before it is a wedge rather than
#: a busy machine. Four, because a read competing with a resident model for
#: the same hardware was measured taking about three times its idle cost, and
#: a bound at the thing being measured refuses the first read that reaches it.
LONGER_THAN_USUAL = 4.0


def _how_long_a_look_takes(took: Sequence[float]) -> float:
    """How long to wait for a reading, from how long they have taken here.

    A fixed bound is a guess about a machine. This one is a measurement of the
    machine she is on, and it widens when the machine gets busy — which is
    exactly when a read is slow and exactly when calling it broken is wrong.

    Until she has looked enough times to have an opinion, the standing bound
    applies, which is what every caller assumed before there was anything to
    measure.
    """
    seen = [one for one in took or () if one > 0.0]
    if len(seen) < 3:
        return OBSERVE_TIMEOUT_S
    usual = sorted(seen)[len(seen) // 2]
    return max(OBSERVE_TIMEOUT_S, usual * LONGER_THAN_USUAL)


async def _the_best_reading_available(
    observation: dict[str, Any],
    band: tuple[float, float, float, float] | None,
    *,
    like: Any,
    in_a_browser: bool,
    answering: frozenset[tuple[int, int]] | None = None,
    lattice: Any = None,
) -> Any:
    """Ask the page what it is showing; look at the screen when it will not say.

    A page knows exactly what it is showing and where. LIVE 2026-08-29 on
    play2048.co the screen reading found five of the sixteen places on the
    board, at two distinct columns out of four — no lattice in a handful of
    scattered cells, so no thing to model, so nothing to look ahead over, so
    every move fell through to a full language generation. The board was drawn
    perfectly well the whole time.

    The reader for this was written then and never called by anything. Taken
    only when it sees MORE than the screen does, so a page that answers
    poorly, or an application that is not a browser at all, changes nothing.
    """
    from core.perception.where_it_responds import (
        what_is_there,
        what_the_page_is_showing,
    )

    seen = what_is_there(
        observation, band, like=like, answering=answering, lattice=lattice
    )
    if not in_a_browser:
        return seen
    try:
        from core.perception.what_the_page_says import what_the_page_says

        said = await what_the_page_says()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info",
            action="read the screen because the page would not say",
        )
        return seen
    if not said:
        return seen
    from_page = what_the_page_is_showing(said, band, like=like, lattice=lattice)
    # Whichever shows the THING more clearly, not whichever holds more text.
    #
    # A page that draws its board on a canvas has no text in the board at all,
    # and plenty around it — a score, a best, a New Game, a footer. Preferring
    # the reading with more things in it therefore preferred the furniture and
    # threw the board away. What matters is the laid-out thing inside each
    # reading, which is the question the crop already answers.
    from core.perception.the_thing_itself import the_thing_itself

    theirs = the_thing_itself(from_page)
    mine = the_thing_itself(seen)
    # By how full each one is, not by how much it holds. The crop hands back
    # the reading unchanged when it finds no lattice, so an uncropped page of
    # furniture — ten pieces of text with no grid among them, spread over
    # thirty-five places — beat a real board of eight over twenty every time.
    # A thing laid out is full of its own cells; a page with text scattered
    # about it is not, and that is the difference between them.
    # Both: more of the thing, and more thing than page. One cell read on its
    # own is a full reading by the second measure alone.
    if theirs.occupied() <= mine.occupied() or _how_full(theirs) <= _how_full(mine):
        return seen
    logger.info(
        "the page shows the thing better: %dx%d with %d in it, against %dx%d with %d",
        theirs.rows, theirs.columns, theirs.occupied(),
        mine.rows, mine.columns, mine.occupied(),
    )
    return from_page


#: What each way of being refused a look means for her, and what it takes to
#: change it.
#:
#: Seven distinct refusals were reported as one sentence — "your screen is
#: locked" — whatever had actually happened. A setting switched off was a
#: locked screen. Something private in front was a locked screen. Not being
#: able to tell what was in front was a locked screen. The person was told to
#: unlock a screen that was not locked, and the one thing that would have
#: helped went unsaid.
#:
#: Waiting is right for a condition that passes on its own: a person unlocks a
#: screen, closes a private window, brings something forward. It is wrong for
#: one that does not, and waiting out the whole budget on a setting nobody is
#: going to change during the task is the same as failing, slower.
PASSES_ON_ITS_OWN = frozenset({
    "session_locked",
    "private_foreground",
    "private_visible",
    "foreground_unknown",
    "browser_title_unknown",
})


def _what_being_refused_a_look_means(why: str) -> str:
    """The refusal in her own words, and what would change it."""
    return {
        "session_locked": (
            "your screen is locked, so there is nothing for me to look at yet"
        ),
        "runtime_setting_disabled": (
            "I am not allowed to read the screen at the moment — the setting "
            "that lets me is switched off"
        ),
        "private_foreground": (
            "something private is in front, so I am not reading the screen "
            "while it is there"
        ),
        "private_visible": (
            "something private is on screen, so I am not reading it while it "
            "is showing"
        ),
        "foreground_unknown": (
            "I cannot tell what is in front of me, so I will not act on it"
        ),
        "browser_title_unknown": (
            "I cannot tell which page is in front, so I will not act on it"
        ),
        "policy_unavailable": (
            "I could not check whether I am allowed to read the screen, so I "
            "am not going to"
        ),
    }.get(str(why or ""), f"I am not able to read the screen ({why or 'no reason given'})")


async def wait_for_a_screen_to_look_at(ends_at: float) -> bool:
    """Wait for a locked screen, rather than failing at one.

    A locked screen is a condition that passes, like a model still warming.
    Failing at it turns "ask her, then sit down at the machine" into "ask her
    again once you are there", and the person has no way to know that is what
    happened — LIVE 2026-08-30, a request to play a game came back as a fault.

    Bounded by the deadline the task already has, so nothing waits longer than
    the work was given. Checked about once a second because that is the
    granularity of the thing being waited for: a person reaching over and
    unlocking. Checking faster cannot see it sooner.
    """
    from core.security.screen_capture_policy import (
        evaluate_screen_capture_admission_async,
    )

    # Let the settings land before believing a refusal.
    #
    # Permission reads answer from a snapshot a worker keeps current, and
    # until it has run once there is no snapshot — so every permission reads
    # as denied, which is indistinguishable from the person having switched
    # them all off. Measured on this machine with screen access ON: the first
    # read said off, and the run reported having nothing to look at.
    try:
        from core.runtime.runtime_settings import wait_until_settled  # noqa: PLC0415

        await asyncio.to_thread(wait_until_settled, 1.0)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info",
            action="checked whether she may look before the settings had settled",
        )

    told = ""
    while True:
        try:
            admission = await evaluate_screen_capture_admission_async()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            _WHY_SHE_CANNOT_LOOK["value"] = ""
            return True
        why = str(getattr(admission.reason, "value", admission.reason) or "")
        if admission.allowed:
            if told:
                logger.info("she can see the screen again; carrying on")
            _WHY_SHE_CANNOT_LOOK["value"] = ""
            return True
        _WHY_SHE_CANNOT_LOOK["value"] = why
        if why not in PASSES_ON_ITS_OWN:
            # Nothing about waiting changes this one, and waiting out the
            # whole budget on it is failing slowly.
            logger.info("she cannot look and waiting will not help: %s", why)
            return False
        left = ends_at - time.monotonic()
        if left <= 0.0:
            logger.info("she could not look for the whole of this task: %s", why)
            return False
        if told != why:
            logger.info("she cannot look yet (%s); waiting rather than failing", why)
            # Say it. Waiting in silence for the whole deadline and then
            # explaining is the same information delivered too late to act
            # on — the person is the one who can change it, and they cannot
            # do that if nothing tells them.
            await _narrate(
                f"{_what_being_refused_a_look_means(why).capitalize()}. "
                "I will start the moment that changes."
            )
            told = why
        await asyncio.sleep(min(max(1.0, left / 60.0), left))


#: Why the last look was refused, for the caller that has to say so. Kept
#: beside the wait rather than returned, so every existing caller keeps its
#: boolean and none of them has to learn a new shape to stop lying.
_WHY_SHE_CANNOT_LOOK: dict[str, str] = {"value": ""}


async def clear_what_is_in_front(on_top: str) -> bool:
    """Try to get whatever is covering her work out of the way.

    A dialog in front of the thing she is acting in is an obstacle, not a
    reason to stop — a person closes it and carries on. She could see one and
    name it and had no way to move it, because every key she can send is bound
    to her own window and the dialog is not in it.

    Only ever the key that declines. She may clear something out of her way;
    she may not agree to something on somebody's behalf, and a dialog asking
    for a permission or a consent is exactly the case where those two come
    apart. blocking_overlay.py holds the same line for a dialog inside a page:
    it dismisses and never agrees.
    """
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .screen_pursuit import (
        _everything_on_top,
    )

    if not str(on_top or "").strip():
        return False
    try:
        from core.capabilities.host_automation import get_host_automation

        logger.info("something is in front of her work (%s) — declining it", on_top)
        await get_host_automation().hotkeys([DECLINES_AND_NOTHING_ELSE])
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="could not decline what was in front"
        )
        return False
    # Ask whether THAT is still there, not whether something else is.
    #
    # This called _whats_on_top(on_top), whose first argument names the window
    # to leave OUT — so the check for "did it close" excluded the very thing
    # it was checking for, and reported success whenever the overlay was the
    # only thing above her. Every claim it made was unfalsifiable. LIVE
    # 2026-08-29: "UserNotificationCenter was in front of this. Closed it."
    # four times over, with the notification exactly where it had been.
    above = await _everything_on_top("")
    wanted = str(on_top).strip().lower()
    if any(name.strip().lower() == wanted for name in above):
        logger.info("%r is still in front and will not decline", on_top)
        return False
    logger.info("%s is out of the way", on_top)
    return True


async def _put_her_own_window_away() -> bool:
    """Hide her own application, so what she was asked to act in is visible."""
    from core.config import get_config

    named = ""
    try:
        named = str(getattr(get_config(), "app_name", "") or "").strip()
    except (AttributeError, RuntimeError, TypeError, ValueError) as why:
        logger.debug("could not read her own app name from the config: %s", why)
        named = ""
    for candidate in (named, "Aura"):
        if not candidate:
            continue
        try:
            from core.capabilities.host_automation import get_host_automation

            receipt = await get_host_automation().hide_app(candidate)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("could not put %r away", candidate, exc_info=True)
            continue
        if bool(getattr(receipt, "ok", False) or getattr(receipt, "success", False)):
            logger.info("put her own window away so the thing is visible")
            return True
    return False


async def _bring_the_thing_back_to_the_front(app: str) -> bool:
    """Raise the window she was asked to act in, rather than close what is over it.

    Whatever is in front, the thing she wants is behind it, and asking for it
    is both gentler and more general than closing the other: it works for her
    own window, for a notification, for anything.

    LIVE 2026-08-31, asked to play a game in a browser: her own desktop window
    was frontmost the whole run, so every reading was of her own interface —
    LIVE NEURAL FEED, TELEMETRY, MEMORY, SETTINGS — and not one was of the
    board. She pressed keys into herself for eighteen moves. Moving her own
    surface aside covered only the companion bubble, which is not the window
    that was in the way.
    """
    named = str(app or "").strip()
    if not named:
        return False
    try:
        from core.capabilities.host_automation import get_host_automation

        host = get_host_automation()
        receipt = await host.focus_app(named)
        raised = bool(getattr(receipt, "ok", False) or getattr(receipt, "success", False))
        if not raised:
            # Not in front because it is not running.
            #
            # Raising a window asks the window system for a process by name,
            # and there is no process. Asked to play a game that was not open,
            # she would fail to raise it, read whatever happened to be
            # frontmost, and play that instead — which on this machine means
            # pressing arrow keys into somebody's editor. Starting it is the
            # same request, one step earlier.
            receipt = await host.launch_app(named)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        logger.debug("could not raise %r", named, exc_info=True)
        return False
    raised = bool(getattr(receipt, "ok", False) or getattr(receipt, "success", False))
    if raised:
        logger.info("brought %r back to the front", named)
    return raised


async def _why_nothing_answers(
    mine: str, over: tuple[float, float, float, float] | None = None
) -> str:
    """Why nothing she does is changing anything, before blaming the thing.

    A world that has stopped answering and a world she cannot reach look
    identical from inside the loop: keys reported sent, her window reported in
    front, and nothing moving. LIVE 2026-08-26, a system permission dialog sat
    above every window, took the keyboard, and swallowed every keystroke for
    an hour — and what she said was that the game had ended.

    Something else holding the keyboard is a different thing from a thing that
    is finished, and it has a different answer: one of them somebody can fix.
    """
    from .screen_pursuit import (
        _whats_on_top,
    )

    on_top = await _whats_on_top(mine, over=over)
    if on_top:
        # Hers first. A window she owns is moved, not declined — declining
        # does nothing to it and stopping because of it is stopping because
        # of herself.
        hers = await window_bounds(mine) if over else None
        if await _move_her_own_surface_aside(over, hers):
            return (
                "My own window was over the board. I have moved it aside — "
                "carrying on."
            )
        # Ask for the thing back before closing anything. It is gentler and it
        # covers every occluder rather than the one kind she can place.
        if await _bring_the_thing_back_to_the_front(mine):
            return f"{on_top} was in front. I have brought {mine} back — carrying on."
        # Try to move it before saying it cannot be moved.
        if await clear_what_is_in_front(on_top):
            return f"{on_top} was in front of this. I have closed it — carrying on."
        return (
            f"Nothing I do is reaching this — {on_top} is in front of it and taking "
            "the keyboard, and it will not close. Nothing I press is getting through."
        )
    return "Nothing I do is changing anything here — this attempt is over."


def _covers(
    window: Any, over: tuple[float, float, float, float], mine: tuple[int, int, int, int]
) -> bool:
    """Whether a window actually overlaps the part of her window she is using.

    Being above her window is not the same as being in her way. A notification
    banner sits in a corner; the thing she is acting on is somewhere else, and
    nothing about the banner stops her.

    ``over`` is a band, and a band means a share of the WINDOW she is driving —
    that is the space read_screen measures in, deliberately, so that a band is
    portable across window sizes and monitors. So the overlay's rectangle,
    which the window server gives in screen pixels, is put into that same space
    before the two are compared. Measured against the screen instead, a banner
    halfway down the display reads as sitting on a board halfway down a window
    that starts lower, and the answer is wrong in both directions.
    """
    try:
        bounds = window.get("kCGWindowBounds") or {}
        if not bounds:
            # No bounds is not a window of no size. It is a window she cannot
            # place, and she cannot place it in front of or beside anything.
            return True
        x, y = float(bounds.get("X", 0.0)), float(bounds.get("Y", 0.0))
        wide, tall = float(bounds.get("Width", 0.0)), float(bounds.get("Height", 0.0))
    except (AttributeError, TypeError, ValueError, KeyError):
        # Unreadable bounds mean she cannot tell, and cannot tell is in the way.
        return True
    ox, oy, ow, oh = (float(edge) for edge in mine)
    if ow <= 0.0 or oh <= 0.0:
        return True
    left, top, right, bottom = over
    return not (
        (x - ox) / ow >= right
        or (x + wide - ox) / ow <= left
        or (y - oy) / oh >= bottom
        or (y + tall - oy) / oh <= top
    )


def _her_reasoning(stakes: float) -> Any:
    """Her own judgement, sized to what rides on the move."""
    from core.agency.her_reasoning import reasoning_for

    return reasoning_for(stakes)


def _reasoning_for_a_plan() -> Any:
    """Her judgement on how to go about something, which is not a move."""
    from core.agency.her_reasoning import reasoning_for_a_plan

    return reasoning_for_a_plan()


def _say_intent(
    key: str, chosen: Any = None, *, out_loud: bool = False, following_on: bool = False
) -> None:
    """Say what she is about to do, before her body does it.

    A commentary that only ever reports finished moves is a log. Somebody
    watching wants the intention and then the action, in that order, because
    that is the order a person doing something narrates it in.

    The reason this was ever the other way round is real and does not go
    away: an intention is not an act, and a keystroke refused for focus would
    be described as a move she made. So the intention is said here and the
    body is watched afterwards — an intention that was not carried out is
    corrected out loud by :func:`_say_it_did_not_land`, and the RECORD of
    what she did is still written only from what landed.
    """
    from .screen_pursuit import (
        _publish_decision,
        _tell,
    )

    said = f"Going {str(key).strip().lower()}"
    # A reason she did not give does not erase the one she has.
    #
    # This read the other way round and the assignment was unconditional, so
    # a choice carrying no rationale of its own — the ordinary case — wiped
    # out "same plan" and left a bare keystroke. LIVE 2026-08-26: a whole
    # game of "Going up", "Going down", with every reason she had for them
    # discarded one line before it was said.
    because = str(getattr(chosen, "rationale", "") or "") if chosen is not None else ""
    if not because and following_on:
        because = "same plan"
    _publish_decision(said, because, _expected_of(chosen), chosen)
    if out_loud:
        _tell(f"{said} — {because}" if because else said)


def _say_it_did_not_land(key: str, *, out_loud: bool = False) -> None:
    """Say that what she meant to do did not happen.

    Without this, saying the intention first would let the commentary drift
    from the body the moment anything refused a keystroke — which is the
    failure that put the narration after the act in the first place.
    """
    from .screen_pursuit import (
        _publish_decision,
        _tell,
    )

    said = f"{str(key).strip().capitalize()} did not land"
    _publish_decision(said, "the window did not take it", "", None)
    if out_loud:
        _tell(said)


def _expected_of(chosen: Any) -> str:
    option = getattr(chosen, "chosen", None)
    expectation = getattr(option, "expectation", None)
    return str(getattr(expectation, "describes", "") or "")


#: The longest a change has taken to appear, and how long a poll takes, both
#: measured rather than chosen. Waiting is bounded by what this world has
#: actually done, so a slow surface is waited for and a fast one is not.
_ANSWERING_TOOK: dict[str, float] = {"longest": 0.0}


def _how_long_to_wait() -> float:
    """How long to give the world to answer, from how long it has taken.

    Nothing is chosen here. Before she has seen a change there is no
    measurement, so the old default stands; after that it is a little more
    than the longest one she has seen, which is what "long enough" means when
    the thing being waited on is the same thing every time.
    """
    longest = _ANSWERING_TOOK["longest"]
    return max(1.0, longest * 2) if longest else 4.0


async def _settled_after(
    before: dict[str, Any],
    app: str,
    *,
    patience: float = 0.0,
    arrived: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, Any], bool]:
    """The screen once it has changed AND stopped changing, and whether it did.

    An action that moves a surface is not done when the keystroke returns; it
    is done when the surface says so. Waiting on the change rather than on a
    fixed delay means a slow page is waited for and a fast one is not.

    Waiting for it to stop as well as to start is the part that was missing.
    A board mid-slide has half moved, and half a move is not a state any rule
    describes: taken as the result of the move it disagrees with every
    hypothesis, including the true one.

    The second value is the part that matters. A click that lands on nothing
    still reports success — the click happened — so a caller that only waits
    goes on to claim a reset that never occurred. Measured live: "Began again
    1 time(s)" while the score sat unchanged at 996 the whole time.
    """
    # Compared by WHERE things are, not by what words are on the screen.
    #
    # A thing sliding across a surface changes no text at all: the same
    # numbers are there throughout, in different places. So a board mid-slide
    # reads identical to the board before the move, the settle finished on
    # the first look, and whatever was still travelling was read at wherever
    # it had got to.
    #
    # LIVE 2026-09-04 on a correctly read four by four board: the row nearest
    # the direction pressed came back unmoved, move after move, while every
    # other row landed exactly where the rule said. The true rule sat at 59%
    # of 29 — under the bar to be trusted, so nothing ever looked ahead, all
    # game, on a board she was reading perfectly.
    from core.perception.where_it_responds import places_and_text  # noqa: PLC0415

    from .screen_pursuit import (
        read_screen,
    )

    def _reading(observation: dict[str, Any]) -> tuple[Any, str]:
        """Where things are, and what the whole reading says.

        Positions alone miss a change that happens in place. A score going from
        996 to 0 does not move anything, and the reset that produced it is
        exactly what this function was written to notice — measured live,
        "Began again 1 time(s)" while the score sat unchanged. Positions alone
        also cannot be dropped: a board mid-slide has the same words in
        different places, and comparing words alone calls that unchanged.
        """
        return places_and_text(observation), str(observation.get("text") or "")

    was = _reading(before)
    started = time.monotonic()
    seen = before
    moved = False
    while time.monotonic() - started < (patience or _how_long_to_wait()):
        await asyncio.sleep(0.3)
        try:
            now = await asyncio.wait_for(read_screen(app), timeout=OBSERVE_TIMEOUT_S)
        except TimeoutError:
            continue
        said = _reading(now)
        if not moved and said != was:
            moved = True
            _ANSWERING_TOOK["longest"] = max(
                _ANSWERING_TOOK["longest"], time.monotonic() - started
            )
            # Where she foretold the result, recognising it is knowing it has
            # landed.
            #
            # Watching until it stops changing is what she has to do when she
            # cannot say what the change will be. When she can, the arrival of
            # exactly that is the same fact and costs one reading instead of
            # two — and a reading is most of what a move costs. Measured on
            # the real board: about four seconds a move, of which nearly two
            # were the second look.
            if arrived is not None:
                try:
                    if arrived(now):
                        return now, True
                except (AttributeError, TypeError, ValueError) as exc:
                    record_degradation(
                        "screen_pursuit", exc, severity="info",
                        action="waited for stillness rather than for what she foretold",
                    )
        elif moved and said == _reading(seen):
            # Changed, and now the same twice running: it has finished.
            return now, True
        seen = now
    return seen, moved


#: Words a dialog uses when it is asking whether you meant it.
ASKING_TO_CONFIRM = ("are you sure", "do you want to", "confirm", "this will")


def _where_it_asks(observation: dict[str, Any]) -> float | None:
    """How far down the question sits, when something is asking one."""
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip().lower()
        if any(phrase in text for phrase in ASKING_TO_CONFIRM):
            try:
                return float(region.get("center_y", region.get("y")))
            except (TypeError, ValueError) as why:
                # A region whose position will not parse is a region dropped
                # from the reading, and the reading is what everything about
                # the board is worked out from.
                logger.info(
                    "a region was dropped for an unreadable position: %s", why
                )
                return None
    return None


async def _answer_own_confirmation(
    before: dict[str, Any], app: str, label: str
) -> tuple[dict[str, Any], bool]:
    """Finish a reset that asked "are you sure".

    A consequential control usually asks. An agent that clicks it and walks
    away leaves the question open and nothing happens — measured live: the
    click on "New Game" landed correctly, play2048 asked "Are you sure you
    want to start a new game?", and the score sat unchanged while the run
    reported it had begun again.

    The dialog's own button is found by the question, not by classifying the
    dialog: whatever is asking sits next to the control that answers it, and
    the control that answers is not the one already pressed. On that page the
    question is "Are you sure you want to start a new game?" and the answer
    is "Start New Game", directly below it.

    Only ever done when she chose the action being confirmed, and only for a
    control carrying the label she already decided to press.
    """
    seen, changed = await _settled_after(before, app)

    # A change that is a question is not a completed action.
    #
    # The screen does change when a dialog opens, so "it changed" would have
    # been read as "it worked" — and the run would carry on with the question
    # still up and nothing actually done.
    asked_at = _where_it_asks(seen)
    if asked_at is None:
        return seen, changed

    pressed = _where_clicked(before, label)
    best: tuple[float, float, float] | None = None
    for region in seen.get("layout") or []:
        text = str(region.get("text") or "").strip().lower()
        if not any(word in text for word in RESTART_LABELS):
            continue
        try:
            cx = float(region.get("center_x", region.get("x")))
            cy = float(region.get("center_y", region.get("y")))
        except (TypeError, ValueError):
            continue
        if pressed is not None and abs(cx - pressed[0]) < 0.02 and abs(cy - pressed[1]) < 0.02:
            # The control she already pressed. Pressing it again re-asks.
            continue
        # The answer sits below the question and near it.
        if cy <= asked_at:
            continue
        distance = cy - asked_at
        if best is None or distance < best[0]:
            best = (distance, cx, cy)

    if best is None:
        return seen, False
    _distance, cx, cy = best
    if not await click_normalized(cx, cy, expect_app=app, bounds=list(seen.get("bounds") or [])):
        return seen, False
    answered, moved = await _settled_after(seen, app)
    # And it is only done when nothing is still asking.
    return answered, bool(moved and _where_it_asks(answered) is None)


def _where_clicked(observation: dict[str, Any], label: str) -> tuple[float, float] | None:
    """Where the control she pressed was, so it is not pressed again."""
    found = restart_control(observation)
    if found is None:
        return None
    _label, x, y = found
    return (x, y)


def _where(region: Any) -> tuple[str, float, float]:
    """A text run's identity: what it says and roughly where it says it."""
    try:
        return (
            str(region.get("text") or "").strip().lower(),
            round(float(region.get("center_x", region.get("x", 0.0))), 2),
            round(float(region.get("center_y", region.get("y", 0.0))), 2),
        )
    except (TypeError, ValueError, AttributeError):
        return ("", 0.0, 0.0)


async def _say_line(line: str) -> None:
    """Hand one line to every surface a person might be watching. Never raises."""
    try:
        from core.agency.narrator import Narrator

        Narrator.say_everywhere(line)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        await _narrate(line)


async def _narrate(line: str, because: str = "") -> None:
    """Offer one line to whatever surface is listening. Never raises."""
    try:
        from core.perception.ambient_presence import get_ambient_presence

        get_ambient_presence().offer_utterance(
            f"{line} — {because}" if because else line
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        logger.debug("screen pursuit narration unavailable: %s", line)
