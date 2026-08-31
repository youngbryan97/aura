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
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import record_degradation
from core.runtime.watched_goal import PURSUIT_SECONDS, a_cycle_took
from core.runtime.what_she_learned import TRUST_CARRIED_OVER, named, recall, remember
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Aura.ScreenPursuit")

#: Keys the loop is allowed to press, by the name a person would use.
#: Bounded on purpose — a loop that can press anything can press ⌘Q.
PRESSABLE_KEYS = (
    "up", "down", "left", "right",
    "return", "enter", "tab", "space", "escape",
)

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

#: How long a single observation may take before the cycle is abandoned. A
#: screen read finishes well inside this. A capture still running when the
#: timeout expires is wedged, and waiting on it only makes the loop less
#: responsive.
OBSERVE_TIMEOUT_S = 8.0
#: The moves offered when a caller does not name its own. Arrow keys are the
#: universal keyboard affordance: they mean something on a board, a list, a
#: map, a carousel, a form. A caller with a richer vocabulary passes its own.
DEFAULT_MOVES: tuple[str, ...] = ("up", "down", "left", "right")
#: How many graded attempts travel into the next decision. Enough to notice a
#: move that stopped working, short enough that old evidence stops steering.
RECENT_ATTEMPTS = 4
#: How often language is consulted on a run of routine moves.
#:
#: A board changes a little each move, so re-reasoning every single one buys
#: little and costs the whole cycle: measured live, a language pass took
#: about eight seconds and a decision from evidence takes none, on a loop
#: that needs hundreds of moves. Language is asked when the answer is most
#: likely to differ — the first move, after a restart, when what she is doing
#: has stopped working — and periodically in between so the commentary stays
#: hers rather than a run of bandit statistics.
LANGUAGE_EVERY = 5
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


async def window_bounds(app_name: str) -> tuple[int, int, int, int] | None:
    """The front window rectangle of `app_name` in pixels, or None."""
    if not app_name:
        return None
    from core.capabilities.host_automation import get_host_automation

    script = (
        f'tell application "System Events" to tell process {app_name!r} '
        "to get {position, size} of front window"
    ).replace("'", '"')
    try:
        receipt = await get_host_automation().execute_applescript(script)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as why:
        logger.debug("could not read the window's bounds: %s", why)
        return None
    if not getattr(receipt, "success", False):
        logger.debug(
            "could not read the window's bounds: %s",
            getattr(receipt, "error", "") or "the host refused",
        )
        return None
    numbers = re.findall(r"-?\d+", str(getattr(receipt, "result", "") or ""))
    if len(numbers) < 4:
        return None
    x, y, width, height = (int(value) for value in numbers[:4])
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


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
        "at": time.time(),
    }


def _the_part_of(
    window: tuple[int, int, int, int], over: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """The pixels of a window that a band names."""
    x, y, wide, tall = (int(edge) for edge in window)
    left, top, right, bottom = over
    return (
        x + int(left * wide),
        y + int(top * tall),
        max(1, int((right - left) * wide)),
        max(1, int((bottom - top) * tall)),
    )


def _matches(pattern: str, text: str, *, whole_region: bool = False) -> bool:
    """Regex when the pattern is one, plain text when it is not.

    ``whole_region`` requires the text to BE the pattern rather than contain
    it. A bare number is the case that needs it: "128" appears inside
    "SCORE 128" and inside "1284", and neither is the thing being waited for.
    """
    body = str(text or "").strip()
    if whole_region:
        return body.replace(",", "") == str(pattern or "").strip().replace(",", "")
    try:
        return re.search(pattern, body, re.IGNORECASE) is not None
    except re.error:
        return pattern.lower() in body.lower()



#: How far from a number a word can be and still be its label, as a share of
#: the screen. A label sits against the value it names; anything further away
#: is a different thing on the page.
LABEL_REACH = 0.16


def labelled_by(region: dict[str, Any], layout: Sequence[dict[str, Any]]) -> str:
    """The word this number is the value of, if it is a value of anything.

    A bare number beside a word is that word's number. "SCORE" and "128" are
    separate text regions, so wholeness cannot tell a score from a tile — but
    a tile has nothing sitting next to it saying what it counts, and a score
    does.

    LIVE 2026-08-19: asked to play until a 128 tile, she matched the 128 in
    the header and reported the goal met without a move. General to any
    screen: "Total 99", "Items: 42", "BEST 6068".
    """
    try:
        x = float(region.get("x", region.get("center_x", 0.0)))
        y = float(region.get("center_y", region.get("y", 0.0)))
    except (TypeError, ValueError):
        return ""
    height = float(region.get("height", 0.03) or 0.03)
    for other in layout or []:
        if other is region:
            continue
        word = str(other.get("text") or "").strip()
        if not word or not re.search(r"[A-Za-z]", word) or len(word) > 24:
            continue
        try:
            ox = float(other.get("x", other.get("center_x", 0.0)))
            ow = float(other.get("width", 0.0) or 0.0)
            oy = float(other.get("center_y", other.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        same_line = abs(oy - y) <= max(height, 0.02)
        to_the_left = 0.0 <= x - (ox + ow) <= LABEL_REACH
        directly_above = 0.0 < y - oy <= LABEL_REACH and abs(ox - x) <= LABEL_REACH
        if (same_line and to_the_left) or directly_above:
            return word
    return ""


def content_text(
    observation: dict[str, Any],
    *,
    region_top: float = 0.0,
    region_bottom: float = 1.0,
) -> str:
    """The reading inside the content band, with the furniture left out.

    A caller that names a band has already said where the task lives, and
    everything outside it is the application's own chrome. Asking a question
    about a stuck position is where that matters most: a whole-screen reading
    of a game puts the score, the best-ever score and the site footer into
    the question, and a search for those returns nothing about the position.
    """
    if region_top <= 0.0 and region_bottom >= 1.0:
        return str(observation.get("text") or "")
    said: list[str] = []
    for region in observation.get("layout") or []:
        try:
            middle = float(region.get("center_y", region.get("y", 0.0)))
        except (TypeError, ValueError):
            continue
        if region_top <= middle <= region_bottom:
            text = str(region.get("text") or "").strip()
            if text:
                said.append(text)
    return " ".join(said) if said else str(observation.get("text") or "")


def _value_is_on_screen(value: str, observation: dict[str, Any]) -> bool:
    """Whether the screen is showing this value as a thing in its own right.

    A value that is the whole of a text region is a value the screen is
    showing; one inside a longer run is part of a sentence about something
    else, and one sitting beside a word is that word's number.
    """
    regions = list(observation.get("layout") or [])
    return any(
        _matches(value, str(region.get("text") or ""), whole_region=True)
        and not labelled_by(region, regions)
        for region in regions
    )


def goal_reached(
    observation: dict[str, Any],
    success_when: str,
    *,
    region_top: float = 0.0,
    region_bottom: float = 1.0,
) -> bool:
    """Whether this reading shows the goal met.

    Tested against what was actually read, not against a belief about what the
    action should have done. That distinction is the reason to look again at
    all: an action that ran is not an action that worked.

    When a band is given, only text whose measured position falls inside it
    counts. The layout was already being returned by every reading and this
    function ignored it, so the goal could be satisfied by the browser tab
    rather than the content — on play2048.co the word "2048" is in the tab, the
    heading and a welcome modal, and the board is 300 pixels below all three.
    A predicate that cannot say WHERE is a predicate that reports victory on
    the furniture.
    """
    pattern = str(success_when or "").strip()
    if not pattern:
        return False

    # A bare value has to BE something on screen, not appear inside something.
    #
    # LIVE 2026-08-19: asked to play until a 128 tile, she opened the game,
    # read "SCORE 128" from the header, and reported the goal reached in 1.2
    # seconds without making a move. The number was on screen; it was not a
    # tile. A value that is the whole of a text region is a value the screen
    # is showing as a thing; one inside a longer run is part of a sentence
    # about something else.
    bare_value = bool(re.fullmatch(r"[0-9][0-9,]*", pattern))

    band_is_whole_screen = region_top <= 0.0 and region_bottom >= 1.0
    if band_is_whole_screen:
        text = str(observation.get("text") or "")
        if not bare_value:
            if bool(text) and _matches(pattern, text):
                return True
            # A description that names one value is waiting for that value.
            #
            # Her own goal reader turns "play until you get a 128 tile" into
            # "128", so the usual path never sees a sentence. A caller that
            # passes the description straight through would otherwise wait
            # forever with the tile in front of her — measured: 494 moves, a
            # 128 on the board, and the run reported out of time. Only tried
            # once the condition as written has failed, so nothing that
            # matches today changes.
            values = re.findall(r"\b\d[\d,]*\b", pattern)
            if len(values) != 1:
                return False
            return _value_is_on_screen(values[0], observation)
        # With no band and no geometry there is nothing to check a bare value
        # against, so every region is examined instead of the flattened text.
        return _value_is_on_screen(pattern, observation)

    layout = observation.get("layout") or []
    if not layout:
        # A band was asked for and no geometry came back. Refusing is the
        # honest answer: matching the flat text would silently ignore the
        # constraint the caller added precisely because it mattered.
        return False
    top, bottom = (region_top, region_bottom) if region_top <= region_bottom else (
        region_bottom,
        region_top,
    )
    for region in layout:
        try:
            y = float(region.get("center_y", region.get("y", -1.0)))
        except (TypeError, ValueError):
            continue
        if not (top <= y <= bottom):
            continue
        if not _matches(pattern, str(region.get("text") or ""), whole_region=bare_value):
            continue
        if bare_value and labelled_by(region, layout):
            # This number is something's total, not the thing itself.
            continue
        return True
    return False


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


async def _ensure_page(expect_page: str) -> bool:
    """True when the browser is on `expect_page`, restoring it if it can.

    Identity, not appearance. A loop that only reads pixels cannot tell that
    the page changed under it — measured live, a stray click navigated the
    browser to a different site and the run kept reading and acting for
    cycles, every layer working and none of them knowing where they were.

    Restores by tab rather than by reload, because a task's page usually still
    exists in another tab and reloading would throw away whatever progress the
    task had made on it.
    """
    if not expect_page:
        return True
    wanted = expect_page.strip().lower()
    page = await current_page_identity()
    here = f"{page.get('url', '')} {page.get('title', '')}".lower()
    if wanted in here:
        return True
    try:
        from core.capabilities.browser_controller import get_browser_controller

        receipt = await get_browser_controller().focus_tab(expect_page)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as why:
        logger.info("could not bring %r to the front: %s", expect_page, why)
        return False
    if not getattr(receipt, "success", False):
        logger.info(
            "could not bring %r to the front: %s",
            expect_page,
            getattr(receipt, "error", "") or "the browser refused",
        )
        return False
    page = await current_page_identity()
    here = f"{page.get('url', '')} {page.get('title', '')}".lower()
    return wanted in here


async def _ensure_frontmost(app_name: str) -> bool:
    """Bring `app_name` forward if it is not already. True when it is."""
    from core.capabilities.host_automation import get_host_automation

    host = get_host_automation()
    context = await host.get_frontmost_window_context()
    observed = str(getattr(context, "result", "") or "").split("|", 1)[0].strip().lower()
    wanted = app_name.strip().lower()
    if observed and (wanted in observed or observed in wanted):
        return True
    receipt = await host.launch_app(app_name)
    return bool(getattr(receipt, "success", False))


async def click_normalized(
    x: float,
    y: float,
    *,
    expect_app: str = "",
    bounds: Sequence[int] | None = None,
) -> bool:
    """Click a point given in 0..1 coordinates, top-left origin.

    `bounds` is the rectangle those coordinates are normalized AGAINST — the
    window when the reading was scoped to one, the whole display otherwise.

    Passing it is not optional bookkeeping. Scoping perception to a window
    changed what 0..1 means, and this converter still assumed the display, so
    every dismissal click landed hundreds of pixels from its target: the loop
    saw the dialog, decided correctly to close it, clicked somewhere else, and
    tried again for forty cycles. Two halves of one system disagreeing about a
    coordinate frame is silent by construction — both look right in isolation.

    The same focus guard applies as for keystrokes, because a click at the
    wrong window is a click on someone else's document.
    """
    from core.capabilities.host_automation import get_host_automation

    host = get_host_automation()
    if expect_app:
        refusal = await host._refuse_if_not_frontmost(expect_app, "click_at")
        if refusal is not None:
            return False

    if bounds and len(bounds) >= 4:
        left, top, width, height = (int(value) for value in bounds[:4])
    else:
        left, top = 0, 0
        width, height = await _screen_size()
    if not width or not height:
        return False
    receipt = await host.click_at(
        int(round(left + x * width)), int(round(top + y * height))
    )
    return bool(getattr(receipt, "success", False))


async def _screen_size() -> tuple[int, int]:
    """Main display size in pixels, or (0, 0) when it cannot be read."""
    try:
        from AppKit import NSScreen

        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return (0, 0)


async def press(key: str, *, expect_app: str = "") -> bool:
    """Press one of the allowed keys. False if it is not one of them.

    `expect_app` is passed through to the focus guard. A loop that acts on what
    it sees must aim its input at the window it was looking at: measured live,
    a run opened a page in Chrome, read the board correctly, and sent its keys
    to whatever the person had clicked since — reported as success, with the
    board untouched.
    """
    name = str(key or "").strip().lower()
    if name not in PRESSABLE_KEYS:
        return False
    from core.capabilities.host_automation import get_host_automation

    receipt = await get_host_automation().hotkey(name, expect_app=expect_app)
    return bool(getattr(receipt, "success", False))


async def press_many(keys: Sequence[str], *, expect_app: str = "") -> int:
    """Press several allowed keys in order, in one call. Returns how many landed.

    Spawning the automation costs about a third of a second whatever it
    carries, so a loop pressing one key at a time pays that on every move.
    The focus guard still runs, once for each key, because focus can move
    part-way through a batch.

    The count is what a caller narrating its own moves needs. One flag for
    the batch would let her say four moves when the window went away after
    the second, and what she says has to be what her body did.
    """
    wanted = [str(key or "").strip().lower() for key in keys]
    wanted = [key for key in wanted if key in PRESSABLE_KEYS]
    if not wanted:
        return 0
    from core.capabilities.host_automation import get_host_automation

    receipt = await get_host_automation().hotkeys(wanted, expect_app=expect_app)
    evidence = dict(getattr(receipt, "evidence", None) or {})
    if "keys_sent" in evidence:
        try:
            return max(0, min(len(wanted), int(evidence["keys_sent"])))
        except (TypeError, ValueError) as why:
            # How many keys landed is the whole question here, so a count that
            # will not parse is worth saying rather than falling through to a
            # guess from the success flag.
            logger.info("the host reported an unreadable key count: %s", why)
    return len(wanted) if bool(getattr(receipt, "success", False)) else 0


class ScreenPursuitSkill(BaseSkill):
    """Keep looking at the screen and acting until a goal is reached."""

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



def screen_options(keys: Sequence[str] = DEFAULT_MOVES) -> list[Any]:
    """The moves really available on a screen, each carrying its own test.

    An option states what should be different once it lands, so the check is
    made by measurement rather than by asking the same faculty that chose.
    For a keypress the honest claim is narrow: the view is not what it was.
    A key that changes nothing is a key that did nothing, whatever the
    keystroke's own receipt said.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    options: list[Any] = []
    for key in keys:
        name = str(key).strip().lower()
        if name not in PRESSABLE_KEYS:
            continue
        options.append(
            ActionOption(
                name=name,
                detail=f"press {name}",
                expectation=Expectation(
                    changed=True,
                    describes=f"the view to be different after {name}",
                ),
            )
        )
    return options



#: Controls that begin a task again, by the words they are usually labelled
#: with. Matched against what is really on screen — never inferred, and never
#: clicked unless the run has actually stopped getting anywhere.
RESTART_LABELS = ("new game", "restart", "play again", "try again", "start over", "reset")
#: The two ways out of an impasse that are not "keep pressing".
START_OVER = "start over"
SEE_IT_THROUGH = "see it through"


def restart_control(observation: dict[str, Any]) -> tuple[str, float, float] | None:
    """A control on screen that would begin the task again, if there is one."""
    for region in observation.get("layout") or []:
        text = str(region.get("text") or "").strip()
        if not text or len(text) > 40:
            continue
        lowered = text.lower()
        if not any(label in lowered for label in RESTART_LABELS):
            continue
        try:
            return (
                text,
                float(region.get("center_x", region.get("x"))),
                float(region.get("center_y", region.get("y"))),
            )
        except (TypeError, ValueError):
            continue
    return None


def ways_out(observation: dict[str, Any], *, ended: bool = False) -> list[Any]:
    """What she can do about being stuck, as options she chooses between.

    A loop whose only moves are inside the task can only press harder at
    something that has stopped working. These are moves about the task: begin
    it again knowing what she now knows, or finish it badly on purpose,
    because the ending is where the evidence about how it goes wrong is.

    Offered only once the in-task moves have demonstrably stopped working, so
    an ordinary run never sees them and nothing gets restarted casually.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    options: list[Any] = [
        ActionOption(
            name=SEE_IT_THROUGH,
            detail="keep playing this out and learn from how it ends",
            # Needing a reason in words protects live work from being thrown
            # away on a ranking. Where nothing answers any more there is no
            # live work to protect, and the alternative to choosing is
            # pressing keys into something that has finished.
            needs_words=not ended,
            expectation=Expectation(
                changed=False, describes="to reach the end of this attempt and know why it failed"
            ),
        )
    ]
    control = restart_control(observation)
    if control is not None:
        label, x, y = control
        options.insert(
            0,
            ActionOption(
                name=START_OVER,
                params={"label": label, "x": x, "y": y},
                needs_words=not ended,
                detail=f"begin again with {label!r}, knowing what this attempt taught",
                expectation=Expectation(changed=True, describes="a fresh start on the same task"),
            ),
        )
    return options



#: What she can do when her voice falls behind her hands. Moves about
#: herself rather than about the task, in the same shape as the ways out of
#: an impasse: offered only when the situation is real, chosen through the
#: ordinary deliberation, and recorded with a reason.
SLOW_DOWN = "slow down"
SAY_LESS = "say less"
PRESS_ON = "press on"


def narration_backlog() -> dict[str, int]:
    """How far behind the voice is. Empty when there is no surface to speak to."""
    try:
        from core.perception.ambient_presence import get_ambient_presence

        return dict(get_ambient_presence().narration_backlog())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return {}


def pacing_options(backlog: dict[str, int]) -> list[Any]:
    """What she can do about acting faster than she can speak.

    Two faculties running at once will not run at the same speed. Noticing
    that is not enough on its own — noticing without a lever is a status
    line — so each of these is something she can actually do: wait for the
    voice to catch up, say less per move, or carry on and let some of it go
    unsaid. Any of the three is a defensible answer, which is why it is a
    decision and not a rule.
    """
    from core.agency.deliberate_action import ActionOption, Expectation

    waiting = int(backlog.get("waiting", 0) or 0)
    if waiting <= 0:
        return []
    return [
        ActionOption(
            name=SLOW_DOWN,
            detail=f"wait for my commentary to catch up — {waiting} line(s) behind",
            expectation=Expectation(changed=False, describes="my words to catch up with my hands"),
        ),
        ActionOption(
            name=SAY_LESS,
            detail="name each move without explaining it, and keep this pace",
            expectation=Expectation(changed=False, describes="to keep up by saying less"),
        ),
        ActionOption(
            name=PRESS_ON,
            detail="carry on at this pace and let some of it go unsaid",
            expectation=Expectation(changed=False, describes="to keep playing and lose some commentary"),
        ),
    ]


async def let_the_voice_catch_up(before: dict[str, int], *, patience: float = 4.0) -> None:
    """Wait for the backlog to drain, bounded, without inventing a delay.

    Sized by the thing actually being waited for rather than by a number
    somebody picked: it returns as soon as the queue is shorter than it was,
    and gives up after ``patience`` seconds so a surface nobody is reading
    cannot stall the run.
    """
    started = time.monotonic()
    was = int(before.get("waiting", 0) or 0)
    while time.monotonic() - started < patience:
        await asyncio.sleep(0.25)
        now = int(narration_backlog().get("waiting", 0) or 0)
        if now < was:
            return


def _ask_again_after(asked_at: int) -> int:
    """How many moves may pass before the question is put again.

    A first plan can wait longer than a second one, because the first is
    waiting for the screen to say which part of it is the task. The horizon
    it waits to is the one past which an approach nobody revisits is a habit.
    """
    from core.agency.standing_strategy import RECONSIDER_AFTER

    return LANGUAGE_EVERY if asked_at >= 0 else RECONSIDER_AFTER


def _time_left(began: float, max_seconds: float, deadline_at: float) -> float:
    """What is left of the budget, on whichever clock started first.

    A caller that began counting before this action did says so, and its
    deadline wins: otherwise the setup between the two is free time that the
    outer deadline is then blamed for.
    """
    now = time.monotonic()
    ends_at = began + float(max_seconds)
    if deadline_at > 0.0:
        ends_at = min(ends_at, float(deadline_at))
    return max(1.0, ends_at - now)


def _within_the_run(think: Any, ends_at: float) -> Any:
    """Her thinking, bounded by what is left of the run rather than its own budget.

    A cycle checks the clock at its top and then goes away to think. When the
    thought outlasts the run, the deadline is only noticed after it returns,
    and by then the caller outside — which has room to report and nothing
    more — has already cancelled everything. LIVE 2026-08-26: twenty-nine
    narrated moves, a 64 built into the corner, and "Operation took too long.
    Completed 0/0 steps."
    """
    if think is None or ends_at <= 0.0:
        return think

    async def bounded(objective: str, evidence: Any) -> Any:
        left = ends_at - time.monotonic()
        if left <= 1.0:
            raise TimeoutError("the run is out of time to think")
        return await asyncio.wait_for(think(objective, evidence), timeout=left)

    return bounded


def _say_what_kind_of_problem(
    knows: Any, acts: Any, state: Any, toward: str, said_already: dict[str, bool]
) -> None:
    """Name the shape of what she is in, once she has worked out enough to name it.

    Recognising the kind of problem is the general part; what it calls for is
    allowed to be as specialised as the problem is. Said out loud because a
    watcher cannot otherwise tell a mind that recognised its situation from
    one that got lucky in it.
    """
    if said_already.get("shape"):
        return
    try:
        from core.agency.what_kind_of_problem import recognise  # noqa: PLC0415

        suits = recognise(
            acts=[getattr(option, "name", str(option)) for option in acts or ()],
            knows_how_it_moves=getattr(knows, "rules", knows),
            state=state,
            toward=toward,
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="acted without naming the problem"
        )
        return
    if not suits.shape.transition_known:
        return
    said_already["shape"] = True
    _tell(f"I know what kind of thing this is now: {suits.shape.named()}.")


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

#: How many screenfuls down she will look for the thing before deciding the
#: page does not have one. A page is taller than a screen and what she came
#: for is usually below the writing about it — six is enough to clear a
#: heading, a paragraph and an advertising rail without walking a long article
#: end to end.
SCREENFULS_TO_LOOK = 6

#: The least a reading has to hold before it counts as a thing laid out rather
#: than as prose that happens to have numbers in it.
ENOUGH_TO_BE_A_THING = 4


def _is_a_thing_laid_out(reading: Any) -> bool:
    """Whether this reading holds something arranged in rows and columns."""
    rows = int(getattr(reading, "rows", 0) or 0)
    columns = int(getattr(reading, "columns", 0) or 0)
    occupied = getattr(reading, "occupied", None)
    if rows < 2 or columns < 2 or not callable(occupied):
        return False
    return occupied() >= ENOUGH_TO_BE_A_THING


async def _bring_it_into_view(look: Any, read: Any, cannot_see: dict[str, str] | None = None) -> int:
    """Scroll down until what she came for is on screen, or the page runs out.

    A page is taller than a screen. She opened a real sliding puzzle, read the
    heading and the advertising above it, found no part of what she could see
    that answered to her, and reported truthfully that nothing on screen
    offered a move — with the board eleven screenfuls further down. LIVE
    2026-08-27, and the run ended having made none.

    Scrolling commits to nothing. It moves a view and is undone by moving
    back, which is the same line drawn around every other input she is allowed
    to try without knowing what it will do.

    Returns how far down she had to go, so a caller can say so. ``cannot_see``
    is filled in when the screen could not be read at all, which is a
    different fact from finding nothing on it.
    """
    cannot_see = {} if cannot_see is None else cannot_see
    from core.capabilities.host_automation import get_host_automation

    try:
        hands = get_host_automation()
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation("screen_pursuit", exc, action="scroll to find the thing")
        return 0
    for down in range(SCREENFULS_TO_LOOK):
        seen = await look()
        if not seen.get("ok"):
            why = str(seen.get("error") or "no reason given")
            logger.info(
                "looking for what she came for, %d down: nothing could be read (%s)", down, why
            )
            cannot_see["reason"] = why
            return down
        here = read(seen)
        if _is_a_thing_laid_out(here):
            logger.info(
                "what she came for is %d screenful(s) down: %dx%d with %d thing(s) in it",
                down,
                here.rows,
                here.columns,
                here.occupied(),
            )
            return down
        # Say what WAS there. A run that ends "nothing offered a move" names
        # the symptom and hides whether she read a page with no grid on it, a
        # grid too small to count, or nothing at all.
        logger.info(
            "%d down: read %d region(s), %dx%d with %d thing(s) — not a thing laid out yet",
            down,
            len(seen.get("layout") or ()),
            here.rows,
            here.columns,
            here.occupied(),
        )
        try:
            await hands.scroll(dy=-_a_screenful(hands))
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
            record_degradation("screen_pursuit", exc, action="scroll to find the thing")
            return down
        await asyncio.sleep(SETTLE_AFTER_SCROLL_S)
    return SCREENFULS_TO_LOOK


#: What one scroll goes if the screen cannot be measured. Small enough that
#: nothing is stepped over on any display anyone still uses.
A_SCREENFUL_AT_LEAST = 400

#: The share of a screen one scroll moves. Not the whole of it, so a thing
#: sitting across the fold is never skipped between two readings.
MOST_OF_A_SCREEN = 0.8

#: Long enough for a page to finish moving before it is read again.
SETTLE_AFTER_SCROLL_S = 0.35


def _a_screenful(hands: Any) -> int:
    """How far one scroll should go, from the screen she is actually looking at.

    A number picked in advance is wrong on every display but one. Most of a
    screen rather than all of it, so something sitting across the fold is not
    skipped between two readings.
    """
    measure = getattr(hands, "_main_screen_visible_frame", None)
    if not callable(measure):
        return A_SCREENFUL_AT_LEAST
    try:
        height = int(measure()[3])
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, IndexError):
        return A_SCREENFUL_AT_LEAST
    return max(A_SCREENFUL_AT_LEAST, int(height * MOST_OF_A_SCREEN))


def _worth_holding(found: Any, whole: Any, seen: dict[Any, int] | None = None) -> Any:
    """The block worth carrying to the next glance.

    A reading taken before the page settles has no lattice in it — nothing
    drawn yet, or a panel over the top — and what comes back is the whole
    reading rather than a block of it. Held as though it were a block, it
    forces every later reading back to the whole page, and she never finds the
    thing at all. LIVE 2026-08-29: "reading 13x8" for an entire run on a
    four-by-four board, every comparison after the first discarded as
    unreadable, and no rule ever formed.

    And the shape she carries is the one she has SETTLED on, not the one she
    saw a moment ago. Anchoring on the last glance means one bad glance drops
    the anchor and she begins again: live 2026-08-31 the readings went 3x4,
    then unreadable, then 3x4, and one move in three could be compared with
    another. A thing does not change shape, so the shape she has read most
    often is the better guess about it than the shape she read last — and a
    single misreading no longer costs her the thing.
    """
    if found is None or whole is None:
        return None
    inside = found.rows * found.columns < whole.rows * whole.columns
    if not inside:
        return None
    # `is not None`, because an empty tally is falsy and the first shape would
    # never be recorded — so it stayed empty, and nothing was ever settled.
    if seen is not None:
        seen[(found.rows, found.columns)] = seen.get((found.rows, found.columns), 0) + 1
        settled = max(seen, key=lambda shape: (seen[shape], shape[0] * shape[1]))
        if seen[settled] > 1 and (found.rows, found.columns) != settled:
            # This glance disagrees with what it has usually been. Keep the
            # shape rather than the glance; a short reading is placed inside a
            # known shape, where a differently-shaped one cannot be compared
            # at all.
            return _AS_IT_USUALLY_IS.get(settled) or found
        _AS_IT_USUALLY_IS[settled] = found
    return found


#: The last reading of each shape she has settled on, so a glance that
#: disagrees can be placed in one rather than replace it.
_AS_IT_USUALLY_IS: dict[Any, Any] = {}


def _what_there_is_to_aim_at(reading: Any) -> str:
    """What to prefer one situation over another by, when nobody said.

    A request can name a process without naming a finish — "play it and work
    out how it moves" — and then there is nothing to score a future against,
    so she acts and looks for as long as the budget lasts and never uses the
    model she is building. That is a waste of the thing she just worked out.

    What she can read off the world instead is whether it counts, and how far
    it could go. A laid-out thing that combines equal pairs cannot exceed one
    doubling per place it has: sixteen places cannot hold more than two to the
    sixteenth however well it is played. That ceiling is a fact about the thing
    in front of her rather than a number anybody picked, and it is far enough
    above where she is that being nearer to it stays worth something all the
    way through — which a nearer goal does not, because arriving at one makes
    every situation after it look equally good.

    Measured 2026-08-27, played the way the loop plays it — the goal put
    through the same gate, six games each, run to a dead board:

        said "the largest"          median 128, and not one look ahead
        read the ceiling off it     median 768, max 1024, 666 looks ahead

    The first is random. Not because the words are wrong but because nothing
    downstream can use them: worth_comparing refuses a goal it cannot measure,
    the search never runs, and every move is a coin toss on a board she can
    read perfectly well.

    Where the things in front of her are not numbers, nothing here invents a
    purpose: it says so, and she goes back to acting and looking.
    """
    numbers = getattr(reading, "numbers", None)
    places = getattr(reading, "places", None)
    if not callable(numbers) or not callable(places):
        return ""
    if not numbers():
        return ""
    room = int(places() or 0)
    if room <= 0:
        return ""
    return f"{2 ** room}"


def _left_her_better_off(
    before: Any, after: Any, toward: str, approach: str
) -> bool:
    """Whether the move improved the situation, by the measure she is using.

    The same measure that ranks futures ranks what actually happened, so what
    she gets good at is what her own judgement says was worth doing rather
    than a separate opinion about it.
    """
    try:
        from core.agency.how_good_is_this import how_good

        was = how_good(before, toward=toward, approach=approach)
        now = how_good(after, toward=toward, approach=approach)
    except (ImportError, AttributeError, TypeError, ValueError) as why:
        logger.debug("could not weigh whether that left her better off: %s", why)
        return False
    return now >= was


def _what_she_is_not_reading(rules: Any) -> str:
    """Whether her own record proves a quantity she cannot see.

    "How this moves is not worked out yet" is true of two different worlds and
    says nothing about which. In one, a rule is there and she has not found it,
    and more moves are the answer. In the other, the same board and the same
    key came out two ways, so no rule reading only the board can ever fit, and
    more moves are looking where the answer cannot be.

    Watching is perception's; reading what the watching proves is not, so the
    record leaves the model whole and the question is asked here.
    """
    from core.cognition.something_she_cannot_see import what_she_cannot_see

    try:
        record = rules.what_she_saw_happen()
    except (AttributeError, TypeError):
        return ""
    if len(record) < 2:
        return ""
    found = what_she_cannot_see(
        [((seen, did), then) for seen, did, then in record]
    )
    if not found.anything:
        return ""
    if found.she_can_compute_it:
        return (
            f" — and one thing she was not reading, which runs every "
            f"{found.every} moves, so she can"
        )
    return (
        f" — and one thing she was not reading, taking {found.how_many} values "
        "in no cycle, so the world puts something there she does not control"
    )


def _say_what_she_worked_out(knows: Any, said_already: dict[str, bool]) -> None:
    """Say it the once, when she first works out how a thing moves."""
    rules = getattr(knows, "rules", None)
    if rules is None or said_already.get("said"):
        return
    if getattr(rules, "rule", lambda: None)() is None:
        return
    said_already["said"] = True
    logger.info("she can see ahead now: %s", rules.says())
    _tell(f"I can see what my moves do here now — {rules.says()}.")


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


def am_i_there(wanted: str, reading: str, page: str, window: str) -> bool:
    """Whether this is the thing she was asked to act in.

    She was asked to find something and act in it, and nothing anywhere
    checked that she had: a reading is a reading, and anything with text laid
    out in rows reads as something she could push. LIVE 2026-08-26, another
    part of her closed the browser mid-run and she played twelve moves of 2048
    into a chat window, narrating every one of them.

    Identity where there is identity — an address, a title, the name of the
    window — and the reading itself where there is not. What is not accepted
    is silence: a name she was given and cannot find anywhere is a name she
    has not arrived at.
    """
    name = " ".join(str(wanted or "").split()).lower()
    if not name:
        return True
    said = [word for word in re.split(r"[^a-z0-9]+", name) if len(word) > 2]
    if not said:
        return True
    # Identity first, and the reading only when there is no identity.
    #
    # A screen reading is of the screen, not of her window, so anything else
    # visible counts as evidence that she has arrived. LIVE 2026-08-26: the
    # word she was looking for was in a terminal on the same display, the test
    # passed, and she played into that instead. What identifies a thing — an
    # address, a title, the name of the window — cannot be borrowed from
    # somebody else's window the way words on a screen can.
    known = " ".join((str(page or ""), str(window or ""))).lower().strip()
    if known:
        return any(word in known for word in said)
    return any(word in str(reading or "").lower() for word in said)


async def _take_the_run_its_bearings(
    anchor: dict[str, str], *, expect_page: str = "", open_page: str = ""
) -> None:
    """Work out which page and which window this run belongs to.

    Done before anything depends on the answer, because everything does: what
    is brought forward, what a keystroke is bound to, and whether she is in
    the thing she was asked to act in at all.
    """
    page = await current_page_identity()
    if not anchor["page"]:
        anchor["page"] = str(
            expect_page or page.get("url") or page.get("title") or ""
        ).strip()
    if not anchor["app"]:
        # The application that holds the page, when this run is about a page
        # at all. A task about a desktop application would otherwise anchor
        # itself to a browser that happens to be open behind it.
        about_a_page = bool(open_page or expect_page or page.get("url"))
        holder = str(page.get("app") or "") if about_a_page else ""
        anchor["app"] = (holder or await _frontmost() or "").strip()
    if anchor["app"]:
        logger.info(
            "this run belongs to %r on %r", anchor["app"], anchor["page"][:60]
        )


#: The one key she may send without knowing where it will land.
#:
#: Every other keystroke needs to be bound to a window, and the reason is on
#: the record: unbound arrow keys played thirty-five moves of a game into a
#: chat window. Escape is different in kind rather than in degree. It declines,
#: it commits to nothing, it is reversible, and it is the platform-standard
#: way out of a modal on every desktop. Sending it at whatever has the
#: keyboard is the only way to reach a thing that took the keyboard from her.
DECLINES_AND_NOTHING_ELSE = "escape"


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


async def _the_best_reading_available(
    observation: dict[str, Any],
    band: tuple[float, float, float, float] | None,
    *,
    like: Any,
    in_a_browser: bool,
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

    seen = what_is_there(observation, band, like=like)
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
    from_page = what_the_page_is_showing(said, band, like=like)
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

    told = False
    while True:
        try:
            admission = await evaluate_screen_capture_admission_async()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return True
        if admission.allowed or admission.reason.value != "session_locked":
            if told:
                logger.info("the screen is back; carrying on")
            return admission.allowed
        left = ends_at - time.monotonic()
        if left <= 0.0:
            logger.info("the screen stayed locked for the whole of this task")
            return False
        if not told:
            logger.info("the screen is locked; waiting for it rather than failing")
            # Say it. Waiting in silence for the whole deadline and then
            # explaining is the same information delivered too late to act
            # on — the person is the one who can unlock it, and they cannot
            # do that if nothing tells them.
            await _narrate(
                "Your screen is locked, so there is nothing for me to look at "
                "yet. I will start the moment it is unlocked."
            )
            told = True
        await asyncio.sleep(min(max(1.0, left / 60.0), left))


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

        receipt = await get_host_automation().focus_app(named)
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


def _her_reasoning(stakes: float) -> Any:
    """Her own judgement, sized to what rides on the move."""
    from core.agency.her_reasoning import reasoning_for

    return reasoning_for(stakes)


def _reasoning_for_a_plan() -> Any:
    """Her judgement on how to go about something, which is not a move."""
    from core.agency.her_reasoning import reasoning_for_a_plan

    return reasoning_for_a_plan()


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
    from core.agency.deliberate_action import Attempt, confirm, deliberate
    from core.agency.how_good_is_this import a_trial_is_running as _a_trial_is_running
    from core.agency.how_good_is_this import worth_comparing
    from core.agency.looking_ahead import look_ahead
    from core.agency.standing_strategy import Strategy, settle_on_an_approach, still_holds
    from core.agency.task_knowledge import learn_about, stuck, work_out_what_it_means
    from core.agency.what_i_can_do_here import WhatWorksHere
    from core.agency.what_i_cannot_explain import WhatICannotExplain
    from core.agency.what_worked_before import WhatWorkedBefore
    from core.agency.worth_thinking_about import worth_a_pass
    from core.perception.how_it_moves import HowItMoves
    from core.perception.the_thing_itself import the_thing_itself
    from core.perception.what_the_world_does import WhatTheWorldDoes
    from core.perception.where_it_responds import (
        Responsive,
        describe,
        noticed,
        what_is_there,
        within,
    )
    from core.perception.why_nothing_answers import ELSEWHERE, ENDED, work_out_why
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
        return {
            "ok": False,
            "outcome": "no_screen_to_look_at",
            "error": "the screen is locked, so there is nothing for me to look at yet",
            "moves": [],
        }
    moves: list[dict[str, Any]] = []
    history: list[Attempt] = []
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
    #: Where the page says it draws, asked once when the run gets its bearings.
    drawn: dict[str, Any] = {"where": None, "asked": False}
    #: Why the last cycle ended without a move. Eleven different facts used
    #: to arrive at the executor as one silence, and the run then reported
    #: "nothing on screen offered a move" for every one of them — which cost
    #: three wrong diagnoses in a row before this line existed.
    no_move: dict[str, str] = {"because": ""}
    #: Whether a restart control has appeared, which is a thing saying it has
    #: finished. Held so it is said once rather than every cycle.
    offered_a_restart: dict[str, bool] = {"value": False}
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
        # Put the target back in front before looking at it.
        #
        # Over a long run focus wanders: a notification, a click that lands
        # outside the window, the person switching away. Without this the loop
        # refuses every keystroke for the rest of the run and reads whatever
        # replaced its target — technically correct and completely stuck. A
        # task that is meant to last minutes has to be able to recover the
        # conditions it needs rather than only detect that they are gone.
        # The window this run belongs to, named by the caller or learned on
        # the first cycle. Either way it has to be in front to be acted in.
        # Work out what this run belongs to before anything depends on it.
        #
        # This whole block sat inside "if the run has a window", and the only
        # thing that could give it one was inside the block. With nothing
        # named the run never acquired an anchor, never brought anything
        # forward, and sent every key to whatever happened to be in front —
        # for want of a first cycle it could never have. LIVE 2026-08-26:
        # thirty-five moves into a terminal with the game one window back.
        if not anchor["page"] or not anchor["app"]:
            await _take_the_run_its_bearings(
                anchor, expect_page=expect_page, open_page=open_page
            )
        mine = target_app or anchor["app"]
        if mine:
            try:
                await _ensure_frontmost(mine)
                # Anchor to the page this run STARTED on when the caller did
                # not name one.
                #
                # Otherwise a run is bound to an application and nothing more,
                # and an application is not a context: the browser holds the
                # task's page and a dozen others. A run that only knows "Google
                # Chrome" will send its keys to whatever tab is in front, so
                # arrow keys meant for a game land on a video, a form, or
                # someone's mail — each keystroke legitimately delivered to the
                # wrong world. Measured live: a stray click moved the browser to
                # a different site and the loop kept acting there.
                #
                # Anchoring on the first cycle means a caller never has to
                # remember, and drift is always detectable rather than only
                # detectable when someone thought to declare an expectation.
                if not anchor["page"]:
                    page = await current_page_identity()
                    anchor["page"] = str(
                        expect_page or page.get("url") or page.get("title") or ""
                    ).strip()
                    # And the window it is in, so a keystroke has somewhere it
                    # belongs.
                    #
                    # Nothing named meant nothing checked: an empty target is
                    # read as "no constraint" rather than "I do not know where
                    # I am", so every guard passed and every key went to
                    # whatever happened to be in front. LIVE 2026-08-26:
                    # another part of her closed the browser mid-run, and she
                    # played twelve moves of 2048 into a chat window.
                    if not anchor["app"]:
                        # The application that holds the page, when this run is
                        # about a page at all. A task about a desktop
                        # application would otherwise anchor itself to a
                        # browser that happens to be open behind it.
                        about_a_page = bool(open_page or expect_page)
                        holder = str(page.get("app") or "") if about_a_page else ""
                        anchor["app"] = (holder or await _frontmost() or "").strip()
                        if anchor["app"]:
                            logger.info("this run belongs to %r", anchor["app"])
                if anchor["page"] and not await _ensure_page(anchor["page"]):
                    # The page this run is about is no longer in front and
                    # could not be brought back. Reading on would be reading
                    # someone else's page.
                    lost_page["value"] = True
                    return {
                        "ok": False, "text": "", "layout": [],
                        "error": "navigated_away", "at": time.time(),
                    }
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "screen_pursuit",
                    exc,
                    severity="info",
                    action="continued a pursuit without refocusing the target window",
                )
        try:
            return await asyncio.wait_for(
                read_screen(target_app, over=drawn["where"]), timeout=OBSERVE_TIMEOUT_S
            )
        except TimeoutError:
            # A wedged capture is not a reason to keep acting blind.
            logger.info(
                "the screen did not answer inside %.1fs", OBSERVE_TIMEOUT_S
            )
            return {
                "ok": False,
                "text": "",
                "layout": [],
                "error": f"observe_timeout: no reading inside {OBSERVE_TIMEOUT_S:.1f}s",
            }

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
        """A Step that clears whatever is covering the content, or None.

        Tried BEFORE the policy on every cycle, because a dialog owning the
        screen makes every other decision meaningless: the reading is of the
        dialog, and the keys go to the dialog. Measured live — a page opened,
        was read correctly, and six moves in a row changed nothing because a
        modal had focus, with every keystroke reporting success.

        The judgement lives in core/perception/blocking_overlay.py, which
        dismisses and never agrees: a dialog offering only acceptance is
        reported as the person's decision and is left alone. This loop
        inherits that and adds nothing to it.
        """
        try:
            from core.perception.blocking_overlay import assess_overlay
        except ImportError as why:
            logger.info("cannot judge what is covering the content: %s", why)
            return None
        verdict = assess_overlay(observation, intending=intending["value"])
        if verdict.needs_person:
            # A dialog only the person can answer means the task cannot go on.
            #
            # This used to fall through to the policy, which then acted into a
            # dialog that owned the keyboard: measured live, forty moves that
            # each reported success while the board behind the dialog never
            # changed. Continuing past a question addressed to the person is
            # not perseverance, it is acting blind — and it is how a run
            # eventually stumbles into answering that question by accident.
            # Seen twice before the task is handed back.
            #
            # Handing it back ends the run, and a page carries things that go
            # away on their own: an advertising rail rotates, a toast appears
            # and fades, a banner loads late. Measured live, a run stopped
            # after twelve moves over a rail that had already changed by the
            # next reading. A dialog only the person can answer is still
            # there a second later, which is the whole difference.
            same = verdict.needs_person == needs_person["seen"]
            needs_person["seen"] = verdict.needs_person
            needs_person["times"] = needs_person["times"] + 1 if same else 1
            if needs_person["times"] < TWICE_BEFORE_HANDING_BACK:
                return None
            needs_person["reason"] = verdict.needs_person
            return None
        if not verdict.present:
            # Whatever it was is gone, so it was not the thing that stops a run.
            needs_person["seen"] = ""
            needs_person["times"] = 0
            return None

        # A caller may declare its OWN way forward.
        #
        # The detector deliberately refuses to guess at unlabelled controls,
        # because clicking an unknown thing on someone's screen is how a tab
        # gets closed or a row deleted. But a task usually knows its own
        # affordance — "New Game", "Start", "Continue", "Begin" — and that
        # knowledge belongs to whoever set the goal, not to a generic reader.
        #
        # Declared rather than inferred: the loop still never guesses, it just
        # accepts an instruction. Matched case-insensitively against the placed
        # text, and only when something is genuinely in the way.
        if unblock_with:
            wanted = unblock_with.strip().lower()
            candidates: list[tuple[float, float]] = []
            for region in observation.get("layout") or []:
                if wanted not in str(region.get("text") or "").strip().lower():
                    continue
                try:
                    candidates.append(
                        (
                            float(region.get("center_x", region.get("x"))),
                            float(region.get("center_y", region.get("y"))),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            if candidates:
                # The one ON the dialog, when the label appears more than once.
                #
                # "New Game", "Start" and "Continue" routinely name both a
                # dialog's button and a permanent control in the app's own
                # toolbar. Measured live: four regions matched, and the first
                # was the toolbar — clicking it started a game BEHIND the
                # dialog and left the dialog up, so the run stayed blocked
                # while every step reported success.
                from core.perception.blocking_overlay import (
                    overlay_box,
                    overlay_focus,
                )

                box = overlay_box(observation)
                if box is not None:
                    left, top, right, bottom = box
                    # Inside the dialog's horizontal span, at or below its
                    # text. A margin, because a button may sit slightly wider
                    # than the sentence above it; and downward only, because a
                    # dialog's controls are under its message.
                    margin = 0.08
                    inside = [
                        point
                        for point in candidates
                        if left - margin <= point[0] <= right + margin
                        and top - 0.02 <= point[1] <= bottom + 0.35
                    ]
                    if inside:
                        candidates = inside
                focus = overlay_focus(observation)
                if focus is not None:
                    candidates.sort(
                        key=lambda point: (point[0] - focus[0]) ** 2
                        + (point[1] - focus[1]) ** 2
                    )
                ux, uy = candidates[0]
                label = unblock_with
                frame = list(observation.get("bounds") or [])

                async def click_declared() -> bool:
                    return await click_normalized(
                        ux, uy, expect_app=target_app or anchor["app"], bounds=frame
                    )

                return Step(
                    name=f"clear the way with {label!r}", action=click_declared
                )

        if verdict.suggested_key:
            key = verdict.suggested_key

            async def press_away() -> bool:
                return await press(key, expect_app=target_app or anchor["app"])

            return Step(name=f"dismiss overlay with {key}", action=press_away)

        if verdict.click_x is not None:
            label, x, y = verdict.label, verdict.click_x, verdict.click_y

            frame = list(observation.get("bounds") or [])

            async def click_away() -> bool:
                return await click_normalized(
                    x, y, expect_app=target_app or anchor["app"], bounds=frame
                )

            return Step(name=f"dismiss overlay via {label!r}", action=click_away)
        return None

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
        "state": Responsive.from_memory(knew.get("responds") or {}, TRUST_CARRIED_OVER)
    }
    #: Said once, when the thing she is working in stops answering at all.
    said_it_ended: dict[str, bool] = {"value": False}
    #: Whether she has confirmed being in the thing she was asked to act in.
    confirmed_here: dict[str, bool] = {"value": False}
    not_there: dict[str, str] = {"reason": ""}
    #: What she last tried to move out of her way, so a thing that will not
    #: close is not pressed at once a cycle for the rest of the run.
    in_the_way: dict[str, str] = {"last": ""}

    async def decide(observation: dict[str, Any]) -> Step | None:
        # Something in front of her work is cleared before anything else.
        #
        # Not once it has cost her four moves discovering that nothing
        # answers: a dialog that owns the keyboard makes every reading and
        # every keystroke of this cycle meaningless, and a person closes it
        # and carries on rather than playing on underneath it.
        # Over the part of the screen she is using, not merely above her
        # window. Before the answering part is worked out this is None and
        # anything on top counts, which is the honest reading while she does
        # not yet know where the task lives.
        in_front = await _whats_on_top(
            target_app or anchor["app"], over=responds["state"].band()
        )
        if in_front and in_front != in_the_way["last"]:
            in_the_way["last"] = in_front
            if await clear_what_is_in_front(in_front):
                in_the_way["last"] = ""
                if narrate:
                    _tell(f"{in_front} was in front of this. Closed it.")
                no_move["because"] = "a blocker was cleared, so this cycle is spent"
                return None

        blocker = await clear_blocker(observation)
        if blocker is not None:
            # Verified, not assumed. A blocker still present after the previous
            # attempt means that attempt did not work, whatever its receipt
            # said.
            if blocker_attempts["count"] >= MAX_BLOCKER_ATTEMPTS:
                blocker_attempts["last"] = blocker.name
                no_move["because"] = "something is in front of it that will not move"
                return None
            blocker_attempts["count"] += 1
            blocker_attempts["dismissed"] += 1
            blocker_attempts["last"] = blocker.name
            return blocker
        if needs_person["reason"]:
            no_move["because"] = "declining what is in front of it"
            return None
        blocker_attempts["count"] = 0
        if not observation.get("ok"):
            no_move["because"] = "waiting for what is in front of it to go"
            return None

        # What she is looking at, kept to the part that answers to her.
        #
        # A reading of a screen is everything on it. On the page holding a
        # game that is the board, the score, two advertising rails and a
        # copyright line, so what she recalls about "a situation like this
        # one" is dominated by whichever advertisement was loaded — and two
        # readings of the same board look like different situations because
        # the advertising rotated under her.
        #
        # Which part is the task is not written anywhere on the page, but it
        # is answered by what happens when she acts. Until enough acts have
        # answered it, this is the whole reading, because a guess about where
        # the task is would be worse.
        if not drawn["asked"]:
            drawn["asked"] = True
            from core.perception.what_the_page_says import where_the_drawing_is

            drawn["where"] = await where_the_drawing_is()
            if drawn["where"] and narrate:
                _tell("The page told me where it is drawing — I will look there.")

        # Where the task lives, asked rather than worked out, when it can be.
        #
        # The band is normally learned: act, look, and see which places
        # changed. That is right for anything that cannot be questioned, and
        # slow — it takes many moves, and until it settles she is reading
        # browser tabs and advertising rails as part of the thing. A page that
        # draws its content can say exactly where it draws, and then she knows
        # on the first cycle what would otherwise take twenty.
        #
        # LIVE 2026-08-29: play2048.co draws its board on a canvas. She was
        # reading the whole screen and finding five of the sixteen places on
        # it, and no model ever formed.
        # Not applied twice. When the reading IS the part, every position in
        # it is already a share of that part, and filtering again would cut
        # the thing down by its own outline.
        already = str(observation.get("read_within") or "") == "the part"
        band = None if already else (drawn["where"] or responds["state"].band())
        # Whether this reading is OF the thing rather than of everything.
        #
        # It was read off "is there a band", which stopped meaning that the
        # moment a reading could be scoped by photographing only the part —
        # then the band is None BECAUSE she is already looking at the right
        # place, and every guard that tested for one read it as the opposite.
        # She played on without learning anything from a board she was finally
        # reading properly.
        looking_at_the_thing = already or band is not None
        seen = within(observation, band, responds["state"])
        # The same reading, with a place for each thing in it. What she reads
        # is the string; what her claims are checked against is this.
        # The thing she is acting on, not the page it is drawn on.
        #
        # A reading of a screen is a reading of everything on it. Handed all
        # of it, the shape is called open because two hundred places is not
        # small, no rule about movement can match because most of a page never
        # moves, and two readings a second apart disagree about how many rows
        # there are. LIVE 2026-08-29: readings of 12x17 then 7x7 of a board
        # that is four by four, "how this moves is not worked out yet" after
        # eighty-four moves, and therefore a full language generation for
        # every single one of them — about twenty-eight seconds a move.
        whole = await _the_best_reading_available(
            observation, band, like=pending["whole"], in_a_browser=bool(anchor["page"])
        )
        laid_out = the_thing_itself(
            whole,
            like=_worth_holding(
                pending["arranged"], pending["whole"], pending.setdefault("shapes", {})
            ),
        )

        # Is this the thing she was asked to act in.
        #
        # Checked before a key is pressed rather than after, because a
        # keystroke into the wrong window is not something a later cycle can
        # take back.
        if not confirmed_here["value"]:
            confirmed_here["value"] = am_i_there(
                open_page or expect_page, seen, anchor["page"], anchor["app"]
            )
            if confirmed_here["value"]:
                # Arriving at a window is not the same as reading it.
                #
                # This check is about identity — the address, the title, the
                # name of the window — and it is right to be: words on a
                # screen can be borrowed from anyone's window. But the pixels
                # she then reads are whatever is drawn on top, and her own
                # interface is drawn on top of everything by design. LIVE
                # 2026-08-31: Chrome held the address she was sent to, she
                # confirmed she was there, and every reading for eighteen
                # moves was of her own panels — LIVE NEURAL FEED, TELEMETRY,
                # MEMORY, SETTINGS. She pressed keys into herself and her
                # predictions about what would change were correct.
                #
                # So before the first key: her own window goes away and the
                # thing is asked to the front. Asking alone is not enough —
                # hers is drawn above everything and comes straight back —
                # and putting hers away alone leaves the wrong window
                # frontmost. Both, in that order, are what make the pixels
                # agree with the identity.
                await _put_her_own_window_away()
                await _bring_the_thing_back_to_the_front(anchor["app"] or target_app)
            if not confirmed_here["value"]:
                not_there["reason"] = (
                    f"{(open_page or expect_page)!r} is not what is in front of me — "
                    f"{anchor['app'] or 'this window'} is"
                )
                logger.info("she is not where she was asked to be: %s", not_there["reason"])
                no_move["because"] = "she is not where she was asked to be"
                return None

        # Grade the last prediction before making another one.
        #
        # This is the difference between a loop that acts and one that steers.
        # The move it just made claimed something would be different; this
        # reading is the only chance to find out. A prediction that held is
        # weak evidence the move was understood, and one that broke is strong
        # evidence it was not — measured live, a run pressed the same key
        # forty times because nothing ever checked that the board moved.
        previous = pending["deliberation"]
        if previous is not None:
            attempt = confirm(
                previous,
                pending["before"],
                seen,
                spine=spine,
                graph=graph,
                toward=success_when,
                seen_before=pending["arranged"],
                seen_after=laid_out,
            )
            history.append(attempt)
            if previous.chosen is not None:
                # A key that never changes anything is not one of her actions
                # in this world, whoever wrote it down.
                can_do.tried(previous.chosen.name, attempt.verdict.observed_change)
                if can_do.dead() and not foreseen.get("acts"):
                    foreseen["acts"] = True
                    logger.info("what works here: %s", can_do.says())
            # Her own move, and what it did. Three things she already had and
            # threw away after one glance, which is why she could never try a
            # move without making it.
            # Learned from the part that answers to her, once she knows which
            # part that is.
            #
            # Before the band settles a reading is the whole page — the tabs,
            # the address, the score, a Give Feedback button — and no rule
            # about what her own act moves can match one. Learning from it
            # anyway fills the counters with failures that then take longer to
            # recover from than starting clean. LIVE 2026-08-26: nineteen
            # moves in, every hypothesis discredited, and she was choosing
            # blind on a board she could read perfectly well.
            if (
                pending["arranged"] is not None
                and previous.chosen is not None
                and looking_at_the_thing
            ):
                # What a rule said would happen, before it is folded in. The
                # difference between that and what she actually saw is the
                # world's doing, and it is free at exactly this moment.
                foretold = knows.rules.expect(pending["arranged"], previous.chosen.name)
                world.watched(foretold, knows.rules.the_thing(laid_out))
                knows.watched(pending["arranged"], previous.chosen.name, laid_out)
                # And whether it left her better off, against the kind of
                # position it was made from. This is experience turning into
                # skill: the same triple she learns the world's rules from
                # also says which move is worth making again from a shape
                # like that one.
                went_well = _left_her_better_off(
                    pending["arranged"],
                    laid_out,
                    success_when,
                    plan["held"].approach if plan["held"] is not None else "",
                )
                if plan["held"] is not None:
                    lines.learned(A_LINE_HERE, plan["held"].approach, went_well)
                    lines_held[plan["held"].approach] = plan["held"].as_memory()
                # What she was in, what she made of it, and what it came to.
                #
                # Two situations she scores alike, one of which went on to do
                # much better, is a difference her measure cannot account for —
                # and the only honest place a property nobody wrote can come
                # from. Gathered here because this is where both halves exist.
                from core.agency.how_good_is_this import (
                    how_good as _how_good,
                )
                from core.agency.how_good_is_this import (
                    how_the_trial_is_going as _how_the_trial_is_going,
                )

                _worth_here = sum(laid_out.numbers() or (0.0,))
                if began_at["worth"] is None:
                    began_at["worth"] = _worth_here
                began_at["seen"] += 1
                _for = success_when or _what_there_is_to_aim_at(laid_out)
                if pending["arranged"] is not None and _for:
                    cannot_explain.been_here(
                        pending["arranged"],
                        _how_good(
                            pending["arranged"],
                            toward=_for,
                            approach=plan["held"].approach if plan["held"] is not None else "",
                        ),
                        _worth_here,
                    )
                if trying["name"]:
                    verdict = _how_the_trial_is_going(trying["name"], _worth_here)
                    if verdict:
                        if narrate:
                            _tell(
                                f"{trying['name']} {'earned its place' if verdict == 'kept' else 'did not earn its place'}."
                            )
                        trying["name"] = ""
                skilled.learned(
                    pending["arranged"].as_shape(),
                    previous.chosen.name,
                    _left_her_better_off(
                        pending["arranged"],
                        laid_out,
                        success_when,
                        plan["held"].approach if plan["held"] is not None else "",
                    ),
                )
                _say_what_she_worked_out(knows, foreseen)
                _say_what_kind_of_problem(
                    knows, screen_options(move_keys), laid_out, success_when, foreseen
                )
                if len(moves) % 6 == 0 and knows.rules is not None:
                    logger.info(
                        "after %d move(s): %s%s | reading %dx%d",
                        len(moves),
                        knows.rules.says(),
                        _what_she_is_not_reading(knows.rules),
                        laid_out.rows,
                        laid_out.columns,
                    )
            # Learned from the same measurement. A move that changed nothing
            # is the control: whatever still changed across it was changing
            # on its own, and a page whose advertising animates as often as
            # the task does cannot be separated any other way.
            if pending["watched"]:
                noticed(
                    responds["state"],
                    pending["watched"],
                    observation,
                    # Whether the act had an effect, not whether her claim
                    # about it was right.
                    #
                    # These were the same answer while the only claim a move
                    # carried was that the view would differ, and they came
                    # apart the moment a claim could say something. A move
                    # that moved the board and did not do the specific thing
                    # she predicted was being counted as a move that did
                    # nothing — the control for working out which part of the
                    # screen answers to her — so the band stopped settling and
                    # nothing downstream of it could form.
                    worked=attempt.verdict.observed_change,
                )
            if moves:
                moves[-1]["held"] = attempt.verdict.held
                moves[-1]["outcome"] = attempt.verdict.why()
            pending["deliberation"] = None

        if policy is not None:
            try:
                intent = await policy(observation)
            except (RuntimeError, TypeError, ValueError, KeyError) as exc:
                record_degradation(
                    "screen_pursuit",
                    exc,
                    severity="info",
                    action="ended a screen pursuit cycle without a move",
                )
                no_move["because"] = "the policy raised rather than answered"
                return None
            if not intent:
                no_move["because"] = "the policy offered no move"
                return None
            key = str(intent.get("key") or "").strip().lower()
            if key not in PRESSABLE_KEYS:
                no_move["because"] = "the policy named a key nothing can press"
                return None
            because = str(intent.get("because") or "").strip()
        else:
            # Find out how this is done — at the start, and again when what
            # she is doing has stopped working.
            #
            # A loop that only reads the screen in front of it can play badly
            # forever: it has the board, the moves and its own last few
            # outcomes, and none of that contains the thing a person would go
            # and look up. Being stuck is the signal, because a run of broken
            # predictions means the current approach is not working whatever
            # the reason.
            if knowledge["held"] is None or (
                stuck(history) and knowledge["relearned"] < MAX_RELEARNS
            ):
                if knowledge["held"] is not None:
                    knowledge["relearned"] += 1
                relearning = knowledge["held"] is not None
                knowledge["meant"] = []
                knowledge["held"] = await learn_about(
                    goal,
                    search=research,
                    remember=not relearning,
                    because_stuck=relearning,
                    situation=content_text(
                        observation, region_top=region_top, region_bottom=region_bottom
                    ),
                    history=history[-RECENT_ATTEMPTS:],
                )
            learned = knowledge["held"].as_evidence() if knowledge["held"] is not None else []
            # Work out what it means HERE before deciding with it.
            #
            # Retrieving advice is not applying it. "Keep your largest tile in
            # a corner" is a fact about the game; what it means depends on
            # where the tiles actually are, and that comparison is the step
            # between reading something and playing differently.
            if knowledge["held"] is not None and knowledge["held"].known and not knowledge["meant"]:
                knowledge["meant"] = await work_out_what_it_means(
                    knowledge["held"],
                    seen,
                    screen_options(move_keys),
                    think=_within_the_run(think or _her_reasoning(stakes), ends_at),
                    history=history[-RECENT_ATTEMPTS:],
                )
            learned = learned + [meaning.as_evidence() for meaning in knowledge["meant"]]

            # The line she is taking, and the thing that would end it.
            #
            # Choosing a move from what is on screen is reacting: it assumes
            # the world sits still and corrects once the world says
            # otherwise. Anywhere the world keeps moving while she works, an
            # approach is the missing middle — a line held across moves, with
            # the condition that would make it wrong named when she adopts it
            # rather than discovered when it fails. The condition is checked
            # here, before she acts, so a pivot is something she was watching
            # for and not something that happened to her.
            holding, ended = still_holds(plan["held"], seen, len(moves))
            if plan["held"] is not None and holding is False:
                logger.info("the line she was taking stopped holding: %s", ended)
            # A pivot is immediate; a first attempt is not retried every move.
            #
            # The condition breaking is news and is worth the pass that
            # answers it. Having no stated approach yet is not news, and a run
            # that asks for one every cycle pays a full language pass per move
            # for an answer that was not there last time either.
            # Asked when there is something to base an approach on.
            #
            # This asked on the first cycle, when the only thing she has seen
            # is a whole screen: on a page holding a game that is the board,
            # the score, the browser's own tabs and bookmarks, an "Ask Gemini"
            # button and a copyright line. Live 2026-08-26, she was asked how
            # she would play and answered by reading the page back, three
            # runs in a row, because that is what the question was about.
            #
            # The part of the screen that answers to her is known a few moves
            # in, from what changed when she acted. That is the first moment
            # the question has a subject. The count is still a backstop, so a
            # screen that never resolves into anything is not a screen she
            # goes on playing with no line at all.
            # A pivot costs a move: she cannot judge a line she never tried.
            #
            # The condition on a fresh approach is checked on the very next
            # cycle, before she has acted under it, and an anchor bound to a
            # tile that merges away breaks at once. Live 2026-08-26: ten
            # approaches decided for nine moves made, each one a full pass at
            # her reasoning, and the run spent its whole budget deciding how
            # to play rather than playing.
            tried_it = len(moves) > plan["asked_at"]
            time_to_ask = (
                holding is False
                and plan["held"] is not None
                and tried_it
                or len(moves) - plan["asked_at"] >= _ask_again_after(plan["asked_at"])
                or (plan["asked_at"] < 0 and looking_at_the_thing)
            )
            if not holding and time_to_ask:
                plan["asked_at"] = len(moves)
                fresh = await settle_on_an_approach(
                    goal,
                    seen,
                    screen_options(move_keys),
                    # Deciding the line she will hold across a hundred moves
                    # is not the same question as deciding one of them, and
                    # asking it with the thinking that suits a move got the
                    # model's own warm-up handed back as a plan.
                    think=_within_the_run(think or _reasoning_for_a_plan(), ends_at),
                    knowledge=learned,
                    history=history[-RECENT_ATTEMPTS:],
                    previous=plan["held"],
                    moves_made=len(moves),
                )
                if fresh is not None:
                    changing = plan["held"] is not None
                    plan["held"] = fresh
                    plan["changes"] += 1 if changing else 0
                    # Held where the rest of her can see it, not in this loop.
                    doing.going_about_it(
                        fresh.approach,
                        because=fresh.because,
                        watching_for=fresh.holds_while.describes,
                        alternatives=fresh.otherwise,
                        spine=spine,
                        lived=lived,
                    )
                    if narrate:
                        said = fresh.narrate()
                        _tell(f"{said} ({ended})" if changing and ended else said)
            if plan["held"] is not None:
                learned = learned + plan["held"].as_evidence()

            # When nothing in the task is working, the task itself becomes a
            # choice. Both ways out are hers, and both are recorded as
            # decisions with reasons rather than happening to her.
            available = screen_options(can_do.available() or move_keys)
            # The ways out are offered when what she is doing has stopped
            # working, and when the thing itself has stopped responding.
            #
            # Two different facts. Predictions breaking says her moves are
            # wrong; nothing answering at all says the attempt is over — a
            # finished game, an expired session, a form already submitted.
            # Measured live: she played to Game Over and went on pressing
            # arrow keys into a dead board, because a run of broken
            # predictions had not accumulated in the way the first test
            # wanted.
            # Nothing is answering. WHICH of the three is it?
            #
            # "Nothing I do changes anything" is a good ending test and a poor
            # diagnosis: it is equally true of a finished game, a dialog over
            # the board, and somebody else's window in front. Those want
            # opposite responses, and collapsing them is what had her pressing
            # keys into a finished board and narrating moves as though a game
            # were happening. LIVE 2026-08-29: the page said "Game Over, 940
            # points scored in 100 moves" and she went on saying "Going right".
            ended = responds["state"].nothing_answers()
            # A way to start again, offered where there was none before, is the
            # thing saying it has finished.
            #
            # The other test asks whether what she was acting on is gone, and
            # on a board that keeps its tiles under a "Play Again" overlay it
            # never fires — so she went on pressing keys into a game that was
            # over, which is the failure it exists to prevent. A control that
            # appears only at the end is better evidence than the absence of
            # one, and it is general: a finished form, an expired session and
            # a lost game all put one up.
            if not ended and restart_control(observation) is not None:
                if not offered_a_restart["value"]:
                    offered_a_restart["value"] = True
                    logger.info(
                        "a way to start again has appeared, so this has ended"
                    )
                ended = True
            if ended:
                mine_now = target_app or anchor["app"]
                why = work_out_why(
                    mine=mine_now,
                    in_front=await _frontmost(),
                    on_top=await _whats_on_top(mine_now, over=responds["state"].band()),
                    still_there=_is_a_thing_laid_out(laid_out),
                )
                if why.can_fix:
                    # Not an ending. Something she can do something about, and
                    # the doing is the answer rather than the reporting.
                    if narrate and not said_it_ended["value"]:
                        _tell(why.says())
                    mended = (
                        await _ensure_page(anchor["page"])
                        if why.because == ELSEWHERE
                        else await clear_what_is_in_front(why.what)
                    )
                    if mended:
                        # Reachable again, so this cycle is not an ending. The
                        # rest of it proceeds normally: she has her window
                        # back and there is a move to choose.
                        responds["state"].began_again()
                        ended = False
                elif why.because == ENDED and narrate and not said_it_ended["value"]:
                    said_it_ended["value"] = True
                    _tell(why.says())
            if (stuck(history) or ended) and not seen_through["value"]:
                out = ways_out(observation, ended=ended)
                if ended and out:
                    # Pressing a move key into something that has finished is
                    # not one of the things she could do. Offering it beside
                    # the real choices made every cycle look like a decision
                    # between four keys and a restart, so she went on playing
                    # a game that was over — measured live, thirty-nine moves
                    # after Game Over, each one costing a language pass
                    # because the situation was unusual.
                    available = out
                else:
                    available = available + out
                if ended and not said_it_ended["value"]:
                    said_it_ended["value"] = True
                    if narrate:
                        _tell(
                            await _why_nothing_answers(
                                target_app or anchor["app"],
                                over=responds["state"].band(),
                            )
                        )
            # Her own pacing is hers to decide, once there is really a gap.
            behind = narration_backlog() if narrate else {}
            # A pace chosen because the commentary was behind ends when it
            # is not. Left standing, one choice governs the whole run and she
            # never gets asked again.
            if pacing["choice"] and not behind.get("waiting"):
                pacing["choice"] = ""
                pacing["brief"] = False
            offered_pacing = bool(behind.get("waiting")) and not pacing["choice"]
            if offered_pacing:
                available = available + pacing_options(behind)

            # Effort follows what rides on this one. A routine move is a
            # routine move; a run that has stopped getting anywhere, or one
            # weighing whether to start over, is worth more than one pass.
            # What is unusual is the situation, not the number of buttons.
            #
            # This counted the options: a way out is appended whenever the
            # screen has one, and a game page has a New Game button on it
            # permanently — so every ordinary move was treated as a moment
            # worth weighing, and paid a language pass for it. Measured live
            # 2026-08-26: seventeen passes for fourteen moves, twenty seconds
            # a move, and a run that spent its budget deliberating over a
            # button she was never going to press.
            held_line = plan["held"].approach if plan["held"] is not None else ""
            unusual = stuck(history) or ended or offered_pacing
            weight = stakes if unusual else min(stakes, 0.3)
            # Where each move would lead, when she has worked out how this
            # moves and there is anything to prefer one future over another by.
            ahead: dict[str, tuple[float, str]] = {}
            # What she is playing for, which the request does not always say.
            aiming_at = success_when or _what_there_is_to_aim_at(laid_out)
            if worth_comparing(aiming_at, held_line):
                # As far ahead as there is time to look, which is decided from
                # what a level of looking has been measured costing.
                ahead = look_ahead(
                    knows.rules,
                    laid_out,
                    [option.name for option in available],
                    toward=aiming_at,
                    approach=held_line,
                    budget_s=max(0.05, min(2.0, (ends_at - time.monotonic()) * 0.02)),
                    world=world,
                )
            # And a routine move in a fast loop does not always need words.
            #
            # What a thought is worth here, rather than how long since the
            # last one. A counter cannot tell a forced move from the one that
            # decides the shape of the next thirty, so it spends the same on
            # both and is wrong about both.
            # What has worked before from a position of this kind, if anything.
            #
            # Recognition is what frees her from deciding. Where it disagrees
            # with the arithmetic, that disagreement is the surest sign this
            # position is not the routine one it looked like, and it buys a
            # thought rather than saving one.
            kind = laid_out.as_shape() if laid_out is not None else ""
            recognised = (
                skilled.suggests(kind, tuple(option.name for option in available))
                if kind
                else ""
            )
            asking, because_of = worth_a_pass(
                ahead,
                stakes=weight,
                since_words=len(moves) - asked["at"],
                horizon=LANGUAGE_EVERY,
                unusual=unusual or not moves or restarts["count"] > asked["after_restarts"],
                recognised=recognised,
            )
            if recognised and not asking:
                skilled.took(kind)
            if asking != last_call["asked"] or last_call["why"] != because_of:
                last_call.update({"asked": asking, "why": because_of})
                logger.info("%s: %s", "thinking about this one" if asking else "no need to think", because_of)
            if asking:
                asked["at"] = len(moves)
                asked["after_restarts"] = restarts["count"]
            chosen = await deliberate(
                goal,
                seen,
                available,
                foresight=ahead or None,
                seeing=laid_out,
                think=_within_the_run(think or _her_reasoning(weight), ends_at) if asking else None,
                knowledge=learned,
                history=history[-RECENT_ATTEMPTS:],
                stakes=stakes,
                control_point="screen_pursuit.next_move",
                # Her plan reaches the moves she does not put into words,
                # which is most of them.
                approach=plan["held"].approach if plan["held"] is not None else "",
                lived=lived,
                spine=spine,
                graph=graph,
                # Reported once, when the body acts, with the reasoning on it.
                announce=False,
            )
            if not chosen.reached:
                # Stop rather than press something for no reason. A loop that
                # keeps acting once its judgement is out of reach is the exact
                # failure this decision path was built to end.
                undecided["reason"] = chosen.reason
                no_move["because"] = "she could not settle on one"
                return None
            key = chosen.chosen.name
            because = chosen.rationale

            if key in {SLOW_DOWN, SAY_LESS, PRESS_ON}:
                # A decision about herself. It changes how the next moves are
                # made rather than being one, so the cycle ends here and the
                # next one acts on it.
                pacing["choice"] = key
                pacing["because"] = because
                pacing["brief"] = key == SAY_LESS
                if key == SLOW_DOWN:
                    await let_the_voice_catch_up(behind)
                    pacing["waits"] += 1
                    # Chosen once and then re-decided as the gap changes,
                    # rather than committing the rest of the run to one pace.
                    pacing["choice"] = ""
                no_move["because"] = "she chose a pace rather than a move"
                return None

            if key == SEE_IT_THROUGH:
                # Chosen once. It says "stop offering me the way out", not
                # "do something", so the loop carries on with the moves it has.
                seen_through["value"] = True
                seen_through["because"] = because
                no_move["because"] = "she chose to see it through"
                return None

            if key == START_OVER:
                params = dict(chosen.chosen.params)
                label = str(params.get("label") or "")
                rx, ry = float(params.get("x", 0.0)), float(params.get("y", 0.0))
                frame = list(observation.get("bounds") or [])
                # Always a reason, even when her wording was unusable. The
                # filters drop an echo of the evidence, which is right, and a
                # decision with no recorded reason is not much better than an
                # unexplained one.
                restarts["because"] = because or "nothing here was moving the board"
                intending["value"] = START_OVER
                history.clear()
                was_showing = seen

                async def begin_again() -> bool:
                    clicked = await click_normalized(
                        rx, ry, expect_app=target_app or anchor["app"], bounds=frame
                    )
                    if not clicked:
                        return False
                    # Deciding to start again is not starting again.
                    #
                    # The verdict that nothing answers was cleared when she
                    # CHOSE to restart, so a click that landed on nothing left
                    # her believing the world was fresh — and she went back to
                    # pressing keys into a finished game. LIVE 2026-08-26:
                    # "Nothing I do is changing anything here — this attempt is
                    # over", and then "Going right", at a board reading Game
                    # Over the whole time.
                    #
                    # A click reports success for having happened. Only the
                    # screen can say whether it did anything.
                    after = await observe()
                    now_showing = within(after, responds["state"].band(), responds["state"])
                    if now_showing.strip() == was_showing.strip():
                        logger.info("the restart did not take — the screen is unchanged")
                        return False
                    restarts["count"] += 1
                    responds["state"].began_again()
                    logger.info("began again: %s", label or "restart")
                    return True

                return Step(name=f"begin again with {label!r}", action=begin_again)

            pending["deliberation"] = chosen
            pending["before"] = seen
            pending["arranged"] = laid_out
            pending["whole"] = whole
            # Kept whole for the comparison that finds where she has effects,
            # which cannot use a band it has not worked out yet.
            pending["watched"] = observation

        # The first move is recorded when it lands, like every other one.
        #
        # Written here, before the body was asked to do anything, it counted
        # whether or not the keystroke reached a window. LIVE 2026-08-26:
        # thirty-five moves in the record, a board that had not changed once,
        # and no correction said out loud — because only the FOLLOW-ONS of a
        # sequence were written from what landed, and a plan of one has no
        # follow-ons.
        about_to = {"key": key, "because": because, "at": time.time()}
        # Nothing is said here on purpose.
        #
        # Every decision is published to the deliberation stream as it is
        # made, and a narrator — if one is running — speaks about it on its
        # own schedule. Saying the line inline made the next move wait on
        # language, which is backwards: she should be able to play at full
        # speed and describe it, play silently, or narrate something else
        # entirely, and the loop should read the same in all three cases.

        made = pending["deliberation"]
        follow_on = [option.name for option in getattr(made, "then", ()) or ()] if made else []
        logger.info(
            "about to press %r then %s (brief=%s, made=%s)",
            key,
            follow_on,
            pacing["brief"],
            made is not None,
        )

        async def act() -> bool:
            # Said after the body did it, never before.
            #
            # The line has to correspond to what actually happened. Announcing
            # a decision announces an intention: a keystroke refused for focus,
            # or sent to the wrong window, would have been described as a move
            # she made. What she says she did is now what her body did, in the
            # order it did it.
            # One call for the whole sequence she settled on.
            #
            # Looking and deciding cost about three seconds; the keystroke
            # itself costs a third of one, and spawning the automation is most
            # of that. A run that re-reads the board between every key of a
            # pattern it has already chosen pays the whole cycle per move —
            # measured live, about one move every three seconds, where a
            # person plays several a second.
            # Saying less is about her mouth, not her hands.
            #
            # Choosing it collapsed every committed sequence to a single key
            # for the rest of the run, so one decision about commentary cost
            # her all multi-move play: measured live 2026-08-26, forty-eight
            # cycles that had committed to two to four moves each produced
            # fifty-three moves between them, one screen reading apiece.
            sequence = [key, *follow_on] if follow_on else [key]
            # Intent, then action. Said before the body moves, because that is
            # the order a person doing something narrates it in.
            #
            # Only the first one carries the reason she gave. The rest are
            # the same decision continuing, and repeating its reason under
            # each of them says something false: live, she committed to
            # left-then-right and narrated "Going right — left has worked."
            for position, step in enumerate(sequence):
                reason = (None if pacing["brief"] else made) if position == 0 else None
                # Under a quiet pace only the first of a sequence is spoken.
                # The moves still happen; what she asked for is fewer words.
                aloud = narrate and (position == 0 or not pacing["brief"])
                _say_intent(step, reason, out_loud=aloud, following_on=position > 0)
            if len(sequence) > 1:
                # Only the keys that really landed are spoken for. Focus can
                # move part-way through a batch, and a commentary describing
                # moves the window never received is the disconnect this
                # whole path exists to avoid.
                arrived = await press_many(sequence, expect_app=target_app or anchor["app"])
            else:
                arrived = 1 if await press(key, expect_app=target_app or anchor["app"]) else 0
            for position, step in enumerate(sequence[:arrived]):
                if position == 0:
                    about_to["at"] = time.time()
                    moves.append(about_to)
                else:
                    moves.append(
                        {"key": step, "because": "part of the same plan", "at": time.time()}
                    )
                doing.a_step_taken()
            for step in sequence[arrived:]:
                # An intention she stated and did not carry out is corrected
                # out loud. The record of what she did is written only from
                # what landed, so the two can never drift.
                _say_it_did_not_land(step, out_loud=narrate)
            if arrived and pacing["choice"] == SLOW_DOWN:
                await let_the_voice_catch_up(narration_backlog())
            return arrived > 0

        return Step(name=f"press {key}", action=act)

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

    # What she worked out about this thing, for the next time she is in it.
    remember(
        this_world,
        {
            "responds": responds["state"].as_memory(),
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
    said = f"{str(key).strip().capitalize()} did not land"
    _publish_decision(said, "the window did not take it", "", None)
    if out_loud:
        _tell(said)


def _expected_of(chosen: Any) -> str:
    option = getattr(chosen, "chosen", None)
    expectation = getattr(option, "expectation", None)
    return str(getattr(expectation, "describes", "") or "")


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
        task = loop.create_task(coroutine)
        task.add_done_callback(lambda done: done.exception())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("screen_pursuit", exc, severity="info", action="acted without saying so")


async def _settled_after(
    before: dict[str, Any], app: str, *, patience: float = 4.0
) -> tuple[dict[str, Any], bool]:
    """The screen once it has changed, and whether it changed at all.

    An action that resets a surface is not done when the click returns; it is
    done when the surface says so. Waiting on the change rather than on a
    fixed delay means a slow page is waited for and a fast one is not.

    The second value is the part that matters. A click that lands on nothing
    still reports success — the click happened — so a caller that only waits
    goes on to claim a reset that never occurred. Measured live: "Began again
    1 time(s)" while the score sat unchanged at 996 the whole time.
    """
    was = str(before.get("text") or "")
    started = time.monotonic()
    seen = before
    while time.monotonic() - started < patience:
        await asyncio.sleep(0.3)
        try:
            seen = await asyncio.wait_for(read_screen(app), timeout=OBSERVE_TIMEOUT_S)
        except TimeoutError:
            continue
        if str(seen.get("text") or "") != was:
            return seen, True
    return seen, False



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
