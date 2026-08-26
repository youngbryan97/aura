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
from core.runtime.watched_goal import PURSUIT_SECONDS
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
    success_when: str = Field(..., min_length=1, max_length=200)
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
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not getattr(receipt, "success", False):
        return None
    numbers = re.findall(r"-?\d+", str(getattr(receipt, "result", "") or ""))
    if len(numbers) < 4:
        return None
    x, y, width, height = (int(value) for value in numbers[:4])
    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


async def read_screen(app_name: str = "") -> dict[str, Any]:
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

    bounds = await window_bounds(app_name) if app_name else None
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
        "at": time.time(),
    }


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
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return {"url": "", "title": "", "error": "unavailable"}


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
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    if not getattr(receipt, "success", False):
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
        except (TypeError, ValueError):
            pass
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
    from core.agency.standing_strategy import settle_on_an_approach, still_holds
    from core.agency.task_knowledge import learn_about, stuck, work_out_what_it_means
    from core.perception.where_it_responds import Responsive, describe, noticed, within
    from core.skills.fluid_executor import FluidExecutor, Step

    # The clock starts when she takes this on, not when the first key lands.
    #
    # Finding the page and opening it is part of the task and used to be free:
    # the pursuit's own budget began afterwards, so the outer deadline — the
    # one that only has room to report — was reached first and won. Live
    # 2026-08-26: sixty-five narrated moves, nine approaches held, cancelled
    # from outside and reported as "Completed 0/0 steps".
    began = time.monotonic()
    moves: list[dict[str, Any]] = []
    history: list[Attempt] = []
    pending: dict[str, Any] = {"deliberation": None, "before": "", "watched": {}}
    undecided: dict[str, str] = {"reason": ""}
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
        if target_app:
            try:
                await _ensure_frontmost(target_app)
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
                read_screen(target_app), timeout=OBSERVE_TIMEOUT_S
            )
        except TimeoutError:
            # A wedged capture is not a reason to keep acting blind.
            return {"ok": False, "text": "", "layout": [], "error": "observe_timeout"}

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
        except ImportError:
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
                        ux, uy, expect_app=target_app, bounds=frame
                    )

                return Step(
                    name=f"clear the way with {label!r}", action=click_declared
                )

        if verdict.suggested_key:
            key = verdict.suggested_key

            async def press_away() -> bool:
                return await press(key, expect_app=target_app)

            return Step(name=f"dismiss overlay with {key}", action=press_away)

        if verdict.click_x is not None:
            label, x, y = verdict.label, verdict.click_x, verdict.click_y

            frame = list(observation.get("bounds") or [])

            async def click_away() -> bool:
                return await click_normalized(
                    x, y, expect_app=target_app, bounds=frame
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
    anchor: dict[str, str] = {"page": expect_page.strip()}
    #: Where her actions have been having their effects.
    responds: dict[str, Any] = {"state": Responsive()}
    #: Said once, when the thing she is working in stops answering at all.
    said_it_ended: dict[str, bool] = {"value": False}

    async def decide(observation: dict[str, Any]) -> Step | None:
        blocker = await clear_blocker(observation)
        if blocker is not None:
            # Verified, not assumed. A blocker still present after the previous
            # attempt means that attempt did not work, whatever its receipt
            # said.
            if blocker_attempts["count"] >= MAX_BLOCKER_ATTEMPTS:
                blocker_attempts["last"] = blocker.name
                return None
            blocker_attempts["count"] += 1
            blocker_attempts["dismissed"] += 1
            blocker_attempts["last"] = blocker.name
            return blocker
        if needs_person["reason"]:
            return None
        blocker_attempts["count"] = 0
        if not observation.get("ok"):
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
        band = responds["state"].band()
        seen = within(observation, band, responds["state"])

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
                previous, pending["before"], seen, spine=spine, graph=graph, toward=success_when
            )
            history.append(attempt)
            # Learned from the same measurement. A move that changed nothing
            # is the control: whatever still changed across it was changing
            # on its own, and a page whose advertising animates as often as
            # the task does cannot be separated any other way.
            if pending["watched"]:
                noticed(
                    responds["state"],
                    pending["watched"],
                    observation,
                    worked=attempt.verdict.held,
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
                return None
            if not intent:
                return None
            key = str(intent.get("key") or "").strip().lower()
            if key not in PRESSABLE_KEYS:
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
                    think=think or _her_reasoning(stakes),
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
                or (plan["asked_at"] < 0 and band is not None)
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
                    think=think or _reasoning_for_a_plan(),
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
            available = screen_options(move_keys)
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
            ended = responds["state"].nothing_answers()
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
                        _tell("Nothing I do is changing anything here — this attempt is over.")
            # Her own pacing is hers to decide, once there is really a gap.
            behind = narration_backlog() if narrate else {}
            if behind.get("waiting") and not pacing["choice"]:
                available = available + pacing_options(behind)

            # Effort follows what rides on this one. A routine move is a
            # routine move; a run that has stopped getting anywhere, or one
            # weighing whether to start over, is worth more than one pass.
            unusual = stuck(history) or len(available) > len(move_keys)
            weight = stakes if unusual else min(stakes, 0.3)
            # And a routine move in a fast loop does not always need words.
            #
            # Asking costs the cycle; deciding from evidence costs nothing,
            # and on a board that changes a little at a time the two usually
            # agree. So language is consulted where it is most likely to
            # change the answer, and often enough in between that what she
            # says stays her own reasoning.
            asking = (
                unusual
                or not moves
                or restarts["count"] > asked["after_restarts"]
                or len(moves) - asked["at"] >= LANGUAGE_EVERY
            )
            if asking:
                asked["at"] = len(moves)
                asked["after_restarts"] = restarts["count"]
            chosen = await deliberate(
                goal,
                seen,
                available,
                think=(think or _her_reasoning(weight)) if asking else None,
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
                return None

            if key == SEE_IT_THROUGH:
                # Chosen once. It says "stop offering me the way out", not
                # "do something", so the loop carries on with the moves it has.
                seen_through["value"] = True
                seen_through["because"] = because
                return None

            if key == START_OVER:
                params = dict(chosen.chosen.params)
                label = str(params.get("label") or "")
                rx, ry = float(params.get("x", 0.0)), float(params.get("y", 0.0))
                frame = list(observation.get("bounds") or [])
                restarts["count"] += 1
                # Always a reason, even when her wording was unusable. The
                # filters drop an echo of the evidence, which is right, and a
                # decision with no recorded reason is not much better than an
                # unexplained one.
                restarts["because"] = because or "nothing here was moving the board"
                intending["value"] = START_OVER
                history.clear()

                async def begin_again() -> bool:
                    return await click_normalized(rx, ry, expect_app=target_app, bounds=frame)

                return Step(name=f"begin again with {label!r}", action=begin_again)

            pending["deliberation"] = chosen
            pending["before"] = seen
            # Kept whole for the comparison that finds where she has effects,
            # which cannot use a band it has not worked out yet.
            pending["watched"] = observation

        moves.append({"key": key, "because": because, "at": time.time()})
        doing.a_step_taken()
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
            sequence = [key, *follow_on] if follow_on and not pacing["brief"] else [key]
            # Intent, then action. Said before the body moves, because that is
            # the order a person doing something narrates it in.
            #
            # Only the first one carries the reason she gave. The rest are
            # the same decision continuing, and repeating its reason under
            # each of them says something false: live, she committed to
            # left-then-right and narrated "Going right — left has worked."
            for position, step in enumerate(sequence):
                reason = (None if pacing["brief"] else made) if position == 0 else None
                _say_intent(step, reason, out_loud=narrate, following_on=position > 0)
            if len(sequence) > 1:
                # Only the keys that really landed are spoken for. Focus can
                # move part-way through a batch, and a commentary describing
                # moves the window never received is the disconnect this
                # whole path exists to avoid.
                arrived = await press_many(sequence, expect_app=target_app)
            else:
                arrived = 1 if await press(key, expect_app=target_app) else 0
            for step in sequence[:arrived]:
                if step != key:
                    moves.append({"key": step, "because": "part of the same plan", "at": time.time()})
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
                    think=think or _her_reasoning(stakes),
                    control_point="screen_pursuit.pre_met",
                    lived=lived,
                    spine=spine,
                    graph=graph,
                )
                if settle.reached and settle.chosen is not None and settle.chosen.name == START_OVER:
                    label, rx, ry = fresh
                    intending["value"] = START_OVER
                    frame = list(first.get("bounds") or [])
                    if await click_normalized(rx, ry, expect_app=target_app, bounds=frame):
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
            max_seconds=max(1.0, max_seconds - (time.monotonic() - began)),
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
    # How the line she took turned out, written where consequences live, so
    # an approach that keeps failing is harder to reach for next time.
    doing.how_it_went(
        bool(receipt.completed),
        f"{len(moves)} move(s), {plan['changes']} change(s) of approach",
        graph=graph,
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
            except (TypeError, ValueError):
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
