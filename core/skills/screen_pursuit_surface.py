"""The screen as a surface: what is on it, and how to touch it.

Reading a window, labelling what is in it, pressing a key, clicking a point in
its own coordinates rather than the display's. Nothing here decides anything —
each one is a primitive the rest of the pursuit is written in, and every one of
them says what it could not do rather than returning a shape that looks like
success.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ScreenPursuit")


#: Keys the loop is allowed to press, by the name a person would use.
#: Bounded on purpose — a loop that can press anything can press ⌘Q.
PRESSABLE_KEYS = (
    "up", "down", "left", "right",
    "return", "enter", "tab", "space", "escape",
)


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


def _a_pass_in_moves(costs: dict[str, float]) -> float:
    """What a language pass costs, counted in moves not made.

    A pass on a small model costs a fraction of a move and one on a resident
    model under memory pressure costs the time of ten. Whether an answer is
    worth asking for depends on both halves, and only one of them was ever
    looked at.

    One until both have been measured, which is what the caller assumed all
    along.
    """
    passes, quiet = costs.get("passes", 0.0), costs.get("quiet", 0.0)
    if passes < 1.0:
        return 1.0
    a_pass = costs.get("pass_s", 0.0) / passes
    if quiet >= 1.0:
        a_quiet_move = costs.get("quiet_s", 0.0) / quiet
        if a_quiet_move > 0.0:
            return max(1.0, a_pass / a_quiet_move)
    # Nothing to compare a pass against, because there has not been a quiet
    # move — and there never will be while a pass is priced at one.
    #
    # Both halves had to be measured before either counted, so a run that
    # thought about its first move could not find out that thinking was
    # expensive: no quiet move, so a pass costs one, so the bar stays where a
    # pass is cheap, so she thinks again. Live 2026-09-07: forty-eight passes
    # for nineteen moves, every one of them about ten seconds.
    #
    # What she has instead is the pass itself against what the rest of a cycle
    # takes — looking, deciding, pressing. A pass that takes longer than
    # everything else put together is expensive whether or not she has ever
    # done without one.
    a_cycle = costs.get("cycle_s", 0.0) / max(1.0, costs.get("cycles", 0.0))
    without_it = a_cycle - a_pass
    if a_cycle > 0.0 and without_it > 0.0:
        return max(1.0, a_pass / without_it)
    return 1.0


def _looks_like(foretold: Any, band: Any, lattice: Any) -> Any:
    """A test for "this is the arrangement she said the move would make".

    None where she foretold nothing, which is the ordinary case early on and
    the honest answer for a world she cannot predict — then waiting for
    stillness is all she has.
    """
    if foretold is None or lattice is None or not getattr(lattice, "held", False):
        return None
    from core.perception.how_it_moves import prediction_held  # noqa: PLC0415
    from core.perception.where_it_responds import what_is_there  # noqa: PLC0415

    def it_did(now: dict[str, Any]) -> bool:
        placed = what_is_there(now, band.band(), None, None, lattice)
        return bool(prediction_held(foretold, placed))

    return it_did


def _who_the_screen_belongs_to(app: str) -> tuple[str, bool]:
    """Who owns the front window now, and whether ``app`` is drawing one at all.

    The window server answers both from one list: it reports only the windows
    being drawn at this moment, front to back. Two different ways a picture of
    her window's rectangle is a picture of something else — another
    application over her, and her window sitting on a Space that is not the
    one on screen — and the second leaves her bounds exactly as they were.

    An unanswerable question says nothing: no window server, no verdict.
    """
    try:
        import Quartz  # noqa: PLC0415

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "screen_pursuit", exc, severity="info", action="looked without knowing whose screen it was"
        )
        return "", True
    wanted = str(app or "").strip().lower()
    front, drawing = "", False
    for window in windows or []:
        try:
            layer = int(window.get("kCGWindowLayer", 0) or 0)
            owner = str(window.get("kCGWindowOwnerName", "") or "")
        except (TypeError, ValueError, AttributeError):
            continue
        # Ordinary windows only. The menu bar and the dock are always drawn
        # and answer neither question.
        if layer != 0 or not owner:
            continue
        if not front:
            front = owner
        lowered = owner.lower()
        if wanted and (wanted in lowered or lowered in wanted):
            drawing = True
    return front, drawing


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
    # Imported here rather than at module level: the module these
    # came from imports this one. A call-time import also still
    # sees a test's patch of the original.
    from .screen_pursuit import (
        current_page_identity,
    )

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
    at_x, at_y = int(round(left + x * width)), int(round(top + y * height))

    # A place she can see is not always a place she can reach.
    #
    # A window can hang off the edge of the display, and the capture she reads
    # is of the WINDOW — so the part hanging off is in the reading, at
    # perfectly ordinary coordinates, and nothing about it looks unreachable.
    # LIVE 2026-09-02: the game's window ran to 1844 on a display 1728 wide,
    # so the New Game button sat at x=0.89 of the window and the click went to
    # 1763, off the end of the screen. It reported success — the click was
    # made — and three runs in a row ended after one move at a board that had
    # already finished, each of them having pressed a button that was not
    # there.
    #
    # Refused rather than clamped. Sliding it to the nearest visible pixel
    # clicks something she did not choose, which is worse than not clicking.
    across, down = await _screen_size()
    if across and down and not (0 <= at_x < across and 0 <= at_y < down):
        logger.info(
            "not clicking (%d, %d): outside the display, which is %dx%d — "
            "the window hangs off the edge and that part cannot be reached",
            at_x, at_y, across, down,
        )
        return False

    receipt = await host.click_at(at_x, at_y)
    return bool(getattr(receipt, "success", False))


async def _screen_size() -> tuple[int, int]:
    """Main display size in pixels, or (0, 0) when it cannot be read."""
    try:
        from AppKit import NSScreen

        frame = NSScreen.mainScreen().frame()
        return int(frame.size.width), int(frame.size.height)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return (0, 0)


def _bound_to_a_window(key: str, expect_app: str) -> bool:
    """Whether this keystroke knows what will receive it.

    The rule was written down and not enforced: every keystroke but one has to
    be bound to a window, and a run that could not name what it was looking at
    sent its keys with nothing bound and the guard passed them through. That
    is not a weaker version of aiming, it is the unaimed case the rule exists
    for — thirty-five moves of a game played into a chat window, every one of
    them reported as a success.

    Knowing what she is acting on is a precondition of acting, not a detail of
    it. Where she cannot say whether the keyboard belongs to a browser, a
    terminal or the window she was reading, she has no business pressing
    anything into it.
    """
    if str(key or "").strip().lower() == DECLINES_AND_NOTHING_ELSE:
        return True
    return bool(str(expect_app or "").strip())


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
    if not _bound_to_a_window(name, expect_app):
        logger.info("not pressing %r: nothing is bound to receive it", name)
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
    if not all(_bound_to_a_window(key, expect_app) for key in wanted):
        logger.info("not pressing %s: nothing is bound to receive them", wanted)
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


#: The one key she may send without knowing where it will land.
#:
#: Every other keystroke needs to be bound to a window, and the reason is on
#: the record: unbound arrow keys played thirty-five moves of a game into a
#: chat window. Escape is different in kind rather than in degree. It declines,
#: it commits to nothing, it is reversible, and it is the platform-standard
#: way out of a modal on every desktop. Sending it at whatever has the
#: keyboard is the only way to reach a thing that took the keyboard from her.
DECLINES_AND_NOTHING_ELSE = "escape"
