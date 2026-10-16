"""Playing what a page draws: from the page's elements to her eyes, on the same page.

A page that draws a game holds nothing for the browser pursuit to read or press:
LIVE 2026-10-02, a Flash game on webdesignmuseum.org is a 540 by 360 canvas in
Ruffle's shadow root, with its own PLAY button painted on it. The screen pursuit
plays exactly that kind of thing — it read and played 2048 in a browser window —
so when she chooses the drawing, the screen pursuit is handed her own page, with
the part of it the page says it is drawing in.

The page is the surface for the whole hand-over (core/skills/screen_pursuit_on_a_page.py):
she reads it from pictures her own browser takes of it, and her keys and clicks
are delivered to it by her own browser. Nothing goes to whatever window happens
to be in front. LIVE 2 October that window was the person's own Chrome.

Nothing here knows what a game is. The page reports where it draws; she chooses
whether what is drawn is the way on; the screen pursuit finds out how it moves.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from core.verify import invariant

#: The selector the observer gives what a page draws. Not a CSS selector: a move
#: on it is a hand-over, never a click.
DRAWING = "::drawing"

#: What the screen pursuit calls the surface while it plays her page. It names
#: the world she is in for her memory of it, together with the page's address.
HER_BROWSER = "her own browser"

#: Where the page draws, as shares of its viewport, after bringing the largest
#: drawing into view. Shadow roots are walked: a player's canvas lives in one.
_WHERE_IT_DRAWS = """
(function () {
  var best = null, area = 0;
  var walk = function (root) {
    var all = root.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.tagName === 'CANVAS') {
        var r = el.getBoundingClientRect();
        if (r.width * r.height > area) { area = r.width * r.height; best = el; }
      }
      if (el.shadowRoot) walk(el.shadowRoot);
    }
  };
  walk(document);
  if (!best) return '';
  best.scrollIntoView({block: 'center', inline: 'center'});
  var r = best.getBoundingClientRect();
  var W = window.innerWidth || 1, H = window.innerHeight || 1;
  return JSON.stringify({
    left: Math.max(0, r.left) / W,
    top: Math.max(0, r.top) / H,
    right: Math.min(W, r.right) / W,
    bottom: Math.min(H, r.bottom) / H
  });
})()
"""


#: What the page puts over the middle of its drawing, when that is something to
#: press and not the drawing itself: an emulator's power button, a player's
#: "click to play". Shadow roots are looked into, as a player's overlay is in one.
_COVER_AT_POINT = """
function coverAtPoint(x, y) {
  var el = document.elementFromPoint(x, y), depth = 0;
  while (el && el.shadowRoot && depth < 6) {
    var inner = el.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === el) break;
    el = inner; depth++;
  }
  if (!el || el.tagName === 'CANVAS' || el.tagName === 'EMBED' || el.tagName === 'OBJECT') return {element: null, description: ''};
  var pressed = el.closest ? (el.closest('button,a,[role=button]') || el) : el;
  var pressable = getComputedStyle(pressed).cursor === 'pointer' || /^(BUTTON|A|IMG|INPUT|SVG|PATH)$/i.test(pressed.tagName)
    || pressed.getAttribute('role') === 'button' || typeof pressed.onclick === 'function';
  var description = pressable ? (pressed.tagName + ' ' + (pressed.getAttribute('aria-label') || pressed.getAttribute('alt') || pressed.getAttribute('class') || '')).slice(0, 80) : '';
  return {element: pressable ? pressed : null, description: description};
}
"""

_PUT_OVER_IT = '(function (x, y) {' + _COVER_AT_POINT + 'return coverAtPoint(x, y).description;})(%f, %f)'

# The physical press can meet a new overlay after the last hit test. Keep its
# actual public event path, so a press on that overlay does not spend the
# intended control's whole startup allowance.
_COVER_PRESS = '(function (x, y, arm) {' + _COVER_AT_POINT + """
  var observed = coverAtPoint(x, y), previous = window.__auraCoverPress;
  if (arm && observed.element) {
    if (previous) previous.stop();
    var controller = new AbortController(), timeout;
    var receipt = {received: null, stop: function () {
      controller.abort(); clearTimeout(timeout);
    }};
    var expected = observed.element, down = false;
    var atPoint = function (e) {
      return e.isTrusted && e.button === 0 && Math.abs(e.clientX - x) <= 1 && Math.abs(e.clientY - y) <= 1;
    };
    document.addEventListener('mousedown', function (e) {
      if (atPoint(e)) down = e.composedPath().indexOf(expected) >= 0;
    }, {capture: true, signal: controller.signal});
    document.addEventListener('click', function (e) {
      if (!atPoint(e)) return;
      receipt.received = down && e.composedPath().indexOf(expected) >= 0;
      receipt.stop();
    }, {capture: true, signal: controller.signal});
    timeout = setTimeout(receipt.stop, 2500);
    window.__auraCoverPress = receipt;
    previous = receipt;
  }
  return {description: observed.description, received: previous ? previous.received : null};
})(%f, %f, %s)
"""


def _cover_receipt(observation: Any) -> tuple[str, bool | None]:
    """Unknown or malformed delivery remains unmeasured."""
    if not isinstance(observation, Mapping) or not isinstance(observation.get('description'), str):
        return '', None
    received = observation.get('received')
    return observation['description'][:80], received if type(received) is bool else None


@invariant('agency.cover_press_needs_its_actual_event_path', scope='agency',
           owner='core/skills/sovereign_browser_drawing.py', observational=False)
def _cover_press_invariant() -> tuple:
    assert _cover_receipt({'description': 'BUTTON', 'received': False}) == ('BUTTON', False)
    assert all(_cover_receipt({'description': 'BUTTON', 'received': v})[1] is None for v in (1, 'true', [], {}))
    assert _cover_receipt(None) == ('', None)
    return ()

#: The close controls of notices laid over the drawing (a player's warning, a
#: page's banner), by their own words or sign, as viewport points to click.
_CLOSES_OVER_IT = r"""
(function (l, t, r, b) {
  var found = [];
  var hitAt = function (x, y) {
    var hit = document.elementFromPoint(x, y);
    for (var depth = 0; hit && hit.shadowRoot && depth < 16; depth++) {
      var next = hit.shadowRoot.elementFromPoint(x, y);
      if (!next || next === hit) break;
      hit = next;
    }
    return hit;
  };
  var walk = function (root) {
    var all = root.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.shadowRoot) walk(el.shadowRoot);
      var said = ((el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')).toLowerCase();
      var sign = (el.children.length === 0 ? (el.textContent || '') : '').trim();
      if (!/\b(close|dismiss)\b/.test(said) && !/^[\u00d7\u2715\u2716\u2573]$/.test(sign)) continue;
      var rect = el.getBoundingClientRect();
      if (rect.width < 4 || rect.height < 4 || rect.width > 120 || rect.height > 120) continue;
      var x = rect.left + rect.width / 2, y = rect.top + rect.height / 2;
      if (x < l || x > r || y < t || y > b) continue;
      if (getComputedStyle(el).visibility === 'hidden') continue;
      var hit = hitAt(x, y);
      if (!hit || (hit !== el && !el.contains(hit))) continue;
      found.push([x, y]);
    }
  };
  walk(document);
  return JSON.stringify(found.slice(0, 3));
})(%f, %f, %f, %f)
"""

@invariant('agency.notice_controls_keep_javascript_word_boundaries', scope='agency',
           owner='core/skills/sovereign_browser_drawing.py', observational=False)
def _notice_word_boundaries_invariant() -> tuple:
    assert '\x08' not in _CLOSES_OVER_IT and r'/\b(close|dismiss)\b/' in _CLOSES_OVER_IT
    return ()


#: A page she plays on stays where it is. A key the game does not take is the
#: page's to act on, and arrows and space scroll it: LIVE-like 2026-10-05, an
#: archived game's page scrolled to its description under her keys and she read
#: the description as the game. The page's default is held back only for those
#: keys, after the game has had them, and only while she plays.
_HOLD_IT_STILL = """
(function () {
  if (window.__herPlay) return true;
  var stop = new AbortController(), x = window.scrollX, y = window.scrollY;
  var scrolls = [' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End'];
  window.addEventListener('keydown', function (e) {
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if (scrolls.indexOf(e.key) >= 0) e.preventDefault();
  }, {signal: stop.signal});
  window.addEventListener('scroll', function () {
    if (window.scrollX !== x || window.scrollY !== y) window.scrollTo(x, y);
  }, {signal: stop.signal});
  window.__herPlay = stop;
  return true;
})()
"""

_LET_IT_GO = "(function () { if (window.__herPlay) { window.__herPlay.abort(); window.__herPlay = null; } return true; })()"

#: How many times something put over the drawing is pressed, and how long the
#: drawing is given to start after each.
PRESSES_TO_START = 3
STARTS_WITHIN_S = 20.0


async def _start_what_is_covered(page: Any, band: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Press what the page puts over its drawing until the drawing shows, and say where it draws then.

    A person in front of an archived game presses the power button over it
    before anything else; a key pressed at a still cover goes to the page.
    """
    import asyncio
    import time

    from core.perception.what_her_page_shows import the_page_size

    for press in range(PRESSES_TO_START):
        wide, tall = await the_page_size(page)
        x, y = (band[0] + band[2]) / 2 * wide, (band[1] + band[3]) / 2 * tall
        over, _received = _cover_receipt(await page.evaluate(_COVER_PRESS % (x, y, 'true')))
        if not over:
            break
        if press == 0:  # said once: pressing it again is the same act, not news
            _tell("The game has a button over it to start it; pressing that first.")
        await page.mouse.click(x, y)
        began = time.monotonic()
        missed = False
        while time.monotonic() - began < STARTS_WITHIN_S:
            await asyncio.sleep(0.1)
            current, received = _cover_receipt(await page.evaluate(_COVER_PRESS % (x, y, 'false')))
            missed = received is False
            if current != over or missed:
                break
        await asyncio.sleep(0.1 if missed else 2.0)
        band = _the_band(await page.evaluate(_WHERE_IT_DRAWS)) or band
    wide, tall = await the_page_size(page)
    closes = json.loads(str(await page.evaluate(_CLOSES_OVER_IT % (band[0] * wide, band[1] * tall, band[2] * wide, band[3] * tall)) or "[]"))
    for x, y in closes:
        _tell("A notice is over the game; closing it.")
        await page.mouse.click(float(x), float(y))
    return band


#: Pages whose game was played to an end she said (won, its share of time up, not getting better), and that end.
_PLAY_OVER: dict[str, str] = {}
#: When play at each address began for the asking in hand: its share runs from then, however often it is handed back.
_SHARE_BEGAN: dict[str, float] = {}
#: Pages whose drawing she has played in this sitting, how often she has gone back to playing each without asking her
#: model, and the most times she does.
_PLAYED_HERE: set[str] = set()
_PLAYED_ON: dict[str, int] = {}
PLAY_ON_AT_MOST = 3


def _address(url: str) -> str:
    return str(url or "").split("#", 1)[0].rstrip("/")


def played_first(observation: Mapping[str, Any], goal: str, *, take: bool = True) -> dict[str, Any] | None:
    """The first move where the page in front was seen to run what the goal is to run: play what it draws, unasked.

    A person who has just watched a game draw its start screen does not read the
    page around it before playing. LIVE 2026-10-06 her model read each game's
    page for sixty to ninety seconds and then chose to play what it draws. Once
    a page, and only where it was seen running (core/skills/whether_a_page_serves.py).
    """
    from core.skills.whether_a_page_serves import seen_running, what_the_task_needs

    url = str(observation.get("url") or "")
    # Its game played to an end she said: that is the end of what she was doing here, not a page to read afresh.
    # LIVE 2026-10-07 after "I've given this one its share of the time" her model read the page again and played on.
    ended = _PLAY_OVER.pop(_address(url), "") if take else _PLAY_OVER.get(_address(url), "")
    if ended:
        why = "Played, and won." if ended == "won" else f"Played; {ended}." if ended == NOTHING_TO_WIN else f"Played, and not won; {ended}."
        return {"done": True, "actions": [], "why": why}
    # Her play here came back without a win or an end she said: a person goes on playing, and does not stop to think
    # what else to do. LIVE 2026-10-10 her play paused for a game's instructions, her model was asked what next while
    # busy reading the game's code, the answer was refused, and she closed the game.
    address = _address(url)
    if (what_the_task_needs(goal) == "run" and address in _PLAYED_HERE and not seen_running(url)
            and _PLAYED_ON.get(address, 0) < PLAY_ON_AT_MOST):
        if take:
            _PLAYED_ON[address] = _PLAYED_ON.get(address, 0) + 1
        return {"resolved_actions": [{"selector": DRAWING, "said": "I'll keep playing it."}], "why": "I'll keep playing it.",
                "done": False}
    if what_the_task_needs(goal) != "run" or not seen_running(url):
        return None
    seen = seen_running(url, take=take)
    said = f"What I came for is here and runs ({seen}); starting it."
    return {"resolved_actions": [{"selector": DRAWING, "said": said}], "why": said, "done": False}


def chose_the_drawing(moves: list[tuple[Any, Any]]) -> bool:
    """Whether any of this round's moves is the drawing."""
    return any(str(getattr(action, "selector", "") or "") == DRAWING for action, _said in moves)


#: How long a page's player is given to put up the drawing it plays in, and how often it is asked.
DRAWING_COMES_UP_S = 15.0
ASKED_EVERY_S = 0.5


async def _where_it_draws(page: Any) -> tuple[float, float, float, float] | None:
    """Where the page draws, asked again while its player is still putting the drawing up: offline 2026-10-10 a game's
    page was asked before its player had made its drawing, and she gave it up as a page that no longer draws."""
    import asyncio
    import time

    gives_up_at = time.monotonic() + DRAWING_COMES_UP_S
    while True:
        band = _the_band(await page.evaluate(_WHERE_IT_DRAWS))
        if band is not None or time.monotonic() >= gives_up_at:
            return band
        await asyncio.sleep(ASKED_EVERY_S)


def _the_band(said: Any) -> tuple[float, float, float, float] | None:
    """The page's answer to where it draws, as shares of what it is measured in."""
    try:
        where = json.loads(str(said or ""))
        band = (float(where["left"]), float(where["top"]), float(where["right"]), float(where["bottom"]))
    except (TypeError, ValueError, KeyError):
        return None
    left, top, right, bottom = band
    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        return None
    return band


async def played_on_the_drawing(
    browser: Any, goal: str, observation: Mapping[str, Any]
) -> dict[str, Any]:
    """Play what her own page draws, through her own page, and say what came of it.

    Heard as one piece of work: what was said of the last game is not what this one has said, and the names in
    the goal are words when she reads them off its screen (core/language/what_a_watcher_hears.py).
    """
    from core.language.what_a_watcher_hears import a_fresh_watcher, done_watching

    watching = a_fresh_watcher(goal)
    try:
        return await _on_the_drawing(browser, goal, observation)
    finally:
        done_watching(watching)


async def _on_the_drawing(browser: Any, goal: str, observation: Mapping[str, Any]) -> dict[str, Any]:
    from core.runtime.errors import record_degradation

    url = str(observation.get("url") or "")
    step: dict[str, Any] = {"chose": ["what the page draws"], "url": url, "landed": 0}
    page = getattr(browser, "page", None)
    if page is None:
        return {**step, "error": "her browser has no page to play on"}
    try:
        band = await _where_it_draws(page)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="ask her page where it draws")
        band = None
    if band is None:
        return {**step, "error": "the page no longer says where it draws"}
    try:
        band = await _start_what_is_covered(page, band)
        await page.evaluate(_HOLD_IT_STILL)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="start the drawing and hold its page still")
    context, closing = getattr(page, "context", None), _Closing(page)
    if context is not None:
        context.on("page", closing.opened)
    try:
        _PLAYED_HERE.add(_address(url))
        played = await _played(page, band, goal, url, {**step, "runtime_contract": observation.get("runtime_contract") or {}})
        # Play that ended for a reason she said, or won, ends what she was doing on this page too.
        if played.get("won") or played.get("stopped_because"):
            _PLAY_OVER[_address(url)] = "won" if played.get("won") else str(played["stopped_because"])
            _SHARE_BEGAN.pop(_address(url), None)
        return played
    finally:
        if context is not None:
            context.remove_listener("page", closing.opened)
        try:
            await page.evaluate(_LET_IT_GO)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            pass  # a page gone is a page that no longer needs letting go


class _Closing:
    """Tabs the game opens on its own while she plays it, closed, and her game brought back in front.

    A link out or an advert in a game opens a tab of its own over the game:
    LIVE 2026-10-07 a blank tab stood in front of the game she was playing,
    and whoever was watching saw nothing. A person closes it and goes back.
    """

    def __init__(self, page: Any) -> None:
        self.page, self.said = page, False

    def opened(self, tab: Any) -> None:
        import asyncio

        asyncio.ensure_future(self._close(tab))

    async def _close(self, tab: Any) -> None:
        try:
            await tab.close()
            await self.page.bring_to_front()
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return  # a tab already gone, or a page that is: nothing in front of the game to close
        if not self.said:
            self.said = True
            _tell("The game opened another tab of its own; I closed it and went back to the game.")


async def _played(page: Any, band: tuple[float, float, float, float], goal: str, url: str,
                  step: dict[str, Any]) -> dict[str, Any]:
    """The runs of one game, until it is won, the time is up, or it cannot be gone on with."""
    import time

    from core.agency.a_page_gone_blank import GONE_BLANK
    from core.language.how_a_game_ended import asks_to_win, requested_attempts
    from core.language.what_counts_as_done import MADE, MEASURE, terms_asked, without_conditions
    from core.skills.a_new_attempt_must_be_seen import (
        AttemptObservation,
        a_game_was_played,
        attempt_activity_was_seen,
    )
    from core.skills.screen_pursuit_as_it_happens import (
        MADE_NOT_WON,
        MAKING_STALLED,
    )

    # What the person said counts as done for each shape of thing: an end reached, a measure bettered, a thing made.
    terms = terms_asked(goal)
    until_won = asks_to_win(without_conditions(goal)) or terms.until_the_end
    attempts = requested_attempts(without_conditions(goal))
    where_none = terms.measure_runs
    # Asked to carry it through to its end, as many runs as that takes; else a count said, or the most in one hand-over.
    limit = (min(attempts, MOST_RUNS) if attempts is not None else float("inf") if terms.until_the_end
             else MOST_RUNS if until_won else 1)
    if where_none:
        limit = max(limit, where_none)
    # One of several things asked for ("the first of the 3 picked") has its share of the time, not all of it; asked to
    # be carried on until its end is reached, it has all of it.
    share = PLAY_UNTIL_WON_S if terms.until_the_end else max(LEAST_SHARE_S, PLAY_UNTIL_WON_S / _how_many_asked(goal))
    deadline = _the_share_ends(url, share, share if until_won or limit > 1 else _one_run_s())
    contract = step.get("runtime_contract") or {}
    keep: dict[str, Any] = {"required_edges": contract.get("required_edges") or [], "all_of_it_ends": deadline,
                            "edge_provenance": contract.get("provenance") or "", "stock": await _a_stocktaking(goal, page)}
    keep['attempt_observation'] = AttemptObservation(time.time())
    runs: list[dict[str, Any]] = []
    moves: list[Any] = []
    result: dict[str, Any] = {}
    measure_mode = False
    while time.monotonic() < deadline and sum(bool(r.get("reached_end")) for r in runs) < limit:
        started, steps_before = time.monotonic(), _steps_done(keep)
        # Played to be won, a round is looked at every few minutes: still getting somewhere, it goes on.
        round_ends = min(deadline, started + ROUND_S) if until_won else deadline
        result, reflexes = await _one_run(page, band, goal, url, round_ends, keep)
        keep = reflexes.keep
        moves += list(result.get("moves") or [])
        moves += [{"key": "played as it happened"} for stretch in reflexes.stretches if stretch.get("pictures")]
        run = _how_the_run_went(reflexes, result)
        run["reached_end"] = _run_reached_an_end(reflexes, run, result)
        unobserved_ending = not attempt_activity_was_seen(reflexes) and run['ended'] in ('won', 'lost', 'drawn', 'finished')
        if not attempt_activity_was_seen(reflexes):
            run['ended'], run['reached_end'] = '', False
        run["took_s"] = time.monotonic() - started
        run["gains"] = sum(int(stretch.get("gains") or 0) for stretch in reflexes.stretches)
        run["observations"] = [
            {key: stretch.get(key) for key in ("pictures", "pictures_a_second", "observations", "standing", "settled", "runtime_checks",
                                               "input_key_downs", "responsive_pictures", "control_probe_retries",
                                               "control_attribution", "attribution_changes", "identification_evidence")}
            for stretch in reflexes.stretches if stretch.get("pictures")
        ]
        run["score"] = _the_score_of(run, reflexes)
        # The counsel she played this round by, so that a round much worse than her best can go back to it.
        run["counsel"] = str(keep.get("counsel") or "")
        runs.append(run)
        if unobserved_ending and not a_game_was_played(reflexes):
            result['stopped_because'] = 'an ending was shown without new activity being observed'
            break
        violations = [v for s in reflexes.stretches for v in (s.get("runtime_checks") or {}).get("violations", [])]
        if violations:
            result["runtime_violations"] = violations
            break
        if reflexes.over_because == MAKING_STALLED:
            result["stopped_because"] = MAKING_STALLED
            result["completed"] = False
            run["ended"], run["reached_end"] = "", False
            _tell("I could not finish making something here; I have not verified a result to show.")
            break
        if reflexes.over_because == GONE_BLANK:
            band, again = await _begun_again_where_it_went_blank(page, band, run, runs, keep, deadline)
            if again:
                continue
            result["stopped_because"] = GONE_BLANK
            break
        shape = MADE if reflexes.over_because == MADE_NOT_WON else _the_shape_of(reflexes, run)
        if shape == MADE and (terms.made_once or terms.tell_what_was_made) and _a_creation_handed_back(reflexes):
            # Made once, and what was made said, as the person asked of a thing with no end and no measure.
            run["ended"] = "finished"
            _tell(await _what_was_made_as_seen(page, band, run, moves, list(getattr(reflexes, "words", []) or []), keep))
            result["stopped_because"] = MADE_AS_ASKED
            break
        nothing_to_win = shape == MEASURE or _for_points_only(reflexes, goal) or _only_a_score(reflexes, run)
        if nothing_to_win and where_none:
            measure_mode = True
            # Where there is nothing to win, what the person asked for that case: so many runs, and the best of them.
            if not run["reached_end"]:
                # A clock-bounded hand-over can return while the same run is still in play. Its score is evidence of
                # progress, not another finished attempt; keep playing it without beginning it again.
                if time.monotonic() < deadline:
                    continue
                break
            run["ended"] = run["ended"] if run["ended"] in ("won", "lost") else "finished"
            finished_runs = [r for r in runs if r.get("reached_end")]
            if len(finished_runs) < where_none and time.monotonic() < deadline:
                _tell(f"{_what_there_is_to_play_for(reflexes)} Run {len(finished_runs)} of {where_none}: {_scored(run)}. Again.")
                next_band = await _restart_attempt(page, band, keep)
                if next_band is None:
                    result['stopped_because'] = 'the next attempt did not begin'
                    break
                band = next_band
                continue
            _tell(_the_best_of(finished_runs))
            result["stopped_because"] = (PLAYED_FOR_ITS_BEST if len(finished_runs) >= where_none else
                                         f"only {len(finished_runs)} of {where_none} requested runs reached an end before time ran out")
            break
        if reflexes.over_because == MADE_NOT_WON:
            # Asked to win a thing that has no winning, the honest end is to say so, in its own words.
            _tell(f"There's nothing to win in this one; it's for making things: {_what_it_is_for(reflexes.words)!r}. "
                  "I tried it out, and I leave it here.")
            run["ended"] = "finished"
            result["stopped_because"] = NOTHING_TO_WIN
            break
        # The round's own clock stops a little before the slice's end: a round that ran most of its slice ran out of it.
        # Looked at before a round with no end of its own is taken as the last: LIVE 2026-10-07 this came after that
        # break, could never be reached, and a game was left after four minutes with nothing said.
        if (until_won and run["ended"] != "won" and not reflexes.over_because
                and time.monotonic() - started >= ROUND_S - 20.0 and time.monotonic() < deadline):
            # A game in turns that is taking hers is going on, whatever is gained in it: LIVE 2026-10-11 a draughts
            # game was begun again from the top in its fourth minute, she having made thirty moves in it.
            if (run["gains"] > 0 or reflexes.new_screens or _steps_done(keep) > steps_before
                    or any(stretch.get("turns") for stretch in reflexes.stretches)
                    or await _stopped_to_take_stock(STUCK_HERE, goal, reflexes, run, runs, keep, deadline)):
                runs.pop()  # the same round, still getting somewhere, or with something new to go by: played on
                continue
            if terms.until_the_end and shape != MEASURE:
                # Asked to carry it through to its end, a round that gained nothing is begun again from the top.
                _tell(f"{round(ROUND_S / 60)} minutes in this round and nothing gained; I'm starting it again from the top.")
                next_band = await _restart_attempt(page, band, keep)
                if next_band is None:
                    result['stopped_because'] = 'the next attempt did not begin'
                    break
                band = next_band
                continue
            _tell(f"{round(ROUND_S / 60)} minutes in this round and nothing gained; I'll leave this one here.")
            result["stopped_because"] = "getting nowhere in it"
            break
        if (until_won and run["ended"] == "won") or sum(bool(r.get("reached_end")) for r in runs) >= limit or not reflexes.over_because:
            _the_last_run_said(run, runs, keep)
            break
        if until_won and not_getting_better(runs) and not terms.until_the_end and not await _stopped_to_take_stock(
                STUCK_HERE, goal, reflexes, run, runs, keep, deadline):
            # A person keeps at a game while they are getting better at it, and says so when they are not.
            _tell(f"That's {len(runs)} rounds, and the last two went no better than my best; I'll leave this one here.")
            run["ended"] = run["ended"] or "lost"
            result["stopped_because"] = "not getting better at it"
            break
        if _for_points_only(reflexes, goal):
            # Nobody wins a game that only counts points: a finished run is the end of it.
            run["ended"] = "finished"
            _tell(f"This game has no winner, only a score, and the run is done: {run['words'][:80]!r}.")
            break
        # The screen's own words the first time; after that, the round counted: six of the same line read as one.
        await _between_runs(goal, reflexes, run, runs, keep, deadline)
        if run['reached_end'] and time.monotonic() < deadline:
            next_band = await _restart_attempt(page, band, keep)
            if next_band is None:
                result['stopped_because'] = 'the next attempt did not begin'
                break
            band = next_band
    if time.monotonic() >= deadline - RAN_OUT_WITHIN_S and not any(r["ended"] == "won" for r in runs) and not result.get("stopped_because"):
        result["stopped_because"] = await _said_as_the_time_ran_out(page, band, terms, runs, moves, reflexes, keep)
    elif until_won and not any(r["ended"] == "won" for r in runs) and not result.get("stopped_because"):
        # Played to be won and stopped short of it, with no end said: an end is said, and the page is not begun again.
        result["stopped_because"] = "I could not get further with it"
    result["as_it_happened"] = "; ".join(r["said"] for r in runs if r["said"])
    last_seen = str(result.get("last_seen") or "")
    won = any(r["ended"] == "won" for r in runs)
    finished_runs = [r for r in runs if r.get("reached_end")]
    counted = (until_won and won) or attempts is None or len(finished_runs) >= attempts
    complete = (won if until_won else bool(finished_runs)) and counted and not result.get("runtime_violations")
    # Nothing to win, and played as the person said to play such a thing: done as asked.
    fulfilled_measure = result.get("stopped_because") == PLAYED_FOR_ITS_BEST and _measure_runs_are_done(runs, where_none)
    complete = complete or ((fulfilled_measure or result.get("stopped_because") == MADE_AS_ASKED) and not result.get("runtime_violations"))
    if measure_mode:
        complete = bool(fulfilled_measure and not result.get("runtime_violations"))
    if result.get("stopped_because") == MAKING_STALLED:
        complete = False
    scores = [r["score"] for r in finished_runs if r.get("score") is not None]
    return {
        **step,
        "landed": len(moves),
        "moved": bool(moves),
        "played": str(result.get("outcome") or ""),
        "completed": complete,
        "last_seen": last_seen,
        **{key: dict(result[key]) for key in ('last_observation', 'last_read_words') if key in result},
        "runs": [r["ended"] or "unread" for r in runs],
        "run_details": runs,
        "requested_attempts": attempts,
        "requested_measure_runs": where_none,
        "completed_runs": len(finished_runs),
        "won": won,
        "finished": any(r["ended"] in ("won", "finished") for r in runs),
        "did": what_the_play_came_to(len(moves), result),
        "ok": bool(moves) and complete,
        "runtime_violations": result.get("runtime_violations") or [],
        "stopped_because": str(result.get("stopped_because") or ""),
        "the_ask_does_not_apply": result.get("stopped_because") in (NOTHING_TO_WIN, PLAYED_FOR_ITS_BEST, MADE_AS_ASKED),
        "best_score": max(scores) if scores else None,
    }


async def _said_as_the_time_ran_out(page: Any, band: tuple[float, float, float, float], terms: Any,
                                    runs: list[dict[str, Any]], moves: list[Any], reflexes: Any, keep: dict[str, Any]) -> str:
    """What is said when her time with a thing is up, and why she stopped."""
    if runs and moves and (terms.made_once or terms.tell_what_was_made) and _a_place_for_making(keep):
        if not _a_creation_handed_back(reflexes):
            from core.skills.screen_pursuit_as_it_happens import MAKING_STALLED

            _tell("My time with this one is up. I have not verified a created result to show.")
            return MAKING_STALLED
        # A person whose time is up in a thing for making shows what they made in it: offline 2026-10-12 a
        # scrapbook of photographs was worked for five minutes and left with nothing said of what was in it.
        _tell("My time with this one is up. " + await _what_was_made_as_seen(
            page, band, runs[-1], moves, list(getattr(reflexes, "words", []) or []), keep))
        return MADE_AS_ASKED
    _tell("I've given this one its share of the time, and I leave it here." + _where_it_stood(runs))
    return "its share of the time is up"


async def _begun_again_where_it_went_blank(page: Any, band: tuple[float, float, float, float], run: dict[str, Any],
                                           runs: list[dict[str, Any]], keep: dict[str, Any],
                                           deadline: float) -> tuple[tuple[float, float, float, float], bool]:
    """A run whose page went blank is no run finished: the page is loaded again and the thing on it begun again, a
    few times at most (core/agency/a_page_gone_blank.py). The band it draws in, and whether play goes on."""
    import time

    from core.agency.a_page_gone_blank import BEGUN_AGAIN_AT_MOST

    run["ended"], run["reached_end"] = "", False
    keep.pop("blank_since", None)
    keep["begun_again_blank"] = int(keep.get("begun_again_blank") or 0) + 1
    if keep["begun_again_blank"] > BEGUN_AGAIN_AT_MOST or time.monotonic() >= deadline:
        _tell("The game has gone blank again, and I leave it here." + _where_it_stood(runs))
        return band, False
    _tell("The game has gone blank, so I'm loading it again.")
    next_band = await _restart_attempt(page, band, keep)
    return (next_band, True) if next_band is not None else (band, False)


#: Why she stops to take stock when a round gains nothing (core/cognition/taking_stock.py).
STUCK_HERE = "I'm getting nowhere"


async def _a_stocktaking(goal: str, page: Any) -> Any:
    """Taking stock in the thing on the page, named as the request names it, else by the page's own title: LIVE
    2026-10-09 asked for a game by its address, she asked the web "how to win https://archive.org/details/...".
    """
    from core.skills.sovereign_browser_taking_stock import Stocktaking, the_thing

    try:
        title = str(await page.title() or "")
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
        title = ""
    return Stocktaking(the_thing(goal, title or str(getattr(page, "url", "") or "")), url=str(getattr(page, "url", "") or ""))


async def _stopped_to_take_stock(why: str, goal: str, reflexes: Any, run: dict[str, Any], runs: list[dict[str, Any]],
                                 keep: dict[str, Any], deadline: float) -> bool:
    """Taken stock, while the thing waits for her, with something found to go by (core/skills/sovereign_browser_taking_stock.py)."""
    stock = keep.get("stock")
    if stock is None or stock.taken >= 1 and why == STUCK_HERE and stock.counsel is not None:
        return False
    return bool(await stock.take(why, goal, list(reflexes.words), run["words"], runs, keep, deadline))


def _the_last_run_said(run: dict[str, Any], runs: list[dict[str, Any]], keep: dict[str, Any]) -> None:
    """The run that ends it, told as it ends (a win heard of only in the reply afterwards was not seen by anyone
    watching), and what she went by in it judged by how it ended: a win after counsel is counsel that helped."""
    if run["ended"]:
        _tell(f"That one ended {run['words'][:80]!r}: {run['ended']}.")
    stock = keep.get("stock")
    if stock is not None and stock.counsel is not None and run["ended"] == "won":
        stock.judge(runs)


async def _between_runs(goal: str, reflexes: Any, run: dict[str, Any], runs: list[dict[str, Any]], keep: dict[str, Any],
                        deadline: float) -> None:
    """A run lost and another begun: how it ended said, the counsel in force judged, and stock taken where she keeps
    losing (core/skills/sovereign_browser_taking_stock.py)."""
    # The screen's own words the first time; after that, the round counted: six of the same line read as one. And what
    # cost her, by name, where play came to keep clear of it: a person going again says what they will do differently.
    _tell((f"That one ended {run['words'][:80]!r}: {run['ended'] or 'unread'}. Again, with what I learned." if len(runs) == 1
           else f"Round {len(runs)}: {run['ended'] or 'over'}. Again.") + _what_i_do_differently(reflexes))
    # A try that ended without the end reached says the plan was wrong somewhere: it is made again with what happened.
    from core.cognition.a_plan_to_an_end import plan_again

    plan_again(_the_guide_in_hand(keep), f"it ended on “{run['words'][:120]}” ({run['ended'] or 'unread'})")
    _back_to_what_went_best(runs, keep)
    stock = keep.get("stock")
    if stock is not None:
        stock.judge(runs, keep)
        why = stock.due(runs)
        if why:
            await stock.take(why, goal, list(reflexes.words), run["words"], runs, keep, deadline)


def _steps_done(keep: dict[str, Any]) -> int:
    """How many steps of the place's lesson and of her plan are behind her: a round in which more are is getting somewhere,
    though nothing was scored. LIVE 2026-10-10 a trap-building game, where what is gained is the build, was begun again
    from the top every four minutes for gaining nothing."""
    guide = keep.get("guide")
    return sum(len(getattr(p, "done", ()) or ()) for p in (getattr(guide, "rules", None), getattr(guide, "plan", None)) if p)


def _how_well(run: Mapping[str, Any]) -> float:
    """How well a round went, by its score where it has one, else by what it gained."""
    score = run.get("score")
    return float(score) if isinstance(score, (int, float)) else float(run.get("gains") or 0)


def _back_to_what_went_best(runs: list[dict[str, Any]], keep: dict[str, Any]) -> None:
    """Where the round just played went far worse than her best, the counsel she played her best round by, back in
    force: what she changed since made it worse. As the agents that improve most over attempts at unfamiliar games do
    (OmniGameArena, 2026): a revised way of playing that drops below half the best is rolled back to the best."""
    if len(runs) < 2:
        return
    best = max(runs[:-1], key=_how_well)
    if _how_well(best) <= 0 or _how_well(runs[-1]) >= 0.5 * _how_well(best):
        return
    if str(best.get("counsel") or "") == str(keep.get("counsel") or ""):
        return
    keep["counsel"] = str(best.get("counsel") or "")
    _tell(f"That round went much worse than my best ({_scored(runs[-1])} against {_scored(best)}), so I'm going back to "
          "how I played my best one.")


def _what_i_do_differently(reflexes: Any) -> str:
    """What cost her in the run just played, as play came to keep clear of it, said as what she does differently."""
    cost = list(dict.fromkeys(name for stretch in getattr(reflexes, "stretches", []) or []
                              for name in stretch.get("came_to_avoid") or [] if name and name != "other"))[:2]
    if not cost:
        return ""
    return f" The {' and the '.join(cost)} cost me; this time I keep clear of {'it' if len(cost) == 1 else 'them'}."


#: How long she keeps playing a game she was asked to win, in all.
PLAY_UNTIL_WON_S = 1200.0

#: The most runs of one game in one hand-over.
MOST_RUNS = 12


#: How near its deadline a run that stopped is taken to have run out of time: the run's own clock stops a little before.
RAN_OUT_WITHIN_S = 20.0


def _the_share_ends(url: str, share: float, this_time: float) -> float:
    """When play here stops: this hand-over's time, and never past the share the game has had since it was begun for
    the asking in hand. LIVE 2026-10-12 a name machine handed back for "I'll keep playing it" was given four fresh
    minutes each time, its run stopping a moment before its deadline and so never said to be out of time, and one game
    of three held the asking for eighteen minutes."""
    import time

    now = time.monotonic()
    began = _SHARE_BEGAN.setdefault(_address(url), now)
    if now - began > 2 * PLAY_UNTIL_WON_S:
        began = _SHARE_BEGAN[_address(url)] = now  # begun for an asking long gone: this is a new one
    return min(now + this_time, began + share)


def _one_run_s() -> float:
    from core.runtime.watched_goal import PURSUIT_SECONDS

    return PURSUIT_SECONDS


async def _one_run(page: Any, band: tuple[float, float, float, float], goal: str, url: str,
                   deadline: float, keep: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from core.cognition.guidance_lifetime import guidance_work

    async with guidance_work() as guidance:
        return await _one_run_with_guidance(page, band, goal, url, deadline, keep, guidance)


async def _one_run_with_guidance(page: Any, band: tuple[float, float, float, float], goal: str, url: str,
                               deadline: float, keep: dict[str, Any], guidance: Any) -> tuple[dict[str, Any], Any]:
    """One run of the game: menus by the screen pursuit, play by its reflexes, until the run is over."""
    import time

    from core.cognition.a_guide_to_a_place import THE_GUIDE
    from core.skills.screen_pursuit import pursue_on_screen
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, PlayingAsItHappens
    from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage

    stock = keep.get("stock")
    guide = await _the_guide(page, keep)
    if stock is not None and not stock.taken and not stock.remembered():
        # Optional research runs beside the work already grounded in this
        # page. Its operation owner retires it when this run ends.
        from core.cognition.guidance_lifetime import create_guidance_task
        from core.cognition.taking_stock import BEFORE

        create_guidance_task(stock.take(BEFORE, goal, [], "", [], keep, deadline))
    _what_the_guide_says(guide, keep, goal)
    reflexes = PlayingAsItHappens(page=page, band=band, goal=goal, ends_at=deadline, keep=keep)
    held = HER_OWN_PAGE.set(on := OnAPage(page=page, name=HER_BROWSER))
    quick = AS_IT_HAPPENS.set(reflexes)
    guided = THE_GUIDE.set(guide)
    try:
        result = await pursue_on_screen(
            goal=goal,
            success_when="",
            target_app=HER_BROWSER,
            expect_page=url,
            drawn_at=band,
            max_seconds=max(1.0, deadline - time.monotonic()),
        )
    finally:
        await guidance.close()
        THE_GUIDE.reset(guided)
        AS_IT_HAPPENS.reset(quick)
        HER_OWN_PAGE.reset(held)
        await reflexes.close()
        await on.stop_watching()
        _keep_the_guide(guide)
    return result, reflexes


async def _the_guide(page: Any, keep: dict[str, Any]) -> Any:
    """The guide to the thing on the page (core/cognition/a_guide_to_a_place.py): the one in hand, else what she kept of
    it before, else one made now from the page's words and its program."""
    from core.cognition.a_guide_to_a_place import Guide, carry_over
    from core.runtime.what_she_learned import named, recall
    from core.skills.sovereign_browser_guide import guide_to_the_page

    guide = keep.get("guide")
    if isinstance(guide, Guide):
        return guide
    stock = keep.get("stock")
    thing = str(getattr(stock, "thing", "") or "")
    # Made afresh from the page and its program every visit (what a page says of itself can change); what play taught
    # her of it before (what she came to think, its names, what helped) is carried over.
    guide = await guide_to_the_page(page, thing)
    kept = recall(named("a guide", thing)) if thing else None
    if kept:
        carry_over(guide, Guide.from_memory(kept, thing))
    keep["guide"] = guide
    return guide


def _what_the_guide_says(guide: Any, keep: dict[str, Any], goal: str = "") -> None:
    """What was looked up taken into the guide, and what the guide holds said once, before she begins; and what she
    knows herself of the place asked of her model beside her play (core/cognition/what_i_know_of_a_place.py)."""
    from core.cognition.what_i_know_of_a_place import ask_in_the_background
    from core.rebuilding.her_model import ask_her_model

    stock = keep.get("stock")
    counsel = getattr(stock, "counsel", None)
    if counsel is not None and getattr(counsel, "told", ""):
        guide.take_in_counsel(counsel.told)
    if not guide.said and guide.says():
        guide.said = True
        _tell(guide.says())
    ask_in_the_background(guide, task=goal, said=" ".join(guide.goals[:2]), ask=ask_her_model, tell=_tell)


def _keep_the_guide(guide: Any) -> None:
    from core.runtime.what_she_learned import named, remember

    if guide is not None and guide.place:
        remember(named("a guide", guide.place), guide.as_memory())


def _how_the_run_went(reflexes: Any, result: Mapping[str, Any]) -> dict[str, str]:
    from core.language.how_a_game_ended import (
        how_it_ended,
        how_it_ended_in,
        says_an_unresolved_result,
        says_play_is_ongoing,
    )

    words = reflexes.ending_words or str(result.get("last_seen") or "")
    parts = list(getattr(reflexes, "ending_parts", None) or [])
    ended = how_it_ended_in(parts) if parts else how_it_ended(words)
    current = "\n".join([str(result.get("last_seen") or ""), words, *parts])
    ongoing = says_play_is_ongoing(current)
    if ongoing:
        ended = ""
    # Where the end screen's words do not say, the counters may: her score at
    # what wins, theirs at it, nothing left to lose (core/agency/how_the_contest_stands.py).
    played = [s for s in getattr(reflexes, "stretches", []) if s.get("pictures")]
    if any((s.get("runtime_checks") or {}).get("violations") for s in played):
        ended = "interrupted"
    if not ended and played and not ongoing and not says_an_unresolved_result(current) and not _a_clock_is_still_running(reflexes, result):
        predicted = str(played[-1].get("settled") or "")
        ended = predicted if predicted in ("won", "lost", "drawn") else ""
    return {"ended": ended, "words": " ".join(words.split()), "said": reflexes.what_it_came_to()}


def _a_clock_is_still_running(reflexes: Any, result: Mapping[str, Any]) -> bool:
    """Whether recent observations show a clock decreasing with time still left; a frozen final remainder is not play."""
    clock_words = re.compile(r"\b(?:time|timer|clock|countdown)\b", re.I)
    history: dict[str, list[float]] = {}

    def keep(label: str, value: Any, source: str) -> None:
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label))
        name = " ".join(re.sub(r"[^a-z0-9]+", " ", name.lower()).split())
        if not clock_words.search(name) or re.search(r"\b(?:best|record|final|total|elapsed|response|reaction|completion)\b", name):
            return
        try:
            units = [float(part) for part in str(value).strip().split(":")]
        except (TypeError, ValueError):
            return
        if not units or len(units) > 3:
            return
        seconds = sum(unit * 60 ** i for i, unit in enumerate(reversed(units)))
        history.setdefault(f"{source}:{name}", []).append(seconds)

    for stretch in getattr(reflexes, "stretches", []) or []:
        if stretch.get("pictures"):
            for label, value in (stretch.get("counters") or {}).items():
                keep(str(label), value, "counter")
    written_clock = re.compile(r"\b((?:time|timer|clock|countdown)(?:\s+(?:left|remaining))?|remaining time)\s*:?\s*(\d+(?::\d{1,2}){0,2}(?:\.\d+)?)\b", re.I)
    screens = list(getattr(reflexes, "words", None) or [])[-6:]
    current = str(result.get("last_seen") or "")
    if current and (not screens or current != screens[-1]):
        screens.append(current)
    for screen in screens:
        for found in written_clock.finditer(screen):
            keep(found.group(1), found.group(2), "text")
    measured = [values for name, values in history.items() if name.startswith("counter:") and len(values) >= 2]
    # Repeated counter observations can establish that a final positive remainder is frozen. A changed-screen-only
    # OCR history does not record that repeated reading and cannot overrule it with an older decrease.
    recent = measured or [values for name, values in history.items() if name.startswith("text:")]
    return any(len(values) >= 2 and 0 < values[-1] < values[-2] for values in recent)


def _measure_runs_are_done(runs: list[Mapping[str, Any]], requested: int | None) -> bool:
    """Only runs that reached their end can fulfill a requested number of attempts at a measure."""
    return requested is not None and requested > 0 and sum(bool(run.get("reached_end")) for run in runs) >= requested


@invariant("agency.measured_play_requires_requested_completed_runs", scope="agency",
           owner="core/skills/sovereign_browser_drawing.py", observational=False)
def _measure_runs_count_invariant() -> tuple:
    """A visible score or partial hand-over cannot satisfy a requested count of finished runs."""
    for count in range(4):
        runs = [{"reached_end": True, "score": 120} for _ in range(count)]
        runs += [{"reached_end": False, "score": 9999} for _ in range(5)]
        assert _measure_runs_are_done(runs, 3) is (count >= 3)
    return ()


@invariant("agency.stalled_making_is_not_completion", scope="agency",
           owner="core/skills/sovereign_browser_drawing.py", observational=False)
def _stalled_making_invariant() -> tuple:
    import time

    from core.skills.screen_pursuit_as_it_happens import (
        MAKING_STALLED,
        TRYING_OUT_MOST_S,
        PlayingAsItHappens,
    )

    run = PlayingAsItHappens(None, (0, 0, 1, 1), "make one", time.monotonic() + 60)
    run.words, run.for_making_since = ["Create your own card."], time.monotonic() - TRYING_OUT_MOST_S - 1
    assert run.run_is_over({"text": "CARD EDITOR"})
    assert run.over_because == MAKING_STALLED
    return ()


def _run_reached_an_end(reflexes: Any, run: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
    """Whether this hand-over reached the run's end, rather than returning a still-current measure at its own deadline."""
    from core.language.how_a_game_ended import (
        says_a_round_is_over,
        says_play_is_ongoing,
        what_says_the_game_is_over,
    )
    from core.skills.a_new_attempt_must_be_seen import attempt_activity_was_seen

    if not attempt_activity_was_seen(reflexes):
        return False

    words = "\n".join([str(result.get("last_seen") or ""), str(run.get("words") or ""),
                       *list(getattr(reflexes, "ending_parts", None) or [])])
    if says_play_is_ongoing(words) or run.get("ended") == "interrupted":
        return False
    if run.get("ended") in ("won", "lost", "drawn", "finished"):
        return True
    return bool(getattr(reflexes, "over_because", "") and
                (says_a_round_is_over(words) or what_says_the_game_is_over(words)))


@invariant("agency.current_play_vetoes_predicted_completion", scope="agency",
           owner="core/skills/sovereign_browser_drawing.py", observational=False)
def _current_play_veto_invariant() -> tuple:
    """Current turn evidence prevents a board or counter prediction from declaring a win or loss."""
    from types import SimpleNamespace

    for prediction in ("won", "lost"):
        reflexes = SimpleNamespace(ending_words="Your turn to make a move", ending_parts=["Your turn to make a move"],
                                   stretches=[{"pictures": 1, "settled": prediction}], what_it_came_to=lambda: "")
        run = _how_the_run_went(reflexes, {})
        assert run["ended"] == ""
        assert not _run_reached_an_end(reflexes, run, {})
    return ()


#: The least time one of several games asked for is given, in seconds.
LEAST_SHARE_S = 360.0


def _how_many_asked(goal: str) -> int:
    """How many things the goal is one of, where it says ("the first of the 3 picked"); else 1."""
    import re

    said = re.search(r"\bof the (\d+) (?:picked|asked|chosen)\b", goal or "", re.I)
    return max(1, int(said.group(1))) if said else 1


#: How long a round played to be won goes before she looks at whether it is getting anywhere, in seconds.
ROUND_S = 240.0

#: Rounds played before whether she is getting better is judged, and how much longer a round must last to be better.
JUDGED_AFTER = 3
LONGER = 1.15


def not_getting_better(runs: list[dict[str, Any]]) -> bool:
    """Whether the last two rounds went no better than the best before them: no more gained, and not lasting longer.

    LIVE 2026-10-07 she played one game for twelve minutes and another for
    thirteen, each round lost as fast as the last. Better is gaining more, or,
    gaining the same, lasting longer by a sixth; a round won is better than
    any. Judged from the third round on.
    """
    if len(runs) < JUDGED_AFTER:
        return False
    if any(r.get("ended") == "won" for r in runs[-2:]):
        return False
    before = runs[:-2]
    best_gains = max(int(r.get("gains") or 0) for r in before)
    best_took = max(float(r.get("took_s") or 0.0) for r in before if int(r.get("gains") or 0) == best_gains)
    for r in runs[-2:]:
        gains, took = int(r.get("gains") or 0), float(r.get("took_s") or 0.0)
        if gains > best_gains or (gains == best_gains and took > LONGER * best_took):
            return False
    return True


#: Why play to win stopped at a thing that has no winning.
NOTHING_TO_WIN = "there is nothing to win in it, as it is for making things"

#: Why play stopped at a thing with nothing to win, played the number of times the person said to play such a thing.
PLAYED_FOR_ITS_BEST = "there is nothing to win in it, so I played it as many times as you said and kept my best"

#: Why play stopped at a thing for making, made once as the person said to make such a thing.
MADE_AS_ASKED = "it is for making things, so I made one and said what I made"


def _the_shape_of(reflexes: Any, run: Mapping[str, Any]) -> str:
    """Which shape the thing played is, from all its words seen so far and whether a run of it showed a measure."""
    from core.language.what_counts_as_done import shape_in

    return shape_in(" ".join([*getattr(reflexes, "words", []), str(run.get("words") or "")]),
                    a_measure_seen=run.get("score") is not None)


#: What her eyes are asked of a thing that answered a question: its answer, quoted where it is written.
_WHAT_IT_ANSWERED = ("This is a picture of a toy or game that has just answered a question someone typed into it. What "
                     "answer does it give? Quote the answer word for word where it is written on the picture; otherwise "
                     'say in a few words what it shows as its answer. Answer with JSON only: {"answered": "<the answer>"}')
#: What her eyes are asked of a thing made: what it shows, in a sentence.
_WHAT_IT_SHOWS = ("This is a picture someone has just made in a game for making things: a scene, a character, a card, a "
                  "photo. In one short plain sentence, say what it shows: who or what is in it, and what they are doing. "
                  'Answer with JSON only: {"shows": "<one sentence>"}')


async def _what_was_made_as_seen(page: Any, band: tuple[float, float, float, float], run: Mapping[str, Any],
                                 moves: list[Any], words: list[str], keep: Mapping[str, Any] | None = None) -> str:
    """What she made, as her eyes see it at the end, and why, in the thing's own words for what it is for; where her
    eyes cannot say, as her choices and its last words have it (``_what_was_made``). Offline 2026-10-11 she said of a
    scrapbook she had put nothing into "I made it: at the end it shows 'PUBLIC LIBRARY'", and why she made it was not
    said at all."""
    import io

    from core.perception.her_eyes import look_with_her_eyes
    from core.perception.where_i_am_on_screen import _as_jpeg

    shows = ""
    typed = str((keep or {}).get("typed") or "")
    # A name typed into a thing for making is part of what was made, not a question put to it: offline 2026-10-13 an
    # alien made, named and saved was said to have been given her name and to have answered "ERROR Could not connect".
    named = typed if (keep or {}).get("typed_kind") == "a name" and _a_place_for_making(keep) else ""
    try:
        import numpy as np
        from PIL import Image

        size = getattr(page, "viewport_size", None) or {"width": 1280, "height": 800}
        left, top, right, bottom = band
        png = await page.screenshot(clip={"x": left * size["width"], "y": top * size["height"],
                                          "width": (right - left) * size["width"], "height": (bottom - top) * size["height"]})
        frame = np.array(Image.open(io.BytesIO(png)).convert("RGB"))
        # A thing that answered what she asked is asked for its answer, in its own words where it writes them; a thing
        # made, for what it shows.
        asked_it = bool(typed) and not named
        answer = await look_with_her_eyes(_WHAT_IT_ANSWERED if asked_it else _WHAT_IT_SHOWS, _as_jpeg(frame),
                                          max_tokens=80, timeout_s=40.0)
        found = re.search(r"\{.*\}", answer or "", re.S)
        said = json.loads(found.group(0)) if found else {}
        shows = " ".join(str(said.get("answered") or said.get("shows") or "").split()).rstrip(".")
    except Exception as why:  # noqa: BLE001 - her eyes not answering leaves what her choices say
        import logging

        logging.getLogger(__name__).info("her eyes could not say what she made: %s", str(why)[:160])
    if typed and not named:
        # A thing that answers what is put to it: what she asked, and what it answered, as it shows it. Its answer is
        # what is new on it since she put the question, read first from its words, then from her eyes where what they
        # quote was not there before; the label over the answer is not the answer: offline 2026-10-12 a machine that
        # answered "THE FIST" over "YOUR SECRET IDENTITY" was said to have answered "YOUR SECRET IDENTITY", twice.
        from core.skills.screen_pursuit_as_it_happens import _the_new_part

        before = f"{(keep or {}).get('typed_on') or ''} {typed}"
        read = str((keep or {}).get("answered") or "")
        answered = ((_the_new_part(read, before) if read else "") or (_the_new_part(shows, before) if shows else "")
                    or read[:160] or shows or str(run.get("words") or "")[:160])
        gave = (f"I gave it my name, {typed!r}" if (keep or {}).get("typed_kind") == "a name" else f"I asked it {typed!r}")
        # A word in capitals stays in capitals: lowered by its first letter ("lYOUR") it reads as a program's name.
        opening = answered if answered[:2].isupper() else f"{answered[:1].lower()}{answered[1:]}"
        return f"There's nothing to win or score in this one; it answers what you give it. {gave}, and it answered: {opening}."
    gave = f" It asked for a name, and I gave it {named!r}." if named else ""
    if not shows:
        return _what_was_made(run, moves) + gave
    asked = _what_it_is_for(words)
    because = f" I made it that way because it asks to: {asked!r}." if asked else ""
    return (f"There's nothing to win or score in this one; it's for making things. What I made shows "
            f"{shows[:1].lower()}{shows[1:]}.{gave}{because}")


def _what_was_made(run: Mapping[str, Any], moves: list[Any]) -> str:
    """What she made, from what the thing shows at the end, and how she came to it, from the choices she made."""
    chose: list[str] = []
    for move in moves:
        key = str(move.get("key") if isinstance(move, Mapping) else move or "")
        found = re.match(r'click "(.+)"$', key)
        if found and (not chose or chose[-1] != found.group(1)):
            chose.append(found.group(1))
    shows = str(run.get("words") or "").strip()
    what = f"I made it: at the end it shows {shows[:120]!r}." if shows else "I made it; at the end it shows no words to quote."
    how = (f" I chose {', '.join(chose[:8])}" + (" and more" if len(chose) > 8 else "")
           + ", trying each to see what it changed and keeping what I finished with.") if chose else (
        " I worked it by clicking on the picture, keeping what each click made.")
    return f"There's nothing to win or score in this one; it's for making things. {what}{how}"


async def _restart_attempt(page: Any, band: tuple, keep: dict) -> tuple | None:
    """Reset memory only after the page's physical restart succeeds."""
    import time

    from core.skills.a_new_attempt_must_be_seen import AttemptObservation
    from core.skills.screen_pursuit_as_it_happens import _this_game, begin_run

    next_band = await _from_the_top(page, band)
    if next_band is None:
        return None
    begin_run(keep, _this_game(page))
    keep['attempt_observation'] = AttemptObservation(time.time())
    return next_band


async def _from_the_top(page: Any, band: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """The page reloaded and the thing on it begun again, as a person starts again when a go has got nowhere."""
    from core.runtime.errors import record_degradation

    try:
        await page.reload(wait_until="domcontentloaded")
        band = _the_band(await page.evaluate(_WHERE_IT_DRAWS)) or band
        band = await _start_what_is_covered(page, band)
        await page.evaluate(_HOLD_IT_STILL)
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("sovereign_browser", exc, severity="info", action="begin the page's thing again")
        return None
    return band


def _only_a_score(reflexes: Any, run: Mapping[str, Any]) -> bool:
    """Whether a game shows nothing that could be won, anywhere it has been seen, and ends a run with a score.

    An endless game says no "as many points as possible"; it only ends with
    "GAME OVER YOUR SCORE: 120". Seen whole, with no winner, no goal to beat
    and no level or stage on any of its screens, its run is finished, not lost.
    """
    from core.language.how_a_game_ended import names_a_measure, offers_a_win

    seen = " ".join([*getattr(reflexes, "words", []), str(run.get("words") or "")])
    # A score named with its number left undrawn is still a score kept: the run is measured by what she counted.
    return not offers_a_win(seen) and (run.get("score") is not None or names_a_measure(str(run.get("words") or "")))


def _the_score_of(run: Mapping[str, Any], reflexes: Any) -> int | None:
    """A run's score: as its last screen writes it, else as the game's own score counter last read, else None."""
    from core.language.how_a_game_ended import the_score_in

    written = the_score_in(str(run.get("words") or ""))
    if written is not None:
        return written
    for stretch in reversed([s for s in getattr(reflexes, "stretches", []) if s.get("pictures")]):
        for name, value in (stretch.get("counters") or {}).items():
            # A run's measure by whatever name the game keeps it: a score, points, a total, how far or how high it went
            # (offline 2026-10-12 a launching game kept each launch's "Total", and its runs were told as unscored).
            if re.search(r"score|points|pts|total|distance|height", str(name), re.I) and str(value).replace(",", "").isdigit():
                return int(str(value).replace(",", ""))
    return None


def _scored(run: Mapping[str, Any]) -> str:
    score = run.get("score")
    if score is not None:
        return f"scored {score}"
    counted = _by_her_own_count(run)
    return counted if counted else f"{run.get('ended') or 'over'}, its score not shown"


def _by_her_own_count(run: Mapping[str, Any]) -> str:
    """A run whose score was named and not drawn, as she can measure it herself: what she counted in it, and the
    rating its last screen gave it in words; '' where its last screen named no measure."""
    from core.language.how_a_game_ended import names_a_measure, the_rating_in

    words = str(run.get("words") or "")
    if not names_a_measure(words):
        return ""
    rated = the_rating_in(words)
    counted = f"{int(run.get('gains') or 0)} by my own count" if run.get("gains") else "its score not drawn"
    return f"{counted}, rated {rated}" if rated else counted


def _what_there_is_to_play_for(reflexes: Any) -> str:
    from core.skills.screen_pursuit_as_it_happens import MADE_NOT_WON

    if reflexes.over_because == MADE_NOT_WON:
        return f"There's nothing to win in this one; it's for making things: {_what_it_is_for(reflexes.words)!r}."
    return "There's nothing to win in this one, only a score."


def _the_best_of(runs: list[dict[str, Any]]) -> str:
    """The best of the runs, said: the highest score where any was shown, else the longest run."""
    scored = [(r["score"], i) for i, r in enumerate(runs, 1) if r.get("score") is not None]
    if scored:
        best, which = max(scored)
        others = [str(r.get("score")) for r in runs if r.get("score") is not None and r is not runs[which - 1]]
        # Every run is accounted for: LIVE 2026-10-10 "That's 3 runs ... the others: 466", the third begun again
        # before it showed a score and left out.
        unscored = len(runs) - len(scored)
        if unscored:
            others.append("one with no score shown" if unscored == 1 else f"{unscored} with no score shown")
        return f"That's {len(runs)} run{'s' if len(runs) != 1 else ''}. My best was run {which}, with {best}" + (
            f"; the others: {', '.join(others[:-1]) + ' and ' + others[-1] if len(others) > 1 else others[0]}."
            if others else ".")
    counted = [(int(r.get("gains") or 0), i) for i, r in enumerate(runs, 1)]
    if any(count for count, _which in counted):
        # No number was drawn for any run: the best is the one she counted most in, said as her own count, with what
        # each was rated in words where the game rated it.
        best, which = max(counted)
        others = [_by_her_own_count(r) or f"{int(r.get('gains') or 0)} by my own count"
                  for i, r in enumerate(runs, 1) if i != which]
        rated = _by_her_own_count(runs[which - 1]).partition(", rated ")[2]
        return (f"That's {len(runs)} run{'s' if len(runs) != 1 else ''}. No score was drawn; by my own count my best was "
                f"run {which}, with {best}" + (f", rated {rated}" if rated else "")
                + (f"; the others: {'; '.join(others)}." if others else "."))
    # No number drawn and none counted: what the game itself rated each run, in its own words, before how long each was.
    from core.language.how_a_game_ended import the_rating_in

    rated = [the_rating_in(str(r.get("words") or "")) for r in runs]
    if any(rated):
        told = (f"each was rated {rated[0]}" if len(set(rated)) == 1 and all(rated)
                else "; ".join(f"run {i} was rated {r}" for i, r in enumerate(rated, 1) if r))
        return f"That's {len(runs)} run{'s' if len(runs) != 1 else ''}. No score was drawn; {told}."
    longest = max(range(len(runs)), key=lambda i: float(runs[i].get("took_s") or 0.0))
    return f"That's {len(runs)} run{'s' if len(runs) != 1 else ''}. No score was shown; my longest was run {longest + 1}, at {float(runs[longest].get('took_s') or 0.0):.0f} seconds."


def _the_guide_in_hand(keep: Mapping[str, Any] | None) -> Any:
    """The guide to the place being played: the one a run has in hand, else the one kept for the place between its
    runs. Between runs and once they are over no run has one in hand: offline 2026-10-13 a thing for making whose time
    ran out was left with "I've given this one its share of the time" and nothing of what was made, the place being
    taken for one with no guide, and a lost run's plan was never made again."""
    from core.cognition.a_guide_to_a_place import THE_GUIDE

    guide = THE_GUIDE.get()
    return guide if guide is not None else (keep or {}).get("guide")


def _a_place_for_making(keep: Mapping[str, Any] | None = None) -> bool:
    """Whether she understands the place before her is for making something (core/agency/how_a_game_works.py)."""
    guide = _the_guide_in_hand(keep)
    if guide is None or not hasattr(guide, "controls"):
        return False
    from core.agency.how_a_game_works import MAKES, understood

    how = understood(guide)
    return how.counts.value not in ('win', 'score') and (
        how.me.known() and how.me.value == MAKES or how.counts.known() and how.counts.value == 'make')


def _a_creation_handed_back(reflexes: Any) -> bool:
    """A creation handoff is separate from recognizing a place or running out of time."""
    from core.skills.screen_pursuit_as_it_happens import MADE_NOT_WON

    return getattr(reflexes, 'over_because', None) == MADE_NOT_WON


@invariant('agency.making_deadlines_cannot_supply_a_creation_handoff', scope='agency',
           owner='core/skills/sovereign_browser_drawing.py', observational=False)
def _making_handoff_invariant() -> tuple:
    from types import SimpleNamespace

    from core.skills.screen_pursuit_as_it_happens import MADE_NOT_WON, MAKING_STALLED

    assert _a_creation_handed_back(SimpleNamespace(over_because=MADE_NOT_WON))
    assert not any(_a_creation_handed_back(SimpleNamespace(over_because=reason))
                   for reason in ('', MAKING_STALLED, 'out_of_time', MADE_AS_ASKED))
    return ()


def _where_it_stood(runs: list[dict[str, Any]]) -> str:
    """Where play stood when her time with a thing ran out, as a person leaving a game says how far they got: the runs
    that ended, the best of them, and the one still in play with its score so far. '' where no score was shown or
    counted. Offline 2026-10-12 games that only keep a score were left with "its share of the time" and nothing of the
    2400 points on the screen."""
    finished = [r for r in runs if r.get("reached_end")]
    said = " " + _the_best_of(finished) if finished else ""
    last = runs[-1] if runs else {}
    if not runs or last.get("reached_end"):
        return said
    score = last.get("score")
    so_far = str(score) if score is not None else f"{int(last.get('gains') or 0)} by my own count" if last.get("gains") else ""
    if not so_far:
        return said
    return said + (f" The run in play stood at {so_far} when the time ran out." if finished else
                   f" My run was still going, at {so_far}.")


def _what_it_is_for(words: list[str]) -> str:
    """The sentence of a thing's own words that sets a player to make something."""
    import re

    from core.language.how_a_game_ended import what_it_asks_of_a_player

    for screen in words:
        for sentence in re.split(r"(?<=[.!?])\s+", screen):
            if what_it_asks_of_a_player(sentence) == "make":
                return sentence.strip()[:160]
    return ""


def _for_points_only(reflexes: Any, goal: str) -> bool:
    from core.language.how_a_game_ended import what_it_asks_of_a_player

    said = what_it_asks_of_a_player(" ".join(reflexes.words))
    return said == "score"


def _tell(line: str) -> None:
    from core.skills.screen_pursuit import _tell as said

    said(line)


def what_the_play_came_to(moves: int, result: Mapping[str, Any]) -> str:
    """One line on what playing the drawing came to, for her next look at the page.

    The page's text is the same after a game as before it, so without this the
    browser pursuit's account of her last move says nothing changed, and she
    cannot tell a game she lost from one she has not started.
    """
    why = (
        str(result.get("played_out_because") or "")
        or str(result.get("why_no_move") or "")
        or str(result.get("outcome") or "")
    )
    said = f"played what the page draws ({moves} move(s)"
    said += f", ended: {why})" if why else ")"
    quick = str(result.get("as_it_happened") or "")
    if quick:
        said += f"; {quick}"
    last_seen = " ".join(str(result.get("last_seen") or "").split())
    if last_seen:
        said += f"; the last thing it showed: {last_seen}"
    return said
