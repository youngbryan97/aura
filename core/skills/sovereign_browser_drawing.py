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
from collections.abc import Mapping
from typing import Any

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
_PUT_OVER_IT = """
(function (x, y) {
  var el = document.elementFromPoint(x, y), depth = 0;
  while (el && el.shadowRoot && depth < 6) {
    var inner = el.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === el) break;
    el = inner; depth++;
  }
  if (!el || el.tagName === 'CANVAS' || el.tagName === 'EMBED' || el.tagName === 'OBJECT') return '';
  var pressed = el.closest ? (el.closest('button,a,[role=button]') || el) : el;
  var pressable = getComputedStyle(pressed).cursor === 'pointer' || /^(BUTTON|A|IMG|INPUT|SVG|PATH)$/i.test(pressed.tagName)
    || pressed.getAttribute('role') === 'button' || typeof pressed.onclick === 'function';
  return pressable ? (pressed.tagName + ' ' + (pressed.getAttribute('aria-label') || pressed.getAttribute('alt') || pressed.className || '')).slice(0, 80) : '';
})(%f, %f)
"""

#: The close controls of notices laid over the drawing (a player's warning, a
#: page's banner), by their own words or sign, as viewport points to click.
_CLOSES_OVER_IT = """
(function (l, t, r, b) {
  var found = [];
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
      found.push([x, y]);
    }
  };
  walk(document);
  return JSON.stringify(found.slice(0, 3));
})(%f, %f, %f, %f)
"""

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

    for _ in range(PRESSES_TO_START):
        wide, tall = await the_page_size(page)
        x, y = (band[0] + band[2]) / 2 * wide, (band[1] + band[3]) / 2 * tall
        over = str(await page.evaluate(_PUT_OVER_IT % (x, y)) or "")
        if not over:
            break
        _tell("The game has a button over it to start it; pressing that first.")
        await page.mouse.click(x, y)
        began = time.monotonic()
        while time.monotonic() - began < STARTS_WITHIN_S:
            await asyncio.sleep(1.0)
            if str(await page.evaluate(_PUT_OVER_IT % (x, y)) or "") != over:
                break
        await asyncio.sleep(2.0)
        band = _the_band(await page.evaluate(_WHERE_IT_DRAWS)) or band
    wide, tall = await the_page_size(page)
    closes = json.loads(str(await page.evaluate(_CLOSES_OVER_IT % (band[0] * wide, band[1] * tall, band[2] * wide, band[3] * tall)) or "[]"))
    for x, y in closes:
        _tell("A notice is over the game; closing it.")
        await page.mouse.click(float(x), float(y))
    return band


def chose_the_drawing(moves: list[tuple[Any, Any]]) -> bool:
    """Whether any of this round's moves is the drawing."""
    return any(str(getattr(action, "selector", "") or "") == DRAWING for action, _said in moves)


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
    """Play what her own page draws, through her own page, and say what came of it."""
    from core.runtime.errors import record_degradation

    url = str(observation.get("url") or "")
    step: dict[str, Any] = {"chose": ["what the page draws"], "url": url, "landed": 0}
    page = getattr(browser, "page", None)
    if page is None:
        return {**step, "error": "her browser has no page to play on"}
    try:
        band = _the_band(await page.evaluate(_WHERE_IT_DRAWS))
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
    try:
        return await _played(page, band, goal, url, step)
    finally:
        try:
            await page.evaluate(_LET_IT_GO)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            pass  # a page gone is a page that no longer needs letting go


async def _played(page: Any, band: tuple[float, float, float, float], goal: str, url: str,
                  step: dict[str, Any]) -> dict[str, Any]:
    """The runs of one game, until it is won, the time is up, or it cannot be gone on with."""
    import time

    from core.language.how_a_game_ended import asks_to_win, requested_attempts
    from core.skills.screen_pursuit_as_it_happens import begin_run

    until_won = asks_to_win(goal)
    attempts = requested_attempts(goal)
    limit = min(attempts, MOST_RUNS) if attempts is not None else MOST_RUNS if until_won else 1
    deadline = time.monotonic() + (PLAY_UNTIL_WON_S if until_won or limit > 1 else _one_run_s())
    keep: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []
    moves: list[Any] = []
    result: dict[str, Any] = {}
    while time.monotonic() < deadline and len(runs) < limit:
        result, reflexes = await _one_run(page, band, goal, url, deadline, keep)
        keep = reflexes.keep
        moves += list(result.get("moves") or [])
        moves += [{"key": "played as it happened"} for stretch in reflexes.stretches if stretch.get("pictures")]
        run = _how_the_run_went(reflexes, result)
        run["observations"] = [
            {key: stretch.get(key) for key in ("pictures", "pictures_a_second", "observations", "standing", "settled")}
            for stretch in reflexes.stretches if stretch.get("pictures")
        ]
        runs.append(run)
        if (until_won and run["ended"] == "won") or len(runs) >= limit or not reflexes.over_because:
            break
        if _for_points_only(reflexes, goal):
            # Nobody wins a game that only counts points: a finished run is the end of it.
            run["ended"] = "finished"
            _tell(f"This game has no winner, only a score, and the run is done: {run['words'][:80]!r}.")
            break
        _tell(f"That one ended {run['words'][:80]!r}: {run['ended'] or 'unread'}. Again, with what I learned.")
        begin_run(keep)
    result["as_it_happened"] = "; ".join(r["said"] for r in runs if r["said"])
    last_seen = str(result.get("last_seen") or "")
    won = any(r["ended"] == "won" for r in runs)
    counted = (until_won and won) or attempts is None or (len(runs) >= attempts and all(r["ended"] in ("won", "lost", "finished") for r in runs))
    complete = (won if until_won else bool(result.get("completed")) or bool(runs and runs[-1]["ended"])) and counted
    return {
        **step,
        "landed": len(moves),
        "moved": bool(moves),
        "played": str(result.get("outcome") or ""),
        "completed": complete,
        "last_seen": last_seen,
        "runs": [r["ended"] or "unread" for r in runs],
        "run_details": runs,
        "requested_attempts": attempts,
        "won": won,
        "finished": any(r["ended"] in ("won", "finished") for r in runs),
        "did": what_the_play_came_to(len(moves), result),
        "ok": bool(moves) and complete,
    }


#: How long she keeps playing a game she was asked to win, in all.
PLAY_UNTIL_WON_S = 1200.0

#: The most runs of one game in one hand-over.
MOST_RUNS = 12


def _one_run_s() -> float:
    from core.runtime.watched_goal import PURSUIT_SECONDS

    return PURSUIT_SECONDS


async def _one_run(page: Any, band: tuple[float, float, float, float], goal: str, url: str,
                   deadline: float, keep: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """One run of the game: menus by the screen pursuit, play by its reflexes, until the run is over."""
    import time

    from core.skills.screen_pursuit import pursue_on_screen
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, PlayingAsItHappens
    from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage

    reflexes = PlayingAsItHappens(page=page, band=band, goal=goal, ends_at=deadline, keep=keep)
    held = HER_OWN_PAGE.set(on := OnAPage(page=page, name=HER_BROWSER))
    quick = AS_IT_HAPPENS.set(reflexes)
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
        AS_IT_HAPPENS.reset(quick)
        HER_OWN_PAGE.reset(held)
        await reflexes.close()
        await on.stop_watching()
    return result, reflexes


def _how_the_run_went(reflexes: Any, result: Mapping[str, Any]) -> dict[str, str]:
    from core.language.how_a_game_ended import how_it_ended, how_it_ended_in

    words = reflexes.ending_words or str(result.get("last_seen") or "")
    parts = list(getattr(reflexes, "ending_parts", None) or [])
    ended = how_it_ended_in(parts) if parts else how_it_ended(words)
    # Where the end screen's words do not say, the counters may: her score at
    # what wins, theirs at it, nothing left to lose (core/agency/how_the_contest_stands.py).
    played = [s for s in getattr(reflexes, "stretches", []) if s.get("pictures")]
    if not ended and played:
        ended = str(played[-1].get("settled") or "")
    return {"ended": ended, "words": " ".join(words.split()), "said": reflexes.what_it_came_to()}


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
