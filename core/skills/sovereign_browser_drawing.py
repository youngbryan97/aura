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

    import time

    from core.runtime.watched_goal import PURSUIT_SECONDS
    from core.skills.screen_pursuit import pursue_on_screen
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, PlayingAsItHappens
    from core.skills.screen_pursuit_on_a_page import HER_OWN_PAGE, OnAPage

    reflexes = PlayingAsItHappens(page=page, band=band, goal=goal, ends_at=time.monotonic() + PURSUIT_SECONDS)
    held = HER_OWN_PAGE.set(OnAPage(page=page, name=HER_BROWSER))
    quick = AS_IT_HAPPENS.set(reflexes)
    try:
        result = await pursue_on_screen(
            goal=goal,
            success_when="",
            target_app=HER_BROWSER,
            expect_page=url,
            drawn_at=band,
            max_seconds=PURSUIT_SECONDS,
        )
    finally:
        AS_IT_HAPPENS.reset(quick)
        HER_OWN_PAGE.reset(held)
    result["as_it_happened"] = reflexes.what_it_came_to()
    moves = list(result.get("moves") or [])
    if reflexes.stretches:
        moves += [{"key": "played as it happened"} for stretch in reflexes.stretches if stretch.get("pictures")]
    last_seen = str(result.get("last_seen") or "")
    return {
        **step,
        "landed": len(moves),
        "moved": bool(moves),
        "played": str(result.get("outcome") or ""),
        "completed": bool(result.get("completed")),
        "last_seen": last_seen,
        "did": what_the_play_came_to(len(moves), result),
        "ok": bool(moves),
    }


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
