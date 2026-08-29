"""Reading a web page from the page, rather than from a photograph of it.

She reads the screen by looking at it — the same instrument whatever is in
front of her, which is right for an application nobody can question. A browser
is not that. A page knows exactly what it is showing and where, and asking it
is both cheaper and exact.

It is not a preference. LIVE 2026-08-29 on play2048.co: the screen reading
found five of the sixteen places on the board, at two distinct columns out of
four. Nothing downstream could recover from that — no lattice in a handful of
scattered cells, so no thing to model; no model, so nothing to look ahead
over; and with nothing to look ahead over every single move fell through to a
full language generation, about twenty-eight seconds each. The board was
drawn perfectly well the whole time. She was squinting at it.

Nothing here knows what a game is, or a board, or a tile. It asks the page for
every element that holds text and nothing else, and hands back what it is told
in the same shape the screen reader uses, so everything downstream is
unchanged. A table, a form, a calendar and a search-results list all come back
the same way.

Read-only: it evaluates an expression that gathers text and geometry and
returns them. It changes nothing on the page.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["MOST_REGIONS", "what_the_page_says"]

logger = logging.getLogger("Aura.WhatThePageSays")

#: How many pieces of text to bring back. A page with more than this on it is
#: not a thing she is acting on, it is a document, and reading all of it costs
#: more than it tells her.
MOST_REGIONS = 400

#: The longest a piece of text can be and still be a thing in a layout rather
#: than a paragraph about one.
A_LABEL_AT_MOST = 120

#: What the page is asked. Every element that holds text of its own, where it
#: sits, as a share of the part of the page she can see.
_ASK_THE_PAGE = """
(function () {
  var out = [];
  var all = document.body ? document.body.getElementsByTagName('*') : [];
  var W = window.innerWidth || 1, H = window.innerHeight || 1;
  for (var i = 0; i < all.length && out.length < %(most)d; i++) {
    var el = all[i];
    var own = '';
    for (var n = 0; n < el.childNodes.length; n++) {
      if (el.childNodes[n].nodeType === 3) own += el.childNodes[n].nodeValue;
    }
    own = own.replace(/\\s+/g, ' ').trim();
    if (!own || own.length > %(longest)d) continue;
    var r = el.getBoundingClientRect();
    if (!r || r.width <= 0 || r.height <= 0) continue;
    if (r.bottom <= 0 || r.top >= H || r.right <= 0 || r.left >= W) continue;
    var s = window.getComputedStyle(el);
    if (s && (s.visibility === 'hidden' || s.display === 'none' || s.opacity === '0')) continue;
    out.push({t: own, y: (r.top + r.height / 2) / H, x: (r.left + r.width / 2) / W});
  }
  return JSON.stringify(out);
})()
"""


async def what_the_page_says(browser: Any = None) -> tuple[tuple[float, float, str], ...]:
    """Every piece of text the page is showing, and where, top-down then across.

    In the same shape the screen reader produces — ``(y, x, text)`` with both
    positions as shares of what she can see — so nothing downstream needs to
    know which instrument was used.

    Returns nothing when there is no page to ask, which is not a failure: it
    is the answer for anything that is not a browser, and the caller looks at
    the screen instead.
    """
    said = await _ask(browser)
    if not said:
        return ()
    try:
        rows = json.loads(said)
    except (ValueError, TypeError):
        return ()
    if not isinstance(rows, list):
        return ()
    read: list[tuple[float, float, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            y, x = float(row.get("y")), float(row.get("x"))
        except (TypeError, ValueError):
            continue
        text = str(row.get("t") or "").strip()
        if not text or not (0.0 <= y <= 1.0 and 0.0 <= x <= 1.0):
            continue
        read.append((y, x, text))
    read.sort(key=lambda cell: (cell[0], cell[1]))
    logger.debug("the page says %d thing(s)", len(read))
    return tuple(read)


async def _ask(browser: Any) -> str:
    """Put the question to whatever browser is in front, or give up quietly."""
    if browser is None:
        try:
            from core.capabilities.browser_controller import get_browser_controller

            browser = get_browser_controller()
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "what_the_page_says", exc, severity="info", action="reach the browser"
            )
            return ""
    ask = getattr(browser, "read_page_text", None)
    if not callable(ask):
        return ""
    try:
        return str(await ask(_ASK_THE_PAGE % {"most": MOST_REGIONS, "longest": A_LABEL_AT_MOST}) or "")
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "what_the_page_says", exc, severity="info", action="ask the page what it says"
        )
        return ""
