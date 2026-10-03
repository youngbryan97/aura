"""The screen pursuit's eyes and hands on a page in her own browser.

The screen pursuit reaches the world through nine things: read the screen,
press a key, press several, click a point, and five questions about windows —
is hers in front, what page is it, what is in front, what is on top of it, and
bring it back. Every test of it replaces exactly those nine with a world of its
own. A page in her own browser is one more world, and this is those nine
answered by the page: a picture the browser takes of what it draws, keys and
clicks the browser delivers to the page itself, and the page's own address.

Nothing reaches the desktop while a page is the surface. The keys go to the
page whatever the person has in front of them, and no window is ever guessed
at: LIVE 2 October, a hand-over that took the front window to be hers would
have played into the person's own Chrome.

The surface is set for the length of one hand-over (``HER_OWN_PAGE``) and each
of the nine checks it first, so a pursuit over the desktop is exactly what it
was.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["HER_OWN_PAGE", "OnAPage", "on_her_page"]

#: Where keys go when a page is the surface: the pursuit's names to the
#: browser's.
_KEY_NAMES = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "return": "Enter",
    "enter": "Enter",
    "tab": "Tab",
    "space": "Space",
    "escape": "Escape",
}

#: Focus the thing the page draws, across shadow roots, so keys reach it. A
#: player in a shadow root takes the keyboard only once it is focused (measured
#: in the pane 2 October: after "Play Game" the focus stayed on the body).
_FOCUS_THE_DRAWING = """
(function () {
  var best = null, area = 0;
  var walk = function (root, host) {
    var all = root.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.tagName === 'CANVAS' || el.tagName === 'EMBED' || el.tagName === 'OBJECT') {
        var r = el.getBoundingClientRect();
        if (r.width * r.height > area) { area = r.width * r.height; best = host || el; }
      }
      if (el.shadowRoot) walk(el.shadowRoot, host || el);
    }
  };
  walk(document, null);
  if (!best) return false;
  if (!best.hasAttribute('tabindex')) best.setAttribute('tabindex', '0');
  best.focus();
  return document.activeElement === best;
})()
"""


@dataclass
class OnAPage:
    """One page of her own browser, as the surface a screen pursuit acts on."""

    page: Any
    name: str
    _focused: bool = field(default=False, init=False)

    async def read(self, over: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
        from core.perception.what_her_page_shows import look_at_a_page

        seen = await look_at_a_page(self.page, over, name=self.name)
        if seen is None:
            return {"ok": False, "text": "", "layout": [], "error": "her page could not be photographed"}
        seen.setdefault("ok", True)
        return seen

    async def _focus(self) -> None:
        if self._focused:
            return
        try:
            self._focused = bool(await self.page.evaluate(_FOCUS_THE_DRAWING))
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            self._focused = False

    async def press(self, key: str) -> bool:
        from core.skills.screen_pursuit_surface import PRESSABLE_KEYS

        name = str(key or "").strip().lower()
        if name not in PRESSABLE_KEYS:
            return False
        await self._focus()
        try:
            await self.page.keyboard.press(_KEY_NAMES.get(name, name))
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        return True

    async def press_many(self, keys: Sequence[str]) -> int:
        landed = 0
        for key in keys:
            if not await self.press(key):
                break
            landed += 1
        return landed

    async def click(self, x: float, y: float, bounds: Sequence[int] | None) -> bool:
        """A click at shares of ``bounds``, in the page's own pixels; never outside them."""
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        if bounds and len(bounds) >= 4:
            left, top, wide, tall = (float(value) for value in bounds[:4])
        else:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left = top = 0.0
        try:
            await self.page.mouse.click(left + x * wide, top + y * tall)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        # A click on the drawing is also what gives it the keyboard.
        self._focused = True
        return True

    async def identity(self) -> dict[str, str]:
        try:
            title = str(await self.page.title() or "")
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            title = ""
        return {"url": str(getattr(self.page, "url", "") or ""), "title": title, "error": ""}


#: The page a hand-over is playing on, for as long as it plays.
HER_OWN_PAGE: contextvars.ContextVar[OnAPage | None] = contextvars.ContextVar(
    "aura_her_own_page", default=None
)


def on_her_page() -> OnAPage | None:
    """The page the pursuit is acting on, when it is acting on one."""
    return HER_OWN_PAGE.get()
