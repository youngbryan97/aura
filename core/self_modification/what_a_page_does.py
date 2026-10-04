"""What a page that is not a moving picture does when it is used: its controls, its errors, its loads.

Watching a game is one way to see a program misbehave (watching_a_program_run.py).
Most programs are not games. A form, a list, a calculator, an editor: each is
a page of controls, and what is wrong with one shows when it is used. This
uses it the way a person trying it would: every control once, a button
pressed, a field typed into, a box ticked, a choice made, and after each,
whether anything on the page changed. And whether the page threw an error
while that happened, and whether everything it asked for loaded.

Three checks, read the way the game's are, so the same repair can use them:

  errors       the page threw while it loaded or was used
  dead         a control that changed nothing when used
  failed       something the page asked for from its own folder did not load

Each is right, wrong, or not measured. Nothing here knows what any page is
for; a page whose button should do nothing visible is a page this cannot
judge, and says so by not finding it right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CONTROLS_AT_MOST", "PageUse", "what_a_page_does"]

#: The most controls used in one look; past this a page is used in part.
CONTROLS_AT_MOST = 16

#: How long after using a control the page is given to answer, in ms.
ANSWER_MS = 350

_THE_CONTROLS = """
(() => {
  const seen = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const all = [...document.querySelectorAll('button, [role=button], input, textarea, select, a[href]')]
    .filter((el) => seen(el) && !el.disabled);
  return all.map((el, index) => {
    el.setAttribute('data-aura-control', String(index));
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const name = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder')
      || el.getAttribute('name') || el.id || tag).trim().slice(0, 40);
    const leaves = tag === 'a' && el.host && el.host !== location.host;
    return {index, tag, type, name, leaves};
  });
})()
"""

_THE_PAGE_NOW = """
(() => document.body ? document.body.innerText.length + ':' + document.body.innerText.slice(0, 4000)
  + '|' + document.body.innerHTML.length + '|' + [...document.querySelectorAll('input,textarea,select')]
  .map((el) => el.type === 'checkbox' || el.type === 'radio' ? String(el.checked) : el.value).join('\\u0001') : '')()
"""


@dataclass
class PageUse:
    """What using a page showed."""

    used: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


async def what_a_page_does(page: Any, *, before: list[str], failed: list[str]) -> PageUse:
    """Use every control on ``page`` once and note what changed; ``before``/``failed`` are being filled by the caller's listeners."""
    use = PageUse()
    try:
        controls = await page.evaluate(_THE_CONTROLS)
    except Exception:  # noqa: BLE001 - a page that cannot be read has nothing to use
        return use
    usable = [c for c in controls[:CONTROLS_AT_MOST] if not c.get("leaves") and c.get("type") not in ("file", "hidden", "password")]
    alive: list[dict[str, Any]] = []
    looked_dead: list[dict[str, Any]] = []
    for control in usable:
        changed = await _changes_something(page, control, use, before)
        if changed is None:
            continue
        use.used.append(_name(control))
        (alive if changed else looked_dead).append(control)
    # A control can do nothing in the state the page is in: "Reset" pressed
    # just after the count came back to nought. Each that looked dead is used
    # again after one that works has changed the page.
    for control in looked_dead:
        for other in alive:
            if await _changes_something(page, other, use, before):
                break
        if await _changes_something(page, control, use, before):
            continue
        use.dead.append(_name(control))
    use.errors.extend(before)
    use.failed.extend(failed)
    return use


def _name(control: dict[str, Any]) -> str:
    return f"{control['tag']} {control['name']!r}"


async def _changes_something(page: Any, control: dict[str, Any], use: PageUse, before: list[str]) -> bool | None:
    """Whether using the control changed the page (or threw); None when it could not be used."""
    errors_before = len(before)
    try:
        now = await page.evaluate(_THE_PAGE_NOW)
        address = page.url
        await _use(page, control)
        await page.wait_for_timeout(ANSWER_MS)
        changed = await page.evaluate(_THE_PAGE_NOW) != now or page.url != address
        if page.url != address:
            await page.go_back()
            await page.wait_for_timeout(ANSWER_MS)
    except Exception as why:  # noqa: BLE001 - a control that cannot be used is noted, not a stop
        use.errors.append(f"using {_name(control)}: {str(why)[:120]}")
        return None
    return changed or len(before) != errors_before


async def _use(page: Any, control: dict[str, Any]) -> None:
    """Use one control the way a person trying the page would."""
    target = page.locator(f"[data-aura-control='{control['index']}']")
    tag, kind = control["tag"], control["type"]
    if tag == "select":
        options = await target.evaluate("(el) => [...el.options].map((o) => o.value)")
        if len(options) > 1:
            await target.select_option(options[1] if await target.input_value() == options[0] else options[0])
        return
    if tag == "textarea" or (tag == "input" and kind not in ("button", "submit", "reset", "checkbox", "radio", "range", "color", "image")):
        await target.fill("12" if kind in ("number", "range") else "ab")
        await target.press("Enter")
        return
    await target.click(timeout=2000)
