"""How finished a program looks and handles, measured rather than judged.

A program can do every feature it was asked for and still be unfinished: two
buttons drawn on top of each other, a label cut off, grey text on grey, a
window that scrolls sideways when it is made narrower, a button with no name a
screen reader could say, a control that does nothing when used. Each of those
is a fact about the page that can be measured, at the size a desktop window
opens at and at a narrow one, and each is a thing a person notices.

``what_is_unfinished`` measures them. ``finishing`` hands what was measured to
her model as the thing to mend, in the program's own style, and keeps a mend
only when there is less unfinished afterwards and every check that held still
holds.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.rebuilding.checks_a_person_makes import Check
from core.rebuilding.the_program_as_built import Part, ProgramAsBuilt

logger = logging.getLogger("Rebuilding.Finish")

__all__ = ["Unfinished", "finishing", "what_is_unfinished"]

#: The window sizes it is looked at in: a desktop window, and one made narrow.
SIZES = ((1280, 860), (820, 860))

#: Text contrast below which text is hard to read (WCAG AA, body text).
LEAST_CONTRAST = 4.5

#: How many mends are tried before it is left as it is.
MENDS = 2

_MEASURED = r"""
(() => {
  const shown = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0.05; };
  const name = (el) => (el.getAttribute('aria-label') || el.getAttribute('title') || el.innerText || el.value || el.getAttribute('placeholder') || '').trim();
  const say = (el) => (name(el) || el.tagName.toLowerCase()).slice(0, 40);
  const controls = [...document.querySelectorAll('button, select, input:not([type=hidden]), textarea, [role=button], [role=menuitem], a[href]')].filter(shown);
  const found = [];
  for (let i = 0; i < controls.length; i++) {
    const a = controls[i].getBoundingClientRect();
    for (let j = i + 1; j < controls.length; j++) {
      if (controls[i].contains(controls[j]) || controls[j].contains(controls[i])) continue;
      const b = controls[j].getBoundingClientRect();
      const w = Math.min(a.right, b.right) - Math.max(a.left, b.left), h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (w > 4 && h > 4) found.push(`"${say(controls[i])}" and "${say(controls[j])}" are drawn over each other`);
    }
    const el = controls[i];
    if (!name(el) && !el.querySelector('img[alt], svg title')) found.push(`a ${el.tagName.toLowerCase()} has no name a person or a screen reader could say`);
    if (/^(BUTTON|A)$/.test(el.tagName) && el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflow !== 'visible') found.push(`the text of "${say(el)}" is cut off`);
  }
  if (document.documentElement.scrollWidth > innerWidth + 2) found.push(`the window scrolls sideways at ${innerWidth} pixels wide`);
  const lum = (c) => { const v = (c.match(/[\d.]+/g) || [0, 0, 0]).slice(0, 3).map((x) => { x = Number(x) / 255; return x <= 0.03928 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); });
    return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]; };
  const opaque = (c) => { const m = c.match(/[\d.]+/g) || []; return m.length >= 3 && (m.length < 4 || Number(m[3]) > 0.5); };
  const behind = (el) => { for (let e = el; e; e = e.parentElement) { const bg = getComputedStyle(e).backgroundColor; if (opaque(bg)) return bg; } return 'rgb(255,255,255)'; };
  const seen = new Set();
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walk.nextNode()) {
    const el = walk.currentNode.parentElement;
    if (!el || seen.has(el) || !walk.currentNode.nodeValue.trim() || !shown(el)) continue;
    seen.add(el);
    const s = getComputedStyle(el);
    const a = lum(s.color), b = lum(behind(el));
    const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    const large = parseFloat(s.fontSize) >= 24 || (parseFloat(s.fontSize) >= 18.5 && Number(s.fontWeight) >= 700);
    if (ratio < (large ? 3 : %LEAST%)) found.push(`"${walk.currentNode.nodeValue.trim().slice(0, 30)}" is hard to read (contrast ${ratio.toFixed(1)} to 1)`);
  }
  return [...new Set(found)].slice(0, 40);
})()
""".replace("%LEAST%", str(LEAST_CONTRAST))

#: Using each control once, after some text is typed and selected, and seeing
#: whether the page changed, opened something, or saved something.
_DOES_SOMETHING = r"""
async () => {
  const shown = (el) => { const r = el.getBoundingClientRect(); return r.width > 1 && r.height > 1; };
  // What the program shows, not the frame's own "Edited" mark, which any command sets.
  const part = (sel) => [...document.querySelectorAll(sel)].map((el) => el.innerHTML + '|' + getComputedStyle(el).cssText.length + getComputedStyle(el).backgroundColor + getComputedStyle(el).color).join('#');
  const state = () => part('#app-work, #app-side, #app-status, #app-notice.shown, [role=dialog]') + ':' + document.documentElement.className + document.body.className
    + ':' + getComputedStyle(document.body).backgroundColor + ':' + (getSelection() + '') + ':' + window.__appPrinted + ':' + app.name;
  const dead = [];
  const ids = [...new Set([...document.querySelectorAll('[data-command]')].map((el) => el.getAttribute('data-command')))];
  for (const id of ids) {
    const c = app.commands.get(id);
    if (!c || !c.run) continue;
    if (app.side) app.side(null);
    const place = document.querySelector('#app-work [contenteditable=true], #app-work textarea, #app-work input');
    if (place) { place.focus(); if (place.isContentEditable && !place.innerText.trim()) document.execCommand('insertText', false, 'Some words to try it on'); document.execCommand('selectAll'); }
    // A command greyed out as not applying here, as the page now stands (a table's rows with no table,
    // Select all with all selected), is not one that does nothing.
    if (app.applies && !app.applies(c)) continue;
    const before = state();
    let downloaded = false;
    const real = URL.createObjectURL; URL.createObjectURL = (b) => { downloaded = true; return real(b); };
    try { await Promise.race([app.run(id), new Promise((r) => setTimeout(r, 600))]); } catch (e) { /* an error is counted elsewhere */ }
    URL.createObjectURL = real;
    await new Promise((r) => setTimeout(r, 150));
    // Asking for a file to open is doing something, though the page shows nothing until one is given.
    const askedForAFile = !!document.querySelector('input[type=file]');
    const after = state();
    document.querySelectorAll('.app-dialog-back').forEach((d) => d.remove());
    document.querySelectorAll('input[type=file]').forEach((d) => d.remove());
    if (before === after && !downloaded && !askedForAFile) dead.push(c.label);
  }
  return dead;
}
"""


@dataclass
class Unfinished:
    """What was measured as unfinished, and where."""

    found: list[str] = field(default_factory=list)
    dead: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def count(self) -> int:
        return len(self.found) + len(self.dead) + len(self.errors)

    def said(self) -> str:
        lines = [*self.found, *(f'using "{d}" changes nothing' for d in self.dead), *(f"the page says: {e}" for e in self.errors)]
        return "; ".join(lines[:30]) or "nothing"


async def what_is_unfinished(path: Path, *, browser: Any = None) -> Unfinished:
    """Measure the program at ``path`` at each window size, and use each of its commands once."""
    owned = None
    if browser is None:
        from playwright.async_api import async_playwright

        owned = await async_playwright().start()
        browser = await owned.chromium.launch(headless=True)
    seen = Unfinished()
    url = await asyncio.to_thread(lambda: Path(path).resolve().as_uri())
    try:
        for wide, tall in SIZES:
            context = await browser.new_context(viewport={"width": wide, "height": tall}, accept_downloads=True)
            page = await context.new_page()
            page.on("pageerror", lambda why: seen.errors.append(str(why)[:200]))
            await page.goto(url)
            await page.wait_for_timeout(300)
            seen.found.extend(f for f in await page.evaluate(_MEASURED) if f not in seen.found)
            if wide == SIZES[0][0]:
                seen.dead = list(await page.evaluate(_DOES_SOMETHING))
            await context.close()
    finally:
        if owned is not None:
            await browser.close()
            await owned.stop()
    seen.errors = sorted(set(seen.errors))[:10]
    return seen


async def finishing(
    program: ProgramAsBuilt, holding: list[Check], ask: Any, tried: Any, scratch: Path, *, browser: Any = None,
) -> tuple[ProgramAsBuilt, Unfinished]:
    """Mend what is measured unfinished, keeping a mend only when less is unfinished and every check still holds."""
    from core.rebuilding.writing_it_part_by_part import FRAME_API, _does_not_parse, _write

    await asyncio.to_thread(program.write, scratch)
    now = await what_is_unfinished(scratch, browser=browser)
    for _ in range(MENDS):
        if not now.count():
            break
        shown = "\n".join(f"--- part {p.name!r}\n{p.code[:2500]}" for p in program.parts)[:14000]
        written = await _write(ask, (
            f"{program.title} works, and is not finished. Measured on it: {now.said()}.\n"
            "Write one part named \"finish\" that mends these: CSS in `style` for how it looks (keep its look and colours, "
            "fix overlaps, cut-off text, contrast and narrow windows), and code that gives controls their missing names "
            "and makes each control that changes nothing do what its name says. Change nothing that works.\n\n"
            f"Its parts:\n{shown}\n\nIts style:\n{program.style[:4000]}\n\n{FRAME_API}"
        ))
        if written is None or await _does_not_parse(written.code):
            continue
        candidate = program.with_part(Part("finish", written.code, ["finish"]))
        candidate.style = "\n".join(s for s in (program.style, written.style) if s)
        _mine, broke, _errors = await tried(candidate, [])
        if broke:
            logger.info("a finishing mend broke %d check(s); not kept", len(broke))
            continue
        await asyncio.to_thread(candidate.write, scratch)
        after = await what_is_unfinished(scratch, browser=browser)
        if after.count() < now.count():
            program, now = candidate, after
            logger.info("finishing: %d left unfinished", now.count())
    await asyncio.to_thread(scratch.unlink, missing_ok=True)
    return program, now

