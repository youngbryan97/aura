"""What a person would do to see that a feature works, written down before the code, and done to the code.

A feature of a program is known by what using it shows: select a word, press
Bold, and the word is bold. A check is that, as typed data: steps a person
takes (click a control by its name, type, press keys, select some text, fill a
field) and what they would then see (some text, a style on some text, an
element of some kind, a file saved, a dialog). Nothing in a check names code;
it can be written from a description of the program alone, before any of it
exists, and run against whatever is built.

A check that holds before the feature exists checks nothing, so a check is
only kept if it fails on the empty frame (``holds_on_nothing``).

Run in a real browser page, one fresh page per check, so one check's
leftovers cannot make another pass.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("Rebuilding.Checks")

__all__ = ["Check", "CheckRun", "Expectation", "Step", "run_checks"]

#: How long the page is given after a step to answer it, in ms.
SETTLE_MS = 120

#: How many checks run at once, each in its own page.
SIDE_BY_SIDE = 4

#: The longest one check may take, in seconds.
CHECK_S = 20.0


class Step(BaseModel):
    do: Literal["click", "type", "press", "select_text", "select_all", "fill", "choose", "wait", "click_text", "give_file"]
    target: str = Field(default="", max_length=200)
    value: str = Field(default="", max_length=600)


class Expectation(BaseModel):
    see: Literal[
        "text", "no_text", "style", "element", "count", "download", "dialog", "value", "printed",
        "text_after_reopen", "name", "moving", "watched",
    ]
    target: str = Field(default="", max_length=200)
    property: str = Field(default="", max_length=60)
    value: str = Field(default="", max_length=600)


class Check(BaseModel):
    feature: str = Field(max_length=120)
    steps: list[Step] = Field(default_factory=list, max_length=14)
    expect: list[Expectation] = Field(default_factory=list, min_length=1, max_length=5)

    def said(self) -> str:
        """The check in a person's words, for the record and for the model."""
        steps = "; ".join(f"{s.do} {s.target!r}" + (f" {s.value!r}" if s.value else "") for s in self.steps)
        seen = "; ".join(f"{e.see} {e.target!r}" + (f" {e.property}" if e.property else "") + (f" = {e.value!r}" if e.value else "") for e in self.expect)
        return f"{steps} -> expect {seen}"


@dataclass
class CheckRun:
    """What doing one check showed."""

    check: Check
    held: bool
    why: str = ""
    errors: list[str] = field(default_factory=list)


#: In the page: finding things the way a person names them, selecting text,
#: and reading styles the way they show.
_HELPERS = r"""
window.__checks = (() => {
  const norm = (s) => String(s || "").replace(/\s+/g, " ").trim().toLowerCase();
  const visible = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none"; };
  const names = (el) => [el.getAttribute("aria-label"), el.getAttribute("title"), el.getAttribute("placeholder"),
    el.getAttribute("name"), el.getAttribute("data-command"), el.innerText, el.value].map(norm).filter(Boolean)
    .map((n) => n.replace(/\s*\(([⌘^]|ctrl)[^)]*\)$/, ""));
  const CONTROLS = "button, [role=menuitem], [role=button], [role=tab], [role=option], a, summary, input[type=button], input[type=submit], [data-command], [onclick]";
  const FIELDS = "input, textarea, select, [contenteditable=true]";
  function find(target, selector) {
    const want = norm(target);
    if (!want) return null;
    const dialog = [...document.querySelectorAll("[role=dialog]")].filter(visible).pop();
    const roots = dialog ? [dialog, document] : [document];
    for (const root of roots) {
      const all = [...root.querySelectorAll(selector)];
      for (const how of [(n) => n === want, (n) => n.startsWith(want), (n) => n.includes(want)]) {
        const hit = all.filter((el) => names(el).some(how));
        const shown = hit.filter(visible);
        if (shown.length) return shown[0];
        if (hit.length) return hit[0];
      }
    }
    return null;
  }
  function fieldFor(target) {
    const want = norm(target);
    const dialog = [...document.querySelectorAll("[role=dialog]")].filter(visible).pop();
    for (const root of dialog ? [dialog, document] : [document]) {
      for (const label of root.querySelectorAll("label")) {
        if (!norm(label.innerText).includes(want)) continue;
        const el = label.control || label.querySelector(FIELDS);
        if (el) return el;
      }
      const el = find(target, FIELDS);
      if (el) return el;
    }
    return null;
  }
  function mark(el) { if (!el) return null; const id = "c" + Math.random().toString(36).slice(2); el.setAttribute("data-check", id); return id; }
  function opener(el) {
    if (!el || visible(el)) return null;
    for (let up = el.parentElement; up; up = up.parentElement) {
      const button = [...up.querySelectorAll(":scope > [aria-haspopup], :scope > summary, :scope > button")]
        .find((b) => b !== el && !b.contains(el) && visible(b));
      if (button) return mark(button);
    }
    return null;
  }
  function editable() {
    const work = document.querySelector("#app-work") || document.body;
    const active = document.activeElement;
    if (active && active !== document.body && (active.isContentEditable || /^(INPUT|TEXTAREA)$/.test(active.tagName))) return active;
    return work.querySelector("[contenteditable=true], [contenteditable=''], textarea, input:not([type=hidden])") || work;
  }
  function textNodes(root) {
    const out = []; const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walk.nextNode()) out.push(walk.currentNode);
    return out;
  }
  function selectText(target) {
    const work = document.querySelector("#app-work") || document.body;
    const nodes = textNodes(work);
    let all = ""; const starts = [];
    for (const n of nodes) { starts.push(all.length); all += n.nodeValue; }
    const at = all.indexOf(target);
    if (at < 0) return false;
    const locate = (offset) => { for (let i = nodes.length - 1; i >= 0; i--) if (starts[i] <= offset) return [nodes[i], offset - starts[i]]; return [nodes[0], 0]; };
    const [a, ao] = locate(at); const [b, bo] = locate(at + target.length);
    const host = a.parentElement && a.parentElement.closest("[contenteditable=true], [contenteditable='']");
    if (host) host.focus();
    const range = document.createRange(); range.setStart(a, ao); range.setEnd(b, bo);
    const s = getSelection(); s.removeAllRanges(); s.addRange(range);
    document.dispatchEvent(new Event("selectionchange"));
    return true;
  }
  function holding(target) {
    const work = document.querySelector("#app-work") || document.body;
    let best = null;
    for (const el of work.querySelectorAll("*")) if (el.textContent.includes(target)) best = el;
    if (best) return best;
    for (const n of textNodes(work)) if (n.nodeValue.includes(target)) return n.parentElement;
    return null;
  }
  const toRgb = (c) => { const el = document.createElement("span"); el.style.color = c; document.body.append(el);
    const v = getComputedStyle(el).color; el.remove(); return v; };
  const rgb = (c) => (String(c).match(/[\d.]+/g) || []).slice(0, 3).map(Number);
  const near = (a, b) => { const x = rgb(a), y = rgb(b); return x.length === 3 && y.length === 3 && x.every((v, i) => Math.abs(v - y[i]) <= 24); };
  function upward(el, fn) { for (let e = el; e && e !== document.body; e = e.parentElement) { const v = fn(e, getComputedStyle(e)); if (v) return v; } return null; }
  function style(target, property, value) {
    const el = holding(target);
    if (!el) return [false, `no text ${JSON.stringify(target)} to look at`];
    const prop = norm(property).replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
    const want = norm(value);
    const s = getComputedStyle(el);
    if (prop === "font-weight") { const w = Number(s.fontWeight) || (s.fontWeight === "bold" ? 700 : 400);
      const bold = w >= 600; return [want === "bold" || Number(want) >= 600 ? bold : !bold, `font-weight ${s.fontWeight}`]; }
    if (prop.startsWith("text-decoration")) { const line = upward(el, (e, st) => (st.textDecorationLine || "").includes(want) ? st.textDecorationLine : null);
      return [want === "none" ? !upward(el, (e, st) => st.textDecorationLine !== "none" ? 1 : null) : !!line, `text-decoration ${line || s.textDecorationLine}`]; }
    if (prop === "color") return [near(s.color, toRgb(value)), `color ${s.color}`];
    if (prop === "background-color" || prop === "background") { const bg = upward(el, (e, st) => rgb(st.backgroundColor).length === 3 && !/, 0\)$/.test(st.backgroundColor) && st.backgroundColor !== "transparent" ? st.backgroundColor : null);
      return [!!bg && near(bg, toRgb(value)), `background ${bg || "none"}`]; }
    if (prop === "font-size") { const px = parseFloat(s.fontSize); const n = parseFloat(value);
      const ok = /pt$/.test(want) ? Math.abs(px - n * 4 / 3) <= 1.5 : /px$/.test(want) ? Math.abs(px - n) <= 1.5 : Math.abs(px - n) <= 1.5 || Math.abs(px - n * 4 / 3) <= 1.5;
      return [ok, `font-size ${s.fontSize}`]; }
    if (prop === "font-family") return [norm(s.fontFamily).includes(want.split(",")[0].replace(/["']/g, "")), `font-family ${s.fontFamily}`];
    if (prop === "text-align") { const a = upward(el, (e, st) => st.display !== "inline" ? st.textAlign : null) || s.textAlign;
      const n = (x) => x.replace("start", "left").replace("end", "right").replace("-webkit-", "");
      return [n(norm(a)) === n(want), `text-align ${a}`]; }
    if (prop === "vertical-align") { const v = upward(el, (e, st) => st.verticalAlign !== "baseline" ? st.verticalAlign : null) || "baseline";
      return [norm(v).includes(want), `vertical-align ${v}`]; }
    const got = upward(el, (e, st) => st.getPropertyValue(prop) && norm(st.getPropertyValue(prop)).includes(want) ? st.getPropertyValue(prop) : null);
    return [!!got, `${prop} ${got || s.getPropertyValue(prop)}`];
  }
  function count(selector) { const work = document.querySelector("#app-work") || document;
    try { return work.querySelectorAll(selector).length; } catch (e) { return -1; } }
  function bodyText() { return norm([...document.body.querySelectorAll("#app-work, #app-side, #app-status, [role=dialog], #app-notice, #app-title")]
    .map((el) => el.innerText + " " + [...el.querySelectorAll("input, textarea")].map((i) => i.value).join(" ")).join(" ")); }
  return { find: (t) => mark(find(t, CONTROLS)), field: (t) => mark(fieldFor(t)), opener: (id) => opener(document.querySelector(`[data-check='${id}']`)),
    editable: () => mark(editable()), selectText, style, count, bodyText, norm,
    dialog: (t) => [...document.querySelectorAll("[role=dialog]")].filter(visible).some((d) => norm(d.getAttribute("aria-label") + " " + d.innerText).includes(norm(t))) };
})();
"""

_KEY_NAMES = {"mod": "ControlOrMeta", "ctrl": "ControlOrMeta", "control": "ControlOrMeta", "cmd": "ControlOrMeta",
              "command": "ControlOrMeta", "meta": "ControlOrMeta", "alt": "Alt", "option": "Alt", "shift": "Shift",
              "enter": "Enter", "return": "Enter", "esc": "Escape", "escape": "Escape", "tab": "Tab", "del": "Delete",
              "delete": "Delete", "backspace": "Backspace", "space": "Space", "up": "ArrowUp", "down": "ArrowDown",
              "left": "ArrowLeft", "right": "ArrowRight", "home": "Home", "end": "End"}


def _keys(said: str) -> str:
    """'Ctrl+B' / 'Cmd+Shift+z' as the browser names keys."""
    parts = [p.strip() for p in re.split(r"\s*\+\s*", str(said).strip()) if p.strip()]
    named = [_KEY_NAMES.get(p.lower(), p.upper() if len(p) == 1 and p.isalpha() else p) for p in parts]
    if len(named) > 1 and len(named[-1]) == 1:
        named[-1] = named[-1].lower()
    return "+".join(named)


class _Doing:
    """One check done on one page."""

    def __init__(self, page: Any, check: Check) -> None:
        self.page = page
        self.check = check
        self.downloads: list[tuple[str, str]] = []
        self.to_give: tuple[str, str] | None = None
        page.on("download", lambda d: asyncio.ensure_future(self._saved(d)))
        page.on("filechooser", lambda chooser: asyncio.ensure_future(self._give(chooser)))

    async def _give(self, chooser: Any) -> None:
        """The file a person picks when the program asks for one: the one the check said to give."""
        if self.to_give is None:
            return
        name, text = self.to_give
        kind = "text/html" if name.lower().endswith((".html", ".htm")) else "text/plain"
        await chooser.set_files({"name": name, "mimeType": kind, "buffer": text.encode("utf-8")})

    async def _saved(self, download: Any) -> None:
        try:
            path = await download.path()
            data = await asyncio.to_thread(Path(path).read_bytes) if path else b""
            text = data[:400_000].decode("utf-8", "ignore")
        except Exception as why:  # noqa: BLE001 - a download that cannot be read is still a download
            text = f"(unreadable: {why})"
        self.downloads.append((download.suggested_filename, text))

    async def _mark(self, how: str, target: str) -> Any:
        found = await self.page.evaluate(f"(t) => window.__checks.{how}(t)", target)
        if not found:
            return None
        if how == "find":
            opener = await self.page.evaluate("(id) => window.__checks.opener(id)", found)
            if opener:
                await self.page.click(f"[data-check='{opener}']", timeout=2000)
        return self.page.locator(f"[data-check='{found}']")

    async def step(self, step: Step) -> str:
        """Do one step; what went wrong, or ''."""
        page, target, value = self.page, step.target, step.value
        if step.do == "click":
            control = await self._mark("find", target)
            if control is None:
                return f"there is no control named {target!r}"
            await control.click(timeout=3000)
        elif step.do in ("type", "select_all"):
            if not await page.evaluate("document.activeElement && document.activeElement !== document.body && (document.activeElement.isContentEditable || /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName))"):
                place = await self._mark("editable", "")
                if place is not None:
                    await place.click(timeout=3000)
            if step.do == "select_all":
                await page.keyboard.press("ControlOrMeta+a")
            else:
                for n, line in enumerate((value or target).split("\n")):
                    if n:
                        await page.keyboard.press("Enter")
                    if line:
                        await page.keyboard.type(line, delay=4)
        elif step.do == "press":
            await page.keyboard.press(_keys(value or target))
        elif step.do in ("select_text", "click_text"):
            if not await page.evaluate("(t) => window.__checks.selectText(t)", target or value):
                return f"there is no text {target or value!r} to select"
            if step.do == "click_text":
                await page.evaluate("() => getSelection().collapseToEnd()")
        elif step.do in ("fill", "choose"):
            box = await self._mark("field", target)
            if box is None:
                return f"there is no field named {target!r}"
            tag = await box.evaluate("(el) => el.tagName + ':' + (el.type || '')")
            if tag.startswith("SELECT"):
                try:
                    await box.select_option(label=value, timeout=2000)
                except Exception:  # noqa: BLE001 - offered by value rather than by its shown name
                    await box.select_option(value=value, timeout=2000)
            elif tag.endswith(":checkbox") or tag.endswith(":radio"):
                await box.set_checked(value.lower() not in ("false", "no", "off", "0"), timeout=2000)
            else:
                await box.fill(value, timeout=2000)
                await box.dispatch_event("change")
        elif step.do == "give_file":
            self.to_give = (target or "document.txt", value)
        elif step.do == "wait":
            await page.wait_for_timeout(min(3000, int(float(value or 300))))
        await page.wait_for_timeout(SETTLE_MS)
        return ""

    async def sees(self, expectation: Expectation, reopen: Any) -> tuple[bool, str]:
        page, e = self.page, expectation
        if e.see in ("text", "no_text"):
            shown = await page.evaluate("window.__checks.bodyText()")
            want = " ".join((e.target or e.value).lower().split())
            there = want in shown
            return (there if e.see == "text" else not there), f"the page shows {shown[:200]!r}"
        if e.see == "style":
            held, said = await page.evaluate("([t, p, v]) => window.__checks.style(t, p, v)", [e.target, e.property, e.value])
            return bool(held), said
        if e.see in ("element", "count"):
            found = await page.evaluate("(s) => window.__checks.count(s)", e.target)
            if found < 0:
                return False, f"{e.target!r} names no kind of element"
            if e.see == "element":
                return found > 0, f"{found} {e.target!r} in the work area"
            wanted = int(re.sub(r"\D", "", e.value) or 0)
            return (found >= wanted if e.value.strip().startswith(">") else found == wanted), f"{found} {e.target!r} in the work area"
        if e.see == "download":
            await page.wait_for_timeout(400)
            for name, text in self.downloads:
                if (not e.value or name.lower().endswith(e.value.lower().lstrip("*")) or e.value.lower() in name.lower()) and (
                    not e.target or e.target.lower() in text.lower()
                ):
                    return True, f"saved {name}"
            return False, f"saved {[n for n, _ in self.downloads] or 'nothing'}"
        if e.see == "dialog":
            return bool(await page.evaluate("(t) => window.__checks.dialog(t)", e.target or e.value)), "dialogs looked at"
        if e.see == "value":
            box = await self._mark("field", e.target)
            if box is None:
                return False, f"there is no field named {e.target!r}"
            got = await box.evaluate("(el) => el.type === 'checkbox' ? String(el.checked) : el.value")
            return str(got).strip().lower() == e.value.strip().lower(), f"{e.target} is {got!r}"
        if e.see == "printed":
            return bool(await page.evaluate("window.__appPrinted")), "print was asked for"
        if e.see == "name":
            got = await page.evaluate("app.name")
            return str(got).strip().lower() == e.value.strip().lower(), f"the document is named {got!r}"
        if e.see == "moving":
            first = await _a_picture_of(page, e.target)
            await page.wait_for_timeout(600)
            second = await _a_picture_of(page, e.target)
            return bool(first) and first != second, "the picture changed" if first != second else "the picture stood still"
        if e.see == "watched":
            # A game's behaviour, watched the way she watches any game
            # (core/self_modification/watching_a_program_run.py): the keys move
            # what is hers as they are named ("controls"), things turn back off
            # her ("went through"), a miss counts for the right side
            # ("credited"), nothing escapes ("escaped"), the other side acts ("idle").
            from core.self_modification.watching_a_program_run import what_it_does

            words = str(await page.evaluate("document.body.innerText") or "")
            keys = [k for k in ("up", "down", "left", "right", "space") if k in words.lower()] or ["up", "down", "left", "right"]
            seen = await what_it_does(page, page.url, words=words, keys=keys, seconds=8.0)
            wanted = e.target.strip().lower() or "controls"
            return wanted in seen.right, f"watched it play: right {sorted(seen.right)}, wrong {seen.findings}"
        if e.see == "text_after_reopen":
            await reopen()
            shown = await page.evaluate("window.__checks.bodyText()")
            want = " ".join((e.target or e.value).lower().split())
            return want in shown, f"after reopening the page shows {shown[:200]!r}"
        return False, f"cannot look for {e.see}"


async def _a_picture_of(page: Any, target: str) -> str:
    """What ``target`` (else the largest canvas, else the work area) shows now, as a picture's bytes in text."""
    shown = await page.evaluate(
        """(t) => { let el = null; try { el = t ? document.querySelector(t) : null; } catch (e) { el = null; }
          if (!el) { let area = 0; for (const c of document.querySelectorAll('canvas')) { const r = c.getBoundingClientRect();
            if (r.width * r.height > area) { area = r.width * r.height; el = c; } } }
          if (el && el.tagName === 'CANVAS') { try { return el.toDataURL('image/png'); } catch (e) { return ''; } }
          return ''; }""",
        target,
    )
    if shown:
        return str(shown)
    place = page.locator(target or "#app-work").first
    return (await place.screenshot(timeout=3000)).hex() if await place.count() else ""


def _time_for(check: Check) -> float:
    """How long a check may take: watching a game play is most of a minute."""
    return CHECK_S + 45.0 * sum(1 for e in check.expect if e.see == "watched")


async def _one(context: Any, url: str, check: Check) -> CheckRun:
    page = await context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda why: errors.append(str(why)[:200]))
    try:
        await page.goto(url)
        await page.evaluate(_HELPERS)
        doing = _Doing(page, check)

        async def reopen() -> None:
            await page.reload()
            await page.evaluate(_HELPERS)

        for step in check.steps:
            wrong = await doing.step(step)
            if wrong:
                return CheckRun(check, False, f"at {step.do} {step.target or step.value!r}: {wrong}", errors)
        for expectation in check.expect:
            held, said = await doing.sees(expectation, reopen)
            if not held:
                return CheckRun(check, False, f"expected {expectation.see} {expectation.target or expectation.value!r}: {said}", errors)
        return CheckRun(check, True, "", errors)
    except Exception as why:  # noqa: BLE001 - a check the page breaks is a check that did not hold
        return CheckRun(check, False, f"{type(why).__name__}: {str(why)[:200]}", errors)
    finally:
        await page.close()


async def run_checks(path: str | Path, checks: list[Check], *, browser: Any = None) -> list[CheckRun]:
    """Do each check on a fresh page of the program at ``path``, several side by side."""
    if not checks:
        return []
    url = await asyncio.to_thread(lambda: Path(path).resolve().as_uri())
    owned = None
    if browser is None:
        from playwright.async_api import async_playwright

        owned = await async_playwright().start()
        browser = await owned.chromium.launch(headless=True)
    try:
        context = await browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 860})
        gate = asyncio.Semaphore(SIDE_BY_SIDE)

        async def bounded(check: Check) -> CheckRun:
            async with gate:
                try:
                    return await asyncio.wait_for(_one(context, url, check), timeout=_time_for(check))
                except TimeoutError:
                    return CheckRun(check, False, f"took longer than {_time_for(check):.0f}s")

        try:
            return list(await asyncio.gather(*(bounded(c) for c in checks)))
        finally:
            await context.close()
    finally:
        if owned is not None:
            await browser.close()
            await owned.stop()
