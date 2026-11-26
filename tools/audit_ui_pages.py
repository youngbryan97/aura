#!/usr/bin/env python3
"""Every page of the interface, measured the ways a person meets it.

`interface/static/a11y_audit.js` loads with index.html and nothing else, so
fifteen of sixteen pages had never been measured. Run on 28 September they
held seven pages with no first-level heading, text at 1.49:1 on the splash,
five unnamed switches, nav links 19 pixels tall, a 545-pixel page on a
375-pixel phone, and a telemetry page whose stylesheet had never existed.

This serves `interface/` on an ephemeral loopback port (no runtime, no port
the live instance uses) and opens every page it finds in a headless Chromium
at 1440x900, 1024x768 and 320x568, in the system's dark and light
appearance, and in every theme and accent the page's own pickers offer. At
each it runs the audit and a horizontal-overflow check. Once per page it
presses Tab through every control, which must look different when focused,
and asks for reduced motion, under which nothing may keep moving. Any
finding fails the run. Pages are discovered, not listed, so a new page is
measured the day it is added. A page whose audit checked nothing at all is a
failure too: that is what an audit that measured nothing looks like.

Not measured: a WebGL canvas that does not keep its buffer (it cannot be
read back), and what a screen reader announces. Text enlarged to 200% is
covered by the 320-pixel width, which is a 1280-pixel window at 400% zoom.

    python tools/audit_ui_pages.py            # exit 1 on any finding
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import logging
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "interface"
STATIC = INTERFACE / "static"
#: 320 CSS pixels is WCAG 1.4.10's reflow width, below every phone in use.
VIEWPORTS = ((1440, 900), (1024, 768), (320, 568))
#: design_tokens.css defines a light theme under prefers-color-scheme, and the
#: desktop app follows the system appearance, so both are what people see.
SCHEMES = ("dark", "light")
#: How long a page is given to build what it builds from its own script
#: before it is measured. Pages that fetch from a runtime fail those fetches
#: here and render their empty state, which is what a first visit sees.
SETTLE_MS = 900


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - the base class's name
        # Each request to debug rather than stderr: what the audit prints is
        # its measurements, and forty page loads would bury them.
        logging.getLogger("aura.audit_ui_pages").debug(format, *args)


def _serve(root: Path = INTERFACE) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_Quiet, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, name="ui-audit-server", daemon=True).start()
    return server, server.server_address[1]


_RUN = """async () => {
  // An entrance animation still running at measurement time is read as the
  // page: a rise with a transform measured a 24-pixel link as 23.99. Finite
  // animations are waited out, bounded; infinite ones (a pulse, a shimmer)
  // are part of how the page looks.
  const finite = document.getAnimations().filter(
    (a) => a.effect && a.effect.getComputedTiming().endTime !== Infinity);
  await Promise.race([
    Promise.all(finite.map((a) => a.finished.catch(() => null))),
    new Promise((resolve) => setTimeout(resolve, 4000)),
  ]);
  if (!window.auraAccessibilityAudit) {
    (0, eval)(await (await fetch('/static/a11y_audit.js')).text());
  }
  const r = window.auraAccessibilityAudit();
  const total = Object.values(r.checked || {}).reduce((a, b) => a + b, 0);
  return {checked: r.checked, total, failures: r.failures,
          findings: r.findings.map(f => [f.kind, f.where, f.ratio || f.size || ''].join(' / ')),
          width: document.documentElement.scrollWidth};
}"""


#: The shell's theme and accent are settings, not the system appearance, so a
#: page with those pickers is measured in every pairing they offer, chosen
#: through the pickers themselves so the page's own code applies them.
PICKERS = ("setting-theme", "setting-accent")
_OPTIONS = """(ids) => ids.map((id) => { const s = document.getElementById(id);
  return s ? [...s.options].map((o) => o.value) : []; })"""
_PICK = """([id, value]) => { const s = document.getElementById(id); s.value = value;
  s.dispatchEvent(new Event('change', {bubbles: true})); }"""


def _settings(page) -> list[tuple[tuple[str, str], ...]]:
    """Every pairing of picker values except the first, which is the default."""
    import itertools

    offered = [
        [(picker, value) for value in values]
        for picker, values in zip(PICKERS, page.evaluate(_OPTIONS, list(PICKERS)), strict=True)
        if values
    ]
    if not offered:
        return []
    return list(itertools.product(*offered))[1:]


_SETTLED = """async () => {
  const finite = document.getAnimations().filter(
    (a) => a.effect && a.effect.getComputedTiming().endTime !== Infinity);
  await Promise.race([
    Promise.all(finite.map((a) => a.finished.catch(() => null))),
    new Promise((resolve) => setTimeout(resolve, 1500)),
  ]);
}"""

#: How an element looks, in the properties a focus indicator is drawn with,
#: and how its two nearest ancestors look: a text field commonly shows focus
#: on the bar around it through :focus-within, and that is a real indicator.
_LOOK = """(el) => { const one = (e) => { if (!e) return ''; const s = getComputedStyle(e);
    return [s.outlineStyle, s.outlineWidth, s.outlineColor, s.boxShadow, s.borderColor,
            s.backgroundColor, s.color, s.textDecorationLine].join('|'); };
  return [one(el), one(el.parentElement), one(el.parentElement && el.parentElement.parentElement)].join('/'); }"""

_MARK = """() => { const look = """ + _LOOK + """;
  // A page that focuses its composer on load would otherwise be measured
  // "unfocused" while focused, and read as having no indicator at all.
  if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
  const all = [...document.querySelectorAll(
    'a[href],button,input,select,textarea,summary,[tabindex],[contenteditable=true]')];
  window.__auditLook = all.map((el, i) => { el.dataset.auditI = String(i); return look(el); });
  return all.length; }"""

_FOCUSED = """() => { const look = """ + _LOOK + """;
  const el = document.activeElement;
  if (!el || el === document.body || el.dataset.auditI === undefined) return null;
  const box = el.getBoundingClientRect();
  const shown = box.width > 1 && box.height > 1 && getComputedStyle(el).visibility !== 'hidden';
  return {i: Number(el.dataset.auditI), same: look(el) === window.__auditLook[Number(el.dataset.auditI)],
          shown, where: el.id || (el.className || '').toString().split(' ')[0] || el.tagName.toLowerCase(),
          name: (el.getAttribute('aria-label') || el.textContent || '').trim().slice(0, 24)}; }"""

#: Keyframes that move something: what reduced motion exists to stop. A pulse
#: in opacity or colour is not motion in that sense and is allowed to stay.
_MOVING = """() => {
  const moves = /^(transform|translate|rotate|scale|left|top|right|bottom|backgroundPosition[XY]?|offsetPath|offsetDistance)$/;
  return document.getAnimations()
    .filter((a) => a.playState === 'running' && a.effect
      && a.effect.getComputedTiming().iterations === Infinity
      && a.effect.getKeyframes().some((k) => Object.keys(k).some((p) => moves.test(p))))
    .map((a) => { const t = a.effect.target;
      const where = t ? (t.id || (t.className || '').toString().split(' ')[0] || t.tagName.toLowerCase()) : '?';
      return `${a.animationName || 'script animation'} on ${where}`; });
}"""


#: A fingerprint of every canvas on screen. getAnimations cannot see what a
#: script draws, so a canvas is read twice and compared; a canvas that cannot
#: be read (a WebGL one without a preserved buffer) reads the same both times
#: and is not judged.
_CANVASES = """() => [...document.querySelectorAll('canvas')].filter((c) => {
    const b = c.getBoundingClientRect();
    return b.width > 0 && b.height > 0 && getComputedStyle(c).visibility !== 'hidden'; })
  .map((c) => { let data = ''; try { data = c.toDataURL(); } catch (e) { data = ''; }
    let h = 2166136261; for (let i = 0; i < data.length; i++) { h = Math.imul(h ^ data.charCodeAt(i), 16777619); }
    return [c.id || (c.className || '').toString().split(' ')[0] || 'canvas', h >>> 0]; })"""

#: Long enough to span forty frames at sixty a second.
_STILL_FOR_MS = 700


def _keyboard(page, where: str) -> list[str]:
    """Tab through the page; every control that takes focus must look focused.

    WCAG 2.4.7. The in-page audit cannot ask this, because only focus that
    came from a keyboard matches :focus-visible, so this presses Tab.
    """
    count = page.evaluate(_MARK)
    found: list[str] = []
    seen: set[int] = set()
    for _ in range(min(count, 80) + 2):
        page.keyboard.press("Tab")
        page.evaluate(_SETTLED)
        now = page.evaluate(_FOCUSED)
        if now is None:
            continue
        if now["i"] in seen:
            break
        seen.add(now["i"])
        if now["shown"] and now["same"]:
            found.append(f"{where}: no visible focus / {now['where']} / {now['name']}")
    return found


def _motion(page, where: str) -> list[str]:
    found = [f"{where}: moves under reduced motion / {one}" for one in page.evaluate(_MOVING)]
    before = page.evaluate(_CANVASES)
    page.wait_for_timeout(_STILL_FOR_MS)
    after = page.evaluate(_CANVASES)
    for (name, one), (_, other) in zip(before, after, strict=False):
        if one != other:
            found.append(f"{where}: moves under reduced motion / canvas {name} redrawn")
    return found


def _measure(page, where: str, width: int) -> list[str]:
    result = page.evaluate(_RUN)
    found: list[str] = []
    if result["total"] == 0:
        found.append(f"{where}: the audit checked nothing")
    found += [f"{where}: {finding}" for finding in result["findings"]]
    if result["width"] > width + 1:
        found.append(f"{where}: page is {result['width']}px wide")
    return found


def audit(pages: list[str] | None = None, root: Path = INTERFACE) -> list[str]:
    """Findings for every page under ``root/static``; ``root`` is a test's seam."""
    from playwright.sync_api import sync_playwright

    names = pages or sorted(p.stem for p in (root / "static").glob("*.html"))
    problems: list[str] = []
    server, port = _serve(root)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                for scheme, (width, height) in ((s, v) for s in SCHEMES for v in VIEWPORTS):
                    context = browser.new_context(
                        viewport={"width": width, "height": height}, color_scheme=scheme
                    )
                    page = context.new_page()
                    for name in names:
                        page.goto(f"http://127.0.0.1:{port}/static/{name}.html", wait_until="load")
                        page.wait_for_timeout(SETTLE_MS)
                        where = f"{name} @ {width}x{height} {scheme}"
                        problems += _measure(page, where, width)
                        pairings = _settings(page)
                        for pairing in pairings:
                            for choice in pairing:
                                page.evaluate(_PICK, list(choice))
                            label = " ".join(f"{k.split('-')[-1]}={v}" for k, v in pairing)
                            problems += _measure(page, f"{where} {label}", width)
                        if pairings:
                            page.evaluate("() => localStorage.clear()")
                    context.close()
                # Keyboard and reduced motion are asked once per page, at the
                # desktop width: neither depends on the viewport.
                context = browser.new_context(
                    viewport={"width": 1440, "height": 900}, reduced_motion="reduce"
                )
                page = context.new_page()
                for name in names:
                    page.goto(f"http://127.0.0.1:{port}/static/{name}.html", wait_until="load")
                    page.wait_for_timeout(SETTLE_MS)
                    page.evaluate(_SETTLED)
                    where = f"{name} @ 1440x900 reduced-motion"
                    problems += _motion(page, where)
                    problems += _keyboard(page, where)
                context.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="*", help="page stems; default: every page")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    problems = audit(args.pages or None)
    if args.json:
        print(json.dumps({"problems": problems}, indent=2))
    elif problems:
        print(f"{len(problems)} interface finding(s):")
        for one in problems:
            print("  ", one)
    else:
        print("every page passes the audit at 1440x900, 1024x768 and 320x568, dark and light, in every theme and accent it offers, by keyboard, and with reduced motion")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
