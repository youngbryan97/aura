"""Whether a page serves what a task needs: told from what its developer tools and code say, and settled by seeing it do it.

A page can arrive and still not do what it is for. A game's page whose game
file is gone draws an empty box; a page that embeds a Flash movie with no
player for it shows nothing at all, or a sentence saying Flash is needed; a
video that cannot be had says so in its player. A person opening such a page
knows at a glance, and a developer knows why from the tools in the browser:
the document's status, the requests that failed and what they were for, the
errors in the console, what the page's own code asks the browser for.

Here the same evidence is gathered, from the moment the page is opened:

- the developer tools: the document's status, each failed request and what
  kind of thing it was for, each console error and uncaught exception
  (``watching``, attached before the page is opened);
- what the page's code asks for: plugin content (Flash, Silverlight, an
  applet) with no player on the page to run it;
- what the page says of itself, in its text and inside its players' own
  shadow roots ("not supported", "could not be loaded", "no longer
  available");
- and what settles it: the thing the task needs, seen. A thing to run is
  looked at twice, and serves when it draws something and is not an error;
  something to read serves when its words are there.

The verdict is that the page serves it (seen), that it does not (with why), or
that it cannot yet be told; a reason is never taken for a sight. Nothing here
knows which site or which kind of thing it is looking at.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Skills.WhetherAPageServes")

__all__ = ["DevTools", "Serves", "seen_running", "watching", "whether_it_serves", "what_the_task_needs"]

#: What a task asks a page to do with the thing on it.
_RUN = re.compile(r"\b(play|run|launch|start|watch|listen|hear|view)\w*\b", re.I)
_READ = re.compile(r"\b(read|look up|find out|learn|tell me|summari[sz]e)\w*\b", re.I)

#: A page saying, in its own words, that it does not work.
_FAILS = re.compile(
    r"\b(?:not supported|no longer (?:available|supported|works?)|(?:is|isn't|is not) available (?:in|on|for) your|unavailable|"
    r"failed to (?:load|start|play)|could(?: not|n't) (?:be )?(?:load|play|start)\w*|error (?:loading|occurred|playing)|"
    r"something went wrong|encountered an? (?:major |serious )?(?:issue|problem|error)|"
    r"requires? (?:the )?(?:adobe )?flash|flash player (?:is )?(?:required|needed|not installed)|install (?:the )?(?:adobe )?flash|"
    r"enable javascript|page not found|404 not found|access denied|has been removed|this content is blocked|content not found)\b",
    re.I,
)

#: Kinds of request a page's own content is made of, and what each is said as when it fails.
_CONTENT = {"media": "a recording it plays", "object": "the program it embeds", "script": "a script it runs", "fetch": "data it loads",
            "xhr": "data it loads", "document": "the page itself", "other": "a file it loads"}

#: Files that are the thing itself, whatever kind of request fetched them.
_A_THING = re.compile(r"\.(?:swf|wasm|unity3d|data|dcr|jar|xap|mp4|webm|ogg|ogv|mp3|m4a|wav|m3u8|mpd)(?:[?#]|$)", re.I)

#: What a page's code embeds, what it is, and what runs it: the page reads its own markup for these.
_EMBEDS = r"""
() => {
  const all = [];
  const look = (root) => { for (const el of root.querySelectorAll("*")) { all.push(el); if (el.shadowRoot) look(el.shadowRoot); } };
  look(document);
  const plugin = all.filter((el) => /^(OBJECT|EMBED|APPLET)$/.test(el.tagName)).map((el) => ({
    tag: el.tagName.toLowerCase(), type: (el.getAttribute("type") || "").toLowerCase(),
    src: el.getAttribute("data") || el.getAttribute("src") || el.getAttribute("code") || "" }))
    .filter((p) => p.tag === "applet" || /shockwave|flash|silverlight|java|x-director/.test(p.type) || /\.(swf|xap|dcr|jar|class)\b/i.test(p.src));
  const players = all.filter((el) => /^RUFFLE-|^CHEERPJ|^DOSBOX|^EMULATOR/.test(el.tagName)).length + (window.RufflePlayer ? 1 : 0)
    + (document.querySelector("#canvas, #emulator, .emulator, [id*=ruffle], [class*=ruffle]") ? 1 : 0);
  const big = (el) => { const r = el.getBoundingClientRect(); return r.width >= 200 && r.height >= 120 && r.bottom > 0 && r.right > 0; };
  // The words a thing shows, not its stylesheet: a player's shadow root begins with its styles, and its error panel
  // came after the first four hundred letters of them (LIVE 2026-10-08, "Something went wrong" unread).
  const shown = (root) => Array.from(root.querySelectorAll("*")).filter((n) => !/^(STYLE|SCRIPT|TEMPLATE)$/.test(n.tagName)
    && !n.children.length && n.getClientRects().length).map((n) => (n.textContent || "").trim()).filter(Boolean).join(" ");
  // What runs: a canvas, a player, an embed, a frame, or a component of the page's own (a custom element) it draws in.
  const runs = all.filter((el) => /^(CANVAS|VIDEO|AUDIO|EMBED|OBJECT|IFRAME)$/.test(el.tagName) || el.tagName.includes("-"))
    .filter((el) => el.tagName === "AUDIO" || big(el))
    .map((el) => { const r = el.getBoundingClientRect(); return { tag: el.tagName.toLowerCase(), x: r.x, y: r.y, w: r.width, h: r.height,
      media: el.tagName === "VIDEO" || el.tagName === "AUDIO" ? { error: el.error ? el.error.code : 0, ready: el.readyState, time: el.currentTime } : null,
      said: shown(el.shadowRoot || el).slice(0, 400) }; })
    .sort((a, b) => b.w * b.h - a.w * a.h);
  const text = document.body ? document.body.innerText : "";
  return { plugin, players, runs, words: (text.match(/\S+/g) || []).length, text: text.slice(0, 3000) };
}
"""


@dataclass
class DevTools:
    """What the browser's developer tools record of a page from when it was opened: its status, failed requests, errors."""

    status: int | None = None
    failed: list[tuple[str, str, str]] = field(default_factory=list)  # (address, kind of request, why)
    errors: list[str] = field(default_factory=list)

    def opened(self) -> None:
        self.status, self.failed, self.errors = None, [], []


def watching(page: Any) -> DevTools:
    """The developer tools' record of ``page``, kept from now on and begun again at each page it opens; attached once."""
    held = getattr(page, "_her_dev_tools", None)
    if isinstance(held, DevTools):
        return held
    tools = DevTools()

    def asked(request: Any) -> None:
        # A page begins again when the browser asks for a new document in the main frame, before any of it answers.
        try:
            if request.is_navigation_request() and request.frame == page.main_frame:
                tools.opened()
        except Exception:  # noqa: BLE001
            return

    def answered(response: Any) -> None:
        try:
            if response.request.resource_type == "document" and response.frame == page.main_frame:
                tools.status = int(response.status)
            elif int(response.status) >= 400:
                tools.failed.append((response.url, response.request.resource_type, f"answered {response.status}"))
        except Exception:  # noqa: BLE001 - a response that cannot be read says nothing
            return

    def refused(request: Any) -> None:
        try:
            why = str((request.failure or "") if not callable(getattr(request, "failure", None)) else request.failure() or "")
            if "ERR_ABORTED" in why or "BLOCKED_BY_CLIENT" in why:
                return  # the page or the browser chose not to fetch it, which is not its failing
            tools.failed.append((request.url, request.resource_type, why or "failed"))
        except Exception:  # noqa: BLE001
            return

    def said(message: Any) -> None:
        try:
            if message.type == "error":
                tools.errors.append(str(message.text)[:300])
        except Exception:  # noqa: BLE001
            return

    try:
        page.on("request", asked)
        page.on("response", answered)
        page.on("requestfailed", refused)
        page.on("console", said)
        page.on("pageerror", lambda error: tools.errors.append(f"uncaught: {str(error)[:300]}"))
        page._her_dev_tools = tools
    except Exception as why:  # noqa: BLE001 - a page that cannot be watched is judged by what it shows
        logger.debug("could not watch the page's developer tools: %s", why)
    return tools


#: Pages seen to serve a task by running what they draw, by address: what was seen there.
_SEEN_RUNNING: dict[str, str] = {}


def _address(url: str) -> str:
    return str(url or "").split("#", 1)[0].rstrip("/")


def seen_running(url: str, *, take: bool = False) -> str:
    """What was seen running at ``url`` when it was judged to serve a task, ''; taken once where ``take``."""
    return _SEEN_RUNNING.pop(_address(url), "") if take else _SEEN_RUNNING.get(_address(url), "")


@dataclass
class Serves:
    """Whether a page serves the task: True seen doing it, False not (with why), None cannot yet be told."""

    ok: bool | None
    why: list[str] = field(default_factory=list)
    seen: str = ""

    def says(self) -> str:
        if self.ok:
            return self.seen or "it does what it is for"
        return "; ".join(self.why) or "it cannot yet be told"


def what_the_task_needs(task: str) -> str:
    """What a task asks a page to do: run something, show something to read, or nothing that can be checked ahead."""
    if _RUN.search(task or ""):
        return "run"
    if _READ.search(task or ""):
        return "read"
    return ""


#: A control of the page's own that starts the thing it holds ("Play Game", "Click to play"), not a link away from it:
#: the largest one seen, as a point to press.
_WHAT_STARTS_IT = r"""
() => {
  const said = (el) => [el.innerText, el.getAttribute("aria-label"), el.getAttribute("title"), el.id, el.className && String(el.className)].join(" ");
  const away = (el) => el.tagName === "A" && el.getAttribute("href") && !/^(#|javascript:)/i.test(el.getAttribute("href"));
  let best = null, area = 0;
  for (const el of document.querySelectorAll("button, [role=button], input[type=button], input[type=submit], a")) {
    if (away(el) || !/\b(play|start|launch|load|run)\b/i.test(said(el))) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 40 || r.height < 20 || !el.getClientRects().length) continue;
    if (r.width * r.height > area) { area = r.width * r.height; best = el; }
  }
  if (!best) return null;
  best.scrollIntoView({block: "center"});
  const r = best.getBoundingClientRect();
  return [r.x + r.width / 2, r.y + r.height / 2, (best.innerText || best.getAttribute("aria-label") || "").trim().slice(0, 40)];
}
"""

#: How long a thing is given to appear after its start control is pressed, in seconds.
STARTS_WITHIN_S = 6.0


async def whether_it_serves(page: Any, task: str, *, look_for_s: float = 2.5) -> Serves:
    """Whether ``page`` serves ``task``, from its developer tools, its code and its words, and by looking at the thing it is for.

    Where the task is to run something and nothing on the page runs yet, its own start control ("Play Game") is
    pressed once, as a person presses it to see: LIVE 2026-10-08 an archived page's player appeared only then, and
    then said it could not load its own files.
    """
    import asyncio

    needs = what_the_task_needs(task)
    tools = watching(page)
    why: list[str] = []
    if tools.status is not None and tools.status >= 400:
        return Serves(False, [f"the page itself answered {tools.status}"])
    try:
        seen = await page.evaluate(_EMBEDS)
        if needs == "run" and not seen["runs"]:
            start = await page.evaluate(_WHAT_STARTS_IT)
            if start:
                logger.info("nothing on the page runs yet; pressing its %r to see", start[2])
                await page.mouse.click(float(start[0]), float(start[1]))
                await asyncio.sleep(STARTS_WITHIN_S)
                seen = await page.evaluate(_EMBEDS)
    except Exception as error:  # noqa: BLE001 - a page that cannot be read cannot be told to work
        return Serves(None, [f"the page could not be read ({str(error)[:120]})"])
    if seen["plugin"] and not seen["players"]:
        kinds = sorted({p["type"] or p["src"].rsplit(".", 1)[-1] for p in seen["plugin"]})
        why.append(f"its code embeds plugin content ({', '.join(kinds)}) that no browser runs, and the page has no player for it")
    content = [(url, kind, failed) for url, kind, failed in tools.failed if _A_THING.search(url) or kind in ("media", "object")]
    for url, kind, failed in content[:3]:
        why.append(f"{_CONTENT.get(kind, 'a file it loads')} could not be had ({failed}: {url.rsplit('/', 1)[-1][:60]})")
    if needs == "run":
        thing = seen["runs"][0] if seen["runs"] else None
        if thing is None:
            why.append("there is nothing on it that runs")
            return Serves(False if why else None, why)
        if thing["said"] and _FAILS.search(thing["said"]):
            why.append(f"its {thing['tag']} says: {_FAILS.search(thing['said']).group(0)}")
        if thing["media"] and thing["media"]["error"]:
            why.append(f"its {thing['tag']} reports error {thing['media']['error']}")
        if why:
            return Serves(False, why)
        drawn = await _what_it_draws(page, thing, look_for_s)
        if drawn == "blank":
            return Serves(False if tools.errors or content else None,
                          ["it draws nothing", *([f"the console says: {tools.errors[0][:160]}"] if tools.errors else [])])
        if drawn:
            seen_it = f"its {thing['tag']} {drawn}"
            if len(_SEEN_RUNNING) > 64:
                _SEEN_RUNNING.clear()
            _SEEN_RUNNING[_address(getattr(page, "url", ""))] = seen_it
            return Serves(True, seen=seen_it + (" (its console has errors, none of them stopping it)" if tools.errors else ""))
        return Serves(None, ["what it draws could not be seen"])
    words = seen["text"]
    if _FAILS.search(words) and seen["words"] < 400:
        why.append(f"it says: {_FAILS.search(words).group(0)}")
    if why:
        return Serves(False, why)
    if needs == "read":
        return Serves(True, seen=f"it has {seen['words']} words to read") if seen["words"] >= 80 else Serves(None, ["it has little to read"])
    return Serves(None)


async def _what_it_draws(page: Any, thing: dict[str, Any], over_s: float) -> str:
    """'blank' when the thing shows one flat colour twice; else what it was seen to do: 'draws, and moves', 'draws its screen'; '' unseen."""
    import asyncio

    try:
        from PIL import Image, ImageStat
    except ImportError:
        return ""
    box = {"x": max(0.0, thing["x"]), "y": max(0.0, thing["y"]), "width": max(1.0, thing["w"]), "height": max(1.0, thing["h"])}
    looks = []
    for n in range(2):
        from core.perception.a_picture_of_her_page import picture_of

        shot = await picture_of(page, box, css=True)
        if shot is None:
            return ""  # off screen, or the page went away
        looks.append(Image.fromarray(shot).convert("L").resize((64, 48)))
        if n == 0:
            await asyncio.sleep(over_s)
    spread = [ImageStat.Stat(look).stddev[0] for look in looks]
    if max(spread) < 3.0:
        return "blank"
    moved = sum(abs(a - b) for a, b in zip(looks[0].tobytes(), looks[1].tobytes(), strict=False)) / (64 * 48)
    return "draws, and moves" if moved > 1.5 else "draws its screen"
