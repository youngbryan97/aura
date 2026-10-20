"""Watching what plays on her page with someone, in real time, and saying what she thinks as it plays.

As a person watches a video with a friend: before it starts she says what she expects from what the page shows; while
it plays she looks at the picture every few seconds and reads the words the page shows over it (its captions), and
says what she thinks when she has seen something new and has not spoken for a while; when it ends she says what it
was and what she made of it. It plays at its own speed in her own browser. Nothing is downloaded and no frame is
read ahead of when it shows.

What she sees and reads goes into her as perception (her perceptual buffer, the perception daemon's moments, her
soma's sensory imprint). What she says is her own turn: her cognitive engine on the desktop lane, with her live
state's advisory frames, her identity and recall, and the steering, endogenous and kNN channels her worker applies,
given the person's request as their turn, what she has said so far as her own, and what she watched as fresh sensory
evidence on the screen channel. It is said through her Will and constitution (``emit_spontaneous_message``). Nothing
here writes a word of hers or tells her model what to say.

Nothing here knows a site. The page plays the largest video on it; its captions are the words over that video; a
control that turns captions on says so in its own label.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.WatchingAlong")

#: How often she looks at what plays, and how long she lets pass between two things she says, at least.
LOOK_EVERY_S = 5.0
QUIET_AT_LEAST_S = 15.0
#: How long an end is given to stay an end. A site that plays an advert first plays it in the same video, which
#: reports itself ended when the advert does and then plays the video asked for (measured on a watch page, 19 October
#: 2026: a 15-second advert in the one video element).
AN_END_HOLDS_S = 4.0
#: The origin her turns while watching carry: a desktop turn, started by the person's request.
ORIGIN = "desktop_watching_along"

#: The largest video on the page (shadow roots walked), where it is, how far it has played, and the words shown over
#: it that are not its controls: its captions, and whatever else the page writes on the picture.
WHAT_PLAYS = r"""() => {
  let best = null, area = 0;
  const walk = (root) => { for (const el of root.querySelectorAll('*')) {
    if (el.tagName === 'VIDEO') { const r = el.getBoundingClientRect(); if (r.width * r.height > area) { area = r.width * r.height; best = el; } }
    if (el.shadowRoot) walk(el.shadowRoot); } };
  walk(document);
  if (!best) return null;
  const r = best.getBoundingClientRect();
  const words = [];
  for (const el of document.querySelectorAll('body *')) {
    if (el.children.length) continue;
    const t = (el.textContent || '').trim();
    if (!t || el.closest('button,[role=button],a,input,select,label,[role=slider],[role=menu],[role=menuitem]')) continue;
    const s = getComputedStyle(el); if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0) continue;
    const b = el.getBoundingClientRect(); if (b.width <= 0 || b.height <= 0) continue;
    if (b.left < r.right && b.right > r.left && b.top < r.bottom && b.bottom > r.top) words.push(t);
  }
  return {x: r.left, y: r.top, w: r.width, h: r.height, vw: innerWidth, vh: innerHeight, at: best.currentTime || 0,
          duration: isFinite(best.duration) ? best.duration : 0, paused: best.paused, ended: best.ended,
          words: words.join(' ').slice(0, 600)};
}"""

#: Start the largest video if it is paused, and turn on captions where a control says it shows them.
PLAY_WITH_CAPTIONS = r"""() => {
  let best = null, area = 0;
  for (const v of document.querySelectorAll('video')) { const r = v.getBoundingClientRect(); if (r.width * r.height > area) { area = r.width * r.height; best = v; } }
  if (best && best.paused) best.play().catch(() => {});
  const named = (el) => [el.getAttribute('aria-label'), el.getAttribute('title'), el.textContent].join(' ');
  for (const el of document.querySelectorAll('button,[role=button]')) {
    if (/subtit|caption/i.test(named(el)) && el.getAttribute('aria-pressed') === 'false' && el.getClientRects().length) { el.click(); break; }
  }
  return !!best;
}"""

#: A control the page offers to skip what plays before the video (an advert), pressed as a person presses it.
SKIP_WHAT_COMES_FIRST = r"""() => {
  for (const el of document.querySelectorAll('button,[role=button]')) {
    const said = [el.getAttribute('aria-label'), el.textContent].join(' ').trim();
    if (/^skip\b/i.test(said) && el.getClientRects().length) { el.click(); return said.slice(0, 40); }
  }
  return '';
}"""

#: What the page says of itself before it plays: its title and the words near the top of it.
WHAT_THE_PAGE_SAYS = r"""() => ({title: document.title, words: ((document.querySelector('main') || document.body).innerText || '').slice(0, 1500)})"""


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class Moment:
    """One look at what plays: when, what her eyes saw, and the words shown over it."""

    at: float
    seen: str = ""
    read: str = ""


@dataclass
class Watching:
    """What one watching came to: her moments and what she said, in order."""

    request: str
    title: str = ""
    moments: list[Moment] = field(default_factory=list)
    said: list[tuple[float, str]] = field(default_factory=list)
    ended: bool = False

    def since(self, looks: int) -> list[Moment]:
        """The moments after the first ``looks``: counted by look, since a video's clock starts again after an advert."""
        return self.moments[looks:]


def what_she_watched(moments: list[Moment], *, room: int = 3000) -> str:
    """Moments as quoted data, newest kept when there is not room for all."""
    lines, last_read = [], ""
    for m in moments:
        read = m.read if m.read and m.read != last_read else ""
        last_read = m.read or last_read
        parts = [f"seen: {m.seen}" if m.seen else "", f"words shown: {read}" if read else ""]
        line = "; ".join(p for p in parts if p)
        if line:
            lines.append(f"{clock(m.at)} {line}")
    text = "\n".join(lines)
    return text[-room:]


async def watch_along(page: Any, request: str, *, look: Callable[[str], Awaitable[str]] | None = None,
                      turn: Callable[[str, str, list[str]], Awaitable[str]] | None = None,
                      say: Callable[[str], Awaitable[bool]] | None = None,
                      perceive: Callable[[str, str], None] | None = None,
                      keep: Callable[[dict[str, Any]], None] | None = None,
                      look_every_s: float = LOOK_EVERY_S, quiet_at_least_s: float = QUIET_AT_LEAST_S,
                      at_most_s: float | None = None) -> Watching:
    """Watch what plays on ``page`` with whoever asked ``request``, saying what she thinks before, during and after."""
    look = look or her_eyes_on(page)
    turn = turn or her_turn
    say = say or say_it
    perceive = perceive or into_her_senses
    keep = keep or (lambda _row: None)
    page_says = await page.evaluate(WHAT_THE_PAGE_SAYS) or {}
    watching = Watching(request=request, title=str(page_says.get("title") or ""))

    async def speak(observation: str, at: float, when: str) -> None:
        words = (await turn(request, observation, [w for _t, w in watching.said])).strip()
        if words and await say(words):
            watching.said.append((at, words))
            keep({"kind": "said", "when": when, "at": at, "wall": time.time(), "words": words})

    # Before it plays: what the page shows of itself.
    first = await look("")
    keep({"kind": "before", "title": watching.title, "seen": first, "wall": time.time()})
    await speak(f"before it plays. The page: {watching.title}. {str(page_says.get('words') or '')[:1200]} "
                f"What the page shows: {first}", 0.0, "before")
    await page.evaluate(PLAY_WITH_CAPTIONS)
    began = time.monotonic()
    spoke_after_looks, spoke_at_wall = 0, time.monotonic()
    speaking: asyncio.Task | None = None
    while at_most_s is None or time.monotonic() - began < at_most_s:
        plays = await page.evaluate(WHAT_PLAYS)
        if not plays:
            break
        if plays.get("ended"):
            await asyncio.sleep(AN_END_HOLDS_S)
            again = await page.evaluate(WHAT_PLAYS)
            if not again or again.get("ended"):
                watching.ended = True
                break
            continue
        skipped = await page.evaluate(SKIP_WHAT_COMES_FIRST)
        if skipped:
            keep({"kind": "skipped", "control": skipped, "wall": time.time()})
        if plays.get("paused"):
            await page.evaluate(PLAY_WITH_CAPTIONS)
        moment = Moment(at=float(plays.get("at") or 0.0), seen=await look("playing"), read=str(plays.get("words") or ""))
        watching.moments.append(moment)
        perceive(moment.seen, moment.read)
        keep({"kind": "moment", "at": moment.at, "seen": moment.seen, "read": moment.read, "wall": time.time()})
        new = watching.since(spoke_after_looks)
        if ((speaking is None or speaking.done()) and new and time.monotonic() - spoke_at_wall >= quiet_at_least_s):
            observation = f"playing, at {clock(moment.at)} of {clock(plays.get('duration') or 0)}.\n{what_she_watched(new)}"
            speaking = asyncio.ensure_future(speak(observation, moment.at, "during"))
            spoke_after_looks, spoke_at_wall = len(watching.moments), time.monotonic()
        await asyncio.sleep(look_every_s)
    if speaking is not None:
        await speaking
    await speak(f"{'it has ended' if watching.ended else 'it is still playing; we stopped watching'}. What played, in "
                f"order:\n{what_she_watched(watching.moments)}", watching.moments[-1].at if watching.moments else 0.0, "after")
    keep({"kind": "after", "ended": watching.ended, "wall": time.time()})
    return watching


def her_eyes_on(page: Any, *, seconds: float = 8.0) -> Callable[[str], Awaitable[str]]:
    """Her eyes on the page: the picture where the video plays while it plays, else the whole page in view."""
    from core.perception.her_eyes import look_with_her_eyes

    async def look(when: str) -> str:
        plays = await page.evaluate(WHAT_PLAYS) if when else None
        clip = ({"x": max(0.0, plays["x"]), "y": max(0.0, plays["y"]), "width": plays["w"], "height": plays["h"]}
                if plays and plays.get("w") and plays.get("h") else None)
        try:
            shot = await page.screenshot(type="jpeg", quality=70, clip=clip) if clip else await page.screenshot(type="jpeg", quality=70)
            return (await look_with_her_eyes("What is in this picture?", base64.b64encode(shot).decode("ascii"),
                                             max_tokens=90, timeout_s=seconds)).strip()
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError, TimeoutError) as why:
            logger.info("a look did not come back: %s", why)
            return ""

    return look


async def her_turn(request: str, observation: str, said: list[str], *, seconds: float = 60.0) -> str:
    """Her own turn: her cognitive engine on the desktop lane, the request as the person's turn, what she has said so
    far as hers, what she watched as fresh sensory evidence on the screen channel."""
    from core.brain.types import ThinkingMode
    from core.container import ServiceContainer
    from core.senses.turn_evidence import TurnSensoryEvidence
    from core.utils.injected_blocks import stamp_runtime_payload

    engine = ServiceContainer.get("cognitive_engine", default=None)
    if engine is None:
        return ""
    evidence = TurnSensoryEvidence(channel="screen", ok=True, request=request, observed_at=time.time(),
                                   observation=observation[:3200], source="watching_along",
                                   scope="what plays on her own page", coverage="partial")
    # The turn that asked her to watch is open while she watches; her remarks are work it is waiting on, as the
    # pursuit's own decisions are (LIVE 2026-09-28, refused as background inside an open turn until they said so).
    context = {"desktop_quick_reply_contract": True, "origin": ORIGIN, "visible_user_message": request,
               "turn_sensory_evidence": evidence.to_dict(), "max_tokens": 320, "serves_current_turn": True,
               "recent_completed_exchanges": [stamp_runtime_payload({"user": request, "aura": " ".join(said)})] if said else []}
    thought = await engine.think(request, context=context, mode=ThinkingMode.FAST, origin=ORIGIN, timeout_s=seconds)
    return str(getattr(thought, "content", "") or "").strip()


async def say_it(words: str) -> bool:
    """Said aloud and in her chat, through her Will and constitution, as anything she says unasked is."""
    from core.container import ServiceContainer

    orchestrator = ServiceContainer.get("orchestrator", default=None)
    if orchestrator is None or not hasattr(orchestrator, "emit_spontaneous_message"):
        return False
    decision = await orchestrator.emit_spontaneous_message(words, modality="both", origin=ORIGIN,
                                                           metadata={"visible_presence": True})
    return not (isinstance(decision, dict) and decision.get("ok") is False)


def into_her_senses(seen: str, read: str) -> None:
    """What she sees and reads, into her as her own senses' readings are: her perceptual buffer, the perception
    daemon's moments, her soma's sensory imprint."""
    from core.container import ServiceContainer
    from core.runtime.errors import record_degradation

    for source, text in (("vision", seen), ("audio", read)):
        if not text:
            continue
        try:
            buffer = ServiceContainer.get("perceptual_buffer", default=None)
            if buffer is not None:
                buffer.append(source, text, {"from": "watching_along"})
            from core.perception.perception_daemon import get_perception_daemon

            get_perception_daemon().register_moment(source, text, {"from": "watching_along"})
            from core.senses.soma import get_soma

            get_soma().update_sensory_imprint(source, text)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("watching_along", exc, severity="info", action="a moment did not reach her senses")


def kept_at(title: str, url: str = "") -> Callable[[dict[str, Any]], None]:
    """Each row of a watching, kept in her data as it happens: what she saw, read and said, when, and where it played."""
    from core.config import config

    folder = Path(config.paths.data_dir) / "watched"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    def keep(row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({**row, "title": title, "url": url}, ensure_ascii=False) + "\n")

    return keep
