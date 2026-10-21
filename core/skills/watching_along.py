"""Watching what plays on her page with someone, in real time, and saying what she thinks as it plays.

As a person watches a video with a friend: before it starts she says what she expects from what the page shows; while
it plays she looks at the picture every few seconds and hears everything said in it, as the words the page shows over
it (its captions) come and go; she says what she thinks when what she has seen and heard since she last spoke holds
more that is new than is usual for this video, at a lull in it, and not over herself; when it ends she says what it
was and what she made of it. It plays at its own speed in her own browser. Nothing is downloaded and no frame or word
is read ahead of when it shows.

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
#: How often an advert is looked at for the moment it can be skipped: the offer comes in seconds.
ADVERT_LOOK_S = 2.0
#: How long an end is given to stay an end. A site that plays an advert first plays it in the same video, which
#: reports itself ended when the advert does and then plays the video asked for (measured on a watch page, 19 October
#: 2026: a 15-second advert in the one video element).
AN_END_HOLDS_S = 4.0
#: How much one look says: a sentence or two. At 140 a look ran to a paragraph of what a picture is ("This picture shows
#: a **video player interface** — likely from"), and every remark's prompt carried four of them (LIVE 20 October 2026).
LOOK_TOKENS = 70
#: The looks a remark as it plays is given: what is on now. Given every look since her last remark, she narrated the
#: oldest of them in their own form, "**0:21–0:28:** The title card reads", at 1:00 (LIVE 20 October 2026).
NOW_LOOKS = 3
#: How long what she made of it waits for the floor at the end, when somebody has it: the two minutes a turn of theirs
#: takes, at most, to be answered (the watching waits the same for her answer to the request).
FLOOR_WAITED_S = 120.0
#: The origin her turns while watching carry: a desktop turn, started by the person's request.
ORIGIN = "desktop_watching_along"
#: How fast she speaks, in words a second, to know when what she said has been said.
WORDS_A_SECOND = 2.5

#: Listens to what is said in what plays: four times a second, the words shown over the largest video (as WHAT_PLAYS
#: reads them) are compared with the last, and the words that came are kept with the video's time. Captions roll (a
#: line shown again with one more under it), so only what follows the overlap with the last reading is new. Read
#: and emptied at each look (HEARD), so nothing said between two looks is missed (LIVE 19 October 2026: looking every
#: five seconds caught one caption line in two).
LISTEN = r"""() => {
  if (window.__auraListening) return true;
  window.__auraListening = true; window.__auraHeard = []; let last = [];
  const over = () => {
    let best = null, area = 0;
    for (const v of document.querySelectorAll('video')) { const r = v.getBoundingClientRect(); if (r.width * r.height > area) { area = r.width * r.height; best = v; } }
    if (!best) return null;
    const r = best.getBoundingClientRect(), words = [];
    for (const el of document.querySelectorAll('body *')) {
      if (el.children.length) continue;
      const t = (el.textContent || '').trim();
      if (!t || el.closest('button,[role=button],a,input,select,label,[role=slider],[role=menu],[role=menuitem]')) continue;
      const s = getComputedStyle(el); if (s.display === 'none' || s.visibility === 'hidden' || Number(s.opacity) === 0) continue;
      const b = el.getBoundingClientRect(); if (b.width <= 0 || b.height <= 0) continue;
      if (b.left < r.right && b.right > r.left && b.top < r.bottom && b.bottom > r.top) words.push(t);
    }
    return {at: best.currentTime || 0, words: words.join(' ').split(/\s+/).filter(Boolean)};
  };
  setInterval(() => {
    const now = over(); if (!now || !now.words.length) return;
    let k = Math.min(last.length, now.words.length);
    while (k > 0 && last.slice(-k).join(' ') !== now.words.slice(0, k).join(' ')) k--;
    const came = now.words.slice(k);
    if (came.length && now.words.join(' ') !== last.join(' ')) window.__auraHeard.push({at: now.at, words: came.join(' ')});
    last = now.words;
  }, 250);
  return true;
}"""
HEARD = r"""() => { const h = window.__auraHeard || []; window.__auraHeard = []; return h; }"""

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
#: With the name and length the page gives what it plays, where it gives them in the web's standard ways (schema.org's
#: VideoObject, in JSON-LD or microdata, and Open Graph). The video element says the length of what is in it now: LIVE
#: 20 October 2026 that was a six-second advert, the watching took the video for six seconds long and ended it two
#: minutes in.
WHAT_THE_PAGE_SAYS = r"""() => {
  const meta = (sel) => { const m = document.querySelector(sel); return m ? (m.getAttribute('content') || '') : ''; };
  const seconds = (iso) => { const m = /^P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$/.exec(String(iso || ''));
    return m ? (+m[1] || 0) * 86400 + (+m[2] || 0) * 3600 + (+m[3] || 0) * 60 + (+m[4] || 0) : 0; };
  let name = '', length = 0;
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    try { for (const one of [].concat(JSON.parse(s.textContent))) {
      if (one && one['@type'] === 'VideoObject') { name = name || one.name || ''; length = length || seconds(one.duration); } } } catch (e) { /* not ours to read */ }
  }
  length = length || seconds(meta('meta[itemprop="duration"]')) || parseFloat(meta('meta[property="video:duration"]')) || 0;
  name = name || meta('meta[property="og:title"]') || meta('meta[itemprop="name"]');
  return {title: document.title, name: name, length: length,
          words: ((document.querySelector('main') || document.body).innerText || '').slice(0, 1500)};
}"""


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
        parts = [f"seen: {m.seen}" if m.seen else "", f"heard: {read}" if read else ""]
        line = "; ".join(p for p in parts if p)
        if line:
            lines.append(f"{clock(m.at)} {line}")
    text = "\n".join(lines)
    return text[-room:]


def _stems(text: str) -> set[str]:
    import re

    import snowballstemmer

    return set(snowballstemmer.stemmer("english").stemWords([w.lower() for w in re.findall(r"[A-Za-z]{4,}", text or "")]))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


@dataclass
class _WhatIsNew:
    """How much of what she has seen and heard is new to this watching, and when that is worth a remark.

    New is a word (its stem) not seen or heard before in this video. She speaks when what is new since her last
    remark is at least what was new when she last spoke, the middle of her remarks so far (the first in play needs
    only something new), at a lull: the words said in the last look no more than this video's middle. People who
    watch together talk in the gaps of what they watch, and about what has changed."""

    known: set[str] = field(default_factory=set)
    fresh: int = 0
    rates: list[float] = field(default_factory=list)
    at_remarks: list[int] = field(default_factory=list)

    def took_in(self, moment: Moment) -> None:
        came = _stems(f"{moment.seen} {moment.read}")
        self.fresh += len(came - self.known)
        self.known |= came
        self.rates.append(float(len((moment.read or "").split())))

    def worth_a_remark(self) -> bool:
        lull = not self.rates or self.rates[-1] <= _median(self.rates)
        return self.fresh >= max(1.0, _median([float(n) for n in self.at_remarks])) and lull

    def remarked(self) -> None:
        self.at_remarks.append(self.fresh)
        self.fresh = 0


class OnAPage:
    """What plays on a page of her own browser: read and tended through the page's scripts."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.listening = False

    async def says(self) -> dict[str, Any]:
        return await self.page.evaluate(WHAT_THE_PAGE_SAYS) or {}

    async def start(self) -> None:
        await self.page.evaluate(PLAY_WITH_CAPTIONS)
        self.listening = bool(await self.page.evaluate(LISTEN))

    async def state(self) -> dict[str, Any] | None:
        return await self.page.evaluate(WHAT_PLAYS)

    async def tend(self) -> str:
        """An advert's skip control pressed, and, every look, play and captions on again: after an advert the
        video's captions are off (LIVE 19 October 2026, a watch page after two adverts)."""
        skipped = await self.page.evaluate(SKIP_WHAT_COMES_FIRST)
        await self.page.evaluate(PLAY_WITH_CAPTIONS)
        return str(skipped or "")

    async def heard(self) -> list[dict[str, Any]] | None:
        return await self.page.evaluate(HEARD) if self.listening else None

    def eyes(self) -> Callable[[str], Awaitable[str]]:
        return her_eyes_on(self.page)

    async def done(self) -> None:
        return None


async def watch_along(page: Any, request: str, *, look: Callable[[str], Awaitable[str]] | None = None,
                      turn: Callable[[str, str, list[str]], Awaitable[str]] | None = None,
                      say: Callable[[str], Awaitable[bool]] | None = None,
                      perceive: Callable[[str, str], None] | None = None,
                      keep: Callable[[dict[str, Any]], None] | None = None,
                      remember: Callable[[Watching], Awaitable[None]] | None = None,
                      look_every_s: float = LOOK_EVERY_S, quiet_at_least_s: float = QUIET_AT_LEAST_S,
                      at_most_s: float | None = None, before: bool = True,
                      may_speak: Callable[[float], bool] | None = None,
                      on_moment: Callable[[Moment, dict[str, Any]], None] | None = None,
                      on_said: Callable[[str], None] | None = None,
                      worth: Callable[[Moment, float], bool] | None = None) -> Watching:
    """Watch what plays with whoever asked ``request``, saying what she thinks before, during and after. ``page`` is a
    page of her own browser, or a player of another place it plays (core/skills/watching_the_screen.py, the person's
    own screen). Watched as something done together (core/skills/watching_together.py): ``before`` off when her
    answer to the request already said what she expects, ``may_speak`` whether the with leaves her the floor now, and
    ``on_moment`` / ``on_said`` what the with takes in of each look and each remark."""
    player = page if callable(getattr(page, "heard", None)) else OnAPage(page)
    look = look or player.eyes()
    turn = turn or her_turn
    say = say or say_it
    perceive = perceive or into_her_senses
    keep = keep or (lambda _row: None)
    remember = remember or into_her_memory
    page_says = await player.says()
    watching = Watching(request=request, title=str(page_says.get("title") or ""))

    async def speak(observation: str, at: float, when: str) -> None:
        if when == "after" and may_speak is not None:
            await _the_floor_comes_free(may_speak)
        words = (await turn(request, observation, [w for _t, w in watching.said])).strip()
        if words and when == "during" and may_speak is not None and not may_speak(time.time()):
            # Somebody took the floor while it was being thought: swallowed, as a remark is when the other speaks.
            keep({"kind": "held back", "at": at, "wall": time.time(), "words": words})
            return
        if words and await say(words):
            watching.said.append((at, words))
            keep({"kind": "said", "when": when, "at": at, "wall": time.time(), "words": words})
            if on_said is not None:
                on_said(words)

    # Before it plays: what the page shows of itself.
    first = await look("")
    keep({"kind": "before", "title": watching.title, "seen": first, "wall": time.time()})
    if before:
        await speak(f"before it plays. The page: {watching.title}. {str(page_says.get('words') or '')[:1200]} "
                    f"What the page shows: {first}", 0.0, "before")
    await player.start()
    began = time.monotonic()
    spoke_after_looks, spoke_at_wall = 0, time.monotonic()
    speaking: asyncio.Task | None = None
    new = _WhatIsNew()
    in_adverts = 0.0
    while at_most_s is None or time.monotonic() - began - in_adverts < at_most_s:
        plays = await player.state()
        if not plays:
            break
        if plays.get("ended"):
            await asyncio.sleep(AN_END_HOLDS_S)
            again = await player.state()
            if not again or again.get("ended"):
                watching.ended = True
                break
            continue
        if plays.get("advert"):
            # Not what they are watching together: nothing is said of it, what it says is not taken in, its skip is
            # pressed when offered, and its time is not the watching's. LIVE 20 October 2026, rehearsal 7: two remarks
            # on a pillow advert, and "Twenty seconds left" with three and a half minutes still to play.
            advert_began = time.monotonic()
            keep({"kind": "advert", "skipped": await player.tend(), "wall": time.time()})
            await player.heard()
            await asyncio.sleep(ADVERT_LOOK_S)
            in_adverts += time.monotonic() - advert_began
            continue
        skipped = await player.tend()
        if skipped:
            keep({"kind": "skipped", "control": skipped, "wall": time.time()})
        heard = await player.heard()
        read = " ".join(str(h.get("words") or "") for h in heard or []) if heard is not None else str(plays.get("words") or "")
        # Her eyes are told when her voice is using the GPU they share, so the part of a look that needs it can wait
        # (core/skills/watching_the_screen.py); what is said and the words on what plays go on being taken in.
        composing = speaking is not None and not speaking.done()
        moment = Moment(at=float(plays.get("at") or 0.0), seen=await look("speaking" if composing else "playing"), read=read)
        watching.moments.append(moment)
        new.took_in(moment)
        perceive(moment.seen, moment.read)
        keep({"kind": "moment", "at": moment.at, "seen": moment.seen, "read": moment.read, "wall": time.time()})
        if on_moment is not None:
            on_moment(moment, plays)
        since = watching.since(spoke_after_looks)
        said_out = not watching.said or time.monotonic() >= spoke_at_wall + len(watching.said[-1][1].split()) / WORDS_A_SECOND
        if ((speaking is None or speaking.done()) and since and said_out and (worth(moment, time.time()) if worth else new.worth_a_remark())
                and time.monotonic() - spoke_at_wall >= quiet_at_least_s and (may_speak is None or may_speak(time.time()))):
            observation = f"playing, at {clock(moment.at)} of {clock(plays.get('duration') or 0)}.\n{what_she_watched(since[-NOW_LOOKS:])}"
            speaking = asyncio.ensure_future(speak(observation, moment.at, "during"))
            spoke_after_looks, spoke_at_wall = len(watching.moments), time.monotonic()
            new.remarked()
        await asyncio.sleep(look_every_s)
    if speaking is not None:
        await speaking
    # What was said after the last look, up to the end: heard too, before she says what she made of it.
    last_words = " ".join(str(h.get("words") or "") for h in (await player.heard() or []))
    if last_words.strip():
        watching.moments.append(Moment(at=watching.moments[-1].at if watching.moments else 0.0, read=last_words))
        perceive("", last_words)
    await speak(f"{'it has ended' if watching.ended else 'it is still playing; we stopped watching'}. What played, in "
                f"order:\n{what_she_watched(watching.moments)}", watching.moments[-1].at if watching.moments else 0.0, "after")
    keep({"kind": "after", "ended": watching.ended, "wall": time.time()})
    await player.done()
    await remember(watching)
    return watching


async def _the_floor_comes_free(may_speak: Callable[[float], bool], *, at_most_s: float = FLOOR_WAITED_S) -> None:
    """Until nobody has the floor, or ``at_most_s`` has passed: what she made of it is said once it is hers to say."""
    deadline = time.monotonic() + at_most_s
    while not may_speak(time.time()) and time.monotonic() < deadline:  # noqa: ASYNC110 - the floor is a reading, not an event
        await asyncio.sleep(1.0)


def her_eyes_on(page: Any, *, seconds: float = 8.0) -> Callable[[str], Awaitable[str]]:
    """Her eyes on the page: the picture where the video plays while it plays, else the whole page in view."""
    from core.perception.her_eyes import look_with_her_eyes, plainly

    async def look(when: str) -> str:
        plays = await page.evaluate(WHAT_PLAYS) if when else None
        clip = ({"x": max(0.0, plays["x"]), "y": max(0.0, plays["y"]), "width": plays["w"], "height": plays["h"]}
                if plays and plays.get("w") and plays.get("h") else None)
        try:
            from core.perception.browser_presence import without_presence

            async with without_presence(page):
                shot = await page.screenshot(type="jpeg", quality=70, clip=clip) if clip else await page.screenshot(type="jpeg", quality=70)
            return plainly(await look_with_her_eyes("What is happening in this picture?", base64.b64encode(shot).decode("ascii"),
                                                    max_tokens=LOOK_TOKENS, timeout_s=seconds))
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError, TimeoutError) as why:
            logger.info("a look did not come back: %s", why)
            return ""

    return look


async def her_turn(request: str, observation: str, said: list[str], *, seconds: float = 60.0,
                    exchanges: list[dict[str, Any]] | None = None, remark: bool = False) -> str:
    """Her own turn: her cognitive engine on the desktop lane, the request as the person's turn, what she has said so
    far as hers, what she watched as fresh sensory evidence on the screen channel. ``exchanges``, when the watching
    goes on beside a conversation, is that conversation since it began, in place of the request and her remarks.
    ``remark``: a turn after her answer, said to nobody; the request is in her conversation already, and her desktop
    lane reads that conversation itself (LIVE 20 October 2026, it was handed the request again and answered all of
    it again, "Before, what I think it'll be about")."""
    from core.brain.types import ThinkingMode
    from core.container import ServiceContainer
    from core.runtime.structured_input import A_CLOSED_QUESTIONS_FLOOR
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
    # Once the request is in what she reads (her answer to it, or remarks), nobody has just said anything to her.
    # A remark while it plays is said as it comes: its first thought, with its private channel shut (a budget within
    # the closed question's floor, core/brain/llm/chat_format.py). LIVE 20 October 2026 a remark thought for 640 tokens
    # at 8.5 a second, then answered the request's three parts under headings, cut off; stopped at its first
    # paragraph instead, it was "Okay. Here we go." What is said is its first thought (first_thought). Each turn of
    # hers while watching thinks first, within an allowance the decoder holds (core/brain/llm/a_bounded_private_channel.py)
    # and a ceiling over thinking and saying together: left to the channel's own rule her answer thought through its
    # whole ceiling twice, five minutes before the film could open (LIVE 20 October 2026, 11:46 to 11:51).
    as_it_plays = remark and observation.startswith("playing")
    thinks, says = (120, 160) if as_it_plays else (224, 288) if remark else (160, 352)
    context = {"desktop_quick_reply_contract": True, "origin": ORIGIN, "visible_user_message": "" if remark or exchanges or said else request,
               "turn_sensory_evidence": evidence.to_dict(), "max_tokens": max(thinks + says, A_CLOSED_QUESTIONS_FLOOR + 1),
               "private_channel_budget": thinks, "hard_output_token_ceiling": thinks + says,
               "serves_current_turn": True,
               # Hers to say, not a request to act on: LIVE 20 October 2026 the URL in "Watch this with me: http://..."
               # sent such a turn to fetch it twice, and it said nothing; and it was read again as somebody's
               # utterance, which opened a second with with "someone at desktop_watching_along".
               "allow_tools": False, "her_own_turn": True,
               "recent_completed_exchanges": exchanges if exchanges is not None else
               [stamp_runtime_payload({"user": request, "aura": " ".join(said)})] if said else []}
    thought = await engine.think(request, context=context, mode=ThinkingMode.FAST, origin=ORIGIN, timeout_s=seconds)
    words = str(getattr(thought, "content", "") or "").strip()
    return first_thought(words, about=observation) if as_it_plays else whole_paragraphs(words)


def whole_paragraphs(words: str) -> str:
    """What she said, to its last paragraph that ends: LIVE 20 October 2026 what she made of a film ran to its ceiling
    and stopped mid-sentence, under horizontal rules. A rule is not said."""
    import re

    ends = r"[.!?…][\"'\u201d\u2019)*_]*"
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", words or "") if part.strip() and not re.fullmatch(r"[-*_\s]{3,}", part.strip())]
    # The paragraph the ceiling cut ends at its last whole sentence, and goes where it has none (LIVE 21 October 2026,
    # her answer before the video ended "or are they the kind of"); a paragraph alone stays as it is.
    while paragraphs and not re.search(ends + "$", paragraphs[-1]):
        whole = re.match(r"(?s)^(.*" + ends + r")(?:\s|$)", paragraphs[-1])
        if whole:
            paragraphs[-1] = whole.group(1).strip()
        elif len(paragraphs) > 1:
            paragraphs.pop()
        else:
            break
    return "\n\n".join(paragraphs)


#: Fewer words than this in a paragraph is a lead-in to a thought ("Okay. Here we go."), not one.
A_THOUGHT_HAS_WORDS = 6


#: A paragraph this long is a thought of its own; a shorter one is, where it names what she was shown. LIVE 20 October
#: 2026, rehearsal 7, said as remarks: "I'm watching the zoom-out continue." and "I'm right here."; rehearsal 9, 21 and
#: 25 words naming nothing seen or heard: "Oh — so it's already rolling, we're at about two and a half minutes in. Let me
#: catch up with you on where I am." Rehearsal 6's that Bryan called really good each named what was on screen.
A_THOUGHT_STANDS_ALONE = 35
_NOT_ABOUT_ANYTHING = frozenset("""that this with they have what just here there right about from were when into like been
them then than your will would could their which these those some more very really okay still watching along""".split())


def _named(text: str) -> set[str]:
    import re

    return {word for word in re.findall(r"[a-z']{4,}", str(text or "").lower()) if word not in _NOT_ABOUT_ANYTHING}


def first_thought(words: str, about: str = "") -> str:
    """The first paragraph of hers that says something, to its last whole sentence: what a person says of what is
    playing while it plays, before the rest of what they would say about it. Something is said where it is long enough
    to be a thought, or names two things of ``about``, what she was shown. Nothing, where nothing she wrote does."""
    import re

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", words or "") if part.strip()]
    # A heading or a line that leads into what follows is not a thought (LIVE 20 October 2026: "**After it ends — my
    # thoughts:**" was said as one).
    said = [part for part in paragraphs if not re.fullmatch(r"#{1,6}\s[^\n]*|\*\*[^*\n]+\*\*\W*|[^\n]*:\W*", part)]
    shown = _named(about)

    def says_something(part: str) -> bool:
        # Quoted whole, it is what was said in the video, not what she thinks of it (LIVE 20 October 2026, rehearsal 9:
        # '*"In July of 2026, 700 AI agents hacked the infrastructure of Hugging Face..."*' was said as a remark).
        if re.fullmatch(r"[*_\s]*[\"“].*[\"”][.…]*[*_\s]*", part, re.DOTALL):
            return False
        count = len(re.findall(r"[A-Za-z']+", part))
        return count >= A_THOUGHT_HAS_WORDS and (not shown or count >= A_THOUGHT_STANDS_ALONE or len(_named(part) & shown) >= 2)

    thought = next((part for part in said if says_something(part)), "")
    if not thought:
        return ""
    whole = re.match(r"(?s)^(.*[.!?…][\"'”’)*_]*)(?:\s|$)", thought)
    return whole.group(1).strip() if whole and not re.search(r"[.!?…][\"'”’)*_]*$", thought) else thought


async def say_it(words: str) -> bool:
    """Said aloud and in her chat, through her Will and constitution, as anything she says unasked is."""
    from core.container import ServiceContainer

    orchestrator = ServiceContainer.get("orchestrator", default=None)
    if orchestrator is None or not hasattr(orchestrator, "emit_spontaneous_message"):
        return False
    # A commentary the person asked for: LIVE 20 October 2026 every remark was withheld from the chat as a second
    # answer to the request that started the watching (interface/event_bridge.py), and Bryan saw none of them.
    decision = await orchestrator.emit_spontaneous_message(words, modality="both", origin=ORIGIN,
                                                           metadata={"visible_presence": True, "narration": True})
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


async def into_her_memory(watching: Watching, *, url: str = "") -> None:
    """The watching as one episode of her own: what was watched, with whom, and what she said of it, in her words, so
    that asked later what she thought of it she recalls it as she remembers anything she did."""
    from core.container import ServiceContainer
    from core.runtime.errors import record_degradation

    episodic = ServiceContainer.get("episodic_memory", default=None)
    if episodic is None or not hasattr(episodic, "record_episode_async") or not watching.said:
        return
    said = "\n".join(f"{clock(at)} {words}" for at, words in watching.said)
    try:
        await episodic.record_episode_async(
            context=f"Watched {watching.title or 'a video'} as it played, with whoever asked: {watching.request}"[:600],
            action="watching_along", outcome=said[-4000:], success=watching.ended, importance=0.6,
            source="watching_along",
            metadata={"provenance": "her own words while watching", "verified": False, "title": watching.title,
                      "url": url, "ended": watching.ended, "looks": len(watching.moments)})
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("watching_along", exc, severity="info", action="a watching did not reach her memory")


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
