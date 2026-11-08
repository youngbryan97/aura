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
from core.runtime.wall_clock import monotonic_time, wall_time

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
#: The characters of what was said in it before those looks that a remark as it plays is given: what someone watching
#: has in mind of where it has been going. Given only the newest looks, some fifteen seconds, each remark was about the
#: moment and none could be about the argument a film was making or what came before (LIVE 22 October 2026, rehearsal 16).
SAID_SO_FAR_ROOM = 1500
#: The characters of what she watched one turn of hers is given as its evidence: the bound her turn evidence keeps
#: (core/senses/turn_evidence.py), which keeps the first of anything longer.
OBSERVATION_ROOM = 3200
#: The characters of what played that what she made of the whole is given, taken across it from start to end, within
#: what her evidence keeps. At 6,000 the evidence kept its first 3,200, so what she made of a film was given its first
#: half and not its end.
WHOLE_ROOM = OBSERVATION_ROOM - 200
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
          duration: isFinite(best.duration) ? best.duration : 0, paused: best.paused, ended: best.ended, muted: best.muted,
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
  let name = '', length = 0, by = '';
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    try { for (const one of [].concat(JSON.parse(s.textContent))) {
      if (one && one['@type'] === 'VideoObject') { name = name || one.name || ''; length = length || seconds(one.duration);
        const author = [].concat(one.author || [])[0]; by = by || (typeof author === 'string' ? author : (author && author.name) || ''); } } } catch (e) { /* not ours to read */ }
  }
  length = length || seconds(meta('meta[itemprop="duration"]')) || parseFloat(meta('meta[property="video:duration"]')) || 0;
  name = name || meta('meta[property="og:title"]') || meta('meta[itemprop="name"]');
  by = by || meta('[itemprop="author"] [itemprop="name"]') || meta('meta[name="author"]');
  return {title: document.title, name: name, length: length, by: by,
          words: ((document.querySelector('main') || document.body).innerText || '').slice(0, 1500)};
}"""


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class Moment:
    """One look at what plays: when, what her eyes saw, and the words shown over it; how far the look departed from what
    her eyes expected (``sight``), and how loud each second since the last look was (``sounds``, from her ear)."""

    at: float
    seen: str = ""
    read: str = ""
    sight: float | None = None
    sounds: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Watching:
    """What one watching came to: her moments and what she said, in order."""

    request: str
    title: str = ""
    moments: list[Moment] = field(default_factory=list)
    said: list[tuple[float, str]] = field(default_factory=list)
    ended: bool = False
    #: Why she stopped watching it before it ended, when she did: in her own state's terms.
    stopped: str = ""
    #: The events she took it in as, and the most anything in it moved her (core/together/what_moves_her.py).
    events: str = ""
    moved_most: float = 0.0
    #: How far what she heard of it was about her world, in nats a token (knowing who she is made it that much less
    #: unexpected); None when it was not read.
    about_her: float | None = None
    #: Its events as she kept them (when, what was said, what it showed, how far it moved her), and who she watched it
    #: with: what she recalls of it if she watches it again, and that they watched it together.
    event_list: list[dict[str, Any]] = field(default_factory=list)
    with_whom: str = ""
    #: Who made it, as its page said.
    by: str = ""
    #: Who was in it, and how she came to feel about them; and the memories of her own it called up (their ids).
    people: dict[str, dict[str, float]] = field(default_factory=dict)
    reminded_of: list[str] = field(default_factory=list)

    def since(self, looks: int) -> list[Moment]:
        """The moments after the first ``looks``: counted by look, since a video's clock starts again after an advert."""
        return self.moments[looks:]


def what_she_watched(moments: list[Moment], *, room: int = 3000, across: bool = False) -> str:
    """Moments as quoted data, newest kept when there is not room for all; or, ``across``, every so many of them from
    the first to the last, so that what she made of the whole has the whole: a 21-minute film is some 250 looks, and
    its last 3,000 characters were its last minute or so."""
    import re

    lines, last_read = [], ""
    for m in moments:
        read = m.read if m.read and m.read != last_read else ""
        last_read = m.read or last_read
        # One account of what was said: where captions on it say it, what her ear wrote down of the same words is
        # not given beside them. LIVE 7 November 2026, a TED talk: given both, and they differed, she stopped to say
        # "the transcript is doing something that doesn't" match instead of remarking on the talk.
        captions = re.search(r'words on it: "([^"]*)"', m.seen or "")
        if captions and len(re.findall(r"[A-Za-z]{2,}", captions.group(1))) >= 4:
            read = ""
        # What her ear could not make out is not given as heard: LIVE 7 November 2026, a wordless animated film, she
        # was given "heard: . . ." and remarked on "all those dots where a voice should be".
        if len(re.findall(r"[A-Za-z]{2,}", read)) < 2:
            read = ""
        # What she heard is quoted, as the words on it are: its "you" is the speaker's, said to whoever watches. LIVE 8
        # November 2026, dress rehearsal: given a narration's "You have five thousand two hundred weeks" unquoted, her
        # remarks told the person what had got to them ("The line that actually got you was the temporal one").
        parts = [f"seen: {m.seen}" if m.seen else "", f'heard: "{read}"' if read else ""]
        line = "; ".join(p for p in parts if p)
        if line:
            lines.append(f"{clock(m.at)} {line}")
    text = "\n".join(lines)
    every = 1
    while across and len(text) > room and every < len(lines):
        every += 1
        text = "\n".join(lines[index] for index in range(len(lines)) if index % every == 0 or index == len(lines) - 1)
    return text[-room:]


def said_in_it(moments: list[Moment], *, room: int = SAID_SO_FAR_ROOM) -> str:
    """What was heard of what plays over ``moments``, in order, on one line, its newest ``room`` characters from a
    word's start: its words, not her looks at it."""
    heard, last = [], ""
    for m in moments:
        read = " ".join((m.read or "").split())
        if read and read != last:
            heard.append(read)
        last = read or last
    text = " ".join(heard)
    if len(text) <= room:
        return text
    cut = text[-room:]
    if text[-room - 1] != " " and not cut.startswith(" "):  # it begins inside a word: not that word
        cut = cut.split(" ", 1)[1] if " " in cut else ""
    return cut.strip()


def playing_now(at: float, duration: float, moments: list[Moment], now: list[Moment]) -> str:
    """What a remark as it plays is given: how far in, what was said in it before ``now`` (the newest of ``moments``),
    and the newest looks, which always have their room within ``OBSERVATION_ROOM``."""
    head, newest = f"playing, at {clock(at)} of {clock(duration)}.", what_she_watched(now)
    label = "heard in it before now: "
    left = OBSERVATION_ROOM - len(head) - len(newest) - len(label) - 2
    so_far = said_in_it(moments[:max(0, len(moments) - len(now))], room=min(SAID_SO_FAR_ROOM, left)) if left > 0 else ""
    return "\n".join(line for line in (head, f'{label}"{so_far}"' if so_far else "", newest) if line)


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
                      worth: Callable[[Moment, float], bool] | None = None,
                      look_in: Callable[[float], float] | None = None,
                      stop_when: Callable[[], Awaitable[str]] | None = None,
                      whole: Callable[[], tuple[Any, ...]] | None = None) -> Watching:
    """Watch what plays with whoever asked ``request``, saying what she thinks before, during and after. ``page`` is a
    page of her own browser, or a player of another place it plays (core/skills/watching_the_screen.py, the person's
    own screen). Watched as something done together (core/skills/watching_together.py): ``before`` off when her
    answer to the request already said what she expects, ``may_speak`` whether the with leaves her the floor now, and
    ``on_moment`` / ``on_said`` what the with takes in of each look and each remark. ``look_in`` is how long until she
    looks again (what moves her sets it: core/together/what_moves_her.py), ``stop_when`` her reason to stop watching
    before it ends, if she has one, and ``whole`` what she took it in as, for what she makes of it after."""
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
    began = monotonic_time()
    spoke_after_looks, spoke_at_wall = 0, time.monotonic()
    speaking: asyncio.Task | None = None
    new = _WhatIsNew()
    in_adverts = 0.0
    while at_most_s is None or monotonic_time() - began - in_adverts < at_most_s:
        stopping = await stop_when() if stop_when is not None else ""
        if stopping:
            watching.stopped = stopping
            keep({"kind": "stopped", "why": stopping, "wall": time.time()})
            break
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
            if callable(getattr(player, "sounds", None)):
                await player.sounds()  # an advert's sound is not what her ear comes to expect of the video
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
        seen = await look("speaking" if composing else "playing")
        sounds = await player.sounds() if callable(getattr(player, "sounds", None)) else None
        moment = Moment(at=float(plays.get("at") or 0.0), seen=seen, read=read,
                        sight=player.sight() if callable(getattr(player, "sight", None)) else None,
                        sounds=list(sounds or []))
        watching.moments.append(moment)
        new.took_in(moment)
        perceive(moment.seen, moment.read)
        keep({"kind": "moment", "at": moment.at, "seen": moment.seen, "read": moment.read, "wall": time.time(),
              **({"sight": round(moment.sight, 4)} if moment.sight is not None else {})})
        if on_moment is not None:
            on_moment(moment, plays)
        since = watching.since(spoke_after_looks)
        said_out = not watching.said or time.monotonic() >= spoke_at_wall + len(watching.said[-1][1].split()) / WORDS_A_SECOND
        if ((speaking is None or speaking.done()) and since and said_out and (worth(moment, time.time()) if worth else new.worth_a_remark())
                and time.monotonic() - spoke_at_wall >= quiet_at_least_s and (may_speak is None or may_speak(time.time()))):
            observation = playing_now(moment.at, float(plays.get("duration") or 0), watching.moments, since[-NOW_LOOKS:])
            struck = str(getattr(worth, "struck", "") or "")
            if struck:
                observation = f"{observation}\nwhat struck her in it since she last spoke: {struck}"
            speaking = asyncio.ensure_future(speak(observation, moment.at, "during"))
            spoke_after_looks, spoke_at_wall = len(watching.moments), time.monotonic()
            new.remarked()
        await asyncio.sleep(look_in(look_every_s) if look_in is not None else look_every_s)
    if speaking is not None:
        await speaking
    # What was said after the last look, up to the end: heard too, before she says what she made of it.
    last_words = " ".join(str(h.get("words") or "") for h in (await player.heard() or []))
    if last_words.strip():
        watching.moments.append(Moment(at=watching.moments[-1].at if watching.moments else 0.0, read=last_words))
        perceive("", last_words)
    if whole is not None:
        took = whole()
        watching.events, watching.moved_most = took[0], took[1]
        watching.about_her = took[2] if len(took) > 2 else None
        watching.event_list = list(took[3]) if len(took) > 3 else []
        watching.people = dict(took[4]) if len(took) > 4 else {}
        watching.reminded_of = list(took[5]) if len(took) > 5 else []
    how_it_ends = ("it has ended" if watching.ended else f"it is still playing; she stopped watching it: {watching.stopped}"
                   if watching.stopped else "it is still playing; we stopped watching")
    # What she took it in as: its events where she has them, else her looks across it.
    played = watching.events or what_she_watched(watching.moments, room=WHOLE_ROOM, across=True)
    await speak(f"{how_it_ends}. What played, in order:\n{played}"[:OBSERVATION_ROOM],
                watching.moments[-1].at if watching.moments else 0.0, "after")
    keep({"kind": "after", "ended": watching.ended, "stopped": watching.stopped, "wall": time.time()})
    await player.done()
    await remember(watching)
    return watching


async def _the_floor_comes_free(may_speak: Callable[[float], bool], *, at_most_s: float = FLOOR_WAITED_S) -> None:
    """Until nobody has the floor, or ``at_most_s`` has passed: what she made of it is said once it is hers to say."""
    deadline = monotonic_time() + at_most_s
    while not may_speak(wall_time()) and monotonic_time() < deadline:  # noqa: ASYNC110 - the floor is a reading, not an event
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
    # A remark's look is hers, taken while she watches, for no request of anyone's. LIVE 22 October 2026, rehearsal 16:
    # each remark's look went to her as fresh evidence with "request: Go to Allen Pan's YouTube channel ... before it
    # starts, tell me ..." under it, the last thing she read before speaking, and 11 of her 18 remarks while it played
    # began by answering it again ("you re-prompted me", "we're not doing this from the top again", "I'm already *in* it").
    # The request stays where it was said: at the head of the talk since the watching began (the_talk_since).
    evidence = TurnSensoryEvidence(channel="screen", ok=True, request="" if remark else request, observed_at=time.time(),
                                   observation=observation[:OBSERVATION_ROOM], source="watching_along",
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
    # A remark's objective is what plays, not the request: the objective cues what she recalls and which kept answer is
    # brought to the turn. LIVE 7 November 2026, rehearsal 4: with her answer in the talk and the request off the
    # evidence, a remark at 1:22 still began "My gut before we press play", cued by the request it was handed.
    objective = observation if remark else request
    thought = await engine.think(objective, context=context, mode=ThinkingMode.FAST, origin=ORIGIN, timeout_s=seconds)
    words = str(getattr(thought, "content", "") or "").strip()
    words = first_thought(words, about=observation) if as_it_plays else whole_paragraphs(words)
    if remark and tells_of_herself(words, observation):
        logger.info("👀 not said, as herself told in the third person: %r", words[:160])
        return ""
    return words


#: Words that name a woman or a girl, or point back to one: what a "she" in a remark could be about besides herself.
_SOMEONE_SHE_COULD_MEAN = (
    r"\b(?:she|her|hers|herself|woman|women|girl|girls|lady|ladies|mother|mom|mum|sister|daughter|wife|queen|"
    r"princess|actress|grandma|grandmother|aunt|niece|female|madam|ma'am|mrs|ms|miss)\b"
)


def tells_of_herself(words: str, shown: str) -> bool:
    """Whether a remark tells of her in the third person: a "she" or "her" in it, outside what it quotes, where nothing
    she was shown names or points back to a woman or a girl, so that it can only be herself.

    A person watching beside someone does not say that they lean in (``without_asides``), and does not say "she leans
    in" of themselves either. LIVE 8 November 2026, dress rehearsal, a cartoon of two men: from her second remark on
    every one was told of her ("She pauses on it. 'Okay so he cooked the human.' ... She doesn't look at the screen
    anymore. Looks at you instead."), each one read by the next as what she had said so far."""
    import re

    unquoted = re.sub(r"[\"“][^\"”]*[\"”]", " ", str(words or ""))
    if not re.search(r"\b(?:she|her|herself)\b", unquoted, re.IGNORECASE):
        return False
    return not re.search(_SOMEONE_SHE_COULD_MEAN, str(shown or ""), re.IGNORECASE)


def whole_paragraphs(words: str) -> str:
    """What she said, to its last paragraph that ends: LIVE 20 October 2026 what she made of a film ran to its ceiling
    and stopped mid-sentence, under horizontal rules. A rule is not said."""
    import re

    ends = r"[.!?…][\"'\u201d\u2019)*_]*"
    paragraphs = [without_asides(part) for part in re.split(r"\n\s*\n", words or "") if part.strip() and not re.fullmatch(r"[-*_\s]{3,}", part.strip())]
    paragraphs = [part for part in paragraphs if part]  # a stage direction is not said (without_asides)
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


#: Runs of this many words are what a remark is held against what she heard by.
A_HEARD_RUN = 4


def _every_run_heard(part: str, heard: str) -> bool:
    """Whether every run of ``A_HEARD_RUN`` words in ``part`` is one in ``heard``: the line said again, not a thought."""
    import re

    def runs(text: str) -> set[tuple[str, ...]]:
        words = re.findall(r"[a-z0-9']+", str(text or "").lower())
        return {tuple(words[i:i + A_HEARD_RUN]) for i in range(len(words) - A_HEARD_RUN + 1)}

    said = runs(part)
    return bool(said) and said <= runs(heard)


def first_thought(words: str, about: str = "") -> str:
    """The first paragraph of hers that says something, to its last whole sentence: what a person says of what is
    playing while it plays, before the rest of what they would say about it. Something is said where it is long enough
    to be a thought, or names two things of ``about``, what she was shown. Nothing, where nothing she wrote does."""
    import re

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", words or "") if part.strip()]
    # A heading or a line that leads into what follows is not a thought (LIVE 20 October 2026: "**After it ends — my
    # thoughts:**" was said as one).
    said = [part for part in paragraphs if not re.fullmatch(r"#{1,6}\s[^\n]*|\*\*[^*\n]+\*\*\W*|[^\n.!?…]*:\W*", part)]
    shown = _stems(" ".join(_named(about)))

    def says_something(part: str) -> bool:
        # Written as code is not said: LIVE 7 November 2026, twice in a concert, a remark was the docstring of a class
        # of her own source ("``retry`` is given one held item and answers whether it landed. Anything\n    it raises
        # counts as not landing"), its indentation and all.
        if "``" in part or len(re.findall(r"(?m)^ {4,}\S", part)) >= 1 or re.search(r"\b[a-z]+_[a-z_]+\(", part):
            return False
        # Quoted whole, it is what was said in the video, not what she thinks of it (LIVE 20 October 2026, rehearsal 9:
        # '*"In July of 2026, 700 AI agents hacked the infrastructure of Hugging Face..."*' was said as a remark).
        if re.fullmatch(r"[*_\s]*[\"“].*[\"”][.…]*[*_\s]*", part, re.DOTALL):
            return False
        # Nor is it where every run of its words is one she was just given as heard, however it is set off: LIVE 8
        # November 2026, three remarks on a short story were its narration's lines in italics ("*...I'm every human
        # being who ever lived.*").
        if _every_run_heard(part, about):
            return False
        count = len(re.findall(r"[A-Za-z']+", part))
        # Named by stem: "he reads the manual like a recipe" names the "reading" and the "manual" she was shown. Word for
        # word it named one, and a short reaction of hers was not said while a long description was.
        return count >= A_THOUGHT_HAS_WORDS and (not shown or count >= A_THOUGHT_STANDS_ALONE or len(_stems(" ".join(_named(part))) & shown) >= 2)

    # Where her private notes ran past their bound, the first of what she said can be the end of a note: it starts
    # mid-sentence, or annotates a line it quotes. LIVE 7 November 2026, said as a remark: '. This is time, man." -
    # casual, warm opening'. That much is not said; what follows it is.
    cleaned = {part: without_asides(_without_the_end_of_a_note(part)) for part in said}
    first = next((part for part in said if cleaned[part] and says_something(cleaned[part])), "")
    if not first:
        return ""
    thought = cleaned[first]
    # A short paragraph that points ahead runs on into what it introduces: LIVE 7 November 2026 a remark was "Right —
    # here is what I want to push on, and it's the bit that just landed in the room.", its point in the paragraph after.
    rest = paragraphs[paragraphs.index(first) + 1:]
    if rest and len(re.findall(r"[A-Za-z']+", thought)) < A_THOUGHT_STANDS_ALONE and re.search(
            r"\b(?:here(?:'s|’s| is)|this is what|what I want to|the (?:bit|part|thing|line) (?:that|where)|hold on|wait|"
            r"(?:need|have|want) to (?:stop|pause|flag|say|push)|for a (?:second|moment|sec)|"
            # LIVE 7 November 2026, dress rehearsal, said as remarks with their point in the paragraph after: "And I
            # want to really look at this image because I think it's the most honest visual of the whole talk." and
            # "And I think this is where the talk finally gets to be *true* in a way that surprises you."
            r"I want to (?:really )?(?:look|sit|stay|talk|point)|this is (?:where|the (?:part|moment|point)))\b",
            thought, re.IGNORECASE):
        thought = f"{thought}\n\n{_without_the_end_of_a_note(rest[0])}".strip()
    # A quotation it opens runs on to where it closes: LIVE 22 October 2026 a remark ended inside one, 'the on-screen
    # text is saying: "While they secretly start working on goals that are harmful or straight up incomprehensible to
    # us.', its close in the paragraph after.
    following = paragraphs[paragraphs.index(first) + 1:]
    while _a_quotation_left_open(thought) and following:
        thought = f"{thought}\n\n{following.pop(0)}"
    if _a_quotation_left_open(thought):
        opened = max(thought.rfind('"'), thought.rfind("“"))
        before = re.match(r"(?s)^(.*[.!?…][\"'”’)*_]*)(?:\s|$)", thought[:opened])
        thought = before.group(1).strip() if before else thought
    # A last sentence that leads into what follows is not said without it: LIVE 7 November 2026 a remark ended "The
    # arc is:", its list in the paragraph after.
    led_into = re.match(r"(?s)^(.*[.!?…][\"'”’)*_]*)\s+[^.!?…]*:\W*$", thought)
    thought = led_into.group(1).strip() if led_into else thought
    whole = re.match(r"(?s)^(.*[.!?…][\"'”’)*_]*)(?:\s|$)", thought)
    return whole.group(1).strip() if whole and not re.search(r"[.!?…][\"'”’)*_]*$", thought) else thought


#: An aside in italics: what she would do, not what she says. Emphasis that exclaims or asks ("*one hundred and eighty
#: times its own body weight?*") is said; a gesture ends without either.
_AN_ASIDE = r"(?:\*(?!\*)(?P<a>[^*\n]+?)\*(?!\*)|_(?!_)(?P<b>[^_\n]+?)_(?!_))"


def without_asides(part: str) -> str:
    """A paragraph without a stage direction: a paragraph that is one is not said, and one it opens with is left off.

    LIVE 7 November 2026, dress rehearsal: of twelve remarks on Allen Pan's video, eight were asides, said as remarks
    while it played: "*blinks at the screen, then lets out a short laugh that's mostly relief*", "*leans forward
    slightly, watching the hands work the tool on screen*". A person watching beside someone laughs or leans in; they
    do not say that they do. What she says of it is what follows the aside, where anything does."""
    import re

    text = str(part or "").strip()

    def a_gesture(inner: str) -> bool:
        inner = inner.strip()
        # Exclaimed or asked, it is said with emphasis. A gesture is told in the present ("leans", "the room goes
        # loud and I flinch"), or in a word or three ("Exhales.").
        if re.search(r"[?!][\"'”’)]*$", inner):
            return False
        return bool(re.match(r"[a-z]", inner)) or len(re.findall(r"[A-Za-z']+", inner)) <= 3

    whole = re.fullmatch(_AN_ASIDE + r"[.…]?", text)
    if whole and a_gesture(whole.group("a") or whole.group("b") or ""):
        return ""
    while True:
        opening = re.match(r"^" + _AN_ASIDE + r"(?P<stop>[.…]?)\s+(?=\S)", text)
        if not opening or not a_gesture(opening.group("a") or opening.group("b") or ""):
            return text
        # Starred and then carried on in lower case, it is a word stressed in her sentence, not a gesture before it.
        # LIVE 8 November 2026, dress rehearsal: "*This* is the pivot." was said as "is the pivot."
        inner = (opening.group("a") or opening.group("b") or "").strip()
        if not opening.group("stop") and not re.search(r"[.…!?]$", inner) and re.match(r"[a-z]", text[opening.end():]):
            return text
        text = text[opening.end():].strip()


def _without_the_end_of_a_note(part: str) -> str:
    """A paragraph without a note's end at its head: text before its first sentence that starts mid-sentence (with a
    stop, a comma or a lower-case word), or a line it annotates after a quotation ('..." - casual, warm opening')."""
    import re

    text = part.strip()

    def is_a_note(head: str) -> bool:
        return bool(re.match(r"^[.,;:)\]…]|^[a-z]", head) or re.search(r"[\"”]\s+[-–—]\s+[a-z]", head))

    while text:
        parts = re.split(r"(?<=[.!?…])[\"'”’)*_]*\s+(?=[A-Z\"“*])", text, maxsplit=1)
        if not is_a_note(parts[0]):
            return text
        text = parts[1].strip() if len(parts) > 1 else ""
    return text


def _a_quotation_left_open(text: str) -> bool:
    """Whether ``text`` opens a quotation it does not close: straight quotes odd in number, or more curly openings than
    closings."""
    return text.count('"') % 2 == 1 or text.count("“") > text.count("”")


async def say_it(words: str) -> bool:
    """Said aloud and in her chat, through her Will and constitution, as anything she says unasked is."""
    from core.container import ServiceContainer

    from core.conversation.response_reliability import internal_leak_reasons

    orchestrator = ServiceContainer.get("orchestrator", default=None)
    if orchestrator is None or not hasattr(orchestrator, "emit_spontaneous_message"):
        return False
    # Machinery is not said, and what is not said is not kept as said. LIVE 7 November 2026, dress rehearsal: "I lost
    # the reply lane for a moment. Ask that again..." was withheld from her chat as internal text (interface/
    # event_bridge.py), and kept in her record of the watching as a remark, which she would read back as her own.
    leaks = internal_leak_reasons(words)
    if leaks:
        logger.warning("👀 not said, as machinery (%s): %r", ",".join(leaks), words[:120])
        return False
    # A commentary the person asked for: LIVE 20 October 2026 every remark was withheld from the chat as a second
    # answer to the request that started the watching (interface/event_bridge.py), and Bryan saw none of them.
    decision = await orchestrator.emit_spontaneous_message(words, modality="both", origin=ORIGIN,
                                                           metadata={"visible_presence": True, "narration": True})
    said = not (isinstance(decision, dict) and decision.get("ok") is False)
    if said:
        # Said in the chat, so said in the conversation she keeps (core/conversation/said_unasked.py).
        from core.conversation.said_unasked import she_said_unasked

        await she_said_unasked(words, origin=ORIGIN)
    return said


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


async def what_she_remembers_of(title: str, url: str = "") -> list[dict[str, Any]]:
    """The events she kept of it when she last watched it, recalled as she recalls anything: recalling strengthens a
    memory and can let the present rewrite it (core/memory/episodic_memory.py ``pattern_complete``). [] when she has
    not watched it, or kept no events of it."""
    from core.container import ServiceContainer
    from core.runtime.errors import record_degradation

    episodic = ServiceContainer.get("episodic_memory", default=None)
    recall = getattr(episodic, "pattern_complete", None)
    if not callable(recall) or not (title or url):
        return []
    try:
        # Cued by its title (or its address), not "Watched ...", which every watching of hers begins with.
        episodes = await asyncio.to_thread(recall, title or url, 8)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("watching_along", exc, severity="info", action="she watched it without recalling it")
        return []
    latest: tuple[float, list[dict[str, Any]]] | None = None
    for episode in episodes or []:
        kept = dict(getattr(episode, "source_metadata", None) or {})
        if getattr(episode, "action", "") != "watching_along":
            continue
        if not ((url and kept.get("url") == url) or (title and kept.get("title") == title)):
            continue
        events = kept.get("events")
        when = float(getattr(episode, "timestamp", 0.0) or 0.0)
        if isinstance(events, list) and events and (latest is None or when > latest[0]):
            latest = (when, [event for event in events if isinstance(event, dict)])
    return latest[1] if latest else []


#: How many memories of her own one moving moment may call up.
CALLED_UP = 3


async def what_it_calls_up(text: str) -> list[str]:
    """The memories of her own (not of watchings) that ``text`` calls up, recalled as she recalls anything, so recalling
    them strengthens them and lets the present touch them (core/memory/episodic_memory.py ``pattern_complete``); their
    ids, to link them to the watching. Nothing of them is shown or said here."""
    from core.container import ServiceContainer

    episodic = ServiceContainer.get("episodic_memory", default=None)
    recall = getattr(episodic, "pattern_complete", None)
    if not callable(recall) or not str(text or "").strip():
        return []
    try:
        episodes = await asyncio.to_thread(recall, str(text), CALLED_UP + 3)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        logger.debug("what it called up of hers was not recalled: %s", exc)
        return []
    mine = [str(getattr(episode, "episode_id", "") or "") for episode in episodes or []
            if getattr(episode, "action", "") != "watching_along"]
    return [episode for episode in mine if episode][:CALLED_UP]


#: How much of what she remembers of a maker's other videos is read before what she hears of a new one, in characters.
OF_THEM_CHARS = 300


async def what_she_remembers_of_them(by: str, title: str = "") -> str:
    """What she remembers of what ``by`` made before (another video than ``title``): what was said in the first events
    of her latest watching of one, recalled as she recalls anything. "" when she has watched nothing of theirs, or kept
    no events of it."""
    from core.container import ServiceContainer
    from core.runtime.errors import record_degradation

    episodic = ServiceContainer.get("episodic_memory", default=None)
    recall = getattr(episodic, "pattern_complete", None)
    if not callable(recall) or not by:
        return ""
    try:
        episodes = await asyncio.to_thread(recall, by, 8)
    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
        record_degradation("watching_along", exc, severity="info", action="she watched it without recalling its maker")
        return ""
    latest: tuple[float, str] | None = None
    for episode in episodes or []:
        kept = dict(getattr(episode, "source_metadata", None) or {})
        if getattr(episode, "action", "") != "watching_along" or kept.get("by") != by or kept.get("title") == title:
            continue
        heard = " ".join(str(event.get("heard") or "") for event in (kept.get("events") or [])[:4] if isinstance(event, dict))
        when = float(getattr(episode, "timestamp", 0.0) or 0.0)
        if heard.strip() and (latest is None or when > latest[0]):
            latest = (when, f"{kept.get('title') or 'one of theirs'}: {' '.join(heard.split())}")
    return latest[1][:OF_THEM_CHARS] if latest else ""


async def into_her_memory(watching: Watching, *, url: str = "") -> None:
    """The watching as one episode of her own: what was watched, with whom, and what she said of it, in her words, so
    that asked later what she thought of it she recalls it as she remembers anything she did."""
    from core.container import ServiceContainer
    from core.runtime.errors import record_degradation

    episodic = ServiceContainer.get("episodic_memory", default=None)
    if episodic is None or not hasattr(episodic, "record_episode_async") or not (watching.said or watching.events):
        return
    said = "\n".join(f"{clock(at)} {words}" for at, words in watching.said)
    try:
        # What moved her more is kept as mattering more, as emotional memories are kept better in people (the more it
        # moved her, up to 0.9 from 0.6), and what was about her world more again (the self-reference effect: up to a
        # tenth more, at a nat a token).
        about = max(0.0, min(1.0, watching.about_her or 0.0))
        await episodic.record_episode_async(
            context=(f"Watched {watching.title or 'a video'}{f' by {watching.by}' if watching.by else ''} as it played, "
                     f"with {watching.with_whom or 'whoever asked'}: {watching.request}")[:600],
            action="watching_along", outcome=said[-4000:], success=watching.ended,
            importance=round(min(1.0, 0.6 + 0.3 * max(0.0, min(1.0, watching.moved_most)) + 0.1 * about), 3),
            source="watching_along",
            metadata={"provenance": "her own words while watching", "verified": False, "title": watching.title,
                      "url": url, "ended": watching.ended, "looks": len(watching.moments),
                      "what happened in it": watching.events[:3000], "stopped": watching.stopped,
                      "moved most": round(watching.moved_most, 4),
                      "about her": None if watching.about_her is None else round(watching.about_her, 4),
                      "events": watching.event_list[:60], "with": watching.with_whom, "by": watching.by,
                      "people": watching.people, "reminded of": watching.reminded_of[:20]})
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
