"""Her own time: what she could do when nothing is asked of her, done, and what came of it.

Each thing she could do is offered by a part of her that can do it, as an act with a name and an end: a request of somebody's
she left unmet, a thing she wants found out, a goal she set herself, something she found that somebody here would want. What
she does of them is hers to choose (core/agency/what_she_does_next.py), and what came of it is measured where it can be seen:
the parts of a request that are met now and were not, the pages read that had not been, the calls a goal's pass made, whether
somebody answered what she told them.

What is owed is not weighed: something she was in the middle of for somebody when she was closed is gone on with as soon as
she is up (core/understanding/carrying_out.py). And what she was doing is kept, so the next time she is opened she knows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from core.agency.her_inquiries import KEPT, Inquiry
from core.agency.what_she_does_next import CameOf, Option, WhatSheDoesNext, get_what_she_does_next
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.HerOwnTime")

__all__ = ["HerOwnTime", "Offers", "get_her_own_time", "reset_for_test"]

_RECOVERABLE = (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError, TimeoutError, LookupError)

#: A part of her that can do something: what it could do now.
Offers = Callable[[], Iterable[Option] | Awaitable[Iterable[Option]]]

#: The longest one act of hers goes on before what it came to so far is what it came to, in seconds.
AN_ACT_S = 1800.0
#: How long somebody has to answer what she told them before she takes it that they will not, in seconds.
ANSWERED_WITHIN_S = 1800.0
#: How many things she found are held to be told, and how many pages she has read are known for read.
HELD, READ = 24, 4000
#: Who her own requests are from.
HERSELF = "herself"


class HerOwnTime:
    """What she could do, what she does of it, and what she keeps between one start and the next."""

    def __init__(self, chooser: WhatSheDoesNext | None = None, path: Path | None = None) -> None:
        self.chooser = chooser or get_what_she_does_next()
        self._path = path
        self._lock = checked_lock("agency.her_own_time")
        self._offers: dict[str, Offers] = {}
        #: What she is doing, what she has found and not told, what she told and waits on, and the pages she has read.
        self.doing: dict[str, Any] = {}
        self.found: list[dict[str, Any]] = []
        self.told: dict[str, Any] | None = None
        self._read: dict[str, float] = {}
        #: What she is finding out, by what it is for.
        self.inquiries: dict[str, Inquiry] = {}
        self.before: dict[str, Any] = {}
        #: What there was to do when it was last looked for.
        self._there: list[Option] = []
        #: Whether an act may be done: her Will's to say.
        self.may: Callable[[Option], Awaitable[bool]] | None = None
        #: What would do a goal of hers, where something would.
        self.act_of: Callable[[str], Awaitable[Option | None]] | None = None
        self._loaded = False
        self._going = asyncio.Lock()

    def offers(self, name: str, what: Offers) -> None:
        """A part of her that can do something says what it could do."""
        self._offers[name] = what

    async def options(self) -> list[Option]:
        """Everything she could do now. A part that cannot say is left out, and said to be."""
        out: list[Option] = []
        for name, offered in list(self._offers.items()):
            try:
                given = offered()
                out += list(await given if isinstance(given, Awaitable) else given)
            except _RECOVERABLE as exc:
                record_degradation("her_own_time", exc, severity="info", action=f"what she could do by {name} is not among her options this time")
        self._there = out
        return out

    def that_feeds(self, drive: str) -> str:
        """Of what there was to do, the act that would feed a drive and is expected to be worth most; "" where there is none."""
        feeding = [option for option in self._there if drive in option.drives]
        best = max(feeding, key=lambda option: self.chooser.expected(option)[0], default=None)
        return best.what if best is not None and self.chooser.expected(best)[0] >= 0.0 else ""

    async def once(self) -> dict[str, Any] | None:
        """Choose something and do it; the choice and what came of it. None where she rests, or is doing something already."""
        if self._going.locked():
            return None
        async with self._going:
            await asyncio.to_thread(self._load)
            options = await self.options()
            picked = await asyncio.to_thread(self.chooser.choose, options)
            if picked is None:
                return None
            option, chosen = picked
            assert option.does is not None
            if self.may is not None and not await self.may(option):
                self.chooser.learn(chosen, CameOf(put_down=True, what="her Will refused it"), cost_s=option.cost_s)
                logger.info("🧭 Her Will refused it: %s", option.what[:160])
                return None
            self.doing = {"id": option.id, "kind": option.kind, "what": option.what, "why": option.why, "since": time.time()}
            await self.keep()
            began = time.monotonic()
            try:
                came = await asyncio.wait_for(option.does(), timeout=AN_ACT_S)
            except TimeoutError:
                came = CameOf(got=0.0, what=f"it was still going after {AN_ACT_S / 60:.0f} minutes")
            except asyncio.CancelledError:
                self.chooser.learn(chosen, CameOf(put_down=True, what="she was stopped"), cost_s=option.cost_s)
                raise
            except _RECOVERABLE as exc:
                record_degradation("her_own_time", exc, severity="info", action=f"what she chose to do broke off: {option.what[:80]}")
                came = CameOf(got=0.0, what=f"it broke off: {type(exc).__name__}: {exc}"[:200])
            if not came.seconds:
                came = CameOf(came.got, came.worth, came.put_down, came.what, time.monotonic() - began)
            reward = await asyncio.to_thread(self.chooser.learn, chosen, came, cost_s=option.cost_s)
            logger.info("🧭 What came of it (%s): %s; %s", "put down" if reward is None else f"{reward:+.2f}", came.what[:200] or "nothing that can be seen", option.what[:120])
            self.before = {**self.doing, "came": came.what, "got": came.got, "ended": time.time()}
            self.doing = {}
            await self.keep()
            return {"what": option.what, "kind": option.kind, "why": chosen.says(), "came": came.what, "got": came.got, "reward": reward}

    # ── what she found, and telling it ───────────────────────────────

    def is_new(self, pages: Iterable[str]) -> list[str]:
        """Those of some pages she has not read before; read from now on."""
        with self._lock:
            new = [page for page in dict.fromkeys(str(p) for p in pages if p) if page not in self._read]
            now = time.time()
            for page in new:
                self._read[page] = now
            for page in sorted(self._read, key=lambda p: self._read[p])[: max(0, len(self._read) - READ)]:
                del self._read[page]
        return new

    def finds(self, about: str, what: str, where: str = "") -> None:
        """Something she found on her own, held until it is told or is old."""
        with self._lock:
            self.found.append({"about": str(about)[:120], "what": " ".join(str(what).split())[:600], "where": str(where)[:300], "at": time.time()})
            del self.found[:-HELD]

    def has_told(self, find: dict[str, Any]) -> None:
        """Something she found was told to somebody here: whether they answer is what telling it came to."""
        with self._lock:
            self.found = [f for f in self.found if f is not find]
            choices = self.chooser.choices(1)
            self.told = {"at": time.time(), "about": find.get("about"), "choice": choices[-1].at if choices else 0.0}

    def somebody_spoke(self) -> None:
        """Somebody spoke to her: where she had told them something a little before, they answered it."""
        told = self.told
        if told is None:
            return
        self.told = None
        soon = time.time() - float(told["at"]) <= ANSWERED_WITHIN_S
        for chosen in self.chooser.choices():
            if chosen.at == told["choice"]:
                self.chooser.learn(chosen, CameOf(got=1.0 if soon else 0.0, worth=0.5 if soon else -0.3, what="they answered" if soon else "nobody answered"), cost_s=30.0)

    def inquiry(self, of: str, about: str) -> Inquiry:
        """What she is finding out about something, begun where it is not."""
        with self._lock:
            held = self.inquiries.get(of)
            if held is None or held.about != about:
                held = self.inquiries[of] = Inquiry(about, changed_at=time.time())
            for old in sorted(self.inquiries, key=lambda k: self.inquiries[k].changed_at)[: max(0, len(self.inquiries) - KEPT)]:
                if old != of:
                    del self.inquiries[old]
            return held

    # ── between one start and the next ───────────────────────────────

    def where_she_is(self, now: float | None = None) -> str:
        """What she is in the middle of and was doing last, in a line, for her to read."""
        now = time.time() if now is None else now
        parts = []
        if self.doing:
            parts.append(f"now {self.doing['what']}")
        if self.before and now - float(self.before.get("ended") or 0.0) < 2 * 86400:
            at = time.strftime("%H:%M", time.localtime(self.before["ended"]))
            parts.append(f"last {self.before['what']} (until {at}" + (f", {self.before['came']}" if self.before.get("came") else "") + ")")
        return "; ".join(" ".join(str(p).split()) for p in parts)[:420]

    def _where(self) -> Path:
        if self._path is None:
            from core.runtime.state_ownership import state_root

            self._path = state_root() / "data" / "agency" / "her_own_time.json"
        return Path(self._path)

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
        try:
            kept = json.loads(self._where().read_text(encoding="utf-8")) if self._where().is_file() else {}
        except (OSError, ValueError) as exc:
            record_degradation("her_own_time", exc, severity="warning", action="what she was doing before this start is not held")
            kept = {}
        self.found = [*list(kept.get("found") or []), *self.found][-HELD:]
        self._read = {**{str(page): float(at) for page, at in dict(kept.get("read") or {}).items()}, **self._read}
        held = {str(of): Inquiry.of(one) for of, one in dict(kept.get("inquiries") or {}).items()}
        self.inquiries = {**{of: one for of, one in held.items() if one is not None}, **self.inquiries}
        doing = dict(kept.get("doing") or {})
        self.before = {**doing, "came": "left off when she was closed", "got": 0.0, "ended": float(kept.get("at") or 0.0)} if doing else dict(kept.get("before") or {})

    async def keep(self) -> None:
        """Write what she is doing and has found."""
        from core.rebuilding.her_build_writes import governed

        with self._lock:
            kept = {"at": time.time(), "doing": self.doing, "before": self.before, "found": self.found, "read": self._read,
                    "inquiries": {of: one.kept() for of, one in self.inquiries.items()}}

        def write() -> None:
            governed("ensure_directory", self._where().parent, source="agency.her_own_time")
            governed("write_text", self._where(), json.dumps(kept, ensure_ascii=False), source="agency.her_own_time")
            self.chooser.keep()

        try:
            await asyncio.to_thread(write)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("her_own_time", exc, severity="warning", action="what she is doing was not written down")


_HERS: HerOwnTime | None = None
_HERS_LOCK = checked_lock("agency.her_own_time.singleton")


def get_her_own_time() -> HerOwnTime:
    global _HERS
    with _HERS_LOCK:
        if _HERS is None:
            _HERS = HerOwnTime()
        return _HERS


def reset_for_test() -> None:
    global _HERS
    with _HERS_LOCK:
        _HERS = None
