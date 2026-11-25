"""What people have asked her to bring about, and how each stands, from the work itself.

A request is opened when a turn is read as asking for something to be brought about, and it is closed by a check of the
world and by nothing else: not by her saying so, not by a step returning, not by time. Between, it holds what was tried,
what each try came to, and every part that is still missing, so that "where is it?" and "try again" have an answer that
is about the work.

LIVE 23 November 2026: asked to build a game, a lane made an empty folder and said it was done; "where's the game?" and
"why didn't you build it?" were answered from nothing, because nothing was kept of what had been asked.

Kept on disk as it changes. Work that was under way when she stopped is read back as cut short.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Understanding")

__all__ = ["LEFT_OFF", "ASKED", "Ask", "Part", "Request", "RequestsOfHer", "Try", "get_requests_of_her", "line_of"]

#: How a request stands. ``met`` and ``partly`` and ``unmet`` are what a check found; ``cut_short`` is work that stopped
#: before a check; ``set_aside`` is the person's own word.
OPEN, WORKING, MET, PARTLY, UNMET, CUT_SHORT, SET_ASIDE = (
    "open", "working", "met", "partly", "unmet", "cut_short", "set_aside")
#: Ways of standing that leave something to do.
#: Asked about first, and waiting for an answer.
ASKED = "asked"
UNFINISHED = frozenset({OPEN, WORKING, PARTLY, UNMET, CUT_SHORT, ASKED})
#: How many requests are kept, and how many gifts of files a conversation.
KEPT, KEPT_GIVEN = 400, 20


@dataclass
class Part:
    """One thing the request asked for, and what shows whether it is so."""

    what: str
    #: ``met``, ``unmet`` or ``unknown``.
    stands: str = "unknown"
    shown_by: str = ""


@dataclass
class Try:
    at: float
    how: str
    came_to: str = ""
    missing: list[str] = field(default_factory=list)


@dataclass
class Ask:
    """Something she asked whoever made a request, held until they answer it."""

    words: str
    #: What it is about: ``wanted`` (whether it was asked for at all), or ``which`` (something that changes what is made).
    about: str = "which"
    #: What she goes on with until they say; empty where she waits for them.
    goes_with: str = ""
    at: float = 0.0
    answer: str = ""


@dataclass
class Request:
    id: str
    conversation: str
    said: str
    effect: str
    stands: str = OPEN
    materials: list[str] = field(default_factory=list)
    parts: list[Part] = field(default_factory=list)
    tries: list[Try] = field(default_factory=list)
    #: What was made or found: paths, links, names.
    made: dict[str, Any] = field(default_factory=dict)
    #: What was said about it since, in order: answers to a question of hers, corrections, more to go on.
    added: list[str] = field(default_factory=list)
    #: What she has asked them of it.
    asks: list[Ask] = field(default_factory=list)
    #: What she told them as she began it, which the work holds to where it can.
    told: str = ""
    now: str = ""
    opened_at: float = 0.0
    changed_at: float = 0.0

    @property
    def whole(self) -> str:
        """Everything the person said of it."""
        return "\n".join([self.said, *self.added])

    def waits_on(self) -> Ask | None:
        """What she asked of it that has not been answered, the latest first."""
        return next((ask for ask in reversed(self.asks) if not ask.answer), None)

    def ask(self, words: str, *, about: str = "which", goes_with: str = "") -> Ask:
        """Something to ask them, held until it is answered."""
        ask = Ask(" ".join(str(words).split()), about, " ".join(str(goes_with).split()), time.time())
        self.asks.append(ask)
        return ask

    def missing(self) -> list[Part]:
        return [part for part in self.parts if part.stands != "met"]

    def account(self) -> list[str]:
        """How it stands, as facts, for her to speak of."""
        lines = [f"Asked {_when(self.opened_at)}: {_short(self.said)}"]
        stands = {OPEN: "not started", WORKING: "being worked on now", MET: "done, and checked",
                  PARTLY: "partly done", UNMET: "tried, and not done", CUT_SHORT: "cut short before it was finished",
                  SET_ASIDE: "set aside", ASKED: "not begun: waiting for their answer"}[self.stands]
        lines.append(f"It stands: {stands}" + (f" ({self.now})" if self.now and self.stands == WORKING else "") + ".")
        lines += [f"Made: {name}: {value}" for name, value in self.made.items() if value]
        for part in self.parts:
            mark = {"met": "done", "unmet": "not done", "unknown": "not checked"}[part.stands]
            lines.append(f"- {part.what}: {mark}" + (f" ({part.shown_by})" if part.shown_by else ""))
        lines += [f"Asked them: {ask.words}" + (f" They said: {_short(ask.answer)}" if ask.answer else " (not answered yet"
                  + (f"; going on with {ask.goes_with}" if ask.goes_with else "") + ")") for ask in self.asks[-3:]]
        for attempt in self.tries[-3:]:
            lines.append(f"Tried {_when(attempt.at)}: {attempt.how}. " + (f"Came to: {attempt.came_to}" if attempt.came_to else ""))
        return lines


#: How a try ended that was going when she was closed.
LEFT_OFF = "it was under way when she stopped running"


def _short(text: str, room: int = 240) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= room else f"{text[:room].rsplit(' ', 1)[0]} …"


#: How her window names a conversation: who it is with, when she was started, and when the talk was. Who it is with is what
#: stays the same.
_OF_ONE_START = re.compile(r"^(?P<who>.+):[0-9a-f]{8}:[0-9a-f]{8}$")


def line_of(conversation: str) -> str:
    """Who a conversation is with, as it is the same from one start of hers to the next. LIVE 24 November 2026 a person who
    asked after a restart how it was going was asking in a conversation that had been asked for nothing."""
    found = _OF_ONE_START.match(str(conversation))
    return found.group("who") if found else str(conversation)


def _when(at: float) -> str:
    return time.strftime("%H:%M on %d %B", time.localtime(at or time.time()))


class RequestsOfHer:
    """The requests, in the order they were made."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = checked_lock("understanding.requests_of_her")
        self._held: dict[str, Request] = {}
        #: For each conversation, what was given in it, in order: the words it came with and the files.
        self._gave: dict[str, list[dict[str, Any]]] = {}
        self._loaded = False
        self._writing: asyncio.Lock | None = None

    def _where(self) -> Path:
        if self._path is None:
            from core.runtime.state_ownership import state_root

            self._path = state_root() / "data" / "understanding" / "requests.json"
        return self._path

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        path = self._where()
        if not path.is_file():
            return
        try:
            kept = json.loads(path.read_text(encoding="utf-8"))
            for c, g in dict(kept.get("gave") or {}).items():
                self._gave.setdefault(line_of(c), []).extend(g)
            for row in kept.get("requests") or []:
                row = dict(row)
                row["parts"] = [Part(**part) for part in row.get("parts") or ()]
                row["tries"] = [Try(**attempt) for attempt in row.get("tries") or ()]
                row["asks"] = [Ask(**ask) for ask in row.get("asks") or ()]
                request = Request(**{name: value for name, value in row.items() if name in Request.__dataclass_fields__})
                request.conversation = line_of(request.conversation)
                if request.stands == WORKING:
                    request.stands = CUT_SHORT
                    request.tries.append(Try(at=time.time(), how=LEFT_OFF, came_to="cut short"))
                self._held[request.id] = request
        except (OSError, ValueError, TypeError, KeyError) as exc:
            record_degradation("requests_of_her", exc, severity="warning",
                               action="what was asked before this start is not held")

    async def keep(self) -> None:
        """Write what is held."""
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with self._lock:
            kept = {"requests": [asdict(request) for request in list(self._held.values())[-KEPT:]],
                    "gave": {c: given[-KEPT_GIVEN:] for c, given in list(self._gave.items())[-KEPT:]}}
        with local_internal_governed_scope("understanding.requests", domain="file_write"):
            await get_file_write_gateway().write_text_async(self._where(), json.dumps(kept, ensure_ascii=False),
                                                            source="understanding.requests")

    def gave(self, conversation: str, said: str, files: list[str]) -> None:
        """What a person gave with a turn, held for the turns after it."""
        if not files:
            return
        with self._lock:
            self._load()
            self._gave.setdefault(line_of(conversation), []).append(
                {"said": str(said), "files": list(files), "at": time.time()})

    def given(self, conversation: str, *, with_words: list[str] | None = None) -> list[str]:
        """The files last given in a conversation; with ``with_words``, only those given with one of those turns."""
        with self._lock:
            self._load()
            held = list(self._gave.get(line_of(conversation)) or [])
        if with_words is not None:
            wanted = {" ".join(str(words).split()) for words in with_words}
            held = [row for row in held if " ".join(str(row.get("said") or "").split()) in wanted]
        return list(held[-1].get("files") or []) if held else []

    def asked(self, said: str, *, effect: str, conversation: str, materials: list[str] | None = None) -> Request:
        request = Request(id=secrets.token_hex(4), conversation=line_of(conversation), said=str(said), effect=effect,
                          materials=list(materials or ()), opened_at=time.time(), changed_at=time.time())
        with self._lock:
            self._load()
            self._held[request.id] = request
            while len(self._held) > KEPT:
                self._held.pop(next(iter(self._held)))
        return request

    def get(self, request_id: str) -> Request | None:
        with self._lock:
            self._load()
            return self._held.get(str(request_id))

    def of(self, conversation: str) -> list[Request]:
        """The requests of a conversation, the latest last."""
        with self._lock:
            self._load()
            line = line_of(conversation)
            return [r for r in self._held.values() if r.conversation == line]

    def all(self) -> list[Request]:
        with self._lock:
            self._load()
            return list(self._held.values())

    def left_off(self) -> list[Request]:
        """What she was in the middle of when she stopped running, the oldest first."""
        return [r for r in self.all() if r.stands == CUT_SHORT and r.tries and r.tries[-1].how == LEFT_OFF]

    def unfinished(self, conversation: str) -> Request | None:
        """The latest request of a conversation that leaves something to do."""
        return next((r for r in reversed(self.of(conversation)) if r.stands in UNFINISHED), None)

    def latest(self, conversation: str, *effects: str) -> Request | None:
        return next((r for r in reversed(self.of(conversation)) if not effects or r.effect in effects), None)

    async def changed(self, request: Request, *, stands: str | None = None, now: str | None = None) -> None:
        if stands is not None:
            request.stands = stands
        if now is not None:
            request.now = now
        request.changed_at = time.time()
        await self.keep()


_REQUESTS: RequestsOfHer | None = None
_REQUESTS_LOCK = checked_lock("understanding.requests_of_her.singleton")


def get_requests_of_her() -> RequestsOfHer:
    global _REQUESTS
    with _REQUESTS_LOCK:
        if _REQUESTS is None:
            _REQUESTS = RequestsOfHer()
        return _REQUESTS
