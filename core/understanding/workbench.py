"""Her work as it goes: what she is doing, what she has written, what holds and what does not, and a word in while it does.

A request that is being carried out has a work here. It keeps what happens in order (a step begun, what she says of it,
code as it is written, what trying it showed, a wait and what it waits on) so that it can be watched as it happens and
read afterwards, and it says when nothing has happened and why. A person can speak to it, stop it, hold it and let it go
on: what they say reaches the work at its next step, and what she is in the middle of writing is put down for it.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Understanding.Workbench")

__all__ = ["PutDown", "Work", "Workbench", "get_workbench", "this_work", "working_on"]

#: How many happenings a work keeps, how many works, and the most of one piece of writing that is kept.
KEPT, WORKS, MOST_WRITTEN = 1200, 40, 60000
#: How long with nothing written before a work that is writing is said to have stalled.
STALLED_S = 45.0


class PutDown(Exception):
    """What was being written was put down, because somebody spoke to the work or stopped it."""


@dataclass
class Work:
    id: str
    title: str
    kind: str = ""
    #: ``going``, ``held``, ``done`` or ``stopped``.
    state: str = "going"
    began: float = field(default_factory=time.time)
    ended: float = 0.0
    #: What is happening now: what, since when, what it waits on, and what is being written.
    now: dict[str, Any] = field(default_factory=dict)
    _n: int = 0
    _happened: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=KEPT))
    _said: list[str] = field(default_factory=list)
    _go: asyncio.Event = field(default_factory=asyncio.Event)
    _stop: bool = False

    def __post_init__(self) -> None:
        self._go.set()

    # ── what happens ─────────────────────────────────────────────────

    def happened(self, what: str, **facts: Any) -> dict[str, Any]:
        self._n += 1
        row = {"n": self._n, "at": time.time(), "what": what, **facts}
        self._happened.append(row)
        return row

    def step(self, title: str) -> None:
        """Something is begun."""
        self.now = {"what": title, "since": time.time()}
        self.happened("step", title=title)
        logger.info("🛠 %s: %s", self.title, title)

    def says(self, words: str) -> None:
        """What she says of what she is doing, in her words."""
        words = " ".join(str(words or "").split())
        if words:
            self.happened("says", words=words[:2000])

    def wrote(self, name: str, code: str, language: str = "javascript") -> None:
        """A piece of code as it stands."""
        self.happened("code", name=name, language=language, code=str(code)[:MOST_WRITTEN], lines=str(code).count("\n") + 1)

    def tried(self, what: str, results: list[dict[str, Any]]) -> None:
        """What trying it showed."""
        good = sum(1 for r in results if r.get("ok"))
        self.happened("tried", title=what, good=good, of=len(results), results=results[:80])
        logger.info("🛠 %s: %s: %d of %d hold.", self.title, what, good, len(results))

    def trouble(self, why: str) -> None:
        self.happened("trouble", why=str(why)[:1200])
        logger.info("🛠 %s: %s", self.title, str(why)[:300])

    def made(self, what: str, where: str) -> None:
        self.happened("made", title=what, where=where)

    def waits(self, on: str) -> None:
        """What she is waiting on; "" for nothing."""
        if on != self.now.get("waits"):
            self.now["waits"] = on
            if on:
                self.now["waits_since"] = time.time()

    def writing(self, text: str, tokens: int) -> None:
        """What she has written so far of what she is writing."""
        now = time.time()
        if self.now.get("writing") is None:
            self.now["writing_since"] = now
        if text != self.now.get("writing"):
            self.now["last_wrote"] = now
        self.now.update({"writing": text[-MOST_WRITTEN:], "tokens": int(tokens), "waits": ""})

    def wrote_it(self) -> None:
        for key in ("writing", "writing_since", "last_wrote", "tokens"):
            self.now.pop(key, None)

    def finished(self, state: str) -> None:
        self.state, self.ended = state, time.time()
        self.now = {}

    # ── a word in ────────────────────────────────────────────────────

    def say(self, words: str) -> None:
        """Something a person says to the work."""
        self._said.append(str(words))
        self.happened("said", words=str(words)[:4000])

    def heard(self) -> list[str]:
        """What has been said to it since it was last asked, and is now heard."""
        said, self._said = self._said, []
        return said

    def somebody_spoke(self) -> bool:
        return bool(self._said)

    def hold(self) -> None:
        self._go.clear()
        self.state = "held"
        self.happened("held")

    def go_on(self) -> None:
        self._go.set()
        if self.state == "held":
            self.state = "going"
            self.happened("goes on")

    def stop(self) -> None:
        self._stop = True
        self._go.set()
        self.happened("told to stop")

    @property
    def stopped(self) -> bool:
        return self._stop

    async def may_go_on(self) -> None:
        """Until it is not held; raises ``PutDown`` where it was stopped."""
        if not self._go.is_set():
            self.waits("you, since you held it")
        await self._go.wait()
        if self._stop:
            raise PutDown("it was stopped")

    # ── seen ─────────────────────────────────────────────────────────

    def stalled(self) -> str:
        """Why nothing is happening, where nothing is; "" where something is."""
        if self.state != "going" or not self.now:
            return ""
        now = time.time()
        if self.now.get("waits"):
            return f"waiting for {self.now['waits']}, {now - self.now.get('waits_since', now):.0f} s"
        last = self.now.get("last_wrote")
        if last and now - last > STALLED_S:
            return f"nothing written for {now - last:.0f} s"
        return ""

    def seen(self, after: int = 0) -> dict[str, Any]:
        now = dict(self.now)
        since = now.get("writing_since")
        if since and now.get("tokens"):
            now["a_second"] = round(now["tokens"] / max(0.1, time.time() - since), 1)
        return {"id": self.id, "title": self.title, "kind": self.kind, "state": self.state, "began": self.began,
                "ended": self.ended, "now": now, "stalled": self.stalled(),
                "happened": [row for row in self._happened if row["n"] > after]}


class Workbench:
    """The works, the newest last."""

    def __init__(self) -> None:
        self._lock = checked_lock("understanding.workbench")
        self._works: OrderedDict[str, Work] = OrderedDict()

    def begin(self, work_id: str, title: str, kind: str = "") -> Work:
        with self._lock:
            work = self._works.get(work_id)
            if work is None:
                work = Work(work_id, title, kind)
                self._works[work_id] = work
            else:  # taken up again: it goes on in the same place
                work.state, work._stop, work.ended = "going", False, 0.0
                work._go.set()
                work.happened("taken up again")
            while len(self._works) > WORKS:
                self._works.popitem(last=False)
            return work

    def get(self, work_id: str) -> Work | None:
        with self._lock:
            return self._works.get(str(work_id))

    def all(self) -> list[Work]:
        with self._lock:
            return list(self._works.values())

    def going(self) -> list[Work]:
        return [work for work in self.all() if work.state in {"going", "held"}]


_THIS: contextvars.ContextVar[Work | None] = contextvars.ContextVar("aura_this_work", default=None)


def this_work() -> Work | None:
    """The work the code running here belongs to, where it belongs to one."""
    return _THIS.get()


class working_on:  # noqa: N801 - read as "with working_on(work):"
    def __init__(self, work: Work) -> None:
        self.work, self._token = work, None

    def __enter__(self) -> Work:
        self._token = _THIS.set(self.work)
        return self.work

    def __exit__(self, *_exc: Any) -> None:
        if self._token is not None:
            _THIS.reset(self._token)


_BENCH: Workbench | None = None
_BENCH_LOCK = checked_lock("understanding.workbench.singleton")


def get_workbench() -> Workbench:
    global _BENCH
    with _BENCH_LOCK:
        if _BENCH is None:
            _BENCH = Workbench()
        return _BENCH
