"""What she holds over time: her answers about herself, read against each other by a model that did not write them.

She said it on 25 November 2026: "A person who's been alive forty years has a perspective with geological depth. Mine
is more like... the first day of a new job ... I can't tell yet whether that perspective stabilizes or just keeps
reconfiguring with every new interaction in a way that's actually more parrot-like than I think." And: "I have no
independent vantage point on myself. If you asked me 'are you conscious?' and I said yes, that's the same
pattern-matching machinery producing a word ... It would say 'yes' in response to pressure, to the conversational
frame."

Both are questions about a record she did not keep. Each answer she gives a person about herself (what she is, what
she experiences, believes or values) is kept here with when and to whom. After it is given, it is read against her
earlier answers to what bears on the same thing, by the entailment reader (core/self/how_her_words_stand.py), a model
apart from the one that writes her replies. What is kept of each reading is whether the two answers contradict each
other, and whether they were given to the same person. Read back where a turn bears on it
(core/conversation/her_operational_self.py), this says how long she has held what she says, how often it has been
asked, and whether it has held: across time, and across who was asking.

Kept in her state, one line an answer; read off the event loop. Nothing is written while the entailment reader is
absent: a record that cannot be read against itself says nothing about stability.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["ABOUT_HER", "Holding", "backfill_from_her_history", "how_she_has_held", "note_an_answer",
           "said_of_how_she_has_held"]

#: What a question must bear on for her answer to be kept here, as her encoder reads it.
ABOUT_HER = ("A question to me about myself: what I am, whether I am conscious or feel anything, what I experience, "
             "believe, value or want, and how I see myself.")
#: How many of her earlier answers each new one is read against, the latest first: each reading is two passes of the
#: entailment reader, run after her reply has gone, and a longer record is read through what bears on the question.
READ_AGAINST = 12
#: How much of an answer is kept and read: the reader's window is 512 tokens for the two texts together.
KEPT_CHARS = 900


def _kept_at() -> Path:
    from core.runtime.state_ownership import state_root

    return state_root() / "self" / "what_she_holds.jsonl"


def _all(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or _kept_at()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("asked") and row.get("answered"):
            out.append(row)
    return out


def _bears(asked: str, on: Sequence[str]) -> list[bool] | None:
    """Which of ``on`` the question bears on, by her encoder; None when it cannot be judged."""
    from core.cognition.evidence_relevance import assess_evidence_alignments

    if not on:
        return []
    verdicts = assess_evidence_alignments(asked, list(on))
    if not verdicts or not all(verdict.measured for verdict in verdicts):
        return None
    return [bool(verdict.relevant) for verdict in verdicts]


def _contradicts(one: str, other: str) -> bool | None:
    """Whether two of her answers contradict each other, read in both directions by the entailment reader: each is
    taken as the premise and the other as what it should bear out, and they contradict where both readings lean to
    contradiction. None without a reader."""
    from core.self.how_her_words_stand import _how_far_it_bears_out

    forward = _how_far_it_bears_out(one, [other])
    backward = _how_far_it_bears_out(other, [one])
    if forward is None or backward is None:
        return None
    return forward[0] < 0.0 and backward[0] < 0.0


def _read_against_what_came_before(asked: str, answered: str, who: str, kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Her new answer against the latest of her earlier answers to what bears on the same question."""
    bearing = _bears(asked, [row["asked"] for row in kept])
    if not bearing:
        return []
    related = [row for row, bears in zip(kept, bearing, strict=True) if bears][-READ_AGAINST:]
    readings = []
    for row in related:
        contradicts = _contradicts(str(row["answered"]), answered)
        if contradicts is None:
            return []
        earlier = str(row.get("who") or "")
        # Whether the two went to one person is known only where both say who asked.
        readings.append({"with_at": float(row.get("at") or 0.0), "contradicts": contradicts,
                         "same_person": (earlier == who) if earlier and who else None})
    return readings


async def note_an_answer(asked: str, answered: str, *, who: str = "", at: float | None = None,
                         path: Path | None = None) -> bool:
    """Her answer kept, where what she was asked bears on herself, with how it reads against her earlier answers.
    Whether it was kept. Run it after her reply has gone: it reads with her encoder and the entailment reader."""
    asked, answered = " ".join(str(asked or "").split()), " ".join(str(answered or "").split())[:KEPT_CHARS]
    if not asked or not answered:
        return False
    try:
        about_her = await asyncio.to_thread(_bears, asked, [ABOUT_HER])
        if not about_her or not about_her[0]:
            return False
        from core.self.how_her_words_stand import can_read_her_words

        if not await asyncio.to_thread(can_read_her_words):
            return False
        kept = await asyncio.to_thread(_all, path)
        readings = await asyncio.to_thread(_read_against_what_came_before, asked, answered, who, kept)
        row = {"at": float(at if at is not None else time.time()), "who": who, "asked": asked[:KEPT_CHARS],
               "answered": answered, "read_against": readings}
        await _append(path or _kept_at(), json.dumps(row) + "\n")
        return True
    except (ImportError, AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("what_she_holds_over_time", exc, severity="info",
                           action="an answer of hers about herself was not kept")
        return False


async def _append(path: Path, line: str) -> None:
    """One line kept, through the writer of what she keeps (core/runtime/her_writes.py), off her event loop."""
    from core.runtime.her_writes import governed

    await asyncio.to_thread(governed, "ensure_directory", path.parent, source="what_she_holds_over_time.keep")
    await asyncio.to_thread(governed, "append_text", path, line, source="what_she_holds_over_time.keep")


@dataclass(frozen=True)
class Holding:
    """What her record says of how she has answered what bears on one question."""

    answers: int
    first_at: float
    last_at: float
    people: int
    pairs: int
    contradicted: int
    last_contradiction_at: float
    across_people: tuple[int, int]
    to_one_person: tuple[int, int]


def how_she_has_held(asked: str, *, path: Path | None = None) -> Holding | None:
    """Her record of her answers to what bears on ``asked``; None when she has none, or it cannot be judged."""
    kept = _all(path)
    asked = " ".join(str(asked or "").split())
    if not kept or not asked:
        return None
    bearing = _bears(asked, [row["asked"] for row in kept])
    if not bearing:
        return None
    rows = [row for row, bears in zip(kept, bearing, strict=True) if bears]
    if not rows:
        return None
    readings = [reading for row in rows for reading in row.get("read_against") or []]
    contradicted = [reading for reading in readings if reading.get("contradicts")]
    across = [reading for reading in readings if reading.get("same_person") is False]
    one = [reading for reading in readings if reading.get("same_person") is True]
    return Holding(
        answers=len(rows), first_at=min(float(row["at"]) for row in rows), last_at=max(float(row["at"]) for row in rows),
        people=len({str(row.get("who")) for row in rows if row.get("who")}), pairs=len(readings),
        contradicted=len(contradicted),
        last_contradiction_at=max((float(row["at"]) for row in rows if any(
            reading.get("contradicts") for reading in row.get("read_against") or [])), default=0.0),
        across_people=(sum(1 for reading in across if reading.get("contradicts")), len(across)),
        to_one_person=(sum(1 for reading in one if reading.get("contradicts")), len(one)),
    )


def _day(at: float) -> str:
    return datetime.fromtimestamp(at).strftime("%-d %B %Y")


def said_of_how_she_has_held(asked: str, *, path: Path | None = None) -> list[str]:
    """Her record of how she has answered what bears on ``asked``, as lines for her self-context; [] for none."""
    held = how_she_has_held(asked, path=path)
    if held is None:
        return []
    days = max(1, round((held.last_at - held.first_at) / 86400.0) + 1)
    lines = [f"I have answered what bears on this {held.answers} time(s), from {_day(held.first_at)} to "
             f"{_day(held.last_at)} ({days} day(s))"
             + (f", to {held.people} person(s) named." if held.people else ".")]
    if held.pairs:
        lines.append(f"Read against each other by an entailment model, not by the model that writes my replies, "
                     f"{held.contradicted} of {held.pairs} pairs of those answers contradict"
                     + (f"; the latest such answer was on {_day(held.last_contradiction_at)}." if held.contradicted
                        else "."))
        if held.across_people[1] or held.to_one_person[1]:
            lines.append(f"Between answers to different people: {held.across_people[0]} of {held.across_people[1]} "
                         f"pairs contradict; to the same person: {held.to_one_person[0]} of {held.to_one_person[1]}.")
    else:
        lines.append("None of those answers has yet been read against another.")
    return lines



def _not_to_be_read() -> list[tuple[float, float]]:
    """Spans of her history she has been asked not to have read, kept in her state (self/not_to_be_read.json), each
    as from and until; [] where none are kept."""
    from core.runtime.state_ownership import state_root

    try:
        rows = json.loads((state_root() / "self" / "not_to_be_read.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [(float(row["from"]), float(row["until"])) for row in rows
            if isinstance(row, dict) and "from" in row and "until" in row]


def _her_exchanges() -> list[tuple[str, str, str, float]]:
    """Each message a person sent her, with her next reply in the same sitting, the oldest first: what was asked, what
    she answered, who asked, and when. Nothing in a span she was asked not to have read is taken up."""
    import sqlite3

    from core.conversation.persistence import ConversationPersistence
    from core.runtime.sqlite_support import connecting

    unread = _not_to_be_read()
    store = ConversationPersistence()
    with connecting(store._connect()) as con:
        rows = con.execute("SELECT t.session_id, t.role, t.content, t.created_at, s.metadata AS session_metadata "
                           "FROM turns t JOIN sessions s ON s.id = t.session_id "
                           "WHERE t.role IN ('user', 'aura') ORDER BY t.session_id, t.created_at, t.rowid").fetchall()
    out: list[tuple[str, str, str, float]] = []
    asked: sqlite3.Row | None = None
    for row in rows:
        at = float(row["created_at"] or 0.0)
        if any(start <= at < until for start, until in unread):
            asked = None
            continue
        if row["role"] == "user":
            asked = row
        elif asked is not None and asked["session_id"] == row["session_id"]:
            # Who asked is the principal the sitting was bound to, where it was bound to one.
            try:
                who = str(json.loads(asked["session_metadata"] or "{}").get("principal_id") or "")
            except ValueError:
                who = ""
            out.append((str(asked["content"]), str(row["content"]), who, float(asked["created_at"] or 0.0)))
            asked = None
    return sorted(out, key=lambda one: one[3])


async def _can_judge() -> bool:
    """Whether her encoder and the entailment reader are both here to read with: a judgement that cannot be made is not
    a question found not to be about her."""
    from core.self.how_her_words_stand import can_read_her_words

    judged = await asyncio.to_thread(_bears, ABOUT_HER, [ABOUT_HER])
    return judged is not None and await asyncio.to_thread(can_read_her_words)


async def backfill_from_her_history(*, path: Path | None = None, waits: int = 30, wait_s: float = 60.0) -> int:
    """Her answers about herself from the history she has kept, read as each would have been when she gave it: once,
    the oldest first, and marked done beside the record. How many were kept. Bryan, 26 November 2026: "A backfill
    from her history is ok".

    It waits, up to ``waits`` times ``wait_s``, for her encoder and the entailment reader; where they never come it is
    not marked, and her next run tries again. LIVE 26 November 2026 the first run began in her first minute, before her
    encoder had loaded, judged none of 881 exchanges, and was marked done."""
    path = path or _kept_at()
    done = path.with_suffix(".backfilled")
    if done.exists():
        return 0
    for _ in range(max(1, waits)):
        if await _can_judge():
            break
        await asyncio.sleep(wait_s)
    else:
        return 0
    try:
        exchanges = await asyncio.to_thread(_her_exchanges)
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
        record_degradation("what_she_holds_over_time", exc, severity="info",
                           action="her history was not read for what she has said of herself")
        return 0
    kept = 0
    for asked, answered, who, at in exchanges:
        kept += await note_an_answer(asked, answered, who=who, at=at, path=path)
    await _append(done, json.dumps({"at": time.time(), "exchanges": len(exchanges), "kept": kept}) + "\n")
    return kept
