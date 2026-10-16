"""Kinds of problem she learns in use: solved examples she meets, grouped by kind, induced while she is idle.

Her kept procedures so far were induced from solved examples handed to her in
bulk (core/learning/procedures_from_solved_examples.py). Here the examples
are the ones she meets. One is noted only with an answer that is known rather
than believed, and with where it came from:

* ``a_person_gave_it``: a person stated the answer (``note_met`` from
  tools/teach_her_a_kind.py, or any caller a person's answer reaches);
* ``independent_programs_agreed``: programs written by independent strategies
  printed it and her verifier passed it (core/brain/reasoning_amplifier_v2.py).

Her own draft is not a source, however confident, and neither is a kept
procedure's answer: a procedure checked against its own answers proves
nothing.

A met example that a kind she keeps answers is one more known answer for that
kind. Its procedures run on it; when they agree, the kind's count of agreed
answers grows, and when they do not, the kind's procedures are retired from
her book with the counterexample. So the rule a procedure is kept by, that
every known answer agrees, holds after it is kept as well as before.

The rest are grouped by what they mean, in her embedding space
(core/learning/kinds_by_meaning.py). In the order she met them, a problem
joins the group it is nearest when it is at least as near as the loosest bar
of any kind in her book: as near as some kept kind's least typical sealed
problem came to that kind. No group is looser than a kind already shown to be
one. A group is ready when its hash split gives each half at least as many
problems as a request shows.

A ready group's procedure is written one request per step, with the requests,
checks and lines of an induction run (``request_for``, ``judge``,
``next_on_line``), as an autonomy job while she is idle; a person's turn
preempts the generation. A function that agrees with every known answer of
its group joins her book as a new kind, with its meaning and a few examples,
and from then on answers requests of that kind. A group whose rounds are
spent waits until it has grown.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from core.learning.procedures_from_solved_examples import (
    KEPT_BOOK_RELATIVE,
    Candidate,
    Family,
    Outcome,
    ProcedureBook,
    SolvedExample,
    answers_agree,
    dumps,
    judge,
    kept_procedure_book,
    next_on_line,
    request_for,
    signature_of,
    split_examples,
)

__all__ = ["SOURCES", "SHOTS", "ROUNDS", "a_person_answered", "learn_one_step", "mark_programs_checked",
           "met_examples", "next_to_check", "note_met", "note_met_async", "note_to_check_async"]

#: Where a known answer may come from. Nothing else is noted.
SOURCES = frozenset({"a_person_gave_it", "independent_programs_agreed"})
#: Examples a request shows: BIG-Bench Hard's three-shot protocol, as every induction run used.
SHOTS = 3
#: Requests a line gets per set of examples: the three rounds of the first G09 attempt's protocol
#: (docs/evidence/G09_PROCEDURES_PREREGISTRATION_2026-10-08.md).
ROUNDS = 3

MET_RELATIVE = "procedures/met_examples.jsonl"
RETIRED_RELATIVE = "procedures/retired.jsonl"
LEARNING_RELATIVE = "procedures/learning_in_use.json"
#: Requests she answered in chat that independent programs might settle, checked one at a time while she is idle.
TO_CHECK_RELATIVE = "procedures/to_check.jsonl"

Embed = Callable[[list[str]], np.ndarray]
ProposeOne = Callable[[str], Awaitable[str]]


#: How a person says what the answer is, in the turn after she answered: "the answer is 374", "no, it's 374",
#: "actually it should be 374", "no, 374", "yes, it's 374". A bare "no" or "yes" with nothing stated is not
#: an answer.
_STATED = re.compile(
    r"^\s*(?:(?:no|nope|wrong|incorrect|not quite|close|actually|yes|yep|right|correct|exactly)\b"
    r"[\s,.!:;\u2013\u2014-]*)?"
    r"(?:(?:the\s+)?(?:correct\s+|right\s+|real\s+|actual\s+)?answer\s*(?:is|was|should be|'s|:|=)"
    r"|it\s*(?:is|'s|should be|should have been|was)|(?:that\s+)?should be)\s*:?\s*(?P<value>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_DENIED_THEN_VALUE = re.compile(
    r"^\s*(?:no|nope|wrong|incorrect|actually)\b[\s,.!:;\u2013\u2014-]+(?P<value>.+)$", re.IGNORECASE | re.DOTALL
)
#: "It's not 22" says what the answer is not.
_DENIES = re.compile(r"^(?:not|never|no)\b", re.IGNORECASE)
#: Where a stated value ends and a reason or a new request begins.
_VALUE_ENDS = re.compile(r"\s+(?:because|since|as|not)\b|\s+[-\u2014]\s+|[.!?](?:\s|$)|\n", re.IGNORECASE)
_WORD = re.compile(r"[\w()]+")


def a_person_answered(request: str, her_reply: str, said: str) -> str | None:
    """The answer a person states, in this turn, for the request she just answered; None when the turn states none.

    Only an answer of a checkable shape is taken: one with a number in it, or
    one made only of words the request itself holds (the words of a list she
    was asked to sort, an option it offered). "No, it's fine" or "it's
    raining" states no answer to "what is 15% of 340?", and a known answer that
    is wrong would retire a kept procedure that is right
    (``_check_against_kept``), so a turn that cannot be read is left alone.
    """
    if not str(her_reply or "").strip() or len(str(request or "").split()) < 3:
        return None
    text = str(said or "").strip()
    found = _STATED.match(text) or _DENIED_THEN_VALUE.match(text)
    if found is None:
        return None
    value = _VALUE_ENDS.split(found.group("value"), maxsplit=1)[0].strip().strip("\"'`*").rstrip(",;:").strip()
    if not value or len(value) > 200 or _STATED.match(value) or _DENIES.match(value):
        return None
    if any(ch.isdigit() for ch in value):
        return value
    asked = {w.casefold() for w in _WORD.findall(str(request))}
    words = [w.casefold() for w in _WORD.findall(value)]
    return value if words and all(w in asked for w in words) else None


def _root() -> Path:
    from core.utils.paths import aura_data_dir

    return aura_data_dir()


def _key(problem: str) -> str:
    return hashlib.sha256(" ".join(problem.split()).casefold().encode()).hexdigest()[:24]


def _met_line(problem: str, answer: str, source: str) -> str | None:
    problem, answer = str(problem or "").strip(), str(answer or "").strip()
    if source not in SOURCES or not problem or not answer:
        return None
    return json.dumps({"key": _key(problem), "problem": problem, "answer": answer, "source": source,
                       "at": round(time.time(), 3)}) + "\n"


def note_met(problem: str, answer: str, source: str) -> bool:
    """Note a solved example she met; False when the source is not one that knows the answer."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    line = _met_line(problem, answer, source)
    if line is None:
        return False
    path = _root() / MET_RELATIVE
    with local_internal_governed_scope("examples_she_meets.note_met", domain="file_write"):
        gateway = get_file_write_gateway()
        gateway.ensure_directory(path.parent, source="examples_she_meets.note_met")
        gateway.append_text(path, line, source="examples_she_meets.note_met")
    return True


async def note_met_async(problem: str, answer: str, source: str) -> bool:
    """``note_met`` for async callers: the append goes through the gateway's async lane."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    line = _met_line(problem, answer, source)
    if line is None:
        return False
    path = _root() / MET_RELATIVE
    with local_internal_governed_scope("examples_she_meets.note_met", domain="file_write"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(path.parent, source="examples_she_meets.note_met")
        await gateway.append_text_async(path, line, source="examples_she_meets.note_met")
    return True


async def note_to_check_async(problem: str) -> bool:
    """Queue a request she answered, for independent programs to settle while she is idle.

    Her answer in the turn is her own and is not a known answer. When programs
    written by independent strategies print one answer and her verifier passes
    it (core/brain/reasoning_amplifier_v2.py), it is noted as a solved example
    she met, from the source ``independent_programs_agreed``. That search took
    minutes on her model, too long for a turn, so it runs after it.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    problem = str(problem or "").strip()
    if not problem:
        return False
    line = json.dumps({"key": _key(problem), "problem": problem, "at": round(time.time(), 3)}) + "\n"
    path = _root() / TO_CHECK_RELATIVE
    with local_internal_governed_scope("examples_she_meets.note_to_check", domain="file_write"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(path.parent, source="examples_she_meets.note_to_check")
        await gateway.append_text_async(path, line, source="examples_she_meets.note_to_check")
    return True


def next_to_check() -> dict[str, Any] | None:
    """The first queued request no idle step has settled or set aside yet; None when there is none."""
    path = _root() / TO_CHECK_RELATIVE
    if not path.exists():
        return None
    done = _read_learning().get("programs_checked", {})
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("key") and row["key"] not in done:
            return row
    return None


async def mark_programs_checked(key: str, outcome: str) -> None:
    """Record what the idle check of one queued request came to, so it is not checked again."""
    learning = _read_learning()
    learning.setdefault("programs_checked", {})[str(key)] = str(outcome)
    await _write(LEARNING_RELATIVE, dumps(learning))


def met_examples() -> list[dict[str, Any]]:
    """Every example she met, in the order she met them; a problem met twice keeps its first answer."""
    path = _root() / MET_RELATIVE
    if not path.exists():
        return []
    seen: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("key") and row["key"] not in seen:
            seen[row["key"]] = row
    return list(seen.values())


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


def groups_of(rows: Sequence[dict[str, Any]], bar: float, *, documents: Embed, queries: Embed
              ) -> list[list[dict[str, Any]]]:
    """Group met examples by meaning, in the order met: each joins the nearest group it is at least ``bar`` near.

    A group's direction is the mean of its problems embedded as documents and a
    problem is tested embedded as a query, the two sides a kind's bar was
    measured between (core/learning/kinds_by_meaning.py).
    """
    if not rows:
        return []
    problems = [r["problem"] for r in rows]
    docs, asks = _unit(documents(problems)), _unit(queries(problems))
    members: list[list[int]] = []
    sums: list[np.ndarray] = []
    for index in range(len(rows)):
        if sums:
            centroids = _unit(np.stack(sums))
            scores = centroids @ asks[index]
            best = int(np.argmax(scores))
            if float(scores[best]) >= bar:
                members[best].append(index)
                sums[best] = sums[best] + docs[index]
                continue
        members.append([index])
        sums.append(docs[index].copy())
    return [[rows[i] for i in group] for group in members]


def _loosest_bar(book: ProcedureBook) -> float | None:
    bars = [float(m.nearest_required) for m in book.meanings.values()]
    return min(bars) if bars else None


def _examples(group: Sequence[dict[str, Any]]) -> list[SolvedExample]:
    return [SolvedExample(r["key"], r["problem"], r["answer"]) for r in group]


def ready(group: Sequence[dict[str, Any]]) -> bool:
    """Whether a group's hash split gives each half at least as many problems as a request shows."""
    pool, sealed = split_examples(_examples(group))
    return len(pool) >= SHOTS and len(sealed) >= SHOTS


def _candidate_state(candidate: Candidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {"family": candidate.family, "line": candidate.line, "round": candidate.round, "code": candidate.code,
            "sha256": candidate.sha256, "shown_keys": candidate.shown_keys, "pool_total": candidate.pool_total,
            "sealed_total": candidate.sealed_total, "pool": [asdict(o) for o in candidate.pool],
            "sealed": [asdict(o) for o in candidate.sealed]}


def _candidate_from(state: dict[str, Any] | None) -> Candidate | None:
    if not state:
        return None
    return Candidate(family=state["family"], line=state["line"], round=state["round"], code=state["code"],
                     sha256=state["sha256"], shown_keys=list(state["shown_keys"]), pool_total=state["pool_total"],
                     sealed_total=state["sealed_total"], pool=[Outcome(**o) for o in state["pool"]],
                     sealed=[Outcome(**o) for o in state["sealed"]])


def _read_learning() -> dict[str, Any]:
    path = _root() / LEARNING_RELATIVE
    if not path.exists():
        return {"checked": {}, "groups": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    value.setdefault("checked", {})
    value.setdefault("groups", {})
    return value


async def _write(relative: str, text: str, *, append: bool = False) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    path = _root() / relative
    with local_internal_governed_scope("examples_she_meets.learn", domain="file_write"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(path.parent, source="examples_she_meets.learn")
        if append:
            await gateway.append_text_async(path, text, source="examples_she_meets.learn")
        else:
            await gateway.write_text_async(path, text, source="examples_she_meets.learn")


async def _keep(book: ProcedureBook) -> None:
    """Write her book where her live route reads it, without blocking her loop."""
    await _write(KEPT_BOOK_RELATIVE, dumps(book.to_json()))
    from core.learning import procedures_from_solved_examples as procedures

    procedures._kept_cache.clear()


async def _check_against_kept(book: ProcedureBook, rows: Sequence[dict[str, Any]], learning: dict[str, Any],
                              query: Callable[[str], Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run each new met example through the kind she keeps for it; return those no kept kind answers.

    An example of a kept kind that its procedures agree with raises the kind's
    count. One worded as the kind's own problems are (found by words) that
    they answer otherwise retires the kind's procedures: it is a
    counterexample. One found only by meaning is in other words than they
    were kept on, so a procedure that misreads or declines it has not been
    shown wrong, and it is left unanswered instead (live chat, 16 October:
    a person's gcd answer would have retired the right gcd procedure, which
    cannot read "What's the greatest common divisor of ...?"). Each example
    is checked once.
    """
    import asyncio

    report: dict[str, Any] = {"agreed_with_kept": 0, "retired": []}
    changed = False
    unanswered: list[dict[str, Any]] = []
    for row in rows:
        verdict = learning["checked"].get(row["key"])
        if verdict == "unanswered":
            unanswered.append(row)
            continue
        if verdict is not None:
            continue
        kind, _similarity = book.route(row["problem"])
        by_words = kind is not None
        if kind is None and query is not None:
            vector = await asyncio.to_thread(query, row["problem"])
            kind = book.route_by_meaning(vector)[0] if vector is not None else None
        if kind is None or not book.procedures.get(kind):
            learning["checked"][row["key"]] = "unanswered"
            unanswered.append(row)
            continue
        receipt = await asyncio.to_thread(book._run_kind, kind, row["problem"], {"family": kind})
        returned = receipt.get("answer")
        if returned is not None and answers_agree(str(returned), row["answer"]):
            book.agreed[kind] = int(book.agreed.get(kind, 0)) + 1
            learning["checked"][row["key"]] = f"agreed:{kind}"
            report["agreed_with_kept"] += 1
        elif returned is None or not by_words:
            learning["checked"][row["key"]] = "unanswered"
            unanswered.append(row)
            continue
        else:
            retired = {"kind": kind, "problem": row["problem"], "known": row["answer"], "source": row["source"],
                       "returned": returned, "declined": receipt.get("declined"), "procedures": book.procedures[kind],
                       "agreed_before": book.agreed.get(kind, 0), "at": round(time.time(), 3)}
            await _write(RETIRED_RELATIVE, json.dumps(retired) + "\n", append=True)
            book.procedures.pop(kind, None)
            book.agreed.pop(kind, None)
            learning["checked"][row["key"]] = f"retired:{kind}"
            report["retired"].append(kind)
        changed = True
    if changed:
        await _keep(book)
    return unanswered, report


async def learn_one_step(propose_one: ProposeOne, *, documents: Embed | None, queries: Embed | None,
                         book: ProcedureBook | None = None) -> dict[str, Any]:
    """One bounded step of learning in use: check new examples, then at most one request for one ready group.

    ``propose_one`` asks her model for one reply; ``documents`` and ``queries``
    are her embedder's two sides, or None when no embedder is held (then only
    the check by words runs and nothing is grouped).
    """
    import asyncio

    rows = met_examples()
    if not rows:
        return {"status": "nothing_met"}
    book = book if book is not None else (kept_procedure_book() or ProcedureBook({}, {}))
    learning = _read_learning()

    def one_query(text: str) -> Any:
        return queries([text])[0] if queries is not None else None

    unanswered, report = await _check_against_kept(book, rows, learning, one_query if queries else None)
    out: dict[str, Any] = {"met": len(rows), "unanswered": len(unanswered), **report}
    bar = _loosest_bar(book)
    if documents is None or queries is None or bar is None:
        await _write(LEARNING_RELATIVE, dumps(learning))
        return {**out, "status": "no_embedder_held" if documents is None else "no_kind_to_measure_a_group_by"}

    grouped = await asyncio.to_thread(groups_of, unanswered, bar, documents=documents, queries=queries)
    groups = [g for g in grouped if ready(g)]
    chosen = None
    for group in groups:
        state = learning["groups"].get(group[0]["key"])
        keys = sorted(r["key"] for r in group)
        if state is None or (state.get("spent") and state.get("keys") != keys and not state.get("kept")):
            state = {"keys": keys, "attempt": 0, "requests": 0, "previous": None, "spent": False, "kept": None}
            learning["groups"][group[0]["key"]] = state
        if not state["spent"] and not state["kept"]:
            chosen = (group, state)
            break
    if chosen is None:
        await _write(LEARNING_RELATIVE, dumps(learning))
        return {**out, "status": "no_group_ready", "groups": len(groups)}

    group, state = chosen
    name = f"met_{group[0]['key'][:12]}"
    pool, sealed = split_examples(_examples(group))
    family = Family(name=name, pool=pool, sealed=sealed, signature=signature_of(pool, sealed))
    previous = _candidate_from(state["previous"])
    from core.learning.procedures_from_solved_examples import her_kept_steps

    steps = her_kept_steps()
    request, shown = request_for(family, 0, previous, state["attempt"], shots=SHOTS, steps=steps)
    text = await propose_one(request)
    candidate = await asyncio.to_thread(judge, family, 0, state["requests"], shown, text, shots=SHOTS, steps=steps)
    state["requests"] += 1
    out.update({"status": "requested", "group": name, "size": len(group), "request": state["requests"]})
    if candidate is not None and candidate.admitted:
        from core.learning.kinds_by_meaning import meanings_of

        book.signatures[name] = family.signature
        book.procedures[name] = [candidate.code]
        book.agreed[name] = len(pool) + len(sealed)
        book.meanings.update(await asyncio.to_thread(
            meanings_of, {name: ([e.problem for e in pool], [e.problem for e in sealed])},
            documents=documents, queries=queries))
        book.examples[name] = [{"problem": e.problem, "answer": e.answer} for e in pool[:SHOTS]]
        await _keep(book)
        state.update({"kept": name, "previous": None})
        for row in group:
            learning["checked"][row["key"]] = f"agreed:{name}"
        out.update({"status": "kept", "agreed_known": book.agreed[name]})
    else:
        following, state["attempt"] = next_on_line(candidate, state["attempt"])
        state["previous"] = _candidate_state(following)
        state["spent"] = state["requests"] >= ROUNDS
        if candidate is not None:
            out["agreed"] = [candidate.pool_agreed + candidate.sealed_agreed, len(pool) + len(sealed)]
    await _write(LEARNING_RELATIVE, dumps(learning))
    return out
