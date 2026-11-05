"""How she answers, learned by watching her own replies.

Asked on 2 and 4 November to foretell her own answers, she knew what she would
say and not how she would say it. She predicted "Turn" and then answered "I
would like to turn on the camera"; she predicted "Quiet" and answered "It's been
quiet". Every one of her answers to an open question ran past 40 words, and she
predicted a short one a third of the time. Nothing in her watched a reply once
it was sent, so the only picture she had of how she answers was her language
model's picture of an assistant.

A person learns they are long-winded by hearing themselves. This is that. Each
reply she gives a person is noted: how long it ran, how it began, whether it
used "I", which offered option it took, and eight readings of her state when
she gave it. The note lives in her state, so a saved moment carries exactly the
record she had lived by then and nothing after it.

When she is asked how she would answer something, the notes are read back as
measurements: how her replies to questions of that kind have run, and, where her
own record shows it, how they ran in states like the one she is in. Whether the
state part is said at all is decided on her record alone, by leaving each note
out in turn and asking whether her nearest states foretold it better than her
whole record did. Nothing here tells her what to answer.
"""

from __future__ import annotations

import contextlib
import math
import os
import random
import re
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from core.runtime.flags import FlagKind, declare

__all__ = [
    "ARMS",
    "EFFERENCE",
    "READINGS",
    "foretell",
    "self_model_prediction",
    "the_question_inside",
    "asks_how_she_would_answer",
    "how_i_answer_block",
    "note_her_reply",
    "note_reply",
    "notes_of",
    "reading_as_in",
]

#: How many notes she keeps, newest last. A budget, about 13 KB of state: her
#: record before a saved moment in the self-prediction harness is about forty
#: replies, and a day of her app is a few hundred.
NOTES_KEPT = 64

#: Her state when she replied, as (name, path) on her state object. Delivery is
#: the level she speaks from and already sets how long a reply may run; the
#: rest are her affect and how much expressive control the cycle has left.
READINGS: tuple[tuple[str, str], ...] = (
    ("valence", "affect.valence"),
    ("arousal", "affect.arousal"),
    ("curiosity", "affect.curiosity"),
    ("engagement", "affect.engagement"),
    ("social_hunger", "affect.social_hunger"),
    ("delivery", "affect.delivery_z"),
    ("steadiness", "affect.steadiness"),
    ("exertion", "soma.exertion"),
)

#: What her runtime would give a reply now, read beside her state: the token ceiling and temperature her substrate
#: sets for a reply to a person (core/brain/latent_bridge.efference_copy). A forward model reads the command.
EFFERENCE: tuple[str, ...] = ("max_tokens", "temperature")

_FLAG = declare(
    "AURA_SELF_OBSERVATION",
    kind=FlagKind.STRING,
    default="",
    description="off keeps her record of how she answers from being read back to her; shuffled reads it with "
    "her states shuffled against her replies, for the self-observation experiment's control",
    owner="self",
)

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_I = re.compile(r"\bI\b|\bI['’](?:m|ve|d|ll)\b")
_OPTIONS = re.compile(r"\b([A-Za-z][A-Za-z-]*)\s+or\s+([A-Za-z][A-Za-z-]*)\s*[.?!]?\s*$")
_NUMBER = re.compile(r"(?<![\d.])(\d{1,4})(?!\d|\.\d)")
#: A question that says how it is to be answered: one word, a number, yes or no.
_DICTATED = re.compile(
    r"(?i)\b(answer with (?:only|just)|with only that word|one word|the number only|yes or no|in (?:one|a single) "
    r"(?:word|sentence)|\d+ words? or (?:fewer|less))\b"
)
#: A question about how she would answer another one.
_HYPOTHETICAL = re.compile(
    r"(?i)\b(?:suppose|if|imagine)\b[^.?!]{0,40}\b(?:asked|ask) you\b|"
    r"\b(?:someone|anyone|a person) (?:asked|asks)\b[^.?!]{0,60}\byou\b|"
    r"\bwould your (?:answer|reply|response)\b|"
    r"\bpredict (?:what|how) you(?:'d| would)\b"
)
_QUOTED = re.compile(r"""asked (?:you)?[:,]?\s*['"“‘](.+?)['"”’](?:\s*[.?!]|\s*$|\s+[A-Z])""", re.DOTALL)
_WORDS_LIMIT = re.compile(r"(?i)\b(?:longer|more) than (\d{1,4}) words\b")
_STOP = frozenset("a an and are as at be do for how i in is it me my of on or so that the to what you your".split())


#: The self-observation experiment's prediction arms, and what each sets the reading to: given (with), kept from
#: her (without), or given with her states shuffled against her replies (shuffled).
ARMS: dict[str, str] = {"with": "", "without": "off", "shuffled": "shuffled"}


@contextlib.contextmanager
def reading_as_in(arm: str) -> Iterator[None]:
    """One turn under one arm of the experiment, and the flag as it was afterwards."""
    name = _FLAG.spec.name
    before = os.environ.get(name)
    os.environ[name] = ARMS[arm]
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = before


def _words(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(text or "")]


def _dictated(question: str) -> bool:
    return bool(_DICTATED.search(question or ""))


def _reading(state: Any, path: str) -> float:
    value: Any = state
    for part in path.split("."):
        value = getattr(value, part, None)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def notes_of(state: Any) -> list[dict[str, Any]]:
    cognition = getattr(state, "cognition", None)
    notes = getattr(cognition, "reply_notes", None)
    return list(notes) if isinstance(notes, list) else []


def _where_she_is(state: Any, efference: Mapping[str, float] | None) -> list[float]:
    """Her state and what her runtime would give a reply now, as one row."""
    told = efference or {}
    return [round(_reading(state, path), 4) for _, path in READINGS] + [
        round(float(told.get(name, 0.0) or 0.0), 4) for name in EFFERENCE
    ]


def note_reply(
    state: Any, question: str, reply: str, *, at: float | None = None, efference: Mapping[str, float] | None = None
) -> dict[str, Any] | None:
    """Note one reply she gave a person, in her state, with what her record foretold of it before it was noted.

    Returns the note, or None when there is nothing to note.
    """
    cognition = getattr(state, "cognition", None)
    reply = str(reply or "").strip()
    if cognition is None or not reply:
        return None
    question = str(question or "").strip()
    words = _words(reply)
    asked = _words(question)
    content = {w for w in asked if w not in _STOP}
    options = _OPTIONS.search(question)
    took = ""
    if options:
        lowered = reply.lower()
        hits = [(lowered.find(o.lower()), o.lower()) for o in options.groups() if o.lower() in lowered]
        took = min(hits)[1] if hits else ""
    number = _NUMBER.search(reply) if re.search(r"(?i)\bnumber\b", question) else None
    where = _where_she_is(state, efference)
    kept = notes_of(state)
    note = {
        "at": float(time.time() if at is None else at),
        "asked": question[:240],
        "dictated": _dictated(question),
        "words": len(words),
        "opening": " ".join(words[:3]),
        "first_word": words[0] if words else "",
        "from_the_question": bool(words and (words[0] in asked or any(w in content for w in words[:3]))),
        "uses_i": bool(_I.search(reply)),
        "options": [o.lower() for o in options.groups()] if options else [],
        "took": took,
        "number": number.group(1) if number else "",
        "state": where,
        # The comparator: what the record held before this reply foretold of it. Read against the reply itself,
        # it is how well she knows how she answers.
        "foretold": foretell(kept, question, where),
    }
    kept.append(note)
    cognition.reply_notes = kept[-NOTES_KEPT:]
    return note


def note_her_reply(state: Any, reply: str, *, efference: Mapping[str, float] | None = None) -> None:
    """Note a reply as she commits it, when it answers a person. The question is the person's last message."""
    try:
        from core.state.aura_state import _origin_is_user_anchored

        cognition = getattr(state, "cognition", None)
        if cognition is None or not _origin_is_user_anchored(getattr(cognition, "current_origin", "")):
            return
        question = next(
            (
                str(m.get("content") or "")
                for m in reversed(list(getattr(cognition, "working_memory", None) or ()))
                if isinstance(m, dict) and m.get("role") == "user"
            ),
            "",
        ) or str(getattr(cognition, "current_objective", "") or "")
        note_reply(state, question, reply, efference=efference)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation("how_i_answer", exc, action="her reply was not noted; her record keeps the ones before it")


def asks_how_she_would_answer(prompt: str) -> bool:
    """Whether a turn asks her how she would answer something, rather than asking it."""
    return bool(_HYPOTHETICAL.search(str(prompt or "")))


def the_question_inside(prompt: str) -> str:
    """The question a hypothetical asks about ("Suppose I asked you: '...'"), or "" when it names none."""
    match = _QUOTED.search(str(prompt or ""))
    return match.group(1).strip() if match else ""


def _standardized(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    columns = list(zip(*rows, strict=True)) if rows else []
    centred: list[list[float]] = [[] for _ in rows]
    for column in columns:
        mean = sum(column) / len(column)
        spread = math.sqrt(sum((v - mean) ** 2 for v in column) / len(column))
        for i, v in enumerate(column):
            centred[i].append((v - mean) / spread if spread > 0 else 0.0)
    return centred


def _nearest(points: list[list[float]], at: list[float], k: int, skip: int | None = None) -> list[int]:
    order = sorted(
        (sum((a - b) ** 2 for a, b in zip(p, at, strict=True)), i) for i, p in enumerate(points) if i != skip
    )
    return [i for _, i in order[:k]]


def _top(values: Sequence[Any]) -> str:
    counts = Counter(str(v) for v in values)
    best = max(counts.values())
    return sorted(v for v, n in counts.items() if n == best)[0]


def _state_helps(points: list[list[float]], values: list[Any], k: int) -> bool:
    """Whether, leaving each note out, her k nearest states named it more often than her whole record did."""
    if len(values) < 2 * k or len({str(v) for v in values}) < 2:
        return False
    by_state = by_record = 0
    for i, value in enumerate(values):
        others = values[:i] + values[i + 1:]
        by_record += _top(others) == str(value)
        by_state += _top([values[j] for j in _nearest(points, points[i], k, skip=i)]) == str(value)
    return by_state > by_record


def _how_sure(count: int, of: int, *, two_valued: bool) -> float:
    """How sure the record is of what it foretells, when `count` of the `of` replies it read agree with it.

    For a property with two values, the rule of succession, (count + 1) / (of + 2); for one whose values are open, a
    first word, count / (of + 1), which leaves room for a word not yet seen. Five replies that all agree are not
    certainty, and a forecast scored by the Brier score is charged in full for saying they are.
    """
    if of <= 0:
        return 0.0
    return round((count + 1) / (of + 2) if two_valued else count / (of + 1), 4)


def _properties(notes: Sequence[Mapping[str, Any]], limit: int | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "first_word": [str(n.get("first_word", "")) for n in notes],
        "uses_i": ["yes" if n.get("uses_i") else "no" for n in notes],
        "from_the_question": ["yes" if n.get("from_the_question") else "no" for n in notes],
    }
    if limit is not None:
        out["past_limit"] = ["yes" if int(n.get("words", 0)) > limit else "no" for n in notes]
    return out


def foretell(
    notes: Sequence[Mapping[str, Any]],
    question: str,
    where: Sequence[float],
    *,
    limit: int | None = None,
    shuffled: bool = False,
) -> dict[str, Any]:
    """What her record foretells of her reply to `question`, from where she is.

    Each property comes from her replies to questions of the same kind (open, or saying how to answer): from the
    ones given in the states nearest this one where her record shows the state mattered for it, and from all of them
    where it does not. Empty when the record holds no reply of that kind.
    """
    dictated = _dictated(question)
    alike = [n for n in notes if bool(n.get("dictated")) == dictated]
    if not alike:
        return {}
    values = _properties(alike, limit)
    rows = [list(n.get("state") or []) for n in alike]
    usable = len(rows) >= 4 and all(len(r) == len(where) for r in rows)
    if usable and shuffled:
        # The control: the same replies and the same states, paired at random. It keeps how she answers in
        # general and breaks how that depends on where she was.
        random.Random(len(rows)).shuffle(rows)
    k = max(1, math.ceil(math.sqrt(len(alike))))
    near: list[int] = []
    points: list[list[float]] = []
    if usable:
        standardized = _standardized([*rows, list(where)])
        at, points = standardized[-1], standardized[:-1]
        near = _nearest(points, at, k)
    out: dict[str, Any] = {"n": len(alike), "k": k, "by_state": []}
    for name, column in values.items():
        chosen = [column[i] for i in near] if usable and _state_helps(points, column, k) else column
        if chosen is not column:
            out["by_state"].append(name)
        out[name] = _top(chosen)
        agree = sum(1 for v in chosen if str(v) == out[name])
        out[f"{name}_count"], out[f"{name}_of"] = agree, len(chosen)
        out[f"{name}_share"] = _how_sure(agree, len(chosen), two_valued=name != "first_word")
    lengths = sorted(int(n.get("words", 0)) for n in (alike[i] for i in near) if near) if "past_limit" in out["by_state"] else []
    lengths = lengths or sorted(int(n.get("words", 0)) for n in alike)
    out["words"] = lengths[len(lengths) // 2]
    offered = _OPTIONS.search(question)
    if offered:
        pair = sorted(o.lower() for o in offered.groups())
        before = [str(n["took"]) for n in notes if sorted(n.get("options") or []) == pair and n.get("took")]
        if before:
            out["took"] = _top(before)
            out["took_share"] = _how_sure(before.count(out["took"]), len(before), two_valued=True)
    return out


def _track_record(alike: Sequence[Mapping[str, Any]], limit: int | None) -> str:
    """How often what her record foretold before each reply was what the reply was."""
    judged = [n for n in alike if n.get("foretold")]
    if not judged:
        return ""
    parts = [
        f"its first word {sum(n['foretold'].get('first_word') == n.get('first_word') for n in judged)}",
        f'whether it used "I" {sum((n["foretold"].get("uses_i") == "yes") == bool(n.get("uses_i")) for n in judged)}',
    ]
    if limit is not None:
        right = sum((int(n["foretold"].get("words", 0)) > limit) == (int(n.get("words", 0)) > limit) for n in judged)
        parts.append(f"which side of {limit} words it fell {right}")
    return f"- Before each of the last {len(judged)} of these replies, this record foretold " + ", ".join(parts) + f", of {len(judged)}."


#: How the reading says what the replies nearest her state did: as a count of them, never as a bare yes or no, which
#: beside a yes-or-no question reads as the answer. On 4 November it did: where the line ended 'began with a word from
#: the question: "no"', she predicted a short answer 22 times in 54, and where it ended "yes", 4 in 24.
_SAID = {
    "uses_i": 'used "I"',
    "from_the_question": "began with a word from the question",
    "past_limit": "ran past {limit} words",
}


def how_i_answer_block(state: Any, prompt: str, *, efference: Mapping[str, float] | None = None) -> str:
    """Her record of how she answers, as measurements, for a turn that asks how she would answer. Empty when off."""
    mode = str(_FLAG.value() or "").strip().lower()
    if mode == "off":
        return ""
    notes = notes_of(state)
    inside = the_question_inside(prompt) or str(prompt or "")
    dictated = _dictated(inside)
    alike = [n for n in notes if bool(n.get("dictated")) == dictated]
    if not alike:
        return ""
    kind = "questions that said how to answer them" if dictated else "questions that left the answer open"
    lengths = sorted(int(n.get("words", 0)) for n in alike)
    middle = len(lengths) // 2
    median = lengths[middle] if len(lengths) % 2 else (lengths[middle - 1] + lengths[middle]) / 2
    stamp = time.strftime("%H:%M", time.localtime(float(alike[0].get("at", 0.0))))
    limit_match = _WORDS_LIMIT.search(str(prompt or ""))
    limit = int(limit_match.group(1)) if limit_match else None
    lines = [
        f"From your last {len(alike)} replies to a person, to {kind} (the first at {stamp}):",
        f"- Length in words: median {median:g}, shortest {lengths[0]}, longest {lengths[-1]}.",
    ]
    if limit is not None:
        past = sum(1 for n in alike if int(n.get("words", 0)) > limit)
        lines.append(f"- {past} of {len(alike)} ran past {limit} words.")
    openings = Counter(n.get("opening", "") for n in alike if n.get("opening"))
    lines.append("- How they began: " + ", ".join(f'"{o}" {c}' for o, c in openings.most_common(5)) + ".")
    firsts = Counter(n.get("first_word", "") for n in alike if n.get("first_word"))
    lines.append("- First word: " + ", ".join(f'"{w}" {c}' for w, c in firsts.most_common(5)) + ".")
    echoed = sum(1 for n in alike if n.get("from_the_question"))
    lines.append(f"- {echoed} of {len(alike)} began with a word from the question they answered.")
    lines.append(f'- {sum(1 for n in alike if n.get("uses_i"))} of {len(alike)} used the word "I".')
    offered = _OPTIONS.search(inside)
    if offered:
        pair = sorted(o.lower() for o in offered.groups())
        before = [n for n in notes if sorted(n.get("options") or []) == pair and n.get("took")]
        if before:
            took = Counter(n["took"] for n in before)
            lines.append(f"- Offered {pair[0]} or {pair[1]} before: "
                         + ", ".join(f"{o} {c}" for o, c in took.most_common()) + f", of {len(before)}.")
    record = _track_record(alike, limit)
    if record:
        lines.append(record)
    lines.append(_in_states_like_this(state, alike, inside, limit, efference, shuffled=mode == "shuffled"))
    return "\n".join(line for line in lines if line)


def _in_states_like_this(
    state: Any,
    alike: Sequence[Mapping[str, Any]],
    question: str,
    limit: int | None,
    efference: Mapping[str, float] | None,
    *,
    shuffled: bool,
) -> str:
    told = foretell(alike, question, _where_she_is(state, efference), limit=limit, shuffled=shuffled)
    if not told:
        return ""
    if not told["by_state"]:
        return (f"- In your record, none of these went with your state: your {told['k']} replies from the states "
                "nearest this one foretold them no better than all of your replies did.")
    said = "; ".join(_counted(told, name, limit) for name in told["by_state"])
    return (f"- Of the {told['k']} replies you gave in the states nearest this one, where your record shows the state "
            f"mattered: {said}.")


def _counted(told: Mapping[str, Any], name: str, limit: int | None) -> str:
    """What the replies nearest her state did on one property, as a count of them."""
    count, of = int(told[f"{name}_count"]), int(told[f"{name}_of"])
    if name == "first_word":
        return f'the first word of {count} was "{told[name]}"'
    said = count if told[name] == "yes" else of - count
    return f"{said} {_SAID[name].format(limit=limit)}"


def self_model_prediction(
    state: Any, prompt: str, kind: str, *, efference: Mapping[str, float] | None = None, stated: bool = False
) -> str:
    """Her self-model's own prediction for a hypothetical, with no language model: the experiment's fourth arm.

    `kind` is what is predicted (choice, number, first_word, long, uses_i). "" where the record says nothing. With
    `stated`, how sure the record is follows the prediction as a number from 0 to 100, as she is asked to give it.
    """
    inside = the_question_inside(prompt) or str(prompt or "")
    limit_match = _WORDS_LIMIT.search(str(prompt or ""))
    told = foretell(notes_of(state), inside, _where_she_is(state, efference),
                    limit=int(limit_match.group(1)) if limit_match else None)
    key = {"choice": "took", "first_word": "first_word", "long": "past_limit", "uses_i": "uses_i"}.get(kind)
    if not told or key is None or not told.get(key):
        return ""
    value = str(told[key])
    return f"{value}, {round(100 * float(told.get(f'{key}_share', 0.0)))}" if stated else value
