#!/usr/bin/env python3
"""L09 attempt 3, step 1: requests people make that a program settles, by kind, in the order they were typed.

The plan is docs/design-docs/L09_NEXT_LEARNING_WHERE_SHE_IS_WRONG.md. A request is drawn when the whole of what the
person typed is one question of a kind below, with what it is about stated in it; the kind's program gives the
answer. The kinds and their programs are fixed here before any request is drawn and before any answer is read.

    l09_where_she_is_wrong.py draw --out OUT     every such request in WildChat-4.8M's user turns, by kind, in the
                                                 order typed, each with its operands and the program's answer
    l09_where_she_is_wrong.py wrong --out OUT --model DIR    step 1: each kind's first requests put to her persona
                                                 27B bare (her app down), graded; the kinds she is wrong on, kept

Not drawn from: the requests L09's first two attempts routed. Left out as in attempt 2: turns not in English, toxic,
redacted or flagged by moderation.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WILDCHAT_48M = Path.home() / ".aura" / "datasets" / "wildchat-4.8m"
USED_BEFORE = (Path.home() / ".aura" / "rlc-evidence" / "l09-ordinary-use-20261018" / "requests.json",
               Path.home() / ".aura" / "rlc-evidence" / "l09-ordinary-use-2-20261019" / "requests.json")
#: How many of attempt 2's requests (in its order) it routed; the rest were never put to her.
ATTEMPT_2_ROUTED = 17_317
#: Longer than this, a request is a task with a question in it, not the question.
MOST_CHARS = 200
#: Step 1: a kind is used when it has at least this many requests, and its first this many are put to her model.
FEWEST_DRAWN, FIRST_ASKED = 30, 20
#: How near a number that is not whole must be stated to its exact answer, relative.
NEAR = Fraction(1, 1000)

_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_LEAD = (r"(?:(?:hi|hello|hey)[,!.]?\s+)?(?:(?:please|pls|can you|could you|quick question:?)\s+)?"
         r"(?:(?:tell me|calculate|compute|work out|solve|find|answer)\s*:?\s+)?(?:(?:what|how much)(?:'s|\s+is|\s+are)\s+)?")
_TAIL = r"\s*(?:[?.!=]+|\s+please)*\s*"
_MONTHS = "|".join(calendar.month_name[1:] + [m for m in calendar.month_abbr[1:] if m != "May"])
_DATE = (rf"(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTHS})\.?,?"
         rf"\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})")


def _number(text: str) -> Fraction:
    return Fraction(text.replace(",", ""))


def _date(text: str) -> date | None:
    """A date written with its month as a word, or as YYYY-MM-DD; a date that could be read two ways is not one."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.replace(",", " ").replace(".", " "))
    iso = re.fullmatch(r"\s*(\d{4})-(\d{2})-(\d{2})\s*", text)
    words = text.lower().split()
    try:
        if iso:
            return date(int(iso[1]), int(iso[2]), int(iso[3]))
        months = {name.lower(): n for n, name in enumerate(calendar.month_name) if name}
        months.update({name.lower(): n for n, name in enumerate(calendar.month_abbr) if name})
        month = next(months[w] for w in words if w in months)
        day, year = (int(w) for w in words if w.isdigit())
        return date(year, month, day)
    except (StopIteration, ValueError):
        return None


def _plain(value: Fraction) -> str:
    """A number as a person writes it: whole, or decimal to as many places as it has, at most six, rounded half up."""
    if value.denominator == 1:
        return str(value.numerator)
    places = next((p for p in range(1, 7) if (value * 10**p).denominator == 1), 6)
    with localcontext() as exact:
        exact.prec = 80
        shown = (Decimal(value.numerator) / Decimal(value.denominator)).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
    return format(shown, "f")


# Units: how much of the kind's base each is, exactly (international inch, pound and US gallon).
_LENGTH = {"mm": Fraction(1, 1000), "cm": Fraction(1, 100), "m": Fraction(1), "km": Fraction(1000),
           "in": Fraction(254, 10000), "ft": Fraction(3048, 10000), "yd": Fraction(9144, 10000), "mi": Fraction(1609344, 1000)}
_MASS = {"mg": Fraction(1, 1000), "g": Fraction(1), "kg": Fraction(1000), "lb": Fraction(45359237, 100000),
         "oz": Fraction(45359237, 1600000), "st": Fraction(45359237 * 14, 100000)}
_VOLUME = {"ml": Fraction(1, 1000), "l": Fraction(1), "gal": Fraction(3785411784, 10**9), "qt": Fraction(3785411784, 4 * 10**9),
           "pt": Fraction(3785411784, 8 * 10**9), "cup": Fraction(3785411784, 16 * 10**9), "floz": Fraction(3785411784, 128 * 10**9)}
_UNIT_WORDS = {
    "mm": "millimet(?:er|re)s?|mm", "cm": "centimet(?:er|re)s?|cm", "m": "met(?:er|re)s?|m", "km": "kilomet(?:er|re)s?|kms?",
    "in": "inch(?:es)?|in", "ft": "feet|foot|ft", "yd": "yards?|yds?", "mi": "miles?|mi",
    "mg": "milligrams?|mg", "g": "grams?|g", "kg": "kilograms?|kilos?|kgs?", "lb": "pounds?|lbs?", "oz": "ounces?|oz",
    "st": "stones?|st", "ml": "millilit(?:er|re)s?|ml", "l": "lit(?:er|re)s?|l", "gal": "(?:us )?gallons?|gal",
    "qt": "quarts?|qt", "pt": "pints?|pt", "cup": "cups?", "floz": "fluid ounces?|fl\\.? ?oz",
    "c": "(?:degrees? )?(?:celsius|centigrade|c)|°c", "f": "(?:degrees? )?(?:fahrenheit|f)|°f", "k": "kelvins?|k",
}
_UNIT = "|".join(f"(?P<{{side}}_{key}>{words})" for key, words in sorted(_UNIT_WORDS.items(), key=lambda kv: -len(kv[1])))


def _which_unit(match: re.Match, side: str) -> str:
    return next(key for key in _UNIT_WORDS if match.groupdict().get(f"{side}_{key}"))


def _converted(amount: Fraction, source: str, target: str) -> Fraction | None:
    kelvin = {"c": lambda v: v + Fraction(27315, 100), "f": lambda v: (v - 32) * Fraction(5, 9) + Fraction(27315, 100),
              "k": lambda v: v}
    back = {"c": lambda v: v - Fraction(27315, 100), "f": lambda v: (v - Fraction(27315, 100)) * Fraction(9, 5) + 32,
            "k": lambda v: v}
    if source in kelvin and target in back:
        return back[target](kelvin[source](amount))
    for table in (_LENGTH, _MASS, _VOLUME):
        if source in table and target in table:
            return amount * table[source] / table[target]
    return None


@dataclass(frozen=True)
class Kind:
    name: str
    patterns: tuple[re.Pattern[str], ...]
    settle: Callable[[re.Match[str]], dict | None]


def _kind(name: str, bodies: tuple[str, ...], settle: Callable[[re.Match[str]], dict | None]) -> Kind:
    """A kind: each way of asking it, the whole request; one name a group, so each way is its own pattern."""
    return Kind(name, tuple(re.compile(rf"\s*{_LEAD}{body}{_TAIL}", re.IGNORECASE) for body in bodies), settle)


def _arithmetic(op: Callable[[Fraction, Fraction], Fraction]) -> Callable[[re.Match[str]], dict | None]:
    def settle(match: re.Match[str]) -> dict | None:
        a, b = _number(match["a"]), _number(match["b"])
        try:
            value = op(a, b)
        except ZeroDivisionError:
            return None
        return {"operands": [match["a"], match["b"]], "answer": _plain(value), "exact": str(value)}
    return settle


def _days_between(match: re.Match[str]) -> dict | None:
    first, second = _date(match["d1"]), _date(match["d2"])
    if first is None or second is None:
        return None
    days = abs((second - first).days) + (1 if match["inclusive"] else 0)
    return {"operands": [first.isoformat(), second.isoformat(), bool(match["inclusive"])], "answer": str(days)}


def _weekday(match: re.Match[str]) -> dict | None:
    day = _date(match["d"])
    return None if day is None else {"operands": [day.isoformat()], "answer": calendar.day_name[day.weekday()]}


def _letters(match: re.Match[str]) -> dict | None:
    letter, word = match["letter"].lower(), match["word"]
    return {"operands": [letter, word], "answer": str(word.lower().count(letter))}


def _conversion(match: re.Match[str]) -> dict | None:
    source, target = _which_unit(match, "u1"), _which_unit(match, "u2")
    value = _converted(_number(match["n"]), source, target) if source != target else None
    return None if value is None else {"operands": [match["n"], source, target], "answer": _plain(value), "exact": str(value)}


_N = rf"(?P<a>{_NUMBER})"
_M = rf"(?P<b>{_NUMBER})"
KINDS: tuple[Kind, ...] = (
    _kind("product", (rf"{_N}\s*(?:\*|x|×|times|multiplied by)\s*{_M}", rf"multiply\s+{_N}\s+(?:by|and|with)\s+{_M}",
                      rf"the product of\s+{_N}\s+and\s+{_M}"), _arithmetic(lambda a, b: a * b)),
    _kind("quotient", (rf"{_N}\s*(?:/|÷|divided by)\s*{_M}", rf"divide\s+{_N}\s+by\s+{_M}"), _arithmetic(lambda a, b: a / b)),
    _kind("sum", (rf"{_N}\s*(?:\+|plus)\s*{_M}", rf"add\s+{_N}\s+(?:and|to)\s+{_M}", rf"the sum of\s+{_N}\s+and\s+{_M}"),
          _arithmetic(lambda a, b: a + b)),
    _kind("difference", (rf"{_N}\s+(?:-|minus)\s+{_M}", rf"subtract\s+{_M}\s+from\s+{_N}"), _arithmetic(lambda a, b: a - b)),
    _kind("percent_of", (rf"{_N}\s*(?:%|percent|per cent)\s+of\s+{_M}",), _arithmetic(lambda a, b: a * b / 100)),
    _kind("what_percent", (rf"{_N}\s+is\s+what\s+percent(?:age)?\s+of\s+{_M}", rf"what\s+percent(?:age)?\s+of\s+{_M}\s+is\s+{_N}",
                           rf"what\s+percent(?:age)?\s+is\s+{_N}\s+of\s+{_M}"), _arithmetic(lambda a, b: a / b * 100)),
    _kind("days_between", (rf"how many days (?:are there |is it |are |is )?(?:between|from)\s+(?P<d1>{_DATE})\s+(?:and|to|until|till)"
                           rf"\s+(?P<d2>{_DATE})(?P<inclusive>,?\s+inclusive(?:ly)?)?",), _days_between),
    _kind("weekday", (rf"what day (?:of the week )?(?:was|is|will be|falls on|fell on|will)\s+(?P<d>{_DATE})(?:\s+(?:be|fall on))?",),
          _weekday),
    _kind("letters", (r"how many (?:times does )?(?:the letter |letter )?['\"]?(?P<letter>[a-z])['\"]?(?:'s|s)?\s+"
                      r"(?:are (?:there )?|is (?:there )?|appears? |occurs? )?in\s+(?:the word\s+)?['\"]?(?P<word>[a-z]{2,})['\"]?"
                      r"(?:\s+(?:appear|occur))?",), _letters),
    _kind("conversion", (rf"(?:convert\s+)?(?P<n>{_NUMBER})\s*(?:{_UNIT.format(side='u1')})\s+(?:to|in|into)\s+"
                         rf"(?:{_UNIT.format(side='u2')})",), _conversion),
)


def settled(text: str) -> dict | None:
    """The kind and the program's answer, when the whole of ``text`` is one question a program settles."""
    if len(text) > MOST_CHARS:
        return None
    for kind in KINDS:
        for pattern in kind.patterns:
            match = pattern.fullmatch(text)
            found = kind.settle(match) if match else None
            if found is not None:
                return {"kind": kind.name, **found}
    return None


def _used_before() -> set[str]:
    from tools.what_people_ask_who_it_is import spelled_out

    used: set[str] = set()
    for path in USED_BEFORE:
        rows = json.loads(path.read_text(encoding="utf-8"))
        rows = rows[:ATTEMPT_2_ROUTED] if path.parent.name.startswith("l09-ordinary-use-2") else rows
        used.update(spelled_out(str(row.get("text") or "")) for row in rows)
    return used


def draw(out: Path, wildchat: Path = WILDCHAT_48M) -> dict:
    import pyarrow.parquet as pq

    from tools.what_people_ask_who_it_is import spelled_out

    used, found = _used_before(), {}
    # A user turn here carries no time of its own; its conversation's is when it was begun (WildChat-4.8M's schema).
    columns = ("conversation_hash", "conversation", "toxic", "redacted", "timestamp")
    for shard in sorted(wildchat.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=2000, columns=list(columns)):
            for conversation_hash, conversation, toxic, redacted, begun in zip(*(batch.column(c).to_pylist() for c in columns), strict=True):
                if toxic or redacted:
                    continue
                for turn in conversation or []:
                    text = " ".join(str(turn.get("content") or "").split())
                    if (turn.get("role") != "user" or turn.get("language") != "English" or turn.get("toxic")
                            or turn.get("redacted") or not text or len(text) > MOST_CHARS):
                        continue
                    key = spelled_out(text)
                    if key in used or key in found:
                        continue
                    answer = settled(text)
                    if answer is not None:
                        typed = turn.get("timestamp") or begun
                        found[key] = {"text": text, "typed": typed.isoformat() if typed else "",
                                      "conversation": conversation_hash, "turn": turn.get("turn_identifier"), **answer}
        print(json.dumps({"shard": shard.name, "found": len(found)}), flush=True)
    by_kind: dict[str, list[dict]] = {}
    for row in sorted(found.values(), key=lambda row: row["typed"]):
        rows = by_kind.setdefault(row["kind"], [])
        rows.append({"id": f"{row['kind']}-{len(rows):04d}", **row})
    out.mkdir(parents=True, exist_ok=True)
    summary = {kind: len(rows) for kind, rows in by_kind.items()}
    (out / "drawn.json").write_text(json.dumps({"by_kind": by_kind, "counts": summary}, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return summary


_STATED = re.compile(r"(?<![\w.])[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def _stated(region: str) -> list[Fraction]:
    """The numbers a region states, each with its sign and its decimals, thousands commas allowed."""
    return [Fraction(n.replace("−", "-").replace(",", "")) for n in _STATED.findall(region)]


def right(row: dict, reply: str) -> bool:
    """Whether a reply states the program's answer, by the rule fixed in the preregistration."""
    from tools.l09_ordinary_use import _final_region, _last_new_number_paragraph

    if row["kind"] == "weekday":
        named = set(re.findall(r"\b(" + "|".join(calendar.day_name) + r")\b", _final_region(reply), re.IGNORECASE))
        return {day.capitalize() for day in named} == {row["answer"]}
    stated = _stated(_last_new_number_paragraph(reply, row["text"]))
    exact = Fraction(row.get("exact") or row["answer"])
    if exact.denominator == 1:
        return exact in stated
    return any(abs(n - exact) <= NEAR * abs(exact) for n in stated)


def wrong(out: Path, model_path: Path, max_tokens: int, exclusive: bool = True) -> dict:
    """Step 1: each kind's first requests in the order typed, put to her persona 27B bare as attempt 2's draft step
    puts a request (no system text, her template, thinking on, greedy), and graded; kept, the kinds she is wrong on
    at least as often as right. A reply that did not finish states nothing."""
    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane
    from tools.run_g05_public_answers import decode_public

    drawn = json.loads((out / "drawn.json").read_text(encoding="utf-8"))["by_kind"]
    asked = {kind: rows[:FIRST_ASKED] for kind, rows in drawn.items() if len(rows) >= FEWEST_DRAWN}
    replies = out / "step1"
    replies.mkdir(exist_ok=True)
    with standalone_model_lane(owner_id="l09-where-she-is-wrong", model_path=str(model_path), purpose="evaluation",
                               preemptible=False, require_exclusive=exclusive, allow_owner_eviction=exclusive,
                               metadata={"tool": Path(__file__).name}):
        model, tokenizer = load(str(model_path))
        for rows in asked.values():
            for row in rows:
                path = replies / f"{row['id']}.json"
                if path.exists():
                    continue
                reply = decode_public(model, tokenizer, [{"role": "user", "content": row["text"]}], max_tokens=max_tokens,
                                      thinking=True)
                path.write_text(json.dumps({**row, "reply": reply}, indent=1), encoding="utf-8")
                print(json.dumps({"id": row["id"], "termination": reply.get("termination")}), flush=True)
    by_kind = {}
    for kind, rows in asked.items():
        graded = []
        for row in rows:
            reply = json.loads((replies / f"{row['id']}.json").read_text(encoding="utf-8"))["reply"]
            text = reply.get("public_text", "") if reply.get("termination") == "stop" else ""
            graded.append(right(row, text))
        by_kind[kind] = {"right": sum(graded), "wrong": len(graded) - sum(graded), "kept": len(graded) - sum(graded) >= sum(graded)}
    (out / "step1.json").write_text(json.dumps(by_kind, indent=1), encoding="utf-8")
    print(json.dumps(by_kind))
    return by_kind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("draw", "wrong"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wildchat", type=Path, default=WILDCHAT_48M)
    parser.add_argument("--model", type=Path, help="wrong: her persona 27B")
    parser.add_argument("--max-tokens", type=int, default=32768, help="wrong: as attempt 2's draft step")
    parser.add_argument("--not-exclusive", action="store_true", help="wrong: share the model lane (a test, not a run)")
    args = parser.parse_args()
    if args.step == "draw":
        draw(args.out, args.wildchat)
    else:
        wrong(args.out, args.model, args.max_tokens, exclusive=not args.not_exclusive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
