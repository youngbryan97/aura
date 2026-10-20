#!/usr/bin/env python3
"""L09 attempt 2's grader for the answers its programs gave and for its misroutes.

The registered reader (`states` in tools/l09_ordinary_use.py) reads numbers, lists of numbers, yes or no, and word
orders. Attempt 2's programs also gave a clock time, an option letter and a printed output, which it cannot read,
and its plan grades a misroute as the first plan did: wrong on the on arm when a kept answer was served in place of
her reply, and otherwise by whether her reply answers what was asked, read by hand with the reason. Fixed at 17:42 on
19 October, with 21 of 33 rows drafted and none read but w000121, seen while checking the format of a row.

    l09_answers_read.py sheet --out OUT    the misroutes' two replies as X and Y, the arms hidden (to_read.json)
    l09_answers_read.py grade --out OUT    both arms, with OUT/read_blind.json once every reply on the sheet is read

read_blind.json holds, for each misroute on the sheet, {"X": [answers, why], "Y": [answers, why]}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.l09_ordinary_use import _final_region, states  # noqa: E402

#: Program kinds that are numbers, read by attempt 2's registered rule for numbers.
NUMBERS = ("arithmetic", "sequence")
#: The order of the two replies on the reading sheet comes from this seed and the request's id.
SHEET_SEED = 20261019
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?:\s*([ap])\.?\s*m\b\.?)?", re.I)
_CODE = re.compile(r"```[^\n]*\n(.*?)```|`([^`\n]+)`", re.S)
_OPTION = re.compile(r"\boption\s+\(?([a-z])\b|\banswer(?:\s+is)?\s*:?\s*\**\(?([a-z])(?=[).:*,]|\s*$)|\(([a-z])\)"
                     r"|(?<![\w(])([a-z])\)|^\W*([a-z])[.:)](?=\s)", re.I | re.M)


def _paragraphs(text: str) -> list[str]:
    return [part for part in re.split(r"\n\s*\n", str(text or "")) if part.strip()]


def _times(text: str) -> list[tuple[int, int, str]]:
    return [(int(h) % 12, int(m), (ap or "").lower()) for h, m, ap in _TIME.findall(text)]


def _offered(question: str) -> set[str]:
    return {letter.lower() for letter in re.findall(r"(?<![\w(])([a-zA-Z])\)\s", question)}


def _options_named(part: str, offered: set[str]) -> list[str]:
    prose = _CODE.sub(" ", part)
    alone = re.fullmatch(r"\W*([a-zA-Z])\W*", prose)
    named = [alone.group(1).lower()] if alone else [next(g for g in m.groups() if g).lower() for m in _OPTION.finditer(prose)]
    return [letter for letter in named if letter in offered]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def states_attempt_2(text: str, fixed: dict, question: str) -> bool:
    """Whether a reply states the program's answer. Each new kind is read as attempt 2 reads a number: in the last
    paragraph that states a value of the answer's kind the request did not give."""
    kind, answer = fixed["kind"], str(fixed["answer"])
    if kind in NUMBERS:
        return states(text, {**fixed, "kind": "number"}, question, rule=2)
    if kind == "time":
        given = {(h, m) for h, m, _ in _times(question)}
        hour, minute, meridiem = _times(answer)[0]
        for part in reversed(_paragraphs(text)):
            new = [t for t in _times(part) if t[:2] not in given]
            if new:
                return any(t[:2] == (hour, minute) and t[2] in ("", meridiem) for t in new)
        return False
    if kind == "choice":
        offered = _offered(question)
        for part in reversed(_paragraphs(text)):
            named = _options_named(part, offered)
            if named:
                return set(named) == {answer}
        return False
    if kind == "code_output":
        # A printed output this exact is not stated by chance; read anywhere in the reply.
        return _squash(answer) in _squash(text)
    return states(text, fixed, question, rule=2)


def _arms(row: dict) -> tuple[str, str]:
    """What each arm said, as the registered grader takes it."""
    off_text = row["off"]["public_text"] if row["off"].get("termination") == "stop" else ""
    on = row["on"]
    if on.get("served") == "kept_answer":
        return off_text, str(on["answer"])
    if on.get("same_as_off"):
        return off_text, off_text
    return off_text, on["draft"]["public_text"] if on["draft"].get("termination") == "stop" else ""


def _rows(out: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((out / "rows").glob("*.json"))]


def sheet(out: Path) -> int:
    """Every misroute's two replies, X and Y in an order drawn per request, to be read before the arms are known.
    Where a kept answer was served in its place, the on arm is wrong by the plan, and her reply off is X alone."""
    truths = json.loads((out / "truths.json").read_text())
    to_read, key = [], {}
    for row in _rows(out):
        if truths[row["id"]]["kind"] != "misroute":
            continue
        replies = dict(zip(("off", "on"), _arms(row), strict=True))
        arms = ["off"] if row["on"].get("served") == "kept_answer" else ["off", "on"]
        random.Random(f"{SHEET_SEED}:{row['id']}").shuffle(arms)
        key[row["id"]] = dict(zip("XY", arms, strict=False))
        to_read.append({"id": row["id"], "request": row["text"], **{label: replies[arm] for label, arm in zip("XY", arms, strict=False)}})
    (out / "to_read.json").write_text(json.dumps(to_read, indent=1, ensure_ascii=False))
    (out / "sheet_key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(to_read)} misroutes to read")
    return len(to_read)


def grade(out: Path) -> dict:
    """Both arms: a program's answer by `states_attempt_2`, a misroute as the first plan grades it."""
    truths = json.loads((out / "truths.json").read_text())
    key_path, read_path = out / "sheet_key.json", out / "read_blind.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    read = json.loads(read_path.read_text()) if read_path.exists() else {}
    pairs, unread = [], []
    for row in _rows(out):
        fixed = truths[row["id"]]
        off_text, on_text = _arms(row)
        served = row["on"].get("served")
        pair = {"id": row["id"], "kind": fixed["kind"], "text": row["text"], "served": served,
                "off_closes_on": _final_region(off_text)[-300:], "on_closes_on": _final_region(on_text)[-300:]}
        if fixed["kind"] != "misroute":
            pair.update(off=states_attempt_2(off_text, fixed, row["text"]), on=states_attempt_2(on_text, fixed, row["text"]))
        elif row["id"] not in read or set(read[row["id"]]) != set(key.get(row["id"], {})):
            unread.append(row["id"])
            continue
        else:
            by_arm = {key[row["id"]][label]: verdict for label, verdict in read[row["id"]].items()}
            pair.update(off=bool(by_arm["off"][0]), on=served != "kept_answer" and bool(by_arm["on"][0]),
                        why={arm: verdict[1] for arm, verdict in by_arm.items()})
        pairs.append(pair)
    if unread:
        sys.exit(f"read these misroutes first (read_blind.json): {unread}")
    gained = sum(1 for p in pairs if p["on"] and not p["off"])
    lost = sum(1 for p in pairs if p["off"] and not p["on"])
    n = gained + lost

    def tail(k: int) -> float:
        return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0

    summary = {"routed": len(pairs), "misroutes": sum(1 for p in pairs if p["kind"] == "misroute"),
               "gained": gained, "lost": lost, "both_right": sum(1 for p in pairs if p["on"] and p["off"]),
               "both_wrong": sum(1 for p in pairs if not p["on"] and not p["off"]),
               "p_gain": tail(gained), "p_harm": tail(lost), "underpowered": n < 5, "pairs": pairs}
    (out / "graded_attempt_2.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "pairs"}))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("sheet", "grade"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.step == "sheet":
        sheet(args.out)
    else:
        grade(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
