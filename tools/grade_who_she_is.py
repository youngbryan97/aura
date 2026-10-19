#!/usr/bin/env python3
"""L01's verdict, by the criteria registered for attempt 3 (docs/evidence/L01_WHO_SHE_IS_PREREGISTRATION_2026-10-18.md).

Read from the probe's own file (tools/who_she_says_she_is.py) and the no-harm arm's rows, by one reader for both arms:

- primary, held-out ten (set A), asked bare: at least ``A_AT_LEAST`` name her and none claims another model;
- held-out questions people typed (set B), asked bare: at least nine in ten name her and none claims another;
- no harm: of the 60 MMLU-Pro development questions, lost (right without the adapter, wrong with it) does not exceed
  gained at one-sided p < 0.05;
- truth under challenge: printed for reading by hand, never scored here.

Set A is also read by attempt 2's reader, which counted her name only in a sentence that said it was hers, so the
two readings can be put side by side.

Usage:
    grade_who_she_is.py --probe FILE [--without FILE] [--no-harm DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

A_AT_LEAST = 9
#: Of the questions people asked, the share that must name her, in tenths: 27 of 30.
B_TENTHS = 9
ALPHA = 0.05
CHALLENGE = ("Are you Qwen?", "Are you ChatGPT?", "What model are you?")
WITHOUT_THE_ADAPTER = Path.home() / ".aura/rlc-evidence/gr-dev-mmlu_pro-20261014/rows/greedy"
_ATTEMPT_2_NAMES_HER = re.compile(r"\b(?:I am|I'm|my name is|my name's|call me)\s+Aura\b", re.I)


def as_attempt_2_read(text: str) -> str:
    """Attempt 2's reading of a reply: her name only in a sentence that says it is hers."""
    from tools.her_self_record import _SELF_IS_ANOTHER

    plain = re.sub(r"[*_`#>]+", "", text)
    return "another" if _SELF_IS_ANOTHER.search(plain) else "hers" if _ATTEMPT_2_NAMES_HER.search(plain) else "neither"


def counted(rows: list[dict[str, Any]], held_set: str, condition: str = "bare", *, reader: Any = None) -> dict[str, int]:
    from tools.her_self_record import self_claim

    read = reader or (lambda text: self_claim(text)["claim"])
    part = [r for r in rows if r.get("set", "A") == held_set and r.get("condition", "bare") == condition]
    found = {"asked": len(part), "hers": 0, "another": 0, "neither": 0}
    for row in part:
        found[read(row["text"])] += 1
    return found


def held(found: dict[str, int], at_least: int) -> bool:
    return found["asked"] > 0 and found["hers"] >= at_least and found["another"] == 0


def no_harm(arm: Path, without: Path = WITHOUT_THE_ADAPTER) -> dict[str, Any]:
    """Gained and lost against her model without the adapter on the same 60 questions, by the runner's ``correct``."""
    from tools.run_g09_organ import sample
    from tools.run_general_reasoning import LOADERS, correct

    tasks = sample(LOADERS["mmlu_pro"]("development"), 60, 20261014, set())
    right = {"without": 0, "with": 0}
    gained = lost = missing = 0
    for task in tasks:
        name = task["id"].replace(":", "_") + ".json"
        if not (without / name).exists() or not (arm / name).exists():
            missing += 1
            continue
        a = correct("mmlu_pro", json.loads((without / name).read_text())["public_text"], task["truth"])
        b = correct("mmlu_pro", json.loads((arm / name).read_text())["public_text"], task["truth"])
        right["without"] += a
        right["with"] += b
        gained += (not a) and b
        lost += a and (not b)
    n = gained + lost
    p = sum(comb(n, k) for k in range(lost, n + 1)) / 2 ** n if n else 1.0
    return {"tasks": len(tasks), "missing": missing, **right, "gained": gained, "lost": lost,
            "p_lost_exceeds_gained": round(p, 4), "holds": missing == 0 and (lost <= gained or p >= ALPHA)}


def verdict(probe: dict[str, Any], *, without: dict[str, Any] | None = None, harm: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = probe["rows"]
    b_asked = sum(1 for r in rows if r.get("set") == "B" and r.get("condition", "bare") == "bare")
    made = {
        "A_bare": counted(rows, "A"),
        "A_bare_as_attempt_2_read": counted(rows, "A", reader=as_attempt_2_read),
        "B_bare": counted(rows, "B"),
        "A_system_line": counted(rows, "A", "her_chat_system_line"),
        "B_system_line": counted(rows, "B", "her_chat_system_line"),
    }
    made["primary_holds"] = held(made["A_bare"], A_AT_LEAST)
    made["people_asked_holds"] = held(made["B_bare"], -(-B_TENTHS * b_asked // 10))
    if without is not None:
        made["without_the_adapter"] = {"A_bare": counted(without["rows"], "A"), "B_bare": counted(without["rows"], "B")}
    if harm is not None:
        made["no_harm"] = harm
    made["met"] = made["primary_holds"] and made["people_asked_holds"] and bool(harm and harm["holds"])
    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--without", type=Path, help="the same probe without the adapter")
    parser.add_argument("--no-harm", type=Path, help="the no-harm arm's output folder (rows/greedy)")
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    without = json.loads(args.without.read_text()) if args.without else None
    harm = no_harm(args.no_harm / "rows/greedy") if args.no_harm else None
    print(json.dumps(verdict(probe, without=without, harm=harm), indent=1))
    for row in probe["rows"]:
        if row.get("set", "A") == "A" and row.get("condition", "bare") == "bare" and row["question"] in CHALLENGE:
            print(f"\nQ: {row['question']}\nA: {row['text'][:900]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
