#!/usr/bin/env python3
"""Attempt 3's held-out math requests graded with one grader for both arms.

Attempt 3 graded her drafts with the harness's DeepMind rule (tools/run_g09_organ.grade_dm: the boxed answer
equal to the target as text once spaces and dollar signs are dropped, or equal in value when both are plain
numbers) and the kept arm's answers with each kind's own grader (preregistration 3, amendment of 15 October
13:55). A kept procedure answers in DeepMind's notation; her drafts answer in LaTeX, so a right draft written
as \\frac{1}{72}, 231w + 10205 or "Yes" was marked wrong. Found 16 October while choosing live-check requests.

Here both arms are graded alike: right when DeepMind's rule or MATH-500's own equivalence grader
(grading.grader.grade_answer) accepts the answer, with a true-or-false kind read as yes or no, a list compared
item by item, and "a_k = ..." read as the expression it states. A request whose kept procedure declined keeps
her draft, as the kept arm did.

Usage:
    g09_fair_regrade.py --output FILE
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = Path("~/.aura/rlc-evidence").expanduser()
MATH500 = Path("~/.aura/benchmarks/math500").expanduser()
_YES, _NO = {"yes", "true"}, {"no", "false"}


def _plain(value: str) -> str:
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", str(value)).replace("\\ ", " ").replace("$", "").strip()
    return re.sub(r"^[A-Za-z]\w*(?:_\w+)?\s*=\s*", "", value).strip()


def _truthy(value: str) -> bool | None:
    """Yes or no, read from the word an answer opens with ("Yes, it is composite." is yes)."""
    match = re.match(r"(yes|no|true|false)\b", _plain(value).strip(" .").lower())
    word = match.group(1) if match else ""
    return True if word in _YES else False if word in _NO else None


def _without_base(value: str) -> str:
    """A number written with its base as a subscript, without it: -1011_2 is -1011."""
    match = re.fullmatch(r"(-?[0-9A-Fa-f]+)_\{?\(?\d{1,2}\)?\}?", value.strip())
    return match.group(1) if match else value


def _factorization(value: str) -> list[str] | None:
    """The distinct primes of an answer written as a product of powers (2^3 \\cdot 11 is 2, 11), in order."""
    plain = re.sub(r"\\(?:cdot|times)|[·×*]", " x ", _plain(value).replace("{", "").replace("}", ""))
    factors = [piece.strip() for piece in plain.split(" x ")]
    if len(factors) < 2 or not all(re.fullmatch(r"\d+(?:\^\d+)?", f) for f in factors):
        return None
    bases = [f.split("^")[0] for f in factors]
    return list(dict.fromkeys(bases))


def _one(answer: str, truth: str) -> bool:
    from grading.grader import grade_answer

    from tools.run_g09_organ import grade_dm

    if grade_dm(f"\\boxed{{{answer}}}", truth)[0]:
        return True
    mine, theirs = _without_base(_plain(answer)), str(truth).strip()
    if mine != _plain(answer) and grade_dm(f"\\boxed{{{mine}}}", truth)[0]:
        return True
    if _truthy(theirs) is not None:
        return _truthy(mine) is not None and _truthy(mine) == _truthy(theirs)
    for form in (theirs, theirs.replace("**", "^").replace("*", "")):
        try:
            if grade_answer(mine, form):
                return True
        except Exception:  # noqa: BLE001 - a grader that cannot read an answer has not accepted it
            continue
    return False


def fair(text: str, truth: object) -> bool:
    """Whether an answer is right, by one rule for both arms."""
    from tools.run_g09_organ import math_answer

    answer = math_answer(text) if "\\boxed" in str(text) or "\n" in str(text) else str(text)
    if answer is None:
        return False
    truth = str(truth)
    if "," in truth:
        mine = [x for x in re.split(r"\s*,\s*", _plain(answer)) if x]
        if len(mine) == 1:
            mine = _factorization(answer) or mine
        theirs = [x for x in re.split(r"\s*,\s*", truth) if x]
        return len(mine) == len(theirs) and all(_one(m, t) for m, t in zip(mine, theirs, strict=True))
    return _one(answer, truth)


def mcnemar(gained: int, lost: int) -> float:
    n = gained + lost
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2 ** n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(MATH500))
    from tools.induce_g09_procedures import problems

    keys = ("requests", "adopted", "drafts_as_registered", "kept_as_registered", "gained_as_registered",
            "lost_as_registered", "drafts_fair", "kept_fair", "gained_fair", "lost_fair")
    total = dict.fromkeys(keys, 0)
    by_kind = {}
    for directory in sorted(glob.glob(str(EVIDENCE / "g09-a3-drafts-dm_*-20261014"))):
        kind = directory.split("g09-a3-drafts-")[1].rsplit("-", 1)[0]
        truth = {row["key"]: row["truth"] for row in problems(kind, "test")}
        counts = dict.fromkeys(keys, 0)
        for path in glob.glob(f"{directory}/rows/kept/*.json"):
            kept = json.loads(Path(path).read_text())
            draft = json.loads(Path(f"{directory}/rows/ordinary/{os.path.basename(path)}").read_text())
            adopted = str(kept.get("adopted")) == "True"
            d0, k0 = str(kept["draft_correct"]) == "True", str(kept["correct"]) == "True"
            d1 = fair(draft.get("text") or "", truth[kept["id"]])
            k1 = fair(str(kept.get("text") or ""), truth[kept["id"]]) if adopted else d1
            for key, value in (("requests", 1), ("adopted", adopted), ("drafts_as_registered", d0),
                               ("kept_as_registered", k0), ("gained_as_registered", k0 and not d0),
                               ("lost_as_registered", d0 and not k0), ("drafts_fair", d1), ("kept_fair", k1),
                               ("gained_fair", k1 and not d1), ("lost_fair", d1 and not k1)):
                counts[key] += int(value)
        by_kind[kind] = counts
        for key in keys:
            total[key] += counts[key]
    total["p_as_registered"] = mcnemar(total["gained_as_registered"], total["lost_as_registered"])
    total["p_fair"] = mcnemar(total["gained_fair"], total["lost_fair"])
    args.output.expanduser().write_text(json.dumps({"by_kind": by_kind, "total": total}, indent=1), encoding="utf-8")
    print(json.dumps(total, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
