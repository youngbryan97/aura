#!/usr/bin/env python3
"""How often her book answers a request wrongly, by the nearest kind alone and by the kinds near enough agreeing.

On development problems only: each kind's development half
(tools/induce_g09_procedures.problems) and competition problems from MATH's
training split, posed as the G09 harness poses them. For each request, two
routes over the same book:

* nearest: the nearest kind, if the request clears that kind's bar
  (ProcedureBook.route), and that kind's procedures;
* agreeing: every kind whose bar the request clears; the answer stands only if
  each of those kinds keeps a procedure and they all agree
  (ProcedureBook.answer).

An answer given is graded by the request's own kind's grader. A wrong answer
given is a loss wherever the draft was right, so "wrong" is the harm a route
does; "right" on a kind that keeps no procedure of its own is a gain by
chance.

Usage:
    measure_kept_routing.py --book FILE [--book FILE] --output FILE [--per-kind 12] [--math 300] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MATH_TRAIN = Path("~/.aura/benchmarks/g09/math-train").expanduser()


def _math_training_requests(count: int, seed: int) -> list[dict]:
    import pandas as pd

    from tools.run_g12_math500 import INSTRUCTION, last_boxed

    rows = []
    for path in sorted(MATH_TRAIN.glob("*-train.parquet")):
        for _index, row in pd.read_parquet(path).iterrows():
            truth = last_boxed(row["solution"])
            if truth is not None:
                rows.append({"kind": "math_training", "request": f"{row['problem']}\n\n{INSTRUCTION}",
                             "truth": truth})
    random.Random(seed).shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--math", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from core.learning.procedures_from_solved_examples import (
        SOLVED_EXAMPLE_PROCEDURES_SCHEMA,
        ProcedureBook,
    )
    from tools.induce_g09_procedures import every_kind, grader_for, problems
    from tools.run_g09_organ import grade_math

    book = ProcedureBook.merged([ProcedureBook.from_json(json.loads(p.expanduser().read_text(encoding="utf-8")))
                                 for p in args.book])
    rng = random.Random(args.seed)
    requests: list[dict] = []
    for kind in every_kind():
        try:
            rows = problems(kind, "development")
        except (OSError, KeyError, ValueError):
            continue
        for row in rng.sample(rows, min(args.per_kind, len(rows))):
            requests.append({"kind": kind, "request": row["problem"], "truth": row["truth"]})
    requests += _math_training_requests(args.math, args.seed)

    def graded(kind: str, answer: str, truth) -> bool:
        if kind == "math_training":
            return bool(grade_math(answer, truth)[0])
        try:
            return bool(grader_for(kind)(answer, truth))
        except (KeyError, ValueError, TypeError):
            return False

    def nearest(request: str) -> str | None:
        family, similarity = book.route(request)
        if family is None:
            return None
        receipt = {"schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": family, "similarity": similarity}
        return book._run_kind(family, request, receipt).get("answer")

    counts: dict[str, Counter] = {"nearest": Counter(), "agreeing": Counter()}
    by_class: dict[str, dict[str, Counter]] = {}
    for item in requests:
        kind = item["kind"]
        own = "kept" if book.procedures.get(kind) else ("math_training" if kind == "math_training" else "not_kept")
        for route, answer in (("nearest", nearest(item["request"])), ("agreeing", book.answer(item["request"])["answer"])):
            outcome = "declined" if answer is None else ("right" if graded(kind, answer, item["truth"]) else "wrong")
            counts[route][outcome] += 1
            by_class.setdefault(own, {}).setdefault(route, Counter())[outcome] += 1
    report = {"requests": len(requests), "per_kind": args.per_kind, "math_training": args.math, "seed": args.seed,
              "by_route": {r: dict(c) for r, c in counts.items()},
              "by_class": {k: {r: dict(c) for r, c in v.items()} for k, v in by_class.items()}}
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
