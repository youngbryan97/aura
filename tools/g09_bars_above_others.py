#!/usr/bin/env python3
"""G09 attempt 6's route, chosen on development data: each kind's bars set against problems not of the kind.

Attempt 5's route sent 13 MATH test problems to DeepMind kinds, 9 by their words and 4 by meaning, and 11 of
the kept answers were wrong (docs/evidence/G09_PROCEDURES_RESULT_5_2026-10-17.md). Its development check held
300 MATH training problems and none reached a kind; at the rate the test showed, about one was expected. A
kind's bar is its least typical sealed problem, and nothing set it against problems of other sources.

Here every MATH training problem with an answer is used, split in two by the seed: the calibration half sets
bars, and the measurement half with the kinds' development requests (12 a kind, seed 7, as attempt 5's part 1
drew them) measures them. Three rules, each a book whose bars differ, each routed by words and then, where
words decline, by meaning (tools/g09_routes.py):

* ``as_attempt_5``: the bars as kept.
* ``above_every_other``: each kind's word bar and meaning bar raised above every calibration problem's
  nearness to the kind by that measure.
* ``above_every_wrong``: raised above only the calibration problems that clear the kind's bar and that its
  procedures answer wrongly.

Preregistration 6, part 1, chooses: among the rules that answer no development request wrongly, the one with
the most right; when each answers some wrongly, the one with the fewest wrong. Answers are graded by
tools/g09_one_grader.one_grade. No test request is read.

Usage:
    g09_bars_above_others.py --book FILE --seed N --output DIR
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RULES = ("as_attempt_5", "above_every_other", "above_every_wrong")


def just_above(value: float) -> float:
    """The least bar a problem at ``value`` does not clear (bars are cleared at or above)."""
    return math.nextafter(float(value), math.inf)


def with_bars(book: Any, words: dict[str, float], meaning: dict[str, float]) -> Any:
    """A copy of ``book`` whose word and meaning bars are at least those given."""
    from core.learning.procedures_from_solved_examples import ProcedureBook

    out = ProcedureBook.from_json(book.to_json())
    out.steps = book.steps
    for kind, bar in words.items():
        if kind in out.signatures and bar > out.signatures[kind].nearest_required:
            out.signatures[kind] = dataclasses.replace(out.signatures[kind], nearest_required=bar)
    for kind, bar in meaning.items():
        if kind in out.meanings and bar > out.meanings[kind].nearest_required:
            out.meanings[kind] = dataclasses.replace(out.meanings[kind], nearest_required=bar)
    return out


def choose(counts: dict[str, Counter]) -> str:
    """Preregistration 6's rule: most right among those wrong on none; else fewest wrong, then most right."""
    clean = [rule for rule in RULES if counts[rule]["wrong"] == 0]
    if clean:
        return max(clean, key=lambda rule: (counts[rule]["right"], -RULES.index(rule)))
    return min(RULES, key=lambda rule: (counts[rule]["wrong"], -counts[rule]["right"], RULES.index(rule)))


def development_requests(per_kind: int, seed: int) -> list[dict]:
    """Each kind's development requests, ``per_kind`` a kind, drawn as attempt 5's part 1 drew them."""
    from tools.induce_g09_procedures import every_kind, problems

    rng = random.Random(seed)
    out: list[dict] = []
    for kind in every_kind():
        try:
            rows = problems(kind, "development")
        except (OSError, KeyError, ValueError):
            continue
        for row in rng.sample(rows, min(per_kind, len(rows))):
            out.append({"kind": kind, "request": row["problem"], "truth": row["truth"]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="splits MATH's training problems in two")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--development-seed", type=int, default=7)
    parser.add_argument("--limit-math", type=int, default=0, help="a dry run on this many MATH problems")
    args = parser.parse_args()
    sys.path.insert(0, str(Path("~/.aura/benchmarks/math500").expanduser()))

    import numpy as np

    from core.learning.procedures_from_solved_examples import ProcedureBook, dumps
    from tools.g09_one_grader import one_grade
    from tools.g09_routes import her_query, kept_receipt
    from tools.measure_kept_routing import _math_training_requests

    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    book = ProcedureBook.from_json(json.loads(args.book.expanduser().read_text(encoding="utf-8")))
    math_rows = _math_training_requests(10**9, args.seed)
    if args.limit_math:
        math_rows = math_rows[:args.limit_math]
    half = len(math_rows) // 2
    calibration, measurement = math_rows[:half], math_rows[half:]
    development = development_requests(args.per_kind, args.development_seed)

    vectors: dict[str, Any] = {}

    def query(text: str) -> Any:
        if text not in vectors:
            vector = np.asarray(her_query(text), dtype=np.float32).reshape(-1)
            vectors[text] = vector / max(float(np.linalg.norm(vector)), 1e-9)
        return vectors[text]

    def graded(kind: str, answer: str, truth: Any) -> bool:
        try:
            return one_grade("math" if kind == "math_training" else kind, answer, truth)
        except (KeyError, ValueError, TypeError):
            return False

    answers: dict[tuple[str, str], str | None] = {}

    def kind_answers(kind: str, request: str) -> str | None:
        if (kind, request) not in answers:
            answers[(kind, request)] = book._run_kind(kind, request, {"family": kind}).get("answer")
        return answers[(kind, request)]

    names = list(book.meanings)
    centroids = np.asarray([book.meanings[k].centroid for k in names], dtype=np.float32)
    nearest_other_words: dict[str, float] = {}
    nearest_other_meaning: dict[str, float] = {}
    nearest_wrong_words: dict[str, float] = {}
    nearest_wrong_meaning: dict[str, float] = {}
    wrong_seen: list[dict] = []
    for item in calibration:
        request = item["request"]
        meaning = dict(zip(names, (centroids @ query(request)).tolist(), strict=True))
        for kind, signature in book.signatures.items():
            near = signature.similarity(request)
            nearest_other_words[kind] = max(nearest_other_words.get(kind, -1.0), near)
            if book.procedures.get(kind) and near >= signature.nearest_required:
                answer = kind_answers(kind, request)
                if answer is not None and not graded("math_training", str(answer), item["truth"]):
                    nearest_wrong_words[kind] = max(nearest_wrong_words.get(kind, -1.0), near)
                    wrong_seen.append({"by": "words", "kind": kind, "near": near, "answer": answer,
                                       "truth": item["truth"], "request": request[:300]})
        for kind, near in meaning.items():
            nearest_other_meaning[kind] = max(nearest_other_meaning.get(kind, -1.0), near)
            if book.procedures.get(kind) and near >= book.meanings[kind].nearest_required:
                answer = kind_answers(kind, request)
                if answer is not None and not graded("math_training", str(answer), item["truth"]):
                    nearest_wrong_meaning[kind] = max(nearest_wrong_meaning.get(kind, -1.0), near)
                    wrong_seen.append({"by": "meaning", "kind": kind, "near": near, "answer": answer,
                                       "truth": item["truth"], "request": request[:300]})

    books = {
        "as_attempt_5": book,
        "above_every_other": with_bars(book, {k: just_above(v) for k, v in nearest_other_words.items()},
                                       {k: just_above(v) for k, v in nearest_other_meaning.items()}),
        "above_every_wrong": with_bars(book, {k: just_above(v) for k, v in nearest_wrong_words.items()},
                                       {k: just_above(v) for k, v in nearest_wrong_meaning.items()}),
    }
    for rule, kept in books.items():
        (out / f"book_{rule}.json").write_text(dumps(kept.to_json()), encoding="utf-8")

    counts = {rule: Counter() for rule in RULES}
    by_class: dict[str, dict[str, Counter]] = {}
    wrong_rows: list[dict] = []
    for item in [*development, *({**row, "kind": "math_training"} for row in measurement)]:
        kind = item["kind"]
        own = "kept" if book.procedures.get(kind) else ("math_training" if kind == "math_training" else "not_kept")
        for rule, kept in books.items():
            receipt = kept_receipt(kept, item["request"], "words_then_meaning", query=query)
            answer = None if receipt is None else receipt.get("answer")
            outcome = "declined" if answer is None else ("right" if graded(kind, str(answer), item["truth"]) else "wrong")
            counts[rule][outcome] += 1
            by_class.setdefault(own, {}).setdefault(rule, Counter())[outcome] += 1
            if outcome == "wrong":
                wrong_rows.append({"rule": rule, "class": own, "kind": kind, "family": receipt.get("family"),
                                   "form": receipt.get("form"), "answer": answer, "truth": str(item["truth"]),
                                   "request": item["request"][:300]})
    chosen = choose(counts)
    report = {"seed": args.seed, "calibration": len(calibration), "measurement": len(measurement),
              "development": len(development), "chosen": chosen,
              "by_rule": {r: dict(c) for r, c in counts.items()},
              "by_class": {k: {r: dict(c) for r, c in v.items()} for k, v in by_class.items()},
              "bars_raised": {"above_every_other": {
                                  "words": sum(just_above(v) > book.signatures[k].nearest_required
                                               for k, v in nearest_other_words.items()),
                                  "meaning": sum(just_above(v) > book.meanings[k].nearest_required
                                                 for k, v in nearest_other_meaning.items())},
                              "above_every_wrong": {"words": len(nearest_wrong_words),
                                                    "meaning": len(nearest_wrong_meaning)}},
              "calibration_wrong": wrong_seen, "development_wrong": wrong_rows}
    (out / "rules.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "chosen_rule.txt").write_text(chosen + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("chosen", "by_rule", "by_class", "bars_raised")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
