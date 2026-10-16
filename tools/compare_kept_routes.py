#!/usr/bin/env python3
"""G09 attempt 5's routes compared on development requests: words, meaning, and words then meaning.

The book is attempt 4's with each kind's meaning and shown examples taken from a second book (her live book,
which computed them from the kinds' development halves); --write-book saves it. The requests are
tools/measure_kept_routing.py's: 12 from each kind's development half and MATH training problems, seed 7. Each
answer is graded by the request's own kind with tools/g09_one_grader.one_grade. No test request is read.

Usage:
    compare_kept_routes.py --book FILE --meanings-from FILE --output FILE [--write-book FILE]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", type=Path, required=True)
    parser.add_argument("--meanings-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-book", type=Path)
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--math", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sys.path.insert(0, str(Path("~/.aura/benchmarks/math500").expanduser()))

    from core.learning.kinds_by_meaning import TASK
    from core.learning.procedures_from_solved_examples import ProcedureBook, dumps
    from core.memory.vector_memory_engine import EmbeddingEngine
    from tools.g09_one_grader import one_grade
    from tools.induce_g09_procedures import every_kind, problems
    from tools.measure_kept_routing import _math_training_requests

    book = ProcedureBook.from_json(json.loads(args.book.expanduser().read_text(encoding="utf-8")))
    other = ProcedureBook.from_json(json.loads(args.meanings_from.expanduser().read_text(encoding="utf-8")))
    book.meanings = {k: other.meanings[k] for k in book.signatures if k in other.meanings}
    book.examples = {k: other.examples[k] for k in book.signatures if k in other.examples}
    if args.write_book:
        args.write_book.expanduser().write_text(dumps(book.to_json()), encoding="utf-8")

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

    engine = EmbeddingEngine()
    engine.embed_query("a probe", task=TASK)
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load; nothing measured")

    def graded(kind: str, answer: str, truth) -> bool:
        try:
            return one_grade("math" if kind == "math_training" else kind, answer, truth)
        except (KeyError, ValueError, TypeError):
            return False

    def by_meaning(request: str) -> str | None:
        kind, _similarity, _why = book.route_by_meaning(engine.embed_query(request, task=TASK))
        if kind is None or not book.procedures.get(kind):
            return None
        return book._run_kind(kind, request, {"family": kind}).get("answer")

    counts = {route: Counter() for route in ("words", "meaning", "words_then_meaning")}
    by_class: dict[str, dict[str, Counter]] = {}
    for item in requests:
        kind = item["kind"]
        own = "kept" if book.procedures.get(kind) else ("math_training" if kind == "math_training" else "not_kept")
        words = book.answer(item["request"])["answer"]
        meaning = by_meaning(item["request"])
        for route, answer in (("words", words), ("meaning", meaning),
                              ("words_then_meaning", words if words is not None else meaning)):
            outcome = "declined" if answer is None else ("right" if graded(kind, str(answer), item["truth"]) else "wrong")
            counts[route][outcome] += 1
            by_class.setdefault(own, {}).setdefault(route, Counter())[outcome] += 1
    report = {"requests": len(requests), "kinds_with_meanings": len(book.meanings),
              "by_route": {r: dict(c) for r, c in counts.items()},
              "by_class": {k: {r: dict(c) for r, c in v.items()} for k, v in by_class.items()}}
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
