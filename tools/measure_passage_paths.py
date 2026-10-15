#!/usr/bin/env python3
"""How often the answer reaches her by each way of finding passages: the careful lookup, every paragraph, and both.

For SimpleQA development questions (the same draw as
tools/measure_passage_recall.py), three ways of choosing passages are run:

* careful: the articles the question names, their paragraphs by its rarer
  words (core/knowledge/passages_about.py);
* every paragraph: BM25 over the whole corpus's paragraphs
  (core/knowledge/passage_index.py);
* fused: reciprocal rank fusion of the two.

A question counts within k when every word of its gold answer is in the first
k passages. Words only: her embedder needs the model lane. Each way's time is
reported too.

Usage:
    measure_passage_paths.py --count 200 --seed 7 --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WITHIN = (1, 3, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.knowledge.local_corpus import SEARCH_DEADLINE_S, LocalCorpusStore
    from core.knowledge.passage_index import PassageIndex, fused
    from core.knowledge.passages_about import passages_about
    from tools.run_g09_organ import sample
    from tools.run_knowledge_with_corpus import _words, tasks

    store = LocalCorpusStore()
    index = PassageIndex()
    if not index.available():
        raise SystemExit(f"no passage index at {index.passage_db}")

    def holds(passages: list[tuple[str, str]], gold: str) -> bool:
        text = set(_words(" ".join(f"{t} {p}" for t, p in passages)))
        need = _words(gold)
        return bool(need) and all(w in text for w in need)

    def timed(run):
        began = time.monotonic()
        found = run()
        return found, time.monotonic() - began

    chosen = sample(tasks("development"), args.count, args.seed, set())
    ways = ("careful", "every_paragraph", "fused")
    counts = {way: dict.fromkeys(WITHIN, 0) for way in ways}
    seconds: dict[str, list[float]] = {way: [] for way in ways}
    deepest = max(WITHIN)
    for task in chosen:
        question = task["request"]
        careful, took = timed(lambda q=question: passages_about(q, store=store, limit=deepest))
        seconds["careful"].append(took)
        every, took = timed(lambda q=question: [(p.title, p.paragraph) for p in
                                                index.search(q, limit=deepest, deadline_s=SEARCH_DEADLINE_S)])
        seconds["every_paragraph"].append(took)
        seconds["fused"].append(seconds["careful"][-1] + took)
        both = fused([careful, every], limit=deepest)
        for way, found in (("careful", careful), ("every_paragraph", every), ("fused", both)):
            for k in WITHIN:
                counts[way][k] += holds(found[:k], task["truth"])

    def spread(values: list[float]) -> dict:
        ordered = sorted(values)
        return {"median_ms": round(1000 * ordered[len(ordered) // 2]),
                "p95_ms": round(1000 * ordered[int(0.95 * (len(ordered) - 1))]), "max_ms": round(1000 * ordered[-1])}

    report = {"questions": len(chosen), "seed": args.seed, "answer_within_k": counts,
              "time": {way: spread(seconds[way]) for way in ways}}
    args.output.expanduser().write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
