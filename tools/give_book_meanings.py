#!/usr/bin/env python3
"""Give a book of kept procedures each kind's meaning, embedded by her own engine, and a few shown examples.

For every kind the book knows whose problems the G09 tools can load, the
kind's development problems are split as its induction split them
(``split_examples``): the shown pool's problems, embedded as documents, give
the kind's direction, and the sealed half's, embedded as queries for
"kind_of_problem", give its bar (core/learning/kinds_by_meaning.py). The
first problems of the pool, by the same hash order, are kept with their
answers as the kind's examples, which a request worded otherwise is restated
by.

The embedding is her runtime's own (core/memory/vector_memory_engine.py
EmbeddingEngine), so the vectors are the ones a live turn computes. When the
engine cannot load its model (another process holds the model lane
exclusively) it falls back to TF-IDF, and this refuses rather than write
meanings from a different space.

Each text is its own forward on MPS (batching there changes the vectors), so a
whole book, about 290,000 texts, takes two hours in one process. ``--part I/N``
writes only every N-th kind from the I-th, as a book of those kinds alone, so
N processes can share the work; ProcedureBook.merged joins the parts.

Usage:
    give_book_meanings.py --book FILE [--book FILE ...] --output FILE [--shown 3] [--part I/N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def part_of(kinds: list[str], index: int, parts: int) -> list[str]:
    """Every ``parts``-th kind from the ``index``-th: kinds of one source sit together when sorted, so this
    gives each part a share of the large ones."""
    if not 0 <= index < parts:
        raise ValueError(f"part {index} of {parts}")
    return kinds[index::parts]


def book_of(book, kinds: list[str]):
    """The part of ``book`` that holds only ``kinds``."""
    from core.learning.procedures_from_solved_examples import ProcedureBook

    keep = set(kinds)

    def only(mapping: dict) -> dict:
        return {k: v for k, v in mapping.items() if k in keep}

    return ProcedureBook(signatures=only(book.signatures), procedures=only(book.procedures), agreed=only(book.agreed),
                         meanings=only(book.meanings), examples=only(book.examples), posing=only(book.posing))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shown", type=int, default=3,
                        help="examples kept per kind; induction showed three (BIG-Bench Hard's protocol)")
    parser.add_argument("--only-posed", action="store_true",
                        help="only the kinds whose problems share an opening (kinds_by_meaning.shared_opening); "
                             "the others keep the meanings they have")
    parser.add_argument("--part", default="0/1",
                        help="I/N: only every N-th kind, from the I-th, written as a book of those kinds alone")
    args = parser.parse_args()
    index, parts = (int(x) for x in args.part.split("/"))

    import numpy as np

    from core.learning.kinds_by_meaning import TASK, meanings_of, shared_ending, shared_opening
    from core.learning.procedures_from_solved_examples import (
        ProcedureBook,
        SolvedExample,
        split_examples,
    )
    from core.memory.vector_memory_engine import EmbeddingEngine
    from tools.induce_g09_procedures import problems

    book = ProcedureBook.merged([ProcedureBook.from_json(json.loads(p.expanduser().read_text())) for p in args.book])
    engine = EmbeddingEngine()
    engine.embed_query("a probe", task=TASK)
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load (the model lane is held elsewhere); no meanings written")

    def documents(texts: list[str]) -> np.ndarray:
        return np.asarray(engine.embed_batch(texts), dtype=np.float32)

    def queries(texts: list[str]) -> np.ndarray:
        return np.asarray([engine.embed_query(t, task=TASK) for t in texts], dtype=np.float32)

    if parts > 1:
        book = book_of(book, part_of(sorted(book.signatures), index, parts))
    splits, examples, unknown = {}, {}, []
    for kind in sorted(book.signatures):
        try:
            rows = problems(kind, "development")
        except (KeyError, ValueError, FileNotFoundError):
            unknown.append(kind)
            continue
        # What every problem of the kind opens with is how the benchmark poses it, not the question: it is left
        # out of the meaning and the examples and kept in the book, to be put back before a procedure reads.
        opening = shared_opening([r["problem"] for r in rows])
        # What every problem closes with is the benchmark's instruction, not the question: a person's
        # request never carries it, so it is left out of the meaning and the shown examples too.
        ending = shared_ending([r["problem"] for r in rows])
        if args.only_posed and not opening:
            continue
        pool, sealed = split_examples([SolvedExample(r["key"], r["problem"], str(r.get("answer_text", r["truth"])))
                                       for r in rows])

        def question(text: str) -> str:
            return text[len(opening): len(text) - len(ending)]

        splits[kind] = ([question(e.problem) for e in pool], [question(e.problem) for e in sealed])
        examples[kind] = [{"problem": question(e.problem), "answer": e.answer} for e in pool[: args.shown]]
        if opening:
            book.posing[kind] = opening
        else:
            book.posing.pop(kind, None)
        print(json.dumps({"kind": kind, "pool": len(pool), "sealed": len(sealed), "opening": len(opening)}),
              flush=True)
    book.meanings.update(meanings_of(splits, documents=documents, queries=queries))
    book.examples.update(examples)
    args.output.expanduser().write_text(json.dumps(book.to_json(), indent=1), encoding="utf-8")
    print(json.dumps({"kinds_with_meanings": len(book.meanings), "without_loader": unknown,
                      "bars": {k: round(m.nearest_required, 3) for k, m in sorted(book.meanings.items())}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
