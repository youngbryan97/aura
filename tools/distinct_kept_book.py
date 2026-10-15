#!/usr/bin/env python3
"""A book whose signatures weight each word by how few kinds use it (core/learning/procedures_from_solved_examples.
distinct_signatures), from the kinds' development halves, with the same procedures.

The books are merged as the G09 kept arm merges them; each kind's pool and
sealed half are taken again from its development half exactly as the
induction split them (tools/induce_g09_procedures.problems, split_examples).
No test problem is read.

Usage:
    distinct_kept_book.py --book FILE [--book FILE] --output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.procedures_from_solved_examples import (
        ProcedureBook,
        SolvedExample,
        distinct_signatures,
        dumps,
        split_examples,
    )
    from tools.induce_g09_procedures import problems

    book = ProcedureBook.merged([ProcedureBook.from_json(json.loads(p.expanduser().read_text(encoding="utf-8")))
                                 for p in args.book])
    families = {}
    for kind in book.signatures:
        rows = problems(kind, "development")
        examples = [SolvedExample(r["key"], r["problem"], str(r.get("answer_text", r["truth"]))) for r in rows]
        families[kind] = split_examples(examples)
    signatures = distinct_signatures(families)
    distinct = ProcedureBook(signatures, book.procedures, book.agreed, meanings=book.meanings, examples=book.examples)
    args.output.expanduser().write_text(dumps(distinct.to_json()), encoding="utf-8")
    print(json.dumps({"kinds": len(signatures), "kept": sum(1 for k in signatures if book.procedures.get(k))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
