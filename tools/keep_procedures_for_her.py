#!/usr/bin/env python3
"""Give her live route a book of kept procedures: the books merged by kind, written to her data directory.

Her response phase reads the book at core/learning/procedures_from_solved_examples.KEPT_BOOK_RELATIVE
under her data directory (``AURA_STATE_ROOT``) before the amplifier's search. ``--dry-run`` prints what
would be kept without writing.

Usage:
    keep_procedures_for_her.py --book FILE [--book FILE ...] [--dry-run]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from core.learning.procedures_from_solved_examples import ProcedureBook, keep_for_her

    book = ProcedureBook.merged([ProcedureBook.from_json(json.loads(path.expanduser().read_text(encoding="utf-8")))
                                 for path in args.book])
    print(json.dumps({"kinds": len(book.signatures), "kinds_with_procedures": sorted(book.procedures),
                      "agreed_known": book.agreed}, indent=1, sort_keys=True))
    if not args.dry_run:
        print(json.dumps({"kept_at": str(keep_for_her(book))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
