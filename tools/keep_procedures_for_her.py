#!/usr/bin/env python3
"""Give her live route a book of kept procedures: the books merged by kind, written to her data directory.

Her chat reads the book at core/learning/procedures_from_solved_examples.KEPT_BOOK_RELATIVE under her
data directory before she drafts. That directory is her launched app's: launch_aura.sh exports
``AURA_ROOT`` as the checkout it runs from, the primary one, so her app reads
``<checkout>/data``. Run without ``AURA_ROOT``, this tool would resolve ``~/.aura/data``, which her app
never reads; it takes the launcher's root instead (live 2026-10-16: every book written for her since
8 October had gone there, and her app had none). ``--dry-run`` prints what would be kept without
writing. ``--combine`` lets the books share kinds (``ProcedureBook.combined``): each kind keeps the
procedures of every book that keeps it.

Usage:
    keep_procedures_for_her.py --book FILE [--book FILE ...] [--combine] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def her_app_root() -> Path:
    """The checkout her app is launched from, found as launch_aura.sh finds it: the parent of git's common directory."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                             capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ROOT
    common = Path(out.stdout.strip())
    return common.parent if common.name == ".git" else ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--combine", action="store_true", help="books may share kinds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("AURA_ROOT", str(her_app_root()))

    from core.learning.procedures_from_solved_examples import ProcedureBook, keep_for_her

    books = [ProcedureBook.from_json(json.loads(path.expanduser().read_text(encoding="utf-8"))) for path in args.book]
    book = ProcedureBook.combined(books) if args.combine else ProcedureBook.merged(books)
    print(json.dumps({"kinds": len(book.signatures), "kinds_with_procedures": sorted(book.procedures),
                      "agreed_known": book.agreed}, indent=1, sort_keys=True))
    if not args.dry_run:
        print(json.dumps({"kept_at": str(keep_for_her(book))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
