#!/usr/bin/env python3
"""G09: which requests need her draft, given a book of kept procedures.

A request no kept procedure answers keeps its draft in both arms, so only the
answered ones need drafting. For each held-out kind named with ``--sample``
the test problems are drawn as the preregistration says (``count`` by
``--seed`` from the kind's test half, through the harness's own sampler);
every request of the routing-only sets is routed through the book and kept
when a kept procedure answers it. One file of task ids is written per harness
domain, with a summary.

Usage:
    g09_requests_to_draft.py --book FILE [--book FILE ...] [--signatures-only FILE ...]
        --sample KIND=COUNT [...] [--spread TOTAL:KIND,KIND,...] --seed N --routing-only DOMAIN [...] --output DIR

``--spread`` divides TOTAL test draws as evenly as possible among the listed
kinds that keep a procedure, the remainder going to the first kinds in the
seed's order (attempt 3's rule, G09_PROCEDURES_PREREGISTRATION_3_2026-10-14.md).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def spread(total: int, kinds: list[str], seed: int) -> dict[str, int]:
    """``total`` divided as evenly as possible among ``kinds``, the remainder to the first in the seed's order."""
    kinds = sorted(kinds)
    if not kinds:
        return {}
    base, extra = divmod(total, len(kinds))
    order = list(kinds)
    random.Random(seed).shuffle(order)
    return {kind: base + (kind in order[:extra]) for kind in kinds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path, action="append", default=[], help="a book whose procedures count")
    parser.add_argument("--signatures-only", type=Path, action="append", default=[],
                        help="a book whose kinds route but whose procedures are left out")
    parser.add_argument("--sample", action="append", default=[], help="KIND=COUNT: a held-out kind's test draw")
    parser.add_argument("--spread", action="append", default=[],
                        help="TOTAL:KIND,KIND,...: TOTAL draws over the listed kinds that keep a procedure")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--routing-only", action="append", default=[], help="a harness domain routed whole")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", default="words", choices=("words", "meaning", "words_then_meaning"),
                        help="how a request reaches a kind (tools/g09_routes.py); words is attempt 4's")
    parser.add_argument("--exclude-ids", type=Path, action="append", default=[],
                        help="a folder of an earlier attempt's <kind>.ids; none of those requests is drawn again")
    args = parser.parse_args()

    from core.brain.reasoning_amplifier_v2 import answer_from_kept_procedures
    from core.learning.procedures_from_solved_examples import ProcedureBook
    from tools.induce_g09_procedures import test_ids
    from tools.run_g09_organ import LOADERS, sample

    books = [ProcedureBook.from_json(json.loads(p.expanduser().read_text())) for p in args.book]
    for path in args.signatures_only:
        signatures = ProcedureBook.from_json(json.loads(path.expanduser().read_text()))
        known = {kind for book in books for kind in book.signatures}
        books.append(ProcedureBook({k: s for k, s in signatures.signatures.items() if k not in known}, {}, {}))
    book = ProcedureBook.merged(books)
    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {"kept_kinds": sorted(book.procedures), "domains": {}}

    def answered(tasks: list[dict]) -> list[str]:
        if args.route != "words":
            from tools.g09_routes import kept_receipt

            return [task["id"] for task in tasks if kept_receipt(book, task["request"], args.route) is not None]
        return [task["id"] for task in tasks
                if book.route(task["request"])[0] in book.procedures
                and answer_from_kept_procedures(task["request"], book, own_book=False) is not None]

    def excluded(kind: str) -> set[str]:
        seen: set[str] = set()
        for folder in args.exclude_ids:
            path = folder.expanduser() / f"{kind}.ids"
            if path.exists():
                seen.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        return seen

    for spec in args.spread:
        total, listed = spec.split(":", 1)
        kept = [kind for kind in listed.split(",") if kind in book.procedures]
        args.sample += [f"{kind}={count}" for kind, count in spread(int(total), kept, args.seed).items()]
    for spec in args.sample:
        kind, count = spec.split("=")
        allowed = test_ids(kind)
        drawn = sample([t for t in LOADERS[kind]() if t["id"] in allowed], int(count), args.seed, excluded(kind))
        # Every drawn problem of a kept kind is drafted, answered or not, so the kind's accuracy has a
        # denominator; a held-out kind with nothing kept still drafts any of its problems another kind answers.
        ids = [t["id"] for t in drawn] if kind in book.procedures else answered(drawn)
        (out / f"{kind}.ids").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        summary["domains"][kind] = {"drawn": len(drawn), "to_draft": len(ids)}
    for domain in args.routing_only:
        ids = answered([t for t in LOADERS[domain]() if t["id"] not in excluded(domain)])
        (out / f"{domain}.ids").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        summary["domains"][domain] = {"routed_whole": True, "to_draft": len(ids)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
