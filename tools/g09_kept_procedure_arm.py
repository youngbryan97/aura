#!/usr/bin/env python3
"""G09: her drafts, and the same requests answered through her kept procedures.

For each draft already written, the request goes to
``answer_from_kept_procedures``, the function her response phase calls before
the amplifier's search (ResponseGenerationPhase._maybe_amplify_response). When
a kept procedure answers, its answer replaces the draft and is graded by the
domain's own grader; otherwise the draft stands. No model is loaded and the
amplifier's search is not run, so this measures the kept procedures alone.

Drafts come from tools/run_g09_organ.py (``rows/ordinary``) or, for BBEH Mini,
from tools/run_g12_bbeh.py, whose request is the same text. Rows go to
``rows/kept`` beside the drafts, written once.

Usage:
    g09_kept_procedure_arm.py --domain DOMAIN --drafts DIR --book FILE [--book FILE ...] [--g12-bbeh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def drafts_from_harness(directory: Path, domain: str) -> list[dict[str, Any]]:
    from tools.run_g09_organ import LOADERS

    tasks = {task["id"]: task for task in LOADERS[domain]()}
    out = []
    for path in sorted((directory / "rows" / "ordinary").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task = tasks[row["id"]]
        out.append({"name": path.name, "id": row["id"], "request": task["request"], "truth": task["truth"],
                    "group": task["group"], "draft": row["text"], "draft_correct": bool(row["correct"])})
    return out


def drafts_from_g12_bbeh(directory: Path) -> list[dict[str, Any]]:
    from tools.run_g09_organ import BBEH_SUFFIX
    from tools.run_g12_bbeh import load_cases, saved_results

    mini, catalog = load_cases(Path("~/.aura/benchmarks/g09/bbeh/mini/data.json"))
    out = []
    for key, row in saved_results(directory / "rows", catalog).items():
        case = catalog.by_id[key]
        example = mini[case.source_ordinal]
        draft = row["public_text"] if row.get("termination") == "stop" else ""
        out.append({"name": f"{case.id}.json", "id": case.id, **case.metadata(),
                    "request": f"{example['input']}\n\n{BBEH_SUFFIX}",
                    "truth": example["target"], "group": row.get("task"), "draft": draft,
                    "draft_correct": bool(row["correct"])})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--book", type=Path, action="append", required=True,
                        help="a book from tools/induce_g09_procedures.py; several are merged by kind")
    parser.add_argument("--g12-bbeh", action="store_true", help="the drafts are tools/run_g12_bbeh.py rows")
    args = parser.parse_args()

    from core.brain.reasoning_amplifier_v2 import answer_from_kept_procedures
    from core.learning.procedures_from_solved_examples import ProcedureBook
    from tools.run_g09_organ import GRADERS

    grade = GRADERS[args.domain]
    book = ProcedureBook.merged([ProcedureBook.from_json(json.loads(path.expanduser().read_text(encoding="utf-8")))
                                 for path in args.book])
    directory = args.drafts.expanduser()
    drafts = drafts_from_g12_bbeh(directory) if args.g12_bbeh else drafts_from_harness(directory, args.domain)
    out_dir = directory / "rows" / "kept"
    out_dir.mkdir(parents=True, exist_ok=True)
    book_sha = hashlib.sha256(json.dumps(book.to_json(), sort_keys=True).encode()).hexdigest()
    counts = {"rows": 0, "answered": 0, "kept_correct": 0, "draft_correct": 0}
    for draft in drafts:
        path = out_dir / draft["name"]
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
        else:
            began = time.monotonic()
            kept = answer_from_kept_procedures(draft["request"], book, own_book=False)
            row = {"id": draft["id"], "group": draft["group"], "book_sha256": book_sha, "adopted": kept is not None,
                   "authority": "kept_procedure" if kept is not None else "none"}
            for key in ("source_ordinal", "reference_sha256"):
                if key in draft:
                    row[key] = draft[key]
            if kept is not None:
                row["correct"], row["note"] = grade(kept.answer, draft["truth"])
                row["text"] = kept.answer
                row["procedure"] = kept.receipt.cognitive_operations[0]
            else:
                row["correct"], row["text"] = draft["draft_correct"], None
            row["draft_correct"] = draft["draft_correct"]
            row["seconds"] = round(time.monotonic() - began, 3)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
            temporary.replace(path)
        counts["rows"] += 1
        counts["answered"] += bool(row["adopted"])
        counts["kept_correct"] += bool(row["correct"])
        counts["draft_correct"] += bool(row["draft_correct"])
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
