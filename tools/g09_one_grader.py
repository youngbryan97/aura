#!/usr/bin/env python3
"""A kept arm's rows graded again with one function per domain, applied to both arms of every request.

G09 attempt 3 graded her drafts with the harness's graders and the kept answers with each kind's own matching
(docs/evidence/G09_PROCEDURES_RESULT_3_2026-10-16.md, correction of 16 October). Here one function grades both:

* DeepMind mathematics: right when DeepMind's exact rule or MATH-500's equivalence grader accepts the answer,
  a true-or-false kind read as yes or no, a list item by item (tools/g09_fair_regrade.fair);
* MATH and AIME: MATH-500's own grader on the boxed answer, or on a one-line reply (run_g09_organ.grade_math);
* BIG-Bench Hard: BBH's matching of the answer after the last "the answer is", or of the whole text when it
  states none (a kept procedure returns the answer bare);
* PlanBench: the restated validator (tools/planbench_restated.grade), as both arms were already graded.

A request whose kept procedure declined keeps her draft, as the kept arm did. The rows written beside the
kept arm's carry ``correct`` and ``draft_correct`` from this grading, so tools/g09_analysis.py reads them
unchanged (``--kept <folder>``).

Usage:
    g09_one_grader.py --domain D --drafts DIR --kept kept4 --output-kept kept4_one_grader
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRADER = "one_grader_v1"


def one_grade(domain: str, text: str, truth: object) -> bool:
    """Whether ``text`` answers ``truth``, by the one rule for ``domain``."""
    from tools.induce_g09_procedures import PLANBENCH

    text = str(text or "")
    if domain.startswith("dm_"):
        from tools.g09_fair_regrade import fair

        return fair(text, truth)
    if domain.startswith("bbh_"):
        from tools.run_g09_bbh import extract_answer, matches

        answer = extract_answer(text) or text.strip()
        return bool(answer) and bool(matches(answer, str(truth)))
    if domain in PLANBENCH:
        from tools.planbench_restated import grade

        return bool(grade(text, truth)[0])
    if domain in ("math", "aime"):
        from tools.run_g09_organ import grade_math

        return bool(grade_math(text, truth)[0])
    raise KeyError(f"no one grader for {domain}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--kept", required=True, help="the kept arm's folder under rows/")
    parser.add_argument("--output-kept", required=True, help="the folder under rows/ for the regraded rows")
    args = parser.parse_args()
    sys.path.insert(0, str(Path("~/.aura/benchmarks/math500").expanduser()))
    from tools.g09_kept_procedure_arm import drafts_from_harness

    directory = args.drafts.expanduser()
    out_dir = directory / "rows" / args.output_kept
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "adopted": 0, "draft_correct": 0, "kept_correct": 0, "gained": 0, "lost": 0}
    for draft in drafts_from_harness(directory, args.domain):
        kept = json.loads((directory / "rows" / args.kept / draft["name"]).read_text(encoding="utf-8"))
        adopted = bool(kept.get("adopted"))
        draft_right = one_grade(args.domain, draft["draft"], draft["truth"])
        kept_right = one_grade(args.domain, str(kept.get("text") or ""), draft["truth"]) if adopted else draft_right
        row = {**kept, "correct": kept_right, "draft_correct": draft_right, "grader": GRADER,
               "graded_as_registered": {"correct": kept.get("correct"), "draft_correct": kept.get("draft_correct")}}
        (out_dir / draft["name"]).write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
        counts["rows"] += 1
        counts["adopted"] += adopted
        counts["draft_correct"] += draft_right
        counts["kept_correct"] += kept_right
        counts["gained"] += kept_right and not draft_right
        counts["lost"] += draft_right and not kept_right
    print(json.dumps({"domain": args.domain, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
