#!/usr/bin/env python3
"""A control named in the threat model has to be a control somebody attacks.

docs/THREAT_MODEL.md lists fourteen classes of attack, each with the control
that answers it and the test that runs it. That table is only worth reading
while both halves are true, and a table in a document decays quietly: a test
gets renamed, the row keeps its name, and the document goes on saying the
attack is covered.

So this reads the table and checks it against the tree. Every file in the
"Attacked in" column exists. Every row has a coverage word from a fixed set,
so "checked" cannot quietly become a sentence that means less. And the
document must keep saying, in as many words, that no independent security
engineer has attacked this system, because that is the finding most likely to
be dropped once the table looks full.

    python tools/check_threat_model.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCUMENT = ROOT / "docs" / "THREAT_MODEL.md"
POLICY = ROOT / "SECURITY.md"
SUITE = ROOT / "tests" / "security"

_ROW_RE = re.compile(r"^\|\s*(?P<number>\d+)\s*\|(?P<rest>.*)\|\s*$", re.MULTILINE)
_PATH_RE = re.compile(r"`([^`]+\.py)`")

#: The words a coverage cell may start with. A row that says something else is
#: a row whose meaning nobody can compare with the next one.
COVERAGE_WORDS = ("checked", "partial", "not covered")

#: The sentence that must survive. Written as a check because it is the one a
#: full-looking table makes it tempting to delete.
INDEPENDENT_REVIEW_ADMISSION = "No independent security engineer has attacked this system"


def check() -> list[str]:
    problems: list[str] = []
    if not DOCUMENT.exists():
        return ["docs/THREAT_MODEL.md does not exist"]
    body = DOCUMENT.read_text("utf-8")

    if INDEPENDENT_REVIEW_ADMISSION.lower() not in body.lower():
        problems.append(
            "the threat model no longer states that no independent security "
            "engineer has attacked this system; if that changed, say who and "
            "when instead of deleting the sentence"
        )

    rows = list(_ROW_RE.finditer(body))
    if len(rows) < 10:
        problems.append(f"the attack table has {len(rows)} rows; it had fourteen")

    numbers: list[int] = []
    for row in rows:
        numbers.append(int(row.group("number")))
        cells = [c.strip() for c in row.group("rest").split("|")]
        if len(cells) < 4:
            problems.append(f"row {row.group('number')} has {len(cells)} cells, not four")
            continue
        attack, control, attacked_in, coverage = cells[0], cells[1], cells[2], cells[3]
        if not attack or not control:
            problems.append(f"row {row.group('number')} names no attack or no control")
        referenced = _PATH_RE.findall(attacked_in)
        if not referenced:
            problems.append(
                f"row {row.group('number')} ({attack!r}) names no test file; a "
                "control nobody attacks is a claim"
            )
        for relative in referenced:
            if not (ROOT / relative).exists():
                problems.append(
                    f"row {row.group('number')} ({attack!r}) points at {relative}, "
                    "which does not exist"
                )
        if not coverage.lower().startswith(COVERAGE_WORDS):
            problems.append(
                f"row {row.group('number')} reports coverage as {coverage!r}; it "
                f"must start with one of {COVERAGE_WORDS}"
            )

    if numbers and numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"the attack rows are numbered {numbers}, not 1..{len(numbers)}")

    if not POLICY.exists():
        problems.append("SECURITY.md does not exist, so there is no way to report anything")
    elif "security/advisories/new" not in POLICY.read_text("utf-8"):
        problems.append("SECURITY.md names no private disclosure channel")

    if not SUITE.is_dir() or not list(SUITE.glob("test_*.py")):
        problems.append("tests/security/ holds no attacks")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    problems = check()
    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 1 if problems else 0
    if problems:
        print(f"❌ threat model: {len(problems)} problem(s)")
        for problem in problems:
            print(f"   • {problem}")
        return 1
    rows = len(_ROW_RE.findall(DOCUMENT.read_text("utf-8")))
    print(f"✅ threat model: {rows} attack classes, each with a test that exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
