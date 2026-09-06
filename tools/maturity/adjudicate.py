#!/usr/bin/env python3
"""Every finding in the peer ledger, checked against the code before it is worked.

An external review compared Aura against eleven agent architectures and wrote
202 findings, each naming a place in Aura and a thing to build. A ledger of
that size cannot be worked from the top: some of its findings are already
closed, some name a file that has since moved, and some are right. Which is
which is a question about this repository, and it is answerable mechanically
for the first cut.

Three things are checked per finding, and none of them decides anything on its
own:

* Does the Aura anchor still exist? A finding about a file that is not there is
  either stale or about something that moved, and both need a human reading.
* Do the words of the closure appear anywhere under the watched trees? A hit
  is not proof the closure is done — it is proof there is something to read
  before writing anything.
* Is there a test naming it? The repository's own standard is that a claim
  needs the test that validates it, and the same standard applies to a claim
  that a gap is closed.

The output is a working file, not a verdict. Adjudication is written by hand
into docs/MATURITY_LEDGER.md, which is the append-only record of what was
decided about each one and why.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Trees the closure could plausibly live in. Tests count separately.
WATCHED = ("core", "interface", "skills", "llm", "executors", "security", "tools")

#: Words too common to tell anything apart. Drawn from what the closures
#: actually say rather than from a stop-list somebody wrote for English.
TOO_COMMON = frozenset(
    """a an and are as at be by can cannot every for from has have in into is it its
    may must not of on one only or over per remain remains same so than that the their
    them then there these they this through to under up use used using via was were
    what when where which while who whose will with without aura should each any all
    also both but if no nor other same such take takes them very
    """.split()
)


def _terms(text: str) -> list[str]:
    """The identifier-shaped words of a closure, longest first.

    Identifiers are what can be looked for: `CancellationToken` says something
    a grep can answer, `must propagate` does not.
    """

    found = re.findall(r"[A-Za-z_][A-Za-z0-9_.]{3,}", text or "")
    kept = []
    for one in found:
        bare = one.rstrip(".,;:")
        if len(bare) < 5 or bare.lower() in TOO_COMMON:
            continue
        # An identifier is what a grep can answer. A capitalised English word
        # at the head of a sentence is not one, and admitting it makes every
        # finding look like it is already in the tree.
        camel = re.search(r"[a-z][A-Z]", bare) is not None
        snake = "_" in bare
        dotted = "." in bare and bare.split(".")[0].islower()
        if camel or snake or dotted:
            kept.append(bare)
    return sorted(set(kept), key=len, reverse=True)[:6]


def _exists(anchor: str) -> list[str]:
    """Which of the paths this finding names are still there."""

    found = []
    for piece in re.split(r"[+,;]| and ", anchor or ""):
        piece = piece.strip().split("::")[0].strip()
        if not piece or " " in piece.rstrip("*"):
            continue
        if piece.endswith("*"):
            base = Path(piece.rstrip("*").rstrip("/"))
            if (ROOT / base).exists():
                found.append(piece)
            continue
        if (ROOT / piece).exists():
            found.append(piece)
    return found


def _mentions(term: str, *, tests: bool) -> int:
    where = ["tests"] if tests else list(WATCHED)
    try:
        out = subprocess.run(
            ["grep", "-rl", "--include=*.py", "-F", term, *where],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    return len([one for one in out.stdout.splitlines() if one.strip()])


def look(finding: dict) -> dict:
    anchors = _exists(str(finding.get("aura_anchor", "")))
    terms = _terms(str(finding.get("closure", "")))
    hits = {one: _mentions(one, tests=False) for one in terms}
    tested = {one: _mentions(one, tests=True) for one in terms if hits.get(one)}
    return {
        "num": finding["num"],
        "comparator": finding["comparator"],
        "priority": finding["priority"],
        "title": finding["title"],
        "anchors_that_exist": anchors,
        "anchor_named": finding.get("aura_anchor", ""),
        "closure_terms": hits,
        "terms_with_a_test": tested,
        "reads_as": (
            "nothing by this name"
            if not any(hits.values())
            else "named in the tree" if not any(tested.values())
            else "named and tested"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--priority", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    body = json.loads(args.ledger.read_text())
    findings = [
        one
        for one in body["findings"]
        if not args.priority or one["priority"] == args.priority
    ]
    looked = [look(one) for one in findings]
    text = json.dumps({"looked_at": len(looked), "findings": looked}, indent=1)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    by_reading: dict[str, int] = {}
    for one in looked:
        by_reading[one["reads_as"]] = by_reading.get(one["reads_as"], 0) + 1
    print(f"\n{len(looked)} findings: {by_reading}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
