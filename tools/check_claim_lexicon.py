#!/usr/bin/env python3
"""tools/check_claim_lexicon.py — a loaded name must say what it measures.

A reader meets this project through filenames. `test_consciousness_guarantee.py`,
`core/consciousness/qualia_engine.py`, `test_personhood_battery.py` — each of
those names asserts a conclusion, and each one is met long before any ledger
that qualifies it. TESTING.md says "the filenames are historical", which is true
and does not help: the name is what travels into a screenshot, a search result,
a code review, a conversation.

Renaming 118 files would break every doc link and every reference in a shared
checkout, and it would not stop the next loaded name from being added. So the
rule is not about names. It is:

    A module whose NAME carries a loaded term must state, in its module
    docstring, what it operationally measures — on a line beginning
    "Operationally:" or "What this measures:".

That line is a commitment in the reader's path rather than in a ledger they
would have to know to look for. It is greppable, so it can be checked; and it
cannot be satisfied by atmosphere, because it has to finish the sentence "this
measures ...".

Ratcheted, like layering and the enterprise gate: the count of files still
missing the line may only shrink. Fixing 118 docstrings in one pass would be
either rushed or fake, and a gate nobody can go green against gets deleted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "config" / "claim_lexicon_baseline.json"
SCANNED_ROOTS = ("core", "tests", "interface", "tools")

#: Terms that assert a conclusion about what the system IS or EXPERIENCES.
#:
#: Deliberately narrower than "words that sound impressive". "mind", "memory",
#: "experience" and "introspection" are ordinary engineering vocabulary in this
#: codebase (mind_tick, the experience stream) and flagging them would drown
#: the signal. What is here either names a contested philosophical property, or
#: names a conclusion about general capability.
LOADED_TERMS = (
    "agi",
    "consciousness",
    "emergent",
    "free_will",
    "personhood",
    "phenomenal",
    "qualia",
    "self_aware",
    "sentien",
    "soul",
    "strange_loop",
    "subjective",
    "volition",
)

#: The operational-definition line. Two spellings because a test file naturally
#: says one and an engine module naturally says the other.
DEFINITION_MARKERS = ("Operationally:", "What this measures:")


#: Terms are matched as NAME TOKENS, not substrings. `imagination.py` contains
#: the letters of "agi" and asserts nothing; a gate that cannot tell those apart
#: spends its credibility on false positives and gets switched off.
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def loaded_terms_in(name: str) -> list[str]:
    stem = name.lower().removesuffix(".py")
    tokens = [token for token in _TOKEN_SPLIT.split(stem) if token]
    # Normalised stem with one separator, delimited at both ends, so a term
    # that spans tokens ("strange_loop") can be found the same way a single
    # token is. Splitting alone made multi-word terms unmatchable — they were
    # in the list and could never fire.
    delimited = "_" + "_".join(tokens) + "_"
    found: list[str] = []
    for term in LOADED_TERMS:
        if "_" in term:
            if f"_{term}_" in delimited:
                found.append(term)
            continue
        # A term matches a whole token, or a token that is that term plus an
        # ordinary suffix ("sentien" -> "sentience", "conscious" -> ...).
        if any(token == term or token.startswith(term) for token in tokens):
            found.append(term)
    return found


def module_docstring(path: Path) -> str | None:
    """The module docstring, or None when the file will not parse."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None
    return ast.get_docstring(tree)


def has_operational_definition(docstring: str | None) -> bool:
    if not docstring:
        return False
    return any(marker in docstring for marker in DEFINITION_MARKERS)


def scan(root: Path = ROOT) -> dict[str, Any]:
    flagged: list[dict[str, Any]] = []
    compliant: list[str] = []
    for directory in SCANNED_ROOTS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            terms = loaded_terms_in(path.name)
            if not terms:
                continue
            relative = str(path.relative_to(root))
            if has_operational_definition(module_docstring(path)):
                compliant.append(relative)
            else:
                flagged.append({"file": relative, "terms": terms})
    return {
        "schema": "aura.claim_lexicon.v1",
        "markers": list(DEFINITION_MARKERS),
        "scanned_files_with_loaded_names": len(flagged) + len(compliant),
        "compliant": sorted(compliant),
        "missing_definition": sorted(flagged, key=lambda item: item["file"]),
        "missing_count": len(flagged),
    }


def load_baseline(path: Path = BASELINE_PATH) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = data.get("max_missing_definition")
    return int(value) if isinstance(value, (int, float)) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record the current count as the new ceiling (only ever lower it)",
    )
    args = parser.parse_args(argv)

    report = scan()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    baseline = load_baseline()

    if args.write_baseline:
        if baseline is not None and report["missing_count"] > baseline:
            print(
                f"refusing to raise the baseline: {report['missing_count']} > {baseline}. "
                "This ratchet only shrinks.",
                file=sys.stderr,
            )
            return 1
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "description": (
                        "Files whose NAME carries a loaded term but whose module "
                        "docstring does not say what they operationally measure. "
                        "Reduce this; never raise it."
                    ),
                    "markers": list(DEFINITION_MARKERS),
                    "max_missing_definition": report["missing_count"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"claim lexicon baseline written: {report['missing_count']}")
        return 0

    if baseline is None:
        print(
            f"claim lexicon: no baseline at {BASELINE_PATH}; run with --write-baseline",
            file=sys.stderr,
        )
        return 1

    if report["missing_count"] > baseline:
        print(
            f"claim lexicon FAILED: {report['missing_count']} file(s) carry a loaded "
            f"term in their name without an operational definition (baseline {baseline})",
            file=sys.stderr,
        )
        for item in report["missing_definition"][:20]:
            print(f"  {item['file']}  [{', '.join(item['terms'])}]", file=sys.stderr)
        print(
            "\nAdd a line to the module docstring beginning "
            f"{' or '.join(DEFINITION_MARKERS)!r} that finishes the sentence "
            "'this measures ...' in plain engineering vocabulary.",
            file=sys.stderr,
        )
        return 1

    print(
        f"claim lexicon: {len(report['compliant'])} of "
        f"{report['scanned_files_with_loaded_names']} loaded-name files define what "
        f"they measure ({report['missing_count']} remaining, baseline {baseline})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
