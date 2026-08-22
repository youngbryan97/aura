#!/usr/bin/env python3
"""The facts every file restates, checked against the file that owns them.

`make doc-drift` asks whether a path a document names exists. It cannot ask
whether two documents say the same thing, and that is the other half of the
problem: the image labelled itself ``licenses="MIT"`` while ``LICENSE``
reserves all rights, and nothing in the repository could notice, because both
statements were internally consistent and neither was a path.

A handful of facts get restated in a dozen places — the Python version, the
license, the port, the package version. Each one has exactly one owner here,
and every other file that states it is listed with the pattern that extracts
it. When they disagree, this says which files and what they say.

The platform section is a different shape and deliberately so. "Runs on macOS
with Apple silicon" is not a string another file restates; it is a fact about
what the runtime can do, and the check is that the evidence for it still
exists — the MLX requirements with no environment markers, and the client that
is the only substrate. A document that claims broader operation is contradicted
by the requirements file, not by a regex.

    python tools/check_product_facts.py
    python tools/check_product_facts.py --json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "config" / "product_facts.json"


def _files_for(entry: dict) -> list[Path]:
    if "glob" in entry:
        return sorted(ROOT.glob(entry["glob"]))
    return [ROOT / entry["file"]]


def _extract(pattern: str, text: str) -> list[str]:
    """Every capture the pattern makes, joined when it captures in parts."""
    found: list[str] = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        groups = [g for g in match.groups() if g is not None]
        if not groups:
            continue
        found.append(".".join(groups) if len(groups) > 1 else groups[0])
    return found


def check(payload: dict) -> list[str]:
    problems: list[str] = []

    for name, fact in sorted(payload["facts"].items()):
        expected = str(fact["value"])
        seen_anywhere = False
        for entry in fact["restated_in"]:
            for path in _files_for(entry):
                if not path.exists():
                    problems.append(
                        f"{name}: {path.relative_to(ROOT)} is listed as restating "
                        "this fact and does not exist"
                    )
                    continue
                found = _extract(entry["pattern"], path.read_text("utf-8", errors="ignore"))
                if not found:
                    problems.append(
                        f"{name}: {path.relative_to(ROOT)} no longer states it — "
                        f"the pattern {entry['pattern']!r} matches nothing, so "
                        "either the file changed shape or the fact moved"
                    )
                    continue
                seen_anywhere = True
                for value in found:
                    if value != expected:
                        problems.append(
                            f"{name}: {path.relative_to(ROOT)} says {value!r} and "
                            f"{fact['owner']} says {expected!r}"
                        )
        if not seen_anywhere:
            problems.append(f"{name}: no file states this fact any more")

    platform = payload.get("platform") or {}
    for relative in platform.get("evidence_files", []):
        if not (ROOT / relative).exists():
            problems.append(
                f"platform: {relative} is the evidence for "
                f"{platform.get('primary')!r} and does not exist"
            )
    requirements = ROOT / "requirements.txt"
    if requirements.exists():
        body = requirements.read_text("utf-8")
        unmarked = [
            line.strip()
            for line in body.splitlines()
            if line.strip().startswith(("mlx", "pyobjc")) and ";" not in line
        ]
        if not unmarked and "Apple" in str(platform.get("primary", "")):
            problems.append(
                "platform: requirements.txt no longer pins an unmarked Apple-only "
                "dependency, so the claim that the primary runtime is Apple "
                "silicon has lost its evidence"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(FACTS.read_text("utf-8"))
    problems = check(payload)

    if args.json:
        print(json.dumps({"ok": not problems, "problems": problems}, indent=2))
        return 1 if problems else 0
    if problems:
        print(f"❌ product facts: {len(problems)} disagreement(s)")
        for problem in problems:
            print(f"   • {problem}")
        return 1
    print(f"✅ {len(payload['facts'])} product facts agree everywhere they are stated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
