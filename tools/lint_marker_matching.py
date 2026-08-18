#!/usr/bin/env python3
"""Find keyword markers matched by containment that are fragments of words.

The defect this looks for has been found and fixed one site at a time:

    "in your own words"       launched Microsoft Word
    "notes.txt"               opened the Notes app
    "the latest Claude model" opened a browser conversation with Claude
    "i dont know what to do"  classified as a real-time news query
    "how do you distinguish"  classified as a practical GUI diagnostic

Every one was `marker in text`. Containment does not know where a word begins,
so a marker that is a fragment of an ordinary word will eventually meet that
word. Fixing the site that fired fixes nothing else; the next collision is a
different word.

Only INFIX collisions are reported. A word that BEGINS with the marker is that
marker inflected — "run" claiming "running" is the stem doing its job. A marker
buried mid-word belongs to something else: "test" in "latest", "gui" in
"distinguish", "now" in "know".

The collision vocabulary is this project's own prose. A word the documentation
uses is a word a person types.

Usage:
    python tools/lint_marker_matching.py            # check against baseline
    python tools/lint_marker_matching.py --report   # list every finding
    python tools/lint_marker_matching.py --baseline # tighten (only downward)
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "config" / "marker_matching_baseline.json"
SCAN_ROOTS = ("core", "interface")
SKIP_PARTS = {
    ".venv", "__pycache__", ".git", "archive", "dev_archive", ".claude",
    "artifacts", ".aura_architect", "node_modules", "tests",
}
#: A word has to appear more than once in the prose to count as one the
#: project actually writes; a single occurrence may be a typo.
_MIN_PROSE_USES = 2
#: Letters in the longest ordinary word this project's prose uses unhyphenated
#: ("distinguishes", "installation"). Longer runs are decapitalised identifiers.
_LONGEST_PLAIN_WORD = 13


def prose_vocabulary() -> collections.Counter[str]:
    words: collections.Counter[str] = collections.Counter()
    sources = list((ROOT / "docs").rglob("*.md")) + list(ROOT.glob("*.md"))
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        # Code is not prose. Without this, `FileWriteGateway` becomes the
        # "word" filewritegateway and reports a collision for "write" that no
        # sentence could ever produce.
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`[^`]*`", " ", text)
        text = re.sub(r"\S*[/\\._]\S*", " ", text)
        for word in re.findall(r"[a-z][a-z'-]{2,}", text):
            words[word] += 1
    return words


def collisions(marker: str, vocabulary: collections.Counter[str]) -> list[str]:
    probe = str(marker or "").strip().lower()
    if len(probe) < 3 or not probe.isalpha():
        return []
    found = []
    for word, uses in vocabulary.items():
        if uses < _MIN_PROSE_USES or probe not in word or word == probe:
            continue
        # An unhyphenated run of letters this long is a class name that lost
        # its capitals, not a word anyone types: filewritegateway,
        # perceptionruntime, attributeerror. The longest ordinary words in
        # this project's prose — installation, distinguishes — fit inside it.
        if "-" not in word and len(word) > _LONGEST_PLAIN_WORD:
            continue
        if any(part.startswith(probe) for part in re.split(r"[-']", word)):
            continue
        found.append(word)
    return sorted(found)[:3]


def _literals(node: ast.AST, named: dict[str, list[str]]) -> list[str]:
    if isinstance(node, ast.Tuple | ast.List | ast.Set):
        return [e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    if isinstance(node, ast.Name):
        return named.get(node.id, [])
    return []


def scan_file(path: Path, vocabulary: collections.Counter[str]) -> list[tuple[int, str, list[str]]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (SyntaxError, OSError):
        return []
    named: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    named[target.id] = _literals(node.value, {})
    hits: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.GeneratorExp):
            continue
        if not (isinstance(node.elt, ast.Compare) and node.elt.ops
                and isinstance(node.elt.ops[0], ast.In)):
            continue
        for generator in node.generators:
            for marker in _literals(generator.iter, named):
                found = collisions(marker, vocabulary)
                if found:
                    hits.append((node.lineno, marker, found))
    return hits


def findings() -> dict[str, list[tuple[int, str, list[str]]]]:
    vocabulary = prose_vocabulary()
    out: dict[str, list[tuple[int, str, list[str]]]] = {}
    for scan_root in SCAN_ROOTS:
        for path in (ROOT / scan_root).rglob("*.py"):
            if set(path.parts) & SKIP_PARTS:
                continue
            hits = scan_file(path, vocabulary)
            if hits:
                out[str(path.relative_to(ROOT))] = hits
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="list every finding")
    parser.add_argument("--baseline", action="store_true", help="record a tighter baseline")
    args = parser.parse_args()

    found = findings()
    per_file = {path: len(hits) for path, hits in sorted(found.items())}
    total = sum(per_file.values())

    if args.report:
        for path, hits in sorted(found.items(), key=lambda kv: -len(kv[1])):
            print(path)
            for line, marker, words in hits:
                print(f"   {line:6} {marker!r:16} matches inside {words}")
        print()

    baseline = {"per_file": {}, "total": 0}
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.baseline:
        grown = {p: (baseline["per_file"].get(p, 0), n)
                 for p, n in per_file.items() if n > baseline["per_file"].get(p, 0)}
        if grown and BASELINE_PATH.exists():
            print(f"REFUSED: {len(grown)} file(s) grew; a baseline refresh would launder them:")
            for path, (was, now) in sorted(grown.items())[:10]:
                print(f"   {path}: {was} -> {now}")
            return 1
        BASELINE_PATH.write_text(
            json.dumps({"per_file": per_file, "total": total}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"baseline written: {total} findings across {len(per_file)} files")
        return 0

    print(f"{total} substring markers that collide with words this project writes")
    if total > baseline["total"]:
        grown = {p: (baseline["per_file"].get(p, 0), n)
                 for p, n in per_file.items() if n > baseline["per_file"].get(p, 0)}
        print(f"FAIL: {total} > baseline {baseline['total']}. The ratchet only goes down.")
        for path, (was, now) in sorted(grown.items())[:10]:
            print(f"   {path}: {was} -> {now}")
        print("Use core.conversation.word_markers.names_any instead of `marker in text`.")
        return 1
    if total < baseline["total"]:
        print(f"OK: {total} < baseline {baseline['total']}. Re-run with --baseline to tighten.")
    else:
        print(f"OK: at baseline {total}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
