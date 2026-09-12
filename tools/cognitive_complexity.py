#!/usr/bin/env python3
"""tools/cognitive_complexity.py — the mind should get simpler as the evidence gets stronger.

Four of the compared architectures derive broad behaviour from a small kernel.
Aura derives it from many organs, and the review's word for that is saturation:
adding an organ is the cheapest way to address a deficit and the hardest thing
to undo, so the count only goes up.

This measures the count and ratchets it. Three numbers, and the second is the
one that matters:

* **Organs.** Top-level packages under ``core/`` holding cognitive machinery.
* **Dependency entropy.** Shannon entropy over the cross-package import
  distribution. A tree where everything imports everything has high entropy at
  any organ count, and it is entropy rather than count that makes a system hard
  to reason about - twenty packages in a line are simpler than ten in a mesh.
* **Kernel size.** Lines in the packages an answer cannot be produced without.

The ratchet only goes down, like the writing and layering baselines. Adding an
organ is allowed; adding one and leaving the numbers where they were is not,
because that is the growth the review named.

Redundancy
----------
``--redundant`` lists organs that share an invariant. Two organs implementing
the same computational invariant are candidates for merging, and the map from
organ to invariant is the compression programme's working list rather than a
taxonomy for its own sake.

    python tools/cognitive_complexity.py            # check against the baseline
    python tools/cognitive_complexity.py --write    # lower the baseline
    python tools/cognitive_complexity.py --redundant
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "core"
BASELINE = ROOT / "config/cognitive_complexity_baseline.json"

#: Packages an answer cannot be produced without. The kernel is what has to
#: stay simple; the rest may grow and be pruned.
KERNEL = (
    "brain", "cognition", "consciousness", "evidence", "knowledge",
    "memory", "planning", "reasoning", "runtime", "world_model",
)

#: The computational invariant each cognitive package implements. Two packages
#: sharing one are candidates for merging. Written by hand and deliberately
#: coarse: a taxonomy with one entry per package proves nothing.
INVARIANTS: dict[str, str] = {
    "cognition": "decide under uncertainty and compile what recurs",
    "consciousness": "make one content available to many processes",
    "evidence": "keep belief tied to what it rests on",
    "knowledge": "hold structure that can be queried and rewritten",
    "memory": "keep what will matter and drop what will not",
    "learning": "change in response to outcome",
    "world_model": "predict the next state and be wrong about it usefully",
    "planning": "search a space of futures under a budget",
    "reasoning": "derive what follows",
    "science": "say what has been established and what has not",
    "verify": "check a claim against the thing it claims about",
    "evaluation": "compare arms that were allowed the same resources",
    "perception": "turn a signal into something with an identity",
    "agency": "act, and know that acting is what happened",
    "skills": "act, and know that acting is what happened",
    "governance": "refuse what may not happen",
    "security": "refuse what may not happen",
    "affect": "value a state without deliberating about it",
    "welfare": "value a state without deliberating about it",
    "being": "sense its own state",
    "somatic": "sense its own state",
    "identity": "stay the same one across time",
    "ontogeny": "stay the same one across time",
}


def _packages() -> list[str]:
    return sorted(
        p.name for p in CORE.iterdir()
        if p.is_dir() and not p.name.startswith((".", "__")) and (p / "__init__.py").exists()
    )


def _edges() -> Counter:
    """Cross-package import edges under core/."""
    counts: Counter = Counter()
    for path in CORE.rglob("*.py"):
        relative = path.relative_to(CORE)
        if not relative.parts or relative.parts[0].startswith("__"):
            continue
        source_package = relative.parts[0]
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name if node.names else ""
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "core" and parts[1] != source_package:
                counts[(source_package, parts[1])] += 1
    return counts


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _kernel_lines() -> int:
    total = 0
    for name in KERNEL:
        package = CORE / name
        if not package.exists():
            continue
        for path in package.rglob("*.py"):
            try:
                total += len(path.read_text(errors="replace").splitlines())
            except OSError:
                continue
    return total


def measure() -> dict[str, float | int]:
    packages = _packages()
    edges = _edges()
    return {
        "organs": len(packages),
        "cross_package_edges": len(edges),
        "dependency_entropy": round(_entropy(edges), 4),
        "kernel_lines": _kernel_lines(),
    }


def redundancy() -> dict[str, list[str]]:
    """Organs that share an invariant, which is the merge candidate list."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for package, invariant in INVARIANTS.items():
        if (CORE / package).exists():
            grouped[invariant].append(package)
    return {k: sorted(v) for k, v in sorted(grouped.items()) if len(v) > 1}


def unmapped() -> list[str]:
    """Cognitive packages with no declared invariant.

    Not a violation. A package with no invariant is one nobody has said what it
    is FOR, and that is the list the compression programme starts from.
    """
    return sorted(set(_packages()) - set(INVARIANTS))


def _git(*args: str) -> str:
    """Read something from git, or an empty string if git cannot answer."""
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def head_commit() -> str:
    return _git("rev-parse", "HEAD").strip()


def landed_since(commit: str, *, top: int = 12) -> dict:
    """What grew between ``commit`` and now, per package and per file.

    A raise used to carry one sentence. Over a window of days that sentence
    stands in for a hundred files nobody can name afterwards, and the entry
    reads as a reset rather than a record. This is the same window measured
    rather than described, so a reader can check the sentence against it.
    """
    if not commit:
        return {}
    kernel_dirs = [f"core/{name}" for name in KERNEL]
    numstat = _git("diff", "--numstat", f"{commit}..HEAD", "--", *kernel_dirs)
    if not numstat.strip():
        return {}

    per_file: dict[str, int] = {}
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue  # a binary file, which has no lines to count
        added, deleted, path = int(parts[0]), int(parts[1]), parts[2]
        per_file[path] = per_file.get(path, 0) + added - deleted

    per_package: dict[str, int] = {}
    for path, net in per_file.items():
        package = path.split("/")[1] if path.count("/") >= 1 else path
        per_package[package] = per_package.get(package, 0) + net

    authors: dict[str, int] = {}
    for line in _git("log", "--format=%an", f"{commit}..HEAD", "--", *kernel_dirs).splitlines():
        name = line.strip()
        if name:
            authors[name] = authors.get(name, 0) + 1

    biggest = sorted(per_file.items(), key=lambda item: -item[1])[:top]
    return {
        "since": commit,
        "net_kernel_lines": sum(per_file.values()),
        "files_touched": len(per_file),
        "per_package": dict(sorted(per_package.items(), key=lambda item: -item[1])),
        "largest_files": [{"path": path, "net_lines": net} for path, net in biggest],
        "commits_by_author": dict(sorted(authors.items(), key=lambda item: -item[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="move the baseline to today")
    parser.add_argument(
        "--because", default="",
        help="why the baseline moved. Required when any number goes UP: a ratchet "
             "that can be reset without a reason is a number, not a ratchet.",
    )
    parser.add_argument("--redundant", action="store_true", help="organs sharing an invariant")
    args = parser.parse_args()

    current = measure()
    if args.redundant:
        print(json.dumps({"sharing_an_invariant": redundancy(), "unmapped": unmapped()}, indent=2))
        return 0

    if args.write or not BASELINE.exists():
        previous = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        went_up = [k for k in current if k in previous and current[k] > previous[k]]
        if went_up and not args.because.strip():
            print(
                "cognitive-complexity: "
                + ", ".join(f"{k} rose from {previous[k]} to {current[k]}" for k in went_up)
                + ". Pass --because with the reason; a ratchet that can be reset "
                "without one is a number, not a ratchet.",
                file=sys.stderr,
            )
            return 1
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        history = previous.get("history", [])
        if went_up:
            entry = {"raised": went_up, "from": {k: previous[k] for k in went_up},
                     "to": {k: current[k] for k in went_up},
                     "because": args.because.strip()}
            landed = landed_since(str(previous.get("at_commit", "")))
            if landed:
                entry["landed"] = landed
            history = [*history, entry]
        BASELINE.write_text(
            json.dumps({**current, "at_commit": head_commit(), "history": history}, indent=2)
            + "\n"
        )
        print(f"cognitive-complexity: baseline written {current}")
        return 0

    baseline = json.loads(BASELINE.read_text())
    regressions = [
        f"{key}: {current[key]} > baseline {baseline[key]}"
        for key in current
        if key in baseline and isinstance(baseline[key], (int, float))
        and current[key] > baseline[key]
    ]
    if regressions:
        landed = landed_since(str(baseline.get("at_commit", "")))
        if landed:
            print(
                "cognitive-complexity: since the baseline, "
                f"{landed['net_kernel_lines']:+d} kernel lines over "
                f"{landed['files_touched']} files. Largest: "
                + ", ".join(
                    f"{item['path']} {item['net_lines']:+d}"
                    for item in landed["largest_files"][:5]
                ),
                file=sys.stderr,
            )
    for line in regressions:
        print(f"cognitive-complexity: {line}", file=sys.stderr)
    if regressions:
        print(
            "cognitive-complexity: the mind should get simpler as the evidence gets "
            "stronger. Merge or delete, or justify and re-baseline with --write.",
            file=sys.stderr,
        )
        return 1
    improvements = {
        k: (baseline[k], current[k]) for k in current
        if isinstance(baseline.get(k), (int, float)) and current[k] < baseline[k]
    }
    print(f"cognitive-complexity: within baseline {current}"
          + (f"; improved {improvements}" if improvements else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
