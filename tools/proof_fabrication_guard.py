#!/usr/bin/env python3
"""Fail closed if a proof tool fabricates its own scores.

Two real fabrications were found and removed in the proof layer (2026-06-15):
  - run_prompt_baseline_ablation.py loaded tasks but never ran them, hardcoded
    baseline scores, and asserted victory;
  - run_agi_capability_battery.py generated a 17-category capability scorecard
    from Gaussian noise around base_perf = 0.86 + cpi*0.04.

This guard scans the proof tools (tools/agi, tools/proof) for the two exact
mechanisms so neither can return:

  A. synthetic_score_from_probe — a ``*_score`` / ``*_perf`` variable assigned an
     expression that scales a liveness/probe index (``cpi``). A reported score
     must come from grading real outputs, not from a probe-pass formula.
  B. assert_victory_over_hardcoded_scores — a file that hardcodes >= 2 baseline
     score literals AND asserts a score beats a threshold. Honest proofs either
     grade real tasks or label fixtures as controlled-smoke / effects-not-verified
     and never assert victory over invented numbers.

It is deliberately narrow (low false-positive): the honest tools — including the
sovereignty gauntlet's labeled controlled-smoke fixtures and the negative-control
in run_live_harness_proof.py — do not trip it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROOF_DIRS = ("tools/agi", "tools/proof")

# A score/perf variable assigned an expression that references the probe index.
_SCORE_FROM_PROBE = re.compile(r"^\s*\w*(?:perf|score)\w*\s*=\s*.*\bcpi\b", re.IGNORECASE)
# A hardcoded baseline-style score literal in a dict.
_HARDCODED_SCORE = re.compile(r'"(?:mean|mean_score|score|accuracy)"\s*:\s*0\.\d+')
# An assertion comparing a score/perf/mean term against something (victory check).
_ASSERT_VICTORY = re.compile(r"\bassert\b.*(?:score|perf|mean).*[<>]")


@dataclass(frozen=True)
class Finding:
    kind: str
    file: str
    line: int
    detail: str


def _iter_proof_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in PROOF_DIRS:
        d = root / rel
        if d.is_dir():
            files.extend(p for p in d.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(set(files))


def scan_source(rel: str, source: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = source.splitlines()
    hardcoded_count = 0
    first_assert: tuple[int, str] | None = None
    for line_no, line in enumerate(lines, start=1):
        if _SCORE_FROM_PROBE.search(line):
            findings.append(
                Finding("synthetic_score_from_probe", rel, line_no, line.strip()[:200])
            )
        if _HARDCODED_SCORE.search(line):
            hardcoded_count += 1
        if first_assert is None and _ASSERT_VICTORY.search(line):
            first_assert = (line_no, line.strip()[:200])
    if hardcoded_count >= 2 and first_assert is not None:
        findings.append(
            Finding(
                "assert_victory_over_hardcoded_scores",
                rel,
                first_assert[0],
                first_assert[1],
            )
        )
    return findings


def scan(root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    for path in _iter_proof_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(scan_source(rel, source))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args(argv)
    findings = scan(Path(args.root))
    report = {
        "passed": not findings,
        "findings_count": len(findings),
        "findings": [asdict(f) for f in findings],
    }
    print(json.dumps(report, indent=2))
    if findings:
        print(
            "\nProof fabrication detected. A proof must grade real outputs or "
            "label fixtures as controlled-smoke — never assert victory over "
            "invented numbers.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
