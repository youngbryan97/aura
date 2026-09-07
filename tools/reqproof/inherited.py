"""Inventory inherited ledgers without converting extraction into closure.

The existing docket remains the requirement/evidence authority. This companion
indexes the other ledger forms and preserves every nonblank source span so
prose obligations cannot disappear behind a checkbox-only count. Decisions
must be recorded against exact source hashes in a separate review document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from tools.reqproof.docket import _atomic_write, build_docket_report
from tools.reqproof.evidence import load_evidence_ledger
from tools.reqproof.schema import load_registry

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "docs/AURA_EXECUTION_TRACKER.md",
    "docs/GENERALITY_TODO.md",
    "docs/AGI_GAUNTLET_TRACKER.md",
    "docs/worktodo/TODO.md",
    "docs/gap_atlas/TODO.md",
    "docs/CONNECTOME_TODO.md",
    "docs/LEARNED_LANGUAGE_INTERPRETATION_TODO.md",
    "docs/RECURSIVE_ENDOGENOUS_EXPANSION_TODO.md",
    "docs/AUTONOMOUS_DEVELOPMENTAL_AGENCY_TODO.md",
    "docs/evidence/CLOSEOUT.md",
)
CHECKBOX = re.compile(r"\[([ xX])\]")
OPEN_LANGUAGE = re.compile(
    r"\b(?:TODO|PARTIAL|NOT RUN|IN PROGRESS|DECLARED|BLOCKED|pending|"
    r"unresolved|remaining|still open|still to run|not yet|needs?|"
    r"must|next action|next exact|follow.up|not started)\b", re.I
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan_source(path: str, text: str) -> dict:
    """Keep disjoint nonblank spans; signals select review, never decide it."""
    lines = text.splitlines(keepends=True)
    blocks = []
    heading = ""
    start = None
    fence = None
    boxes = []
    signals = []

    def flush(end: int) -> None:
        nonlocal start, boxes, signals
        if start is None:
            return
        body = "".join(lines[start:end])
        blocks.append({
            "id": f"{path}:{start + 1}",
            "path": path,
            "start_line": start + 1,
            "end_line": end,
            "heading": heading,
            "sha256": digest(body),
            "text": body,
            "checkboxes": boxes,
            "review_signals": sorted(set(signals)),
            "review_status": "unreviewed",
        })
        start, boxes, signals = None, [], []

    for number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped and fence is None:
            flush(number)
            continue
        if line.startswith("#") and fence is None:
            flush(number)
            heading = stripped
        elif fence is None and (stripped.startswith("|") or re.match(r"^(?:[-*] |\d+\. )", line)):
            flush(number)
        if start is None:
            start = number
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is not None:
            continue
        for match in CHECKBOX.finditer(line):
            boxes.append({"line": number + 1, "column": match.start() + 1,
                          "checked": match.group(1).lower() == "x"})
            if match.group(1) == " ":
                signals.append("unchecked")
        if OPEN_LANGUAGE.search(line):
            signals.append("open_language")
        if stripped.startswith("|"):
            signals.append("table")
    flush(len(lines))
    return {"path": path, "sha256": digest(text), "line_count": len(lines),
            "nonblank_lines": sum(bool(line.strip()) for line in lines),
            "blocks": blocks}


def apply_reviews(sources: list[dict], reviews: dict) -> dict:
    """Reject stale decisions and lossy grouping, including unresolved prose."""
    blocks = {block["id"]: block for source in sources for block in source["blocks"]}
    decisions = reviews.get("decisions", [])
    seen = set()
    groups = {}
    for decision in decisions:
        identity = decision["source_id"]
        if identity in seen or identity not in blocks:
            raise ValueError(f"duplicate or unknown review: {identity}")
        seen.add(identity)
        block = blocks[identity]
        if decision["source_sha256"] != block["sha256"]:
            raise ValueError(f"stale review: {identity}")
        status = decision["status"]
        if status not in {"open", "historical", "superseded", "evidence_needed", "verified"}:
            raise ValueError(f"invalid status: {identity}")
        if not decision.get("reason"):
            raise ValueError(f"review lacks rationale: {identity}")
        if status in {"superseded", "verified"} and not decision.get("evidence"):
            raise ValueError(f"review lacks evidence: {identity}")
        mechanism = decision.get("mechanism")
        if not mechanism:
            raise ValueError(f"review lacks mechanism: {identity}")
        block["review_status"] = status
        block["review"] = decision
        groups.setdefault(mechanism, []).append(identity)
    return {key: sorted(values) for key, values in sorted(groups.items())}


def build_report(root: Path, reviews: dict) -> dict:
    sources = [scan_source(path, (root / path).read_text()) for path in SOURCES]
    groups = apply_reviews(sources, reviews)
    registry = load_registry(root / "config/requirement_registry.json")
    docket = build_docket_report(
        root=root, registry=registry,
        ledger=load_evidence_ledger(root / "config/requirement_evidence_ledger.json"),
    )
    atlas = json.loads((root / "docs/gap_atlas/adjudication.json").read_text())["entries"]
    outstanding = {key: value for key, value in atlas.items()
                   if value.get("outstanding") or value.get("status") != "closed"}
    blocks = [block for source in sources for block in source["blocks"]]
    states = Counter(block["review_status"] for block in blocks)
    return {
        "schema_version": 1,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scope": list(SOURCES),
        "summary": {
            "source_lines": sum(source["line_count"] for source in sources),
            "source_blocks": len(blocks),
            "checkboxes": sum(len(block["checkboxes"]) for block in blocks),
            "unchecked": sum(not box["checked"] for block in blocks for box in block["checkboxes"]),
            "unchecked_mapped": sum(
                not box["checked"] for block in blocks
                if block["review_status"] != "unreviewed"
                for box in block["checkboxes"]
            ),
            "review_states": dict(sorted(states.items())),
            "signal_blocks": sum(bool(block["review_signals"]) for block in blocks),
            "atlas_outstanding": len(outstanding),
            "mechanism_groups": len(groups),
            "inventory_review_complete": not states["unreviewed"],
        },
        "sources": sources,
        "mechanisms": groups,
        "requirement_docket": docket,
        "atlas_outstanding": outstanding,
        "non_claims": [
            "Reading source bytes is not a semantic review.",
            "Signals are review candidates, not definitive obligation detection.",
            "A current missing receipt does not prove the implementation is absent.",
            "An atlas mechanism closure does not close its outstanding campaign.",
            "All source blocks remain in scope, including blocks with no signal.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(ROOT, json.loads(args.reviews.read_text()))
    _atomic_write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
