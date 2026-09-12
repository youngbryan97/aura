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

from tools.reqproof import obligations
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

    def a_continuation_follows(after: int) -> bool:
        """Is the next thing a paragraph belonging to the list item above?

        A Markdown list item may hold several paragraphs, separated by blank
        lines and indented under the marker. Ending the block at the blank
        line reads each of those paragraphs as a top-level block of its own,
        with no checkbox to say whether it is done — so an item's own account
        of what it measured came back as an unrouted obligation. 32 of them in
        one document, all continuations of items already ticked.

        Only indentation decides, and a new marker at any depth starts its own
        item rather than continuing this one.
        """
        for ahead in lines[after + 1:]:
            if not ahead.strip():
                continue
            if not ahead.startswith((" ", "\t")):
                return False
            return not re.match(r"^\s+(?:[-*] |\d+\. )", ahead)
        return False

    inside_a_list_item = False
    for number, line in enumerate(lines):
        stripped = line.strip()
        if not stripped and fence is None:
            if inside_a_list_item and a_continuation_follows(number):
                continue
            flush(number)
            inside_a_list_item = False
            continue
        if line.startswith("#") and fence is None:
            flush(number)
            inside_a_list_item = False
            heading = stripped
        elif fence is None and (stripped.startswith("|") or re.match(r"^(?:[-*] |\d+\. )", line)):
            flush(number)
            inside_a_list_item = bool(re.match(r"^(?:[-*] |\d+\. )", line))
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
    for retired in reviews.get("retired_decisions", []):
        if not retired.get("retired_reason") or not retired.get("superseded_by"):
            raise ValueError(f"retired review lacks provenance: {retired['source_id']}")
        current = blocks.get(retired["source_id"])
        if current is not None and current["sha256"] == retired["source_sha256"]:
            raise ValueError(
                f"review retired while its text still stands: {retired['source_id']}")
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


DISMISSAL_RULES = (
    "structural_framing",
    "table_header",
    "atlas_card_closed",
    "all_boxes_checked",
    "lead_in",
)
MECHANISM_MAP = "config/inherited_mechanism_map.json"
AUDIT_GOLD = "config/inherited_obligation_audit.json"
QUEUE_DOC = "docs/AURA_1_0_MASTER_TODO.md"


def _queue_items(root: Path) -> set[str]:
    text = (root / QUEUE_DOC).read_text()
    return set(re.findall(r"^\s*-\s*\[[ xX]\]\s*([A-Z]\d{2})\b", text, re.M))


def classify_and_route(root: Path) -> dict:
    """Partition every block, then route every one that survives dismissal.

    A block leaves the inventory only two ways: a named rule whose predicate
    anyone can re-evaluate, or a mechanism that carries it to a queue item.
    Nothing leaves by not being mentioned.
    """
    vocabulary = obligations.load_vocabulary(root / "config/inherited_status_vocabulary.json")
    rules = [
        (rule["mechanism"], tuple(rule["queue"]), re.compile(rule["pattern"]))
        for rule in json.loads((root / MECHANISM_MAP).read_text())["rules"]
    ]
    atlas_entries = json.loads((root / "docs/gap_atlas/adjudication.json").read_text())["entries"]
    known_items = _queue_items(root)
    unknown_targets = sorted({
        item for _, queue, _ in rules for item in queue if item not in known_items
    })

    sources, records, routes = [], [], {}
    for path in SOURCES:
        text = (root / path).read_text()
        source = scan_source(path, text)
        source["historical"] = obligations.is_historical_source(text)
        sources.append(source)
        blocks = source["blocks"]
        classified = [
            obligations.classify_block(block, vocabulary, source["historical"])
            for block in blocks
        ]
        required = [
            (not source["historical"]) or item["kind"] == "obligation"
            for item in classified
        ]
        context = {
            "atlas": atlas_entries,
            "has_reviewed_successor": {
                blocks[index]["id"] for index in range(len(blocks) - 1) if required[index + 1]
            },
        }
        for index, block in enumerate(blocks):
            record = dict(classified[index])
            record["path"] = path
            record["in_scope"] = required[index]
            record["dismissed_by_rule"] = None
            record["mechanisms"] = []
            record["queue"] = []
            if required[index]:
                for name in DISMISSAL_RULES:
                    if obligations.evaluate_rule(name, block, context):
                        record["dismissed_by_rule"] = name
                        break
                if record["dismissed_by_rule"] is None:
                    subject = f"{block['heading']}\n{block['text']}"
                    for mechanism, queue, pattern in rules:
                        if pattern.search(subject):
                            record["mechanisms"].append(mechanism)
                            record["queue"].extend(queue)
                    record["queue"] = sorted(set(record["queue"]))
                    for item in record["queue"]:
                        routes.setdefault(item, []).append(block["id"])
            records.append(record)
    return {
        "sources": sources,
        "records": records,
        "routes": {key: sorted(value) for key, value in sorted(routes.items())},
        "vocabulary_coverage": obligations.vocabulary_coverage(sources, vocabulary),
        "unknown_queue_targets": unknown_targets,
    }


def build_report(root: Path, reviews: dict) -> dict:
    analysis = classify_and_route(root)
    sources = analysis["sources"]
    records = analysis["records"]
    groups = apply_reviews(sources, reviews)
    registry = load_registry(root / "config/requirement_registry.json")
    docket = build_docket_report(
        root=root, registry=registry,
        ledger=load_evidence_ledger(root / "config/requirement_evidence_ledger.json"),
    )
    atlas = json.loads((root / "docs/gap_atlas/adjudication.json").read_text())["entries"]
    outstanding = {key: value for key, value in atlas.items()
                   if value.get("outstanding") or value.get("status") != "closed"}
    gold_path = root / AUDIT_GOLD
    gold = json.loads(gold_path.read_text()) if gold_path.exists() else {"seed": 0, "labels": {}}
    audit = obligations.dismissal_audit(records, gold, int(gold.get("per_stratum", 40)))
    block_index = {block["id"]: block for source in sources for block in source["blocks"]}
    closure = obligations.closure_reference_audit(records, block_index, root)

    in_scope = [record for record in records if record["in_scope"]]
    dismissed = [record for record in in_scope if record["dismissed_by_rule"]]
    routed = [record for record in in_scope if record["queue"]]
    unrouted = sorted(
        record["id"] for record in in_scope
        if not record["dismissed_by_rule"] and not record["queue"]
    )
    kinds = Counter(record["kind"] for record in records)
    complete = (
        not unrouted
        and analysis["vocabulary_coverage"]["complete"]
        and not analysis["unknown_queue_targets"]
        and audit["complete"]
        and closure["complete"]
    )
    return {
        "schema_version": 2,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scope": list(SOURCES),
        "summary": {
            "source_lines": sum(source["line_count"] for source in sources),
            "source_blocks": len(records),
            "block_kinds": dict(sorted(kinds.items())),
            "in_scope_blocks": len(in_scope),
            "dismissed_by_rule": dict(sorted(
                Counter(record["dismissed_by_rule"] for record in dismissed).items())),
            "routed_blocks": len(routed),
            "unrouted_blocks": unrouted,
            "queue_items_carrying_inheritance": len(analysis["routes"]),
            "unknown_queue_targets": analysis["unknown_queue_targets"],
            "vocabulary_coverage": analysis["vocabulary_coverage"],
            "dismissal_audit": {
                key: value for key, value in audit.items() if key != "strata"
            } | {"strata": {
                kind: {inner: outer for inner, outer in stratum.items() if inner != "sample"}
                for kind, stratum in audit["strata"].items()
            }},
            "closure_reference_audit": closure,
            "checkboxes": sum(len(block["checkboxes"]) for source in sources
                              for block in source["blocks"]),
            "unchecked": sum(not box["checked"] for source in sources
                             for block in source["blocks"] for box in block["checkboxes"]),
            "atlas_outstanding": len(outstanding),
            "mechanism_groups": len(groups),
            "inventory_review_complete": complete,
        },
        "sources": sources,
        "records": records,
        "routes": analysis["routes"],
        "audit": audit,
        "closure_references": closure,
        "mechanisms": groups,
        "requirement_docket": docket,
        "atlas_outstanding": outstanding,
        "non_claims": [
            "Routing an obligation to a queue item does not close it.",
            "A rule dismissal is a predicate anyone can re-evaluate, not a verdict.",
            "Reading source bytes is not a semantic review.",
            "A current missing receipt does not prove the implementation is absent.",
            "An atlas mechanism closure does not close its outstanding campaign.",
            "The audit bounds what the partition missed; it does not prove nothing was missed.",
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
