"""Hold every external review item to a disposition.

"All advisory items were read" is not checkable and not a disposition. This
extracts the items structurally from each corpus and reports the ones that
reach no verdict, so a review cannot be counted as incorporated by saying so.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPORA = ROOT / "config/advisory_corpora.json"
DISPOSITIONS = {"adopted", "superseded", "rejected", "open"}

_NUMBERED = re.compile(r"^(\d+)\.\s+(.*)$")
_TABLE_ROW = re.compile(r"^\|\s*([A-Za-z]?\d{1,3}(?:\.\d{1,2})?)\s*\|(.*)$")
_HEADING = re.compile(r"^#{2,3}\s+(.*)$")
_STRIKE = re.compile(r"~~")

_SECTION_DISPOSITION = (
    ("rejected", "rejected"),
    ("still open", "open"),
    ("not run", "open"),
    ("open", "open"),
    ("superseded", "superseded"),
    ("adopted", "adopted"),
    ("closed since", "adopted"),
    ("built", "adopted"),
)


def _section_disposition(heading: str) -> str | None:
    lowered = heading.lower()
    for token, disposition in _SECTION_DISPOSITION:
        if token in lowered:
            return disposition
    return None


#: A verdict cell often opens with the verdict and then explains it, so the
#: first word carries the disposition on its own. "PASS on a clean tree" was
#: read as undecided because the reader wanted the em dash that usually
#: follows.
_LEADING_VERDICT = re.compile(r"^\W*(pass|fail|closed|open|rejected|superseded)\b", re.I)


def _cell_disposition(text: str) -> str | None:
    lowered = text.lower()
    if "rejected" in lowered:
        return "rejected"
    if "superseded" in lowered:
        return "superseded"
    if any(token in lowered for token in ("built", "done", "pass —", "pass -", "closed")):
        return "adopted"
    if any(token in lowered for token in ("not run", "todo", "declared", "in progress",
                                          "partial", "not started")):
        return "open"
    # The verdict lives in the status cell, which is the last one. Matching the
    # whole row would read the item's own name as its verdict.
    cells = [cell.strip() for cell in text.split("|") if cell.strip()]
    leading = _LEADING_VERDICT.match(cells[-1]) if cells else None
    if leading:
        return {"pass": "adopted", "closed": "adopted", "fail": "open",
                "open": "open", "rejected": "rejected",
                "superseded": "superseded"}[leading.group(1).lower()]
    return None


def extract(corpus: dict, root: Path) -> list[dict]:
    """Pull the items out of one corpus. Structure decides, never a summary line."""
    path = root / corpus["source"]
    if corpus["kind"] == "adjudicated_json":
        cards = json.loads(path.read_text())
        entries = json.loads((root / corpus["adjudication"]).read_text())["entries"]
        items = []
        for card in cards:
            identity = str(card.get("id") or card.get("card_id") or card.get("number"))
            entry = entries.get(identity)
            if entry is None:
                disposition = None
            elif entry.get("outstanding"):
                disposition = "open"
            elif entry.get("status") == "closed":
                disposition = "adopted"
            else:
                disposition = "open"
            items.append({
                "corpus": corpus["id"], "id": identity,
                "text": str(card.get("title") or card.get("subject") or "")[:200],
                "disposition": disposition,
                "evidence": (entry or {}).get("closed_by") or (entry or {}).get("campaign_run"),
                "source": "adjudication.json",
            })
        return items

    items, heading = [], ""
    for line in path.read_text().splitlines():
        match = _HEADING.match(line)
        if match:
            heading = match.group(1)
            continue
        if corpus["kind"] == "numbered_markdown":
            match = _NUMBERED.match(line)
            if not match:
                continue
            disposition = _section_disposition(heading)
            if disposition is None and _STRIKE.search(line):
                disposition = "adopted"
            items.append({
                "corpus": corpus["id"], "id": match.group(1), "text": match.group(2)[:200],
                "disposition": disposition, "evidence": None,
                "source": f"section:{heading}" if disposition else None,
            })
        elif corpus["kind"] == "sectioned_table":
            match = _TABLE_ROW.match(line)
            if not match:
                continue
            body = match.group(2)
            if set(body.replace("|", "").strip()) <= set("-: "):
                continue
            disposition = _cell_disposition(body) or _section_disposition(heading)
            items.append({
                "corpus": corpus["id"], "id": match.group(1),
                "text": body.strip("| ")[:200],
                "disposition": disposition, "evidence": None,
                "source": "status cell" if _cell_disposition(body) else (
                    f"section:{heading}" if disposition else None),
            })
    return items


def apply_dispositions(items: list[dict], recorded: dict) -> list[dict]:
    """Fill in what the corpus does not say for itself, from a reviewed file."""
    for item in items:
        key = f"{item['corpus']}:{item['id']}"
        entry = recorded.get(key)
        if entry is None:
            continue
        if entry["disposition"] not in DISPOSITIONS:
            raise ValueError(f"invalid disposition: {key}")
        if entry["disposition"] == "rejected" and not entry.get("reason"):
            raise ValueError(f"rejection without a reason: {key}")
        if entry["disposition"] in {"adopted", "superseded"} and not entry.get("evidence"):
            raise ValueError(f"{entry['disposition']} without evidence: {key}")
        item["disposition"] = entry["disposition"]
        item["evidence"] = entry.get("evidence")
        item["reason"] = entry.get("reason")
        item["source"] = "config/advisory_dispositions.json"
    return items


def build(root: Path = ROOT) -> dict:
    corpora = json.loads(CORPORA.read_text())["corpora"]
    recorded_path = root / "config/advisory_dispositions.json"
    recorded = (json.loads(recorded_path.read_text())["dispositions"]
                if recorded_path.exists() else {})
    items: list[dict] = []
    for corpus in corpora:
        items.extend(extract(corpus, root))
    apply_dispositions(items, recorded)
    undecided = sorted(f"{item['corpus']}:{item['id']}" for item in items
                       if item["disposition"] is None)
    per_corpus = {}
    for corpus in corpora:
        subset = [item for item in items if item["corpus"] == corpus["id"]]
        per_corpus[corpus["id"]] = {
            "items": len(subset),
            "dispositions": dict(sorted(Counter(
                item["disposition"] or "undecided" for item in subset).items())),
        }
    # Two different facts, and collapsing them was hiding the weaker one. An
    # item can be adopted because a reviewed file names what holds it, or
    # because the corpus's own status cell says so. The second is the corpus
    # marking its own homework, which is worth counting and not worth
    # treating as proof.
    on_its_own_word = sorted(
        f"{item['corpus']}:{item['id']}" for item in items
        if item["disposition"] in {"adopted", "superseded"}
        and not item.get("evidence")
        and item.get("source")
    )
    missing_evidence = sorted(
        f"{item['corpus']}:{item['id']}" for item in items
        if item["disposition"] in {"adopted", "superseded"}
        and not item.get("evidence")
        and not item.get("source")
    )
    return {
        "schema_version": 1,
        "summary": {
            "corpora": len(corpora),
            "items": len(items),
            "dispositions": dict(sorted(Counter(
                item["disposition"] or "undecided" for item in items).items())),
            "per_corpus": per_corpus,
            "undecided": undecided,
            "adopted_without_evidence": missing_evidence,
            "adopted_on_the_corpus_own_word": on_its_own_word,
            "complete": not undecided,
        },
        "items": items,
        "non_claims": [
            "A disposition is a verdict on the item, not proof the item was right.",
            "Adopted means something in this repository holds it, named in the evidence.",
            "Open means it is still owned, and by which queue item the master list says.",
            "Adopted on the corpus's own word is the corpus marking its own homework.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
