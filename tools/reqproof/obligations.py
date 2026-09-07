"""Decide which inherited-ledger blocks carry an unresolved obligation.

The earlier signal was a keyword regex over `must|needs|remaining`, which fires
on ordinary English and so cannot separate an obligation from a narration. This
module replaces it with a structural decision over a closed status vocabulary
that is derived from the corpus, plus the measurement that makes the decision
falsifiable: a random audit of the blocks it dismisses, hand-labelled, with the
miss rate reported. A detector with no null is an assertion.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY_PATH = ROOT / "config/inherited_status_vocabulary.json"

# A status token counts only where a status is written: a table cell, a bolded
# stand-alone marker, an explicit `Status:` line, or the head of a dash clause.
_EMPHASIS = re.compile(r"[*_`]+")
_CHECKBOX = re.compile(r"\[([ xX])\]")
_HISTORICAL = re.compile(r"append-only|historical record|kept as written", re.I)


def load_vocabulary(path: Path | None = None) -> dict:
    data = json.loads((path or VOCABULARY_PATH).read_text())
    open_tokens = {token.upper() for token in data["open"]}
    resolved_tokens = {token.upper() for token in data["resolved"]}
    overlap = open_tokens & resolved_tokens
    if overlap:
        raise ValueError(f"token classified both open and resolved: {sorted(overlap)}")
    neutral = {token.upper() for token in data["neutral"]}
    if neutral & (open_tokens | resolved_tokens):
        raise ValueError("neutral token also carries a status")
    return {
        "open": open_tokens,
        "resolved": resolved_tokens,
        "neutral": neutral,
        "open_symbols": set(data["open_symbols"]),
        "resolved_symbols": set(data["resolved_symbols"]),
        "open_phrases_strong": [phrase.lower() for phrase in data["open_phrases_strong"]],
        "open_phrases_weak": [phrase.lower() for phrase in data["open_phrases_weak"]],
        "threshold": int(data["min_occurrences_requiring_classification"]),
    }


def _clean(cell: str) -> str:
    return _EMPHASIS.sub("", cell).strip()


def status_positions(line: str) -> list[str]:
    """Return the text spans in this line where a status may legitimately sit."""
    spans: list[str] = []
    stripped = line.strip()
    if stripped.startswith("|"):
        spans.extend(_clean(cell) for cell in stripped.strip("|").split("|"))
    for match in re.finditer(r"\*\*([^*]{2,40})\*\*", line):
        spans.append(_clean(match.group(1)))
    match = re.search(r"\bStatus:\s*(.{2,60})", line)
    if match:
        spans.append(_clean(match.group(1)))
    # `— OPEN`, `- NOT RUN — ...`: a marker introduced by a dash.
    for match in re.finditer(r"[—–-]\s*([A-Z][A-Z /]{1,24})\b", line):
        spans.append(_clean(match.group(1)))
    return [span for span in spans if span]


def _span_status(span: str, vocabulary: dict) -> str | None:
    upper = span.upper()
    for symbol in vocabulary["open_symbols"]:
        if symbol in span:
            return "open"
    for symbol in vocabulary["resolved_symbols"]:
        if symbol in span:
            return "resolved"
    # A cell is a status when it *is* the token, or leads with it before a
    # dash/paren explanation. "PARTIAL — the gate is built" is a status;
    # "the partial run" inside a sentence is not.
    head = re.split(r"[—–(:.]|\s-\s", upper, maxsplit=1)[0].strip()
    if head in vocabulary["open"]:
        return "open"
    if head in vocabulary["resolved"]:
        return "resolved"
    return None


_STRUCTURED = re.compile(r"^\s*(?:[-*]\s|\d+[.)]\s|\|)", re.M)

# Five kinds, and every block gets exactly one. The partition is the point:
# a block that nobody classified is a block nobody read.
KINDS = (
    "obligation",              # an open status, an unchecked box, or an open phrase
    "live_structured_unmarked",  # a list/table row in a live ledger with no status
    "resolved",                # a resolved marker or a checked box
    "live_framing",            # prose introducing a live ledger, no enumerated work
    "historical_narrative",    # a block inside a self-declared append-only record
)
REVIEW_REQUIRED = ("obligation", "live_structured_unmarked")


def classify_block(block: dict, vocabulary: dict, historical: bool) -> dict:
    """Classify one source block into exactly one of KINDS.

    An unchecked box outranks prose, an open status outranks a resolved one in
    the same block, and a live ledger's enumerated rows are held for review even
    when they carry no status word at all — that silence is how the surfaces in
    the learned-language ledger went uncounted.
    """
    text = block["text"]
    reasons: list[str] = []
    unchecked = sum(1 for box in block.get("checkboxes", ()) if not box["checked"])
    checked = sum(1 for box in block.get("checkboxes", ()) if box["checked"])
    if unchecked:
        reasons.append(f"unchecked_checkbox x{unchecked}")

    open_hits: list[str] = []
    resolved_hits: list[str] = []
    for line in text.splitlines():
        for span in status_positions(line):
            status = _span_status(span, vocabulary)
            if status == "open":
                open_hits.append(span[:60])
            elif status == "resolved":
                resolved_hits.append(span[:60])
    lowered = text.lower()
    strong_hits = [p for p in vocabulary["open_phrases_strong"] if p in lowered]
    weak_hits = [p for p in vocabulary["open_phrases_weak"] if p in lowered]
    if open_hits:
        reasons.append(f"open_status:{open_hits[0]}")
    if strong_hits:
        reasons.append(f"open_phrase:{strong_hits[0]}")
    if weak_hits:
        reasons.append(f"weak_phrase:{weak_hits[0]}")

    # Precedence: an unchecked box or an explicit open status outranks anything.
    # A loose open phrase does not outrank an explicit resolved marker in a
    # status position — "Known limitations published … ✅" contains the word
    # "failure" only inside a filename.
    # Precedence. An unchecked box, an explicit open status, or an unambiguous
    # forward-looking phrase outranks everything. A weak word does not outrank
    # an explicit resolved marker: "Known limitations published … ✅" contains
    # "failure" only inside a filename.
    if unchecked or open_hits or strong_hits:
        kind = "obligation"
    elif resolved_hits and weak_hits:
        kind = "resolved"
        reasons = []
    elif reasons:
        kind = "obligation"
    elif historical:
        kind = "historical_narrative"
    elif checked or resolved_hits:
        kind = "resolved"
    elif _STRUCTURED.search(text):
        kind = "live_structured_unmarked"
        reasons.append("enumerated_row_without_status")
    else:
        kind = "live_framing"
    return {
        "id": block["id"],
        "kind": kind,
        "reasons": reasons,
        "resolved_markers": resolved_hits[:3],
        "historical_source": historical,
        "review_required": kind in REVIEW_REQUIRED,
        "sha256": block["sha256"],
    }


def is_historical_source(text: str) -> bool:
    """A ledger that declares itself an append-only record in its own preamble."""
    return bool(_HISTORICAL.search("\n".join(text.splitlines()[:25])))


def vocabulary_coverage(sources: list[dict], vocabulary: dict) -> dict:
    """Every frequent status-position token must be classified, or we refuse."""
    counts: Counter[str] = Counter()
    for source in sources:
        for block in source["blocks"]:
            for line in block["text"].splitlines():
                for span in status_positions(line):
                    head = re.split(r"[—–(:.]|\s-\s", span.upper(), maxsplit=1)[0].strip()
                    if re.fullmatch(r"[A-Z][A-Z /]{1,24}", head):
                        counts[head] += 1
    known = vocabulary["open"] | vocabulary["resolved"] | vocabulary["neutral"]
    unclassified = {
        token: count for token, count in counts.items()
        if token not in known and count >= vocabulary["threshold"]
    }
    return {
        "distinct_status_tokens": len(counts),
        "unclassified_frequent": dict(sorted(unclassified.items(), key=lambda kv: -kv[1])),
        "complete": not unclassified,
    }


def audit_strata(records: list[dict]) -> dict[str, list[str]]:
    """The populations a reader is entitled to doubt.

    Two ways a block leaves the inventory, so two families of stratum: it was
    never in scope (a block inside a self-declared append-only record), or a
    named rule dismissed it. Both are sampled; neither is taken on trust.
    """
    strata: dict[str, list[str]] = {}
    for record in records:
        if not record.get("in_scope", True):
            key = f"out_of_scope:{record['kind']}"
        elif record.get("dismissed_by_rule"):
            key = f"rule:{record['dismissed_by_rule']}"
        else:
            continue
        strata.setdefault(key, []).append(record["id"])
    return {key: sorted(value) for key, value in sorted(strata.items())}


def dismissal_audit(records: list[dict], gold: dict, per_stratum: int) -> dict:
    """Sample every dismissed population and score it against hand labels.

    Stratified because the populations differ by three orders of magnitude: a
    uniform draw over ten thousand historical blocks would never reach the
    fifty a rule dismissed. The seed lives in the gold file, so the draw is
    reproducible and cannot be redrawn until a flattering sample appears.
    """
    labels = gold.get("labels", {})
    result, misses_all, unlabelled_all = {}, [], []
    for key, pool in audit_strata(records).items():
        rng = random.Random(f"{gold.get('seed', 0)}:{key}")
        drawn = sorted(rng.sample(pool, min(per_stratum, len(pool))))
        unlabelled = [i for i in drawn if i not in labels]
        misses = [i for i in drawn
                  if labels.get(i, {}).get("carries_obligation") is True]
        scored = len(drawn) - len(unlabelled)
        result[key] = {
            "population": len(pool),
            "sample": drawn,
            "scored": scored,
            "unlabelled": unlabelled,
            "missed_obligations": misses,
            "miss_rate": (len(misses) / scored) if scored else None,
            # Rule of three: zero misses in n draws bounds the true rate at 3/n.
            "miss_rate_upper_95": (3.0 / scored) if scored and not misses else None,
            "missed_population_upper_95": (
                round(len(pool) * 3.0 / scored, 1) if scored and not misses else None
            ),
        }
        misses_all.extend(misses)
        unlabelled_all.extend(unlabelled)
    return {
        "dismissed_total": sum(value["population"] for value in result.values()),
        "per_stratum": per_stratum,
        "strata": result,
        "missed_obligations": sorted(misses_all),
        "unlabelled": sorted(unlabelled_all),
        "complete": not unlabelled_all and not misses_all,
    }



# --- Rule-based decisions -------------------------------------------------
#
# A decision may be made by a named rule instead of by hand, but only when the
# rule is a predicate over the block that anyone can re-evaluate. Every rule
# below is checked against every block it claims, so a rule that stops holding
# fails the gate rather than silently keeping its verdict.

_HEADING_ONLY = re.compile(r"\A(?:#{1,6} .*|-{3,}|={3,}|\|[-: |]+\|)\s*\Z")
_FENCE_ONLY = re.compile(r"\A\s*(?:`{3,}|~{3,})", re.M)


def rule_structural_framing(block: dict, context: dict) -> bool:
    """A bare heading or rule carries no obligation of its own."""
    return bool(_HEADING_ONLY.match(block["text"].strip()))


def rule_all_boxes_checked(block: dict, context: dict) -> bool:
    """Every checkbox in the block is checked and there is at least one."""
    boxes = block.get("checkboxes", ())
    return bool(boxes) and all(box["checked"] for box in boxes)


def rule_atlas_card_closed(block: dict, context: dict) -> bool:
    """A Gap Atlas row whose adjudication entry is closed with nothing outstanding."""
    if not block["path"].endswith("gap_atlas/TODO.md"):
        return False
    entries = context.get("atlas", {})
    identifiers = re.findall(r"\b([A-Z]\d{1,2}\.\d{1,2})\b", block["text"])
    if not identifiers:
        return False
    for identifier in identifiers:
        entry = entries.get(identifier)
        if entry is None or entry.get("status") != "closed" or entry.get("outstanding"):
            return False
    return True


RULES = {
    "structural_framing": rule_structural_framing,
    "all_boxes_checked": rule_all_boxes_checked,
    "atlas_card_closed": rule_atlas_card_closed,
}


def evaluate_rule(name: str, block: dict, context: dict) -> bool:
    try:
        return RULES[name](block, context)
    except KeyError as error:
        raise ValueError(f"unknown rule: {name}") from error


_LEAD_IN = re.compile(r"\A[^\n]{0,220}:\s*\Z")


def rule_lead_in(block: dict, context: dict) -> bool:
    """A one-line sentence ending in a colon introduces the block after it.

    Its obligation is the following block's, so routing it separately would
    double-count. It is only framing when a review-required block follows.
    """
    if not _LEAD_IN.match(block["text"].strip()):
        return False
    return block["id"] in context.get("has_reviewed_successor", ())


RULES["lead_in"] = rule_lead_in


def rule_table_header(block: dict, context: dict) -> bool:
    """A table's column-name row states no obligation of its own."""
    stripped = block["text"].strip()
    if not stripped.startswith("|"):
        return False
    cells = [_clean(cell).lower() for cell in stripped.strip("|").split("|")]
    header_words = {"#", "item", "status", "gate", "state", "unit", "layer",
                    "evidence", "evidence family", "master id", "verdict",
                    "criterion", "verification", "mechanism", "note", "reading",
                    "surface", "site", "next action", "original obligation",
                    "reconciliation", "queue", "what decides meaning today",
                    "ground truth available today", "level", "requirements"}
    return bool(cells) and all(cell in header_words or not cell for cell in cells)


RULES["table_header"] = rule_table_header


_PATH_REFERENCE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|json|md|toml|yaml|yml))`")


def closure_reference_audit(records: list[dict], blocks: dict, root) -> dict:
    """Check that every path a dismissed block cites as its closure still exists.

    A checked box whose evidence has been deleted or renamed is a stale
    checkbox, and stale checkboxes are exactly what the reconciliation is for.
    A bare filename resolves by basename anywhere in the tree, because the
    ledgers cite modules the way a person says them out loud.
    """
    skip = {".git", ".venv", "node_modules", ".claude", "__pycache__"}
    index: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Relative parts, not absolute: a worktree lives under `.claude/`, and
        # testing the absolute path excluded the entire tree.
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts):
            continue
        index.setdefault(path.name, []).append(str(relative))

    checked, missing = 0, []
    for record in records:
        if not record.get("dismissed_by_rule"):
            continue
        for reference in sorted(set(_PATH_REFERENCE.findall(blocks[record["id"]]["text"]))):
            checked += 1
            if (root / reference).exists():
                continue
            if index.get(Path(reference).name):
                continue
            missing.append({"block": record["id"], "reference": reference})
    return {
        "references_checked": checked,
        "unresolved": missing,
        "complete": not missing,
    }
