#!/usr/bin/env python3
"""Move out of her self-reflection stores what she did not conclude, keeping every entry in a file beside them.

By 18 October 2026 her first principles (``first_principles.json``) held 201 entries, 93 of them distinct, the
most kept being what her router returns when her model does not answer ("I heard you. My thinking is running
deeper than my words right now.", twenty-one times). Her inner insights (``identity.json``) were nineteen copies
of a template with a kinship record and retrieval text pasted into it, and two of another template filled with a
belief and an interest. The code that wrote them no longer does (``core/adaptation/abstraction_engine.py``,
``core/agency/agency_core.py``); this moves what it wrote.

A principle is moved when ``training_contamination_reasons`` names it, or when it repeats one kept before it (its
uses are added to that one). An insight is moved when it is one of the two templates. Nothing is deleted: what is
moved is appended to ``<store>.quarantined-<date>.json``.

Usage:
    quarantine_what_she_did_not_conclude.py [--data DIR] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The openings of the two templates that wrote her insights, as the code wrote them.
TEMPLATES = ("I've been reflecting on my bond with ", "Synthesis: Merging '")


def _normal(text: str) -> str:
    return " ".join(str(text).lower().split())


def principles_kept_and_moved(entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from core.learning.live_learner import training_contamination_reasons

    kept: dict[str, dict[str, Any]] = {}
    moved = []
    for entry in entries:
        text = str(entry.get("principle", ""))
        key = _normal(text)
        if training_contamination_reasons(text):
            moved.append({**entry, "moved_because": "a holding line, not a principle"})
        elif len(re.sub(r"</?answer>", " ", text).split()) < 3:
            moved.append({**entry, "moved_because": "nothing said, its answer tags aside"})
        elif key in kept:
            kept[key]["application_count"] = int(kept[key].get("application_count") or 0) + int(entry.get("application_count") or 0)
            moved.append({**entry, "moved_because": "the same as one kept before it"})
        else:
            kept[key] = dict(entry)
    return list(kept.values()), moved


def insights_kept_and_moved(insights: list[Any]) -> tuple[list[Any], list[Any]]:
    kept, moved = [], []
    for insight in insights:
        (moved if str(insight).startswith(TEMPLATES) else kept).append(insight)
    return kept, moved


def _write(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def _append(path: Path, items: list[Any]) -> None:
    before = json.loads(path.read_text()) if path.exists() else []
    _write(path, [*before, *items])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path.home() / ".aura/data")
    parser.add_argument("--apply", action="store_true", help="write; without it, only say what would move")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d")
    report = {}

    store = args.data / "first_principles.json"
    raw = json.loads(store.read_text())
    entries = raw.get("payload") if isinstance(raw, dict) else raw
    kept, moved = principles_kept_and_moved(list(entries or []))
    report["first_principles"] = {"before": len(entries or []), "kept": len(kept), "moved": len(moved)}
    if args.apply and moved:
        _append(store.with_name(f"first_principles.quarantined-{stamp}.json"), moved)
        _write(store, {**raw, "payload": kept} if isinstance(raw, dict) else kept)

    identity_path = args.data / "identity.json"
    identity = json.loads(identity_path.read_text())
    insights = list(identity.get("inner_insights") or [])
    kept_insights, moved_insights = insights_kept_and_moved(insights)
    report["inner_insights"] = {"before": len(insights), "kept": len(kept_insights), "moved": len(moved_insights)}
    if args.apply and moved_insights:
        _append(identity_path.with_name(f"identity.inner_insights.quarantined-{stamp}.json"), moved_insights)
        _write(identity_path, {**identity, "inner_insights": kept_insights})

    print(json.dumps({"applied": args.apply, **report}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
