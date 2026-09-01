#!/usr/bin/env python3
"""tools/gap_atlas.py — the 220-item gap list, and the gate that keeps it honest.

Two independent reviews landed on 2026-09-01. A comparative architecture atlas
put Aura beside ACT-R, Soar, NARS, CLARION, LIDA, Nengo/SPA, OpenCog/Hyperon,
WBAI, SIMA 2, V-JEPA 2 and the frontier model class, and produced 220 cards
(ids 001-220). A cross-system engineering audit compared her against the
frontier agent stacks, OpenHands, DGM, AlphaEvolve, Godel Agent, DreamCoder,
LILO, Soar and Voyager, and produced 194 more (ids A1.1-A13.15). They overlap,
and in several places they disagree. This file holds all 414 cards, Aura's
adjudication of each against her own source, and the status of the work.

Three things it refuses, because a list this long decays into a list of
opinions without them:

* A card with no adjudication. Every one of the 414 gets a verdict, evidence
  paths and a stated bar, or ``--check`` fails.
* A closed item with no test. ``status: closed`` requires ``closed_by``, and
  every path in it must exist. A gap is not closed because code exists; it is
  closed when the named test runs.
* A closed item whose bar names a DEMONSTRATED result - "beats", "hundreds of
  tasks", a percentage - with no campaign named. Building the harness that
  would measure a win is not the win, and an entry that closes on the harness
  has to say in ``outstanding`` which campaign is still to run. That field is
  what stops "closed" reading as "demonstrated".
* A closed item whose bar demands INTEGRATION and whose module nothing calls.
  A bar that says "every", "all" or "architecture-wide" is a claim about the
  system, not about a module, and a module only tests import themselves is at
  EXISTS, not WIRED. Those entries carry ``wired_by`` naming a production
  caller, and ``--check`` verifies that the caller imports the module.
* Evidence that has moved. Every path named in ``evidence`` must still be in
  the tree, so a refactor that deletes the module an adjudication rests on
  fails here rather than leaving a stale judgement standing.

The verdicts are Aura's, not the reviews'. Cards are marked OVERSTATED where
the bar was written for a human subject, a robot arm, a datacentre or a
population of agents, and the entry says what the transferable half is
instead. Cards are marked PRESENT where a review described as missing
something the repository already had - preregistration, matched-budget
refusal, state ownership, sandbox-exec profiles, skill retrieval, computed
engineering design.
"""
from __future__ import annotations

import argparse
import re
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "docs/gap_atlas/cards.json"
ADJUDICATION = ROOT / "docs/gap_atlas/adjudication.json"
TODO = ROOT / "docs/gap_atlas/TODO.md"

REQUIRED_FIELDS = ("verdict", "evidence", "bar", "plan", "wave", "status")
VERDICTS = {
    "PRESENT", "PARTIAL", "ABSENT", "OVERSTATED",
    "PROCESS", "SCALE", "CAPABILITY", "ENGINEERING", "EVIDENCE",
}
STATUSES = {"open", "closed"}

WAVE_TITLES = {
    "W1": "Evidence semantics and the state contract",
    "W2": "Identity: concepts, entities, action receipts",
    "W3": "The cognitive event DAG",
    "W4": "Impasse everywhere, substates, transactions",
    "W5": "One procedure, one operator",
    "W6": "The scientific substrate",
    "W7": "Cost, budget, and neuroscience traceability",
    "W8": "Cross-substrate conversion",
    "W9": "Developmental and environmental evidence",
    "W10": "Metaprogrammable cognition and operator invention",
    "W11": "Agent surface: coding, terminal, sandbox, credentials",
    "W12": "One event and state spine",
    "W13": "Shadow evolution under immutable evaluators",
    "unscheduled": "Unscheduled",
}


def load() -> tuple[list[dict], dict]:
    cards = json.loads(CARDS.read_text())
    adjudication = json.loads(ADJUDICATION.read_text())
    return cards, adjudication


#: Words in a bar that make it a claim about the SYSTEM rather than about a
#: module. An entry whose bar contains one of these has to name a production
#: caller, because a module only its own tests import has not integrated
#: anything.
INTEGRATION_WORDS = (
    "every ", "all ", "architecture-wide", "most cross", "each major", "universal",
)

#: Words that make a bar a claim about a RESULT rather than about a mechanism.
#: A card closing on the harness that would measure such a result has to name
#: the campaign that has not been run.
DEMONSTRATION_WORDS = (
    "beats", "outperform", "better than", "hundreds", "dozens", "million",
    "thousands", "large margin", "improves with experience",
)


def _module_paths(entry: dict) -> list[str]:
    return [
        path for path in entry.get("closed_by", [])
        if path.startswith("core/") and path.endswith(".py")
    ]


def _integration_problems(cid: str, entry: dict) -> list[str]:
    bar = (entry.get("bar") or "").lower()
    if not any(word in bar for word in INTEGRATION_WORDS):
        return []
    modules = _module_paths(entry)
    if not modules:
        return []
    wired_by = entry.get("wired_by") or []
    if not wired_by:
        return [
            f"[{cid}]: the bar claims something about the whole system "
            f"({bar[:60]}...) and names only module(s) {modules}. Name a production "
            "caller in wired_by, or restate the bar as what the module establishes."
        ]
    problems = []
    for caller in wired_by:
        path = ROOT / caller
        if not path.exists():
            problems.append(f"[{cid}]: wired_by names {caller}, which does not exist")
            continue
        text = path.read_text(errors="replace")
        dotted = [m[:-3].replace("/", ".") for m in modules]
        if not any(name in text for name in dotted):
            problems.append(
                f"[{cid}]: wired_by names {caller}, which does not import any of {dotted}"
            )
    return problems


#: A campaign result is only a result if the file it names is on disk and a
#: test reads it. Without both, ``campaign_run`` is a sentence, and a sentence
#: is exactly what ``outstanding`` exists to prevent being mistaken for
#: evidence.
_EVIDENCE_RE = re.compile(r"docs/evidence/[\w./-]+\.json")


def _campaign_problems(cid: str, entry: dict) -> list[str]:
    """Whether a claimed campaign actually left evidence a test checks."""
    note = entry.get("campaign_run") or ""
    if not note:
        return []
    named = _EVIDENCE_RE.findall(note)
    if not named:
        return [
            f"[{cid}]: campaign_run describes a result but names no evidence file. "
            "A result with nothing on disk behind it is a sentence."
        ]
    problems = []
    for relative in named:
        if not (ROOT / relative).exists():
            problems.append(
                f"[{cid}]: campaign_run names {relative}, which is not there."
            )
            continue
        readers = [
            path
            for path in ROOT.glob("tests/test_*.py")
            if relative.rsplit("/", 1)[-1] in path.read_text(encoding="utf-8")
        ]
        if not readers:
            problems.append(
                f"[{cid}]: no test reads {relative}, so nothing notices when the "
                "campaign and its evidence drift apart."
            )
    return problems


def _demonstration_problems(cid: str, entry: dict) -> list[str]:
    bar = (entry.get("bar") or "").lower()
    if not any(word in bar for word in DEMONSTRATION_WORDS):
        return []
    if entry.get("outstanding"):
        return []
    if entry.get("campaign_run"):
        # The campaign was run and left evidence; _campaign_problems checks it.
        return []
    return [
        f"[{cid}]: the bar names a demonstrated result ({bar[:70]}...) and the entry "
        "claims it closed. Name the campaign still to run in `outstanding`, or restate "
        "the bar as the mechanism the tests actually establish."
    ]


def check() -> int:
    cards, adjudication = load()
    entries = adjudication.get("entries", {})
    problems: list[str] = []

    for card in cards:
        cid = card["id"]
        entry = entries.get(cid)
        if entry is None:
            problems.append(f"[{cid}] {card['title']}: no adjudication")
            continue
        for field in REQUIRED_FIELDS:
            if field not in entry or entry[field] in ("", [], None):
                problems.append(f"[{cid}]: adjudication is missing {field}")
        if entry.get("verdict") not in VERDICTS:
            problems.append(f"[{cid}]: verdict {entry.get('verdict')!r} is not one of {sorted(VERDICTS)}")
        if entry.get("status") not in STATUSES:
            problems.append(f"[{cid}]: status {entry.get('status')!r} is not one of {sorted(STATUSES)}")
        for path in entry.get("evidence", []):
            if not (ROOT / path.split(":")[0]).exists():
                problems.append(f"[{cid}]: evidence path {path} no longer exists")
        if entry.get("status") == "closed":
            closed_by = entry.get("closed_by") or []
            if not closed_by:
                problems.append(f"[{cid}]: closed with no closed_by; a gap is not closed because code exists")
            for path in closed_by:
                if not (ROOT / path.split("::")[0]).exists():
                    problems.append(f"[{cid}]: closed_by names {path}, which does not exist")
            problems.extend(_integration_problems(cid, entry))
            problems.extend(_demonstration_problems(cid, entry))
        problems.extend(_campaign_problems(cid, entry))

    orphans = set(entries) - {card["id"] for card in cards}
    problems.extend(f"[{cid}]: adjudicated but not a card in the report" for cid in sorted(orphans))

    for line in problems:
        print(f"gap-atlas: {line}", file=sys.stderr)
    if problems:
        print(f"gap-atlas: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"gap-atlas: {len(cards)} cards, all adjudicated, all evidence present")
    return 0


def render() -> int:
    cards, adjudication = load()
    entries = adjudication["entries"]
    by_wave: dict[str, list[dict]] = defaultdict(list)
    for card in cards:
        by_wave[entries[card["id"]]["wave"]].append(card)

    closed = sum(1 for e in entries.values() if e["status"] == "closed")
    verdicts = Counter(e["verdict"] for e in entries.values())

    out: list[str] = []
    out.append("# Gap Atlas — the 414 bars, and where Aura stands on each")
    out.append("")
    out.append(
        "Generated by `tools/gap_atlas.py --render` from `cards.json` (both reviews, "
        "parsed verbatim) and `adjudication.json` (Aura's verdict on each, against her "
        "own source). Do not edit this file; edit the adjudication and re-render. "
        "`make gap-atlas` fails on a card with no adjudication, a closed card whose "
        "test does not exist, and a closed card whose bar claims something about the "
        "whole system while naming only a module."
    )
    out.append("")
    out.append(f"**{closed} of {len(cards)} closed.** " + ", ".join(
        f"{v} {k}" for k, v in sorted(verdicts.items(), key=lambda kv: -kv[1])))
    out.append("")
    out.append("## The rules this list is kept under")
    out.append("")
    for key, text in adjudication.get("rules", {}).items():
        if isinstance(text, str):
            out.append(f"**{key}** — {text}")
            out.append("")

    for wave in sorted(by_wave, key=lambda w: (w == "unscheduled", w[0], int(w[1:] or 0) if w[1:].isdigit() else 0)):
        group = by_wave[wave]
        done = sum(1 for c in group if entries[c["id"]]["status"] == "closed")
        out.append(f"## {wave} — {WAVE_TITLES.get(wave, wave)} ({done}/{len(group)})")
        out.append("")
        for card in group:
            entry = entries[card["id"]]
            mark = "x" if entry["status"] == "closed" else " "
            out.append(f"- [{mark}] **[{card['id']}] {card['title']}** — _{card['system']}_ · `{entry['verdict']}`")
            if entry.get("card_subject"):
                out.append(f"  - What the rival showed: {entry['card_subject']}")
            out.append(f"  - Bar: {entry['bar']}")
            out.append(f"  - Plan: {entry['plan']}")
            if entry.get("note"):
                out.append(f"  - Note: {entry['note']}")
            if entry.get("closed_by"):
                out.append("  - Closed by: " + ", ".join(f"`{p}`" for p in entry["closed_by"]))
            if entry.get("wired_by"):
                out.append("  - Wired by: " + ", ".join(f"`{p}`" for p in entry["wired_by"]))
            if entry.get("outstanding"):
                out.append(f"  - Still to run: {entry['outstanding']}")
        out.append("")

    TODO.write_text("\n".join(out) + "\n")
    print(f"gap-atlas: wrote {TODO.relative_to(ROOT)} ({closed}/{len(cards)} closed)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail on an unadjudicated card, a missing evidence path, or a closed item with no test")
    parser.add_argument("--render", action="store_true", help="regenerate docs/gap_atlas/TODO.md")
    args = parser.parse_args()
    if args.render:
        return render() or check()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
