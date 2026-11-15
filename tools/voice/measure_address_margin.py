"""Measure the margin her address reading is held to, and how well it does held out by situation.

    python tools/voice/measure_address_margin.py            # writes artifacts/voice/address_margin.json

Her voice session answers an utterance on an open microphone when her address reader predicts the words better after
her conversation than after the room's overheard talk or after nothing, by more than a margin
(core/voice/duplex/address_reading.py). This reads every declared case in config/voice_address_cases.json the way her
session does, through the same reader process (core/voice/duplex/address_process.py): her conversation as her live
context renders it (twice: after her real recent exchanges and in a fresh session with none), the turn's evidence at
the head of the person's turn, the room's talk as plain lines, and nothing.

The margin is the one that makes the fewest mistakes on the cases; among equals, the one that answers least. Each
domain is also held out in turn, its margin chosen on the other domains alone, which is how the certificate says how
the reading does on a kind of situation it was not measured on.

The reader is a small model (about a gigabyte) and runs beside her app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "config" / "voice_address_cases.json"


def _her_recent_exchanges(store: Path, turns: int = 120) -> list[dict[str, str]]:
    """Her latest exchanges, oldest first, read without writing to her store."""
    import sqlite3

    if not store.exists():
        return []
    try:
        with sqlite3.connect(f"file:{store}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT role, content FROM turns ORDER BY created_at DESC LIMIT ?",
                                      (turns,)).fetchall()
    except sqlite3.Error as exc:
        print(f"her conversation store could not be read: {exc}", file=sys.stderr)
        return []
    exchanges, said = [], None
    for role, content in reversed(rows):
        if role == "user":
            said = str(content or "")
        elif role == "aura" and said is not None:
            exchanges.append({"user": said, "aura": str(content or "")})
            said = None
    return exchanges


def _live_turns(conv: list[list[str]], evidence: str, answer_opens: str, turn_ends: str) -> str:
    """A recent exchange as her live context wrote it: evidence at the head of the person's turn, her reply after."""
    text, pending = "", None
    for role, said in conv:
        if role == "user":
            pending = said
        elif pending is not None:
            text += f"<|im_start|>user\n{evidence}{pending}{answer_opens}{said}{turn_ends}"
            pending = None
    return text


def _margin(leads: list[float], to_her: list[bool]) -> float:
    """The margin with the fewest mistakes on these cases; among equals, the highest (she answers least)."""
    ordered = sorted(set(leads))
    candidates = [ordered[0] - 1.0] + [(a + b) / 2.0 for a, b in zip(ordered, ordered[1:], strict=False)] + [ordered[-1] + 1.0]
    best, best_errors = candidates[0], math.inf
    for margin in candidates:
        errors = sum((lead > margin) != hers for lead, hers in zip(leads, to_her, strict=True))
        if errors < best_errors or (errors == best_errors and margin > best):
            best, best_errors = margin, errors
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "voice" / "address_margin.json"))
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--conversations", default=str(Path.home() / ".aura" / "data" / "conversations.db"),
                        help="her conversation store, read only, for the history her live context opens on")
    parser.add_argument("--model", default="", help="the reader model's path (default: her address reader's)")
    args = parser.parse_args()

    from core.voice.duplex.address_process import AddressProcess, address_model_path
    from core.voice.duplex.address_reading import DOCUMENT_BEGINS, AddressReading, spoken_words
    from core.voice.duplex.live_context import (
        _ANSWER_OPENS,
        _TURN_ENDS,
        rendered,
        the_turns_evidence,
    )

    cases_path = Path(args.cases)
    raw = cases_path.read_bytes()
    cases = json.loads(raw)["cases"]
    model_path = str(args.model or address_model_path())

    # What her live context opens on (live_context.what_she_reads_first): her latest exchanges, as many as are read
    # whole for a sitting. That function accepts only pairs her runtime stamped as delivered, which a read of her
    # store does not produce, so the same bound is applied here directly.
    from core.conversation.history_reach import A_SITTING_READ_WHOLE_CHARS, _exchange_chars

    sitting: list[dict[str, str]] = []
    for exchange in reversed(_her_recent_exchanges(Path(args.conversations))):
        if sitting and sum(_exchange_chars(e) for e in sitting) + _exchange_chars(exchange) > A_SITTING_READ_WHOLE_CHARS:
            break
        sitting.insert(0, exchange)
    history = rendered(sitting)
    if not history:
        print("her recent exchanges could not be read; measuring a fresh session only", file=sys.stderr)
    conditions = {"after her recent conversation": history, "in a fresh session": ""}
    if not history:
        conditions.pop("after her recent conversation")
    # Read as an awake session reads it: offline, her conversation lane reads "not up yet", which an awake session's
    # evidence does not say.
    from core.voice.duplex import live_context

    live_context.still_waking = lambda: ""
    evidence = the_turns_evidence(heard=True)
    opening = f"<|im_start|>user\n{evidence}"

    reader = AddressProcess(model_path)
    began = time.perf_counter()
    reader.start()
    print(f"her address reader started in {time.perf_counter() - began:.1f} s: {reader.name}", flush=True)
    readings: list[dict[str, Any]] = []
    try:
        for index, (condition, past) in enumerate(conditions.items()):
            started = time.perf_counter()
            for case in cases:
                words = spoken_words(case["u"])
                hers_prefix = past + _live_turns(case["conv"], evidence, _ANSWER_OPENS, _TURN_ENDS) + opening
                hers = reader.read(hers_prefix, words, f"hers:{index}")
                alone = reader.read(DOCUMENT_BEGINS, words, "alone")
                room = (reader.read(DOCUMENT_BEGINS + "\n".join(case["room"]) + "\n", words, f"room:{index}")
                        if case["room"] else None)
                reading = AddressReading(hers=sum(hers) / len(hers), alone=sum(alone) / len(alone),
                                         room=(sum(room) / len(room)) if room else None, tokens=len(hers))
                readings.append({"condition": condition, "domain": case["domain"], "u": case["u"],
                                 "to_her": case["to_her"], **reading.as_dict()})
            print(f"{condition}: {len(cases)} cases read in {time.perf_counter() - started:.1f} s", flush=True)
    finally:
        reader.close()

    judged = [r for r in readings if r["to_her"] is not None]
    leads = [r["lead"] for r in judged]
    hers = [bool(r["to_her"]) for r in judged]
    margin = _margin(leads, hers)

    held_out: dict[str, dict[str, Any]] = {}
    totals = defaultdict(int)
    for domain in sorted({r["domain"] for r in judged}):
        train = [(r["lead"], bool(r["to_her"])) for r in judged if r["domain"] != domain]
        test = [r for r in judged if r["domain"] == domain]
        domain_margin = _margin([lead for lead, _ in train], [h for _, h in train])
        answered_not_hers = sum(r["lead"] > domain_margin and not r["to_her"] for r in test)
        missed_hers = sum(r["lead"] <= domain_margin and r["to_her"] for r in test)
        held_out[domain] = {"cases": len(test), "correct": len(test) - answered_not_hers - missed_hers,
                            "answered_not_hers": answered_not_hers, "missed_hers": missed_hers,
                            "margin_from_the_others": round(domain_margin, 3),
                            "mean_lead": round(sum(r["lead"] for r in test) / len(test), 3)}
        totals["cases"] += len(test)
        totals["correct"] += held_out[domain]["correct"]
        totals["answered_not_hers"] += answered_not_hers
        totals["missed_hers"] += missed_hers
        totals["not_hers"] += sum(not r["to_her"] for r in test)
        totals["hers"] += sum(bool(r["to_her"]) for r in test)

    measured_at = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True,
                                 check=False).stdout.strip()
    certificate = {
        "schema": 1,
        "what": "the lead her conversation must have, per token, over the room's talk and nothing",
        "model": Path(model_path).name,
        "margin": round(margin, 4),
        "measured_at": measured_at,
        "tool": "tools/voice/measure_address_margin.py",
        "cases": {"path": str(cases_path.relative_to(ROOT)) if cases_path.is_relative_to(ROOT) else str(cases_path),
                  "sha256": hashlib.sha256(raw).hexdigest(), "read": len(readings)},
        "conditions": {name: {"history_chars": len(past)} for name, past in conditions.items()},
        "on_all_cases": {"correct": sum((lead > margin) == h for lead, h in zip(leads, hers, strict=True)), "cases": len(judged)},
        "held_out_by_domain": {"total": dict(totals), "domains": held_out},
        "not_settled_by_the_words": [{"condition": r["condition"], "u": r["u"], "lead": r["lead"]}
                                     for r in readings if r["to_her"] is None],
        "readings": readings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(certificate, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: certificate[k] for k in ("model", "margin", "on_all_cases")}, ensure_ascii=False))
    print(json.dumps(certificate["held_out_by_domain"]["total"]))
    for domain, result in held_out.items():
        print(f"  {domain:16s} {result['correct']}/{result['cases']}  mean lead {result['mean_lead']:+.2f}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
