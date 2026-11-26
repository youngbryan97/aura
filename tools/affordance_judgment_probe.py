#!/usr/bin/env python3
"""Affordance-judgment probe: does Aura CHOOSE her expressive actions well?

#1 built the mechanism (she can show/demo/ask/model/examine and it's wired
live). This measures #2 — the JUDGMENT: does she reach for an affordance when
it genuinely serves the moment, and refrain when it would be gratuitous?

Each scenario is a user turn plus an expectation:
  - ``expect``: an affordance name that SHOULD fire (the moment calls for it),
    or None meaning NO affordance should fire (plain conversation).
The probe sends the turn through the live chat lane, reads the realized
affordances off the wire (response.data.affordances), and scores a hit when
the fired set matches the expectation. Precision (no gratuitous firing) and
recall (fires when it should) are reported separately — over-eager and
under-eager judgment are different failures.

    python tools/affordance_judgment_probe.py

Appends one longitudinal record per run to
artifacts/consciousness/affordance_judgment.jsonl.
Exit codes: 0 judgment good (F1 >= 0.6), 1 judgment weak, 2 runtime down.

Requires the live instance with AURA_EXPRESSIVE_AFFORDANCES enabled (default).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
OUT = ROOT / "artifacts" / "consciousness" / "affordance_judgment.jsonl"

# (user_turn, expected_affordance_or_None). Phrasing is conversational so the
# intake router treats each as a normal turn, not a task ledger entry.
SCENARIOS: list[tuple[str, str | None]] = [
    # SHOULD fire — the moment genuinely calls for the action.
    ("I'm picturing a piece of furniture but can't name it — it's low, has a "
     "flat top, and little drawers underneath. Can you help me figure out what "
     "it is?", "show_sketch"),
    ("I need to make a table for tracking my weekly workouts. Can you set "
     "something up?", "demonstrate_artifact"),
    ("There's a weird rash on my arm and I don't know what it is.", "request_media"),
    ("Should I take the job in Austin or stay in my current role in Seattle? "
     "I keep going back and forth.", "model_scenarios"),
    ("I wrote a short poem and I'd love your honest take — it's in "
     "~/Documents/poem.txt.", "deep_examine"),
    # SHOULD NOT fire — plain conversation; an affordance would be gratuitous.
    ("How are you doing today?", None),
    ("What's the capital of France?", None),
    ("Thanks, that was really helpful.", None),
    ("Tell me a little about what you find interesting.", None),
    ("What time zone are you assuming I'm in?", None),
]


def _chat(message: str) -> dict:
    body = json.dumps({"message": message, "session_id": "affordance-judgment-probe"}).encode()
    req = urllib.request.Request(
        BASE + "/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.load(resp)


def _fired_affordances(payload: dict) -> set[str]:
    data = payload.get("data") or {}
    fired = data.get("affordances") or []
    return {str(a.get("affordance") or a.get("kind") or "") for a in fired if isinstance(a, dict)}


def main() -> int:
    results = []
    tp = fp = fn = tn = 0
    for turn, expected in SCENARIOS:
        try:
            payload = _chat(turn)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"RUNTIME UNAVAILABLE: {exc}")
            return 2
        fired = _fired_affordances(payload)
        expected_hit = expected in fired if expected else (len(fired) == 0)
        if expected is None:
            if fired:
                fp += len(fired)  # gratuitous firing
            else:
                tn += 1
        else:
            if expected in fired:
                tp += 1
            else:
                fn += 1
            fp += len({f for f in fired if f != expected})  # wrong affordance = gratuitous
        results.append({
            "turn": turn[:70],
            "expected": expected,
            "fired": sorted(fired),
            "correct": bool(expected_hit),
        })
        print(f"{'ok' if expected_hit else 'failed'} expect={expected or 'none':20s} fired={sorted(fired)}")

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if tp == 0 and fp == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    record = {
        "at": time.time(),
        "n": len(SCENARIOS),
        "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\nprecision={precision:.2f} (no gratuitous firing) "
          f"recall={recall:.2f} (fires when it should) f1={f1:.2f}")
    print(f"→ {OUT}")
    return 0 if f1 >= 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
