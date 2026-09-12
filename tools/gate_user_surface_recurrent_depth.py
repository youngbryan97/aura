#!/usr/bin/env python3
"""Decide whether a live user surface may run more than one recurrent pass.

The ceiling has been one since an unvalidated depth setting latched the
foreground lane and took the conversation surface down. The comment that set it
says it stays at one "until an accuracy gate says otherwise", and the gate was
never written — so depth above one has been reachable only by setting
AURA_USER_SURFACE_RECURRENT_MAX_LOOPS by hand, which is a person authorising an
experiment rather than evidence authorising a default.

The evidence that does exist says less than it looks. `artifacts/recurrent_depth`
holds `loops1.json` and `loops2.json`, and their response files have the same
SHA-256. The two arms are one arm: the depth was never applied, neither file
records the depth it claims to have run at, and both report 0.625. A comparison
between a run and itself cannot show an improvement, and cannot show the absence
of one either.

So this gate refuses three things before it compares anything:

  an arm with no declared depth     it cannot be an arm of a depth comparison
  two arms at the same depth        there is nothing to compare
  two arms with identical outputs   whatever was varied, it was not the depth

and then requires the deeper arm to be better on a held-out battery by more than
the shallower arm's own spread. What it writes is bound to the model that
produced it, because a depth that helps one checkpoint says nothing about the
next one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RECEIPTS = REPO / "artifacts/recurrent_depth/authorized"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _declared_depth(arm: dict[str, Any]) -> int | None:
    for key in ("loops", "recurrent_loops", "depth", "user_surface_recurrent_loops"):
        value = arm.get(key)
        if value is None:
            continue
        try:
            depth = int(value)
        except (TypeError, ValueError):
            return None
        return depth if depth >= 1 else None
    return None


def _responses_fingerprint(arm: dict[str, Any], path: Path) -> str:
    stated = str(arm.get("responses_sha256") or "").strip()
    if stated:
        return stated
    beside = path.with_suffix("").with_suffix(".responses.jsonl")
    if beside.is_file():
        return hashlib.sha256(beside.read_bytes()).hexdigest()
    return ""


def _accuracy(arm: dict[str, Any]) -> float | None:
    for key in ("accuracy", "score"):
        if key in arm:
            try:
                return float(arm[key])
            except (TypeError, ValueError):
                return None
    result = arm.get("result")
    if isinstance(result, dict) and "accuracy" in result:
        try:
            return float(result["accuracy"])
        except (TypeError, ValueError):
            return None
    return None


def _trials(arm: dict[str, Any]) -> int:
    """How many questions the accuracy was read from.

    The battery's own size first, because it is the number the arm was run
    against. Nesting this inside the `result` check was how an arm that
    carried a battery and no result reported zero trials — and zero trials
    reads as a spread of one, which refuses everything including a real
    improvement.
    """
    battery = arm.get("battery")
    if isinstance(battery, dict) and battery.get("size"):
        try:
            return int(battery["size"])
        except (TypeError, ValueError):
            pass
    result = arm.get("result")
    if isinstance(result, dict):
        for key in ("total", "trials", "n"):
            if key in result:
                try:
                    return int(result[key] or 0)
                except (TypeError, ValueError):
                    return 0
    return 0


def _binomial_spread(accuracy: float, trials: int) -> float:
    """The shallower arm's own sampling spread, as a standard error.

    A battery of thirty-two reads to about nine points either side at 0.6, so
    an improvement of one or two answers is inside the noise of the arm it is
    being compared against.
    """
    if trials <= 1:
        return 1.0
    return math.sqrt(max(0.0, accuracy * (1.0 - accuracy)) / trials)


def adjudicate(arms: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    refusals: list[str] = []
    described: list[dict[str, Any]] = []
    for path, arm in arms:
        depth = _declared_depth(arm)
        accuracy = _accuracy(arm)
        described.append(
            {
                "file": path.name,
                "declared_depth": depth,
                "accuracy": accuracy,
                "trials": _trials(arm),
                "responses_sha256": _responses_fingerprint(arm, path),
                "model": str(arm.get("model") or ""),
            }
        )
        if depth is None:
            refusals.append(f"{path.name}: no declared depth")
        if accuracy is None:
            refusals.append(f"{path.name}: no accuracy")

    depths = {row["declared_depth"] for row in described if row["declared_depth"]}
    if len(described) >= 2 and len(depths) < 2:
        refusals.append("the arms do not differ in depth")

    prints = [row["responses_sha256"] for row in described if row["responses_sha256"]]
    if len(prints) >= 2 and len(set(prints)) < len(prints):
        refusals.append(
            "two arms produced identical responses: whatever was varied, it was "
            "not the depth"
        )

    models = {row["model"] for row in described if row["model"]}
    if len(models) > 1:
        refusals.append("the arms did not run on the same model")

    verdict: dict[str, Any] = {
        "schema": "aura.recurrent_depth.gate.v1",
        "ran_at": time.time(),
        "arms": described,
        "refusals": refusals,
        "authorized_depth": 1,
        "reading": "",
    }
    if refusals:
        verdict["reading"] = (
            "Nothing was compared. " + "; ".join(refusals) + ". The ceiling stays at "
            "one, which is what it means for a gate to be fail-closed."
        )
        return verdict

    ordered = sorted(described, key=lambda row: row["declared_depth"])
    shallow, deep = ordered[0], ordered[-1]
    margin = float(deep["accuracy"]) - float(shallow["accuracy"])
    spread = _binomial_spread(float(shallow["accuracy"]), int(shallow["trials"]))
    verdict["margin"] = round(margin, 5)
    verdict["shallow_spread"] = round(spread, 5)
    if margin > spread:
        verdict["authorized_depth"] = int(deep["declared_depth"])
        verdict["reading"] = (
            f"Depth {deep['declared_depth']} scores {deep['accuracy']:.3f} against "
            f"depth {shallow['declared_depth']}'s {shallow['accuracy']:.3f}, a margin "
            f"of {margin:.3f} against that arm's own spread of {spread:.3f}."
        )
    else:
        verdict["reading"] = (
            f"Depth {deep['declared_depth']} scores {deep['accuracy']:.3f} against "
            f"depth {shallow['declared_depth']}'s {shallow['accuracy']:.3f}. The margin "
            f"of {margin:.3f} does not beat the shallower arm's own spread of "
            f"{spread:.3f}, so the ceiling stays at one."
        )
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", type=Path, nargs="+", help="held-out eval artifacts")
    parser.add_argument("--descriptor", default="", help="active cortex descriptor sha256")
    parser.add_argument("--write-receipt", action="store_true")
    arguments = parser.parse_args(argv)

    loaded = []
    for path in arguments.arms:
        if not path.is_file():
            print(f"no such arm: {path}", file=sys.stderr)
            return 2
        loaded.append((path, _read(path)))

    verdict = adjudicate(loaded)
    verdict["model_descriptor_sha256"] = str(arguments.descriptor or "")
    print(json.dumps(verdict, indent=2))

    if arguments.write_receipt:
        if not verdict["model_descriptor_sha256"]:
            print("a receipt needs the checkpoint it was measured on", file=sys.stderr)
            return 2
        RECEIPTS.mkdir(parents=True, exist_ok=True)
        out = RECEIPTS / f"{verdict['model_descriptor_sha256']}.json"
        out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        print(f"\nreceipt written to {out.relative_to(REPO)}", file=sys.stderr)
    return 0 if verdict["authorized_depth"] > 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
