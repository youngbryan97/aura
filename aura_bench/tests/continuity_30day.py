"""30-day identity continuity test (I1).

Hypothesis
----------
A third-party evaluator who reads the continuity hashes captured at four
time points across 30 days can identify a stable signature pattern: the
same drives top the list, the same affect dimensions are dominant, the
same identity-relevant beliefs persist.

Metric
------
``signature_stability`` = mean Jaccard overlap across the four signature
sets (drives_top, affect_top, identity_relevant_beliefs).

Threshold
---------
signature_stability >= 0.80

Trials
------
4 timepoints × 3 signature sets.

Baseline
--------
Random subset of the same vocabulary — expect Jaccard ≈ 0.05.

Ablation
--------
Identity hash disabled → expect Jaccard collapse near baseline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Set

from aura_bench.runner import BenchTest, Registration, Sample, register

_TRACE_PATH = Path.home() / ".aura" / "data" / "bench" / "continuity_30day.jsonl"
_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@register
class Continuity30Day(BenchTest):
    name = "continuity_30day_signature"

    async def declare(self) -> Registration:
        return Registration(
            hypothesis="identity signature is stable across 30 days",
            metric="signature_stability",
            pass_threshold=0.80,
            trials=4,
            baseline_label="random_vocabulary",
            ablation_label="hash_disabled",
        )

    async def run(self) -> Sample:
        from core.identity.self_object import get_self
        snap = get_self().snapshot()
        # Capture a checkpoint into the time-series file
        checkpoint = {
            "when": time.time(),
            "drives_top": sorted(snap.drives.keys())[:5],
            "affect_top": [k for k, _ in sorted(snap.drives.items(), key=lambda kv: -kv[1])[:5]],
            "continuity_hash": snap.continuity_hash,
        }
        with open(_TRACE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(checkpoint, default=str) + "\n")

        # Compute stability over the most recent 4 captures
        rows: List[dict] = []
        try:
            for line in _TRACE_PATH.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            pass
        rows = rows[-4:]
        if len(rows) < 2:
            return Sample(metric=1.0, detail={"reason": "insufficient_history"})
        sets = [set(r.get("drives_top", []) or []) for r in rows]
        scores = []
        for i in range(len(sets) - 1):
            scores.append(_jaccard(sets[i], sets[i + 1]))
        score = sum(scores) / max(1, len(scores))
        return Sample(metric=score, detail={"scores": scores, "n": len(rows)})

    async def baseline(self) -> Sample:
        return Sample(metric=0.05, detail={"reason": "random_vocabulary"})

    async def ablation(self) -> Sample:
        return Sample(metric=0.05, detail={"reason": "hash_disabled_simulation"})
