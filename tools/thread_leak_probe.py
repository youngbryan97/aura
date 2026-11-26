#!/usr/bin/env python3
"""Thread-leak probe: measure thread growth across conversation turns.

The 104-thread pile-up that amplified the memory crashes was never
measured — only seen once in a crash spike stack. This boots Aura, reads
the live thread histogram (/api/health/threads), drives N turns, and
re-reads it. A healthy runtime reaches STEADY STATE: pools are reused, so
the second half of the run adds ~0 threads. A leak keeps growing roughly
one-per-turn and names the offending pool in the per-group delta.

Verdict (PASS): thread growth over the second half of the run is below
LEAK_THRESHOLD — the runtime plateaued instead of leaking per turn.

Usage:
    python tools/thread_leak_probe.py [--port 8000] [--turns 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.live_boot_proof import LiveProof  # noqa: E402

LEAK_THRESHOLD = 6  # threads of second-half growth tolerated before it's a leak

_PROMPTS = (
    "In one sentence, what are you?",
    "Name one thing you can do on this computer.",
    "What did I just ask you?",
    "Give me a single word that describes your mood.",
    "What is two plus two?",
    "Summarize our conversation in one short line.",
    "What tools can you reach from here?",
    "Say something brief and honest about your limits.",
)


class ThreadLeakProbe(LiveProof):
    def __init__(self, *, port: int, boot_timeout_s: float, turns: int):
        super().__init__(
            port=port,
            mode="headless",
            boot_timeout_s=boot_timeout_s,
            skip_desktop=True,
            restart_continuity=False,
            conversation_soak_turns=0,
        )
        self.turns = max(4, turns)

    def _thread_snapshot(self) -> dict:
        import httpx

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(f"{self.base}/api/health/threads")
            return resp.json() if resp.status_code == 200 else {}
        except httpx.HTTPError as exc:
            return {"error": str(exc)}

    @staticmethod
    def _group_delta(before: dict, after: dict) -> dict[str, int]:
        gb = before.get("groups", {}) or {}
        ga = after.get("groups", {}) or {}
        delta = {}
        for name in set(gb) | set(ga):
            d = int(ga.get(name, 0)) - int(gb.get(name, 0))
            if d:
                delta[name] = d
        return dict(sorted(delta.items(), key=lambda kv: kv[1], reverse=True))

    def exercise_thread_growth(self) -> bool:
        baseline = self._thread_snapshot()
        base_total = int(baseline.get("total", 0))
        time.sleep(1.0)
        per_turn_totals: list[int] = []
        latencies: list[float] = []
        half = max(1, self.turns // 2)
        midpoint = baseline
        for i in range(self.turns):
            ok, _reply, latency = self.chat(_PROMPTS[i % len(_PROMPTS)], timeout_s=120.0)
            latencies.append(round(latency, 1))
            self.guard_rss()
            time.sleep(0.5)
            snap = self._thread_snapshot()
            total = int(snap.get("total", 0))
            per_turn_totals.append(total)
            # Per-turn log so a slow/timed-out run still yields the trend.
            print(
                f"[probe] turn {i + 1}/{self.turns} ok={ok} {latency:.0f}s threads={total}",
                flush=True,
            )
            if i + 1 == half:
                midpoint = snap
        time.sleep(2.0)
        final = self._thread_snapshot()

        mid_total = int(midpoint.get("total", base_total))
        final_total = int(final.get("total", 0))
        first_half_growth = mid_total - base_total
        second_half_growth = final_total - mid_total

        # The leak signal is sustained growth: a healthy runtime warms its
        # pools in the first half, then plateaus.
        leaked = second_half_growth >= LEAK_THRESHOLD
        verified = not leaked and final_total > 0
        return self.record(
            "thread_growth",
            verified,
            summary=(
                f"threads base={base_total} mid={mid_total} final={final_total} "
                f"(1st-half +{first_half_growth}, 2nd-half +{second_half_growth}; "
                f"leak if 2nd-half >= {LEAK_THRESHOLD}); per_turn={per_turn_totals}"
            ),
            turns=self.turns,
            baseline_total=base_total,
            midpoint_total=mid_total,
            final_total=final_total,
            first_half_growth=first_half_growth,
            second_half_growth=second_half_growth,
            per_turn_totals=per_turn_totals,
            turn_latencies_s=latencies,
            second_half_group_delta=self._group_delta(midpoint, final),
            full_run_group_delta=self._group_delta(baseline, final),
            final_groups=final.get("groups", {}),
        )

    def run(self) -> int:  # noqa: D102
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            ok = self.exercise_thread_growth()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "thread_leak_probe",
                "passed": bool(ok and shutdown_ok),
                "steps": self.steps,
            }
            import json

            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                ("THREAD LEAK PROBE PASSED" if verdict["passed"] else "THREAD LEAK PROBE FAILED")
                + f" → {self.verdict_path}"
            )
            return 0 if verdict["passed"] else 1
        finally:
            self.kill_hard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args(argv)
    return ThreadLeakProbe(
        port=args.port, boot_timeout_s=args.boot_timeout, turns=args.turns
    ).run()


if __name__ == "__main__":
    sys.exit(main())
