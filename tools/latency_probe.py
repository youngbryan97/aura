#!/usr/bin/env python3
"""Conversation latency probe — measure p50/p95 under realistic pacing.

You cannot improve (or claim to improve) latency without measuring it,
and you cannot tell whether the cortex-death I saw under rapid-fire load
is a real daily-usage problem without pacing turns like a real user. This
boots Aura headless and sends N turns with a realistic gap between them,
recording each turn's latency and whether it completed. It reports:

  - p50 / p95 / max latency,
  - fast-path vs cortex-generation split,
  - any timed-out turns (a cortex-death / recovery failure signal).

PASS is reported against an SLO target (default p50 <= 12s, no hard
timeouts), but the real product is the latency profile itself — printed
per turn so a slow run still yields the data.

Usage:
    python tools/latency_probe.py [--port 8000] [--turns 12] [--gap 8]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.live_boot_proof import LiveProof  # noqa: E402

SLO_P50_S = 12.0
HARD_TIMEOUT_S = 110.0  # a turn slower than this is treated as a cortex stall

# A mix of fast-path (identity/recall) and real cortex-generation prompts,
# the way daily use actually looks.
_PROMPTS = (
    "What are you, in one sentence?",
    "Give me three concrete ideas for organizing a cluttered Downloads folder.",
    "What did I just ask you about?",
    "Explain the difference between a process and a thread, briefly.",
    "What's a good way to stay focused while coding?",
    "Summarize what we've talked about so far.",
    "Name one limitation you have and one strength.",
    "Suggest a simple weeknight dinner.",
    "What's the capital of Japan?",
    "How would you debug a flaky test?",
    "Write a one-line encouraging note.",
    "What time-management method do you recommend and why?",
)


class LatencyProbe(LiveProof):
    def __init__(self, *, port: int, boot_timeout_s: float, turns: int, gap_s: float):
        super().__init__(
            port=port,
            mode="headless",
            boot_timeout_s=boot_timeout_s,
            skip_desktop=True,
            restart_continuity=False,
            conversation_soak_turns=0,
        )
        self.turns = max(4, turns)
        self.gap_s = max(0.0, gap_s)

    def exercise_latency(self) -> bool:
        latencies: list[float] = []
        completed = 0
        timed_out = 0
        fast_path = 0
        for i in range(self.turns):
            ok, _reply, latency = self.chat(_PROMPTS[i % len(_PROMPTS)], timeout_s=120.0)
            self.guard_rss()
            latencies.append(round(latency, 1))
            if ok:
                completed += 1
            if latency >= HARD_TIMEOUT_S:
                timed_out += 1
            elif latency < 2.0:
                fast_path += 1
            print(
                f"[latency] turn {i + 1}/{self.turns} ok={ok} {latency:.1f}s",
                flush=True,
            )
            if i < self.turns - 1:
                time.sleep(self.gap_s)

        ordered = sorted(latencies)
        p50 = statistics.median(ordered) if ordered else 0.0
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)] if ordered else 0.0
        worst = max(ordered) if ordered else 0.0
        cortex = [x for x in latencies if x >= 2.0 and x < HARD_TIMEOUT_S]
        cortex_p50 = statistics.median(cortex) if cortex else 0.0

        # Reliability: no hard timeouts (cortex stalls) under realistic pacing.
        # SLO: p50 within target. Report both; the profile is the product.
        verified = bool(timed_out == 0 and completed == self.turns and p50 <= SLO_P50_S)
        return self.record(
            "conversation_latency",
            verified,
            summary=(
                f"{self.turns} turns @ {self.gap_s:.0f}s gap — "
                f"p50={p50:.1f}s p95={p95:.1f}s max={worst:.1f}s "
                f"cortex_p50={cortex_p50:.1f}s fast={fast_path} timeouts={timed_out} "
                f"(SLO p50<={SLO_P50_S}s, no timeouts)"
            ),
            turns=self.turns,
            gap_s=self.gap_s,
            completed=completed,
            timed_out=timed_out,
            fast_path_turns=fast_path,
            p50_s=round(p50, 1),
            p95_s=round(p95, 1),
            max_s=round(worst, 1),
            cortex_p50_s=round(cortex_p50, 1),
            latencies_s=latencies,
        )

    def run(self) -> int:  # noqa: D102
        try:
            if not self.boot():
                return 1
            self.snapshot_vitals()
            ok = self.exercise_latency()
            self.snapshot_vitals()
            shutdown_ok = self.shutdown()
            verdict = {
                "proof": "latency",
                "passed": bool(ok and shutdown_ok),
                "steps": self.steps,
            }
            self.verdict_path.write_text(json.dumps(verdict, indent=2, default=str))
            print(
                ("LATENCY PROBE: SLO MET" if verdict["passed"] else " LATENCY PROBE: SLO NOT MET (profile recorded)")
                + f" → {self.verdict_path}"
            )
            return 0 if verdict["passed"] else 1
        finally:
            self.kill_hard()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--boot-timeout", type=float, default=600.0)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--gap", type=float, default=8.0)
    args = parser.parse_args(argv)
    return LatencyProbe(
        port=args.port, boot_timeout_s=args.boot_timeout, turns=args.turns, gap_s=args.gap
    ).run()


if __name__ == "__main__":
    sys.exit(main())
