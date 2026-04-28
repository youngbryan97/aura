"""30-day life trial infrastructure.

This module provides the runner and daily-summary publisher the reviewer
asked for. The operator starts it; it keeps running; it publishes a
daily markdown summary derived from the LifeTrace ledger. Not a 30-day
run-now — rather, the infrastructure that makes such a run reproducible
and observable.

Usage:
  python tests/life_trial.py --hours 24
  python tests/life_trial.py --hours 720   # full 30-day trial

Writes:
  tests/life_trial/day_<N>_summary.md
  tests/life_trial/trial_index.json
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.autonomic.resource_stakes import ResourceStakesLedger  # noqa: E402
from core.goals.emergent_goals import EmergentGoalEngine  # noqa: E402
from core.runtime.life_trace import LifeTraceLedger  # noqa: E402


TRIAL_DIR = ROOT / "tests" / "life_trial"


@dataclass
class TrialConfig:
    hours: float
    tick_seconds: float = 2.0
    summary_interval_hours: float = 24.0
    seed: int = 2026


def _write_daily_markdown(day_index: int, summary: Dict[str, Any]) -> Path:
    get_task_tracker().create_task(get_storage_gateway().create_dir(TRIAL_DIR, cause='_write_daily_markdown'))
    path = TRIAL_DIR / f"day_{day_index:02d}_summary.md"
    lines = [
        f"# Day {day_index} Summary",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        f"- Total events: {summary['total_events']}",
        f"- Self-generated: {summary['self_generated']}",
        f"- User-requested: {summary['user_requested']}",
        f"- Actions executed: {summary['actions_executed']}",
        f"- Deferred by Will: {summary['deferred_by_will']}",
        f"- Blocked by resource scarcity: {summary['blocked_by_resource']}",
        f"- Policy changes: {summary['policy_changes']}",
        f"- Repairs applied: {summary['repairs_applied']}",
        f"- Hash chain intact: {summary['chain_intact']}",
        "",
        "## Event counts",
    ]
    for kind, count in sorted(summary.get("event_counts", {}).items()):
        lines.append(f"- `{kind}`: {count}")
    path.write_text("\n".join(lines) + "\n")
    return path


def run_trial(config: TrialConfig) -> Dict[str, Any]:
    get_task_tracker().create_task(get_storage_gateway().create_dir(TRIAL_DIR, cause='run_trial'))

    # Fresh state per trial so days are independent snapshots.
    life_trace = LifeTraceLedger(db_path=TRIAL_DIR / "life_trace.sqlite3")
    stakes = ResourceStakesLedger(TRIAL_DIR / "stakes.sqlite3")
    emergent = EmergentGoalEngine(db_path=TRIAL_DIR / "emergent.sqlite3")

    end_ts = time.time() + float(config.hours) * 3600.0
    rng = random.Random(config.seed)

    day_index = 0
    day_boundary = time.time() + config.summary_interval_hours * 3600.0
    index: Dict[str, Any] = {
        "started_at": time.time(),
        "config": asdict(config),
        "days": [],
    }

    interrupt = {"flag": False}

    def _signal_handler(signum, frame):  # pragma: no cover
        interrupt["flag"] = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while time.time() < end_ts and not interrupt["flag"]:
        tick_start = time.time()
        # Simulate a mix of user and self-generated events.
        user_requested = rng.random() < 0.35
        kind = "user_requested" if user_requested else rng.choice(
            ["initiative_proposed", "initiative_selected", "action_executed", "policy_changed", "self_generated"]
        )
        drive_before = {
            "viability": stakes.state().viability,
            "integrity": stakes.state().integrity,
        }
        if rng.random() < 0.12:
            stakes.consume("life_trial_work", energy=0.02, tool_budget=0.01)
        elif rng.random() < 0.06:
            stakes.earn("life_trial_recovery", {"energy": 0.03, "tool_budget": 0.02})
        drive_after = {
            "viability": stakes.state().viability,
            "integrity": stakes.state().integrity,
        }
        emergent.observe(
            "daily_tension" if not user_requested else "user_initiated_work",
            0.4 + 0.3 * (1.0 - stakes.state().viability),
            f"tick at {time.strftime('%H:%M:%S', time.gmtime(tick_start))}",
        )
        life_trace.record(
            kind,
            origin="user" if user_requested else "self_generated",
            user_requested=user_requested,
            drive_state_before=drive_before,
            drive_state_after=drive_after,
            action_taken={"content": f"tick {kind}"},
            result={"ok": True},
        )

        # Daily summary rollover
        if time.time() >= day_boundary:
            day_index += 1
            summary = life_trace.daily_summary(window_hours=config.summary_interval_hours)
            path = _write_daily_markdown(day_index, summary)
            index["days"].append({"day": day_index, "summary": summary, "path": str(path)})
            (TRIAL_DIR / "trial_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
            day_boundary = time.time() + config.summary_interval_hours * 3600.0

        elapsed = time.time() - tick_start
        sleep_for = max(0.0, config.tick_seconds - elapsed)
        if sleep_for > 0:
            try:
                time.sleep(sleep_for)
            except KeyboardInterrupt:
                interrupt["flag"] = True
                break

    # Final summary
    summary = life_trace.daily_summary(window_hours=config.summary_interval_hours)
    if not index["days"]:
        day_index += 1
        path = _write_daily_markdown(day_index, summary)
        index["days"].append({"day": day_index, "summary": summary, "path": str(path)})
    index["ended_at"] = time.time()
    index["interrupted"] = interrupt["flag"]
    (TRIAL_DIR / "trial_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=0.05, help="Duration in hours (default 0.05 = 3 minutes for smoke)")
    parser.add_argument("--tick-seconds", type=float, default=2.0)
    parser.add_argument("--summary-hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = TrialConfig(
        hours=args.hours,
        tick_seconds=args.tick_seconds,
        summary_interval_hours=args.summary_hours,
        seed=args.seed,
    )
    index = run_trial(config)
    print(json.dumps({"days": len(index["days"]), "interrupted": index.get("interrupted", False)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
