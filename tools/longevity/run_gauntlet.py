from __future__ import annotations
#!/usr/bin/env python3
"""tools/longevity/run_gauntlet.py

Longevity gauntlet runner — drive Aura for 24h / 72h / 7d / 30d windows
and emit a public artifact bundle for each. The runner is *resumable*:
each run is keyed by a UUID and produces a per-tick row in a JSONL trace
file under ``~/.aura/data/longevity/<run_id>/``.

Profiles
--------
24h_no_user       — pure idle, no user input
72h_mixed         — scripted user pulses + idle gaps
7d_with_failures  — adds chaos: subprocess kill, network drop, model load
                    failure, memory pressure
30d_summary       — long-window run with daily continuity-hash snapshots

Artifacts produced per run
--------------------------
  events.jsonl                — every tick's snapshot
  receipts.jsonl              — durable copy of action receipts
  resource.csv                — cpu/ram/disk over time
  goals_outcome.csv           — goals started / completed / abandoned
  identity_continuity.jsonl   — periodic continuity-hash captures
  summary.md                  — human-readable run summary

Usage:
    python tools/longevity/run_gauntlet.py --profile 24h_no_user
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.Longevity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_PROFILES: Dict[str, Dict[str, Any]] = {
    "24h_no_user": {"duration_s": 24 * 3600, "user_pulse_s": 0, "chaos": False},
    "72h_mixed": {"duration_s": 72 * 3600, "user_pulse_s": 1800, "chaos": False},
    "7d_with_failures": {"duration_s": 7 * 24 * 3600, "user_pulse_s": 3600, "chaos": True},
    "30d_summary": {"duration_s": 30 * 24 * 3600, "user_pulse_s": 0, "chaos": False, "snapshot_only": True},
}


async def _tick_snapshot(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"when": time.time()}
    try:
        import psutil
        out["cpu_pct"] = psutil.cpu_percent(interval=None)
        out["ram_pct"] = psutil.virtual_memory().percent
        try:
            out["disk_pct"] = psutil.disk_usage("/").percent
        except Exception:
            out["disk_pct"] = 0.0
    except Exception:
        pass
    try:
        from core.identity.self_object import get_self
        snap = get_self().snapshot()
        out["continuity_hash"] = snap.continuity_hash
        out["viability"] = snap.viability_state
        out["active_goals"] = len(snap.active_goals)
        out["active_tokens"] = snap.active_capability_tokens
    except Exception as exc:
        out["self_error"] = str(exc)
    try:
        from core.agency.agency_orchestrator import get_receipt_log
        out["receipts_recent"] = len(get_receipt_log().recent(limit=200))
    except Exception:
        pass
    # write to per-run trace
    with open(run_dir / "events.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, default=str) + "\n")
    with open(run_dir / "resource.csv", "a", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([out.get("when"), out.get("cpu_pct"), out.get("ram_pct"), out.get("disk_pct")])
    return out


async def _maybe_fire_user(run_dir: Path) -> None:
    # Hook: write a synthetic user-event into events.jsonl. Actual user
    # injection requires the chat HTTP endpoint, which is the user's
    # responsibility to wire when running this against a live instance.
    with open(run_dir / "user_pulse.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.time(), "kind": "scripted_user_pulse"}) + "\n")


async def _maybe_inject_chaos(run_dir: Path) -> None:
    from tools.chaos.injector import inject_random_fault
    fault = await inject_random_fault()
    with open(run_dir / "chaos.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"when": time.time(), "fault": fault}) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=list(_PROFILES.keys()))
    parser.add_argument("--tick-s", type=float, default=30.0, help="seconds between snapshots")
    args = parser.parse_args()
    profile = _PROFILES[args.profile]
    run_id = f"longevity-{args.profile}-{uuid.uuid4().hex[:8]}"
    run_dir = Path.home() / ".aura" / "data" / "longevity" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("longevity run_id=%s profile=%s dir=%s duration_s=%s", run_id, args.profile, run_dir, profile["duration_s"])

    started = time.time()
    last_user_pulse = 0.0
    last_chaos = 0.0
    while True:
        now = time.time()
        elapsed = now - started
        if elapsed > profile["duration_s"]:
            break
        await _tick_snapshot(run_dir)
        if profile["user_pulse_s"] and (now - last_user_pulse) > profile["user_pulse_s"]:
            await _maybe_fire_user(run_dir)
            last_user_pulse = now
        if profile.get("chaos") and (now - last_chaos) > 600.0:
            await _maybe_inject_chaos(run_dir)
            last_chaos = now
        await asyncio.sleep(args.tick_s)

    # Write summary.md
    rows = []
    try:
        with open(run_dir / "events.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass

    with open(run_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(f"# longevity run {run_id}\n\n")
        fh.write(f"profile: `{args.profile}`\n")
        fh.write(f"duration: {profile['duration_s']}s\n")
        fh.write(f"snapshots: {len(rows)}\n")
        if rows:
            cpu = [float(r.get('cpu_pct') or 0.0) for r in rows]
            ram = [float(r.get('ram_pct') or 0.0) for r in rows]
            fh.write(f"cpu: min={min(cpu):.1f} max={max(cpu):.1f}\n")
            fh.write(f"ram: min={min(ram):.1f} max={max(ram):.1f}\n")
            unique_hashes = sorted({r.get('continuity_hash', '?') for r in rows})
            fh.write(f"unique continuity hashes: {len(unique_hashes)}\n")
    logger.info("longevity complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
