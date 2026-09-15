#!/usr/bin/env python3
"""The v25 carrier run as parallel shard workers and one coordinator.

One process scoring all 511 bipartitions takes most of a day. Each cut's
rollouts depend only on the anchors it forks from, so the cuts are split across
worker processes and merged back:

    python tools/run_subject_core_v25_sharded.py --workers 12 --anchors 32 --rounds 24

Each worker is an ordinary v25 run with `--shard I/N`: its own process, its own
state root, the same baseline, dose matching and anchor collection as the
coordinator, and every N-th cut. The coordinator is an ordinary v25 run with
`--from-shards`: it learns the grain while the workers sweep, waits for every
shard file, merges them only after checking each shard's anchors are
exchangeable with its own, and runs everything that comes after the sweep. A
shard whose anchors are not exchangeable is an authority blocker, not a merge.

The launcher writes `launch.json` with every command, pid and log path, and by
default returns once everything is started.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Arguments passed unchanged to every process, so the workers and the
#: coordinator run one experiment.
SHARED = ("rounds", "anchors", "history_turns", "turns", "cut_rounds", "seed", "domains", "conditions")


def commands(args: argparse.Namespace, base: Path) -> list[tuple[str, list[str]]]:
    """Every process to start, as (name, argv). Pure, so the plan can be read before it runs."""
    shared: list[str] = []
    for name in SHARED:
        value = getattr(args, name)
        if value in (None, "", 0) and name in ("domains", "conditions"):
            continue
        shared += [f"--{name.replace('_', '-')}", str(value)]
    if args.allow_degraded:
        shared.append("--allow-degraded")
    runner = [sys.executable, str(REPO / "tools" / "run_subject_core_v25.py")]
    shard_dir = base / "shards"
    plan = [
        (
            f"worker_{index}",
            [*runner, *shared, "--shard", f"{index}/{args.workers}", "--shard-dir", str(shard_dir),
             "--out", str(base / f"worker_{index}")],
        )
        for index in range(args.workers)
    ]
    plan.append(
        (
            "coordinator",
            [*runner, *shared, "--from-shards", str(shard_dir),
             "--shard-wait-seconds", str(args.hours * 3600.0), "--out", str(base / "coordinator")],
        )
    )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="shard workers; one core each, less one for the coordinator and one for the host")
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--cut-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--domains", type=str, default="")
    parser.add_argument("--conditions", type=int, default=0)
    parser.add_argument("--allow-degraded", action="store_true")
    parser.add_argument("--hours", type=float, default=24.0,
                        help="the bound on every process, and on the coordinator's wait for shards")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_v25_sharded")
    parser.add_argument("--wait", action="store_true", help="stay until every process exits")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and start nothing")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")

    base = args.out / time.strftime("launch_%Y%m%d_%H%M%S")
    plan = commands(args, base)
    if args.dry_run:
        for name, argv in plan:
            print(name, shlex.join(argv))
        return 0

    (base / "logs").mkdir(parents=True, exist_ok=True)
    bound = int(args.hours * 3600)
    started: list[dict[str, object]] = []
    children: list[subprocess.Popen] = []
    for name, argv in plan:
        log_path = base / "logs" / f"{name}.log"
        wrapped = ["caffeinate", "-dims", "perl", "-e", f"alarm {bound}; exec @ARGV", *argv]
        with open(log_path, "wb") as log:
            child = subprocess.Popen(wrapped, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        started.append({"name": name, "pid": child.pid, "log": str(log_path), "argv": argv})
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": args.workers,
        "bound_seconds": bound,
        "processes": started,
    }
    (base / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"started {len(children)} processes; manifest at {base / 'launch.json'}")
    if not args.wait:
        return 0
    return max(child.wait() for child in children)


if __name__ == "__main__":
    raise SystemExit(main())
