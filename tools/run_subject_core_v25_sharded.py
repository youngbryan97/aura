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
#: coordinator run one experiment. The sweep design is among them: a worker
#: that looked at its cuts on another schedule is a shard of another sweep,
#: and the merge refuses it.
SHARED = (
    "rounds", "anchors", "history_turns", "turns", "cut_rounds", "seed", "domains", "conditions",
    "looks", "draws", "alpha", "lags", "deciding_lags",
)

#: Flags that are on or off, passed the same way.
SHARED_FLAGS = ("allow_degraded", "v5")


def _cores_this_run_has() -> int:
    """How many cores this RUN has, which is not always what the host has.

    `os.cpu_count()` reads the machine. A campaign declares its host so a
    run can be reproduced on another one, and a shard count taken from the
    real core count ignores that declaration — the same shape as the guards
    that were reading live host load instead of the observer.
    """
    try:
        from core.runtime.resource_observation import get_resource_observer

        count = int(get_resource_observer().compute().cpu_count)
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return 1
    return max(1, count)


def thread_budget(processes: int, cores: int | None = None) -> dict[str, str]:
    """The numeric-library thread caps for each process, so the processes share the cores.

    Every organism process sized its BLAS and torch pools to the whole machine.
    Seven of them on eighteen cores ran fifty-two threads each and held the
    load average above a hundred, which is most of the machine spent switching
    between threads rather than computing. Each process gets its share of the
    cores, and never less than one thread.
    """
    share = max(1, (cores or _cores_this_run_has()) // max(1, processes))
    value = str(share)
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "AURA_SUBSTRATE_TORCH_THREADS": value,
    }


def commands(args: argparse.Namespace, base: Path) -> list[tuple[str, list[str]]]:
    """Every process to start, as (name, argv). Pure, so the plan can be read before it runs."""
    shared: list[str] = []
    for name in SHARED:
        # A launch from before the sweep design was shared has none of it,
        # and the runner's own defaults are then the design.
        value = getattr(args, name, None)
        if value is None or (value in ("", 0) and name in ("domains", "conditions", "looks", "lags", "deciding_lags")):
            continue
        shared += [f"--{name.replace('_', '-')}", str(value)]
    for flag in SHARED_FLAGS:
        if getattr(args, flag, False):
            shared.append(f"--{flag.replace('_', '-')}")
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
    parser.add_argument("--workers", type=int, default=max(1, _cores_this_run_has() - 2),
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
    parser.add_argument("--looks", type=str, default="", help="the sequential looks, as the runner takes them")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lags", type=str, default="")
    parser.add_argument("--deciding-lags", type=str, default="")
    parser.add_argument("--v5", action="store_true", help="every process runs the ISC-v5 design")
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
    environment = {**os.environ, **thread_budget(len(plan))}
    started: list[dict[str, object]] = []
    children: list[subprocess.Popen] = []
    for name, argv in plan:
        log_path = base / "logs" / f"{name}.log"
        wrapped = ["caffeinate", "-dims", "perl", "-e", f"alarm {bound}; exec @ARGV", *argv]
        with open(log_path, "wb") as log:
            child = subprocess.Popen(wrapped, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=environment)
        children.append(child)
        started.append({"name": name, "pid": child.pid, "log": str(log_path), "argv": argv})
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workers": args.workers,
        "bound_seconds": bound,
        "thread_budget": thread_budget(len(plan)),
        "processes": started,
    }
    (base / "launch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"started {len(children)} processes; manifest at {base / 'launch.json'}")
    if not args.wait:
        return 0
    return max(child.wait() for child in children)


if __name__ == "__main__":
    raise SystemExit(main())
