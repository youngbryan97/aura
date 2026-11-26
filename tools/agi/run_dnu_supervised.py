#!/usr/bin/env python3
"""#50 phase 2 application — run the DNU battery under out-of-process wedge recovery.

The DNU AGI proof battery (tools/agi/run_dnu_agi_proof_battery.py) boots the Aura
runtime IN-PROCESS. When that runtime hits the worst-case Metal-GPU
command-buffer deadlock, the battery process wedges silently forever — its own
in-process watchdog is frozen behind the same GPU semaphore, so the run hangs to
the outer SIGKILL with no result.

This launcher runs the battery as a SUBPROCESS under the proven
WedgeRecoverySupervisor (tools/wedge_recovery_supervisor.py): the supervisor does
no GPU work, monitors the battery's loop-liveness beacon out-of-process, and on a
wedge SIGKILLs + restarts the battery subprocess (bounded restart budget). A
Metal deadlock then kills only the killable battery subprocess and the run
survives instead of hanging.

It does NOT modify the battery internals (the cert-critical file is untouched) —
it composes the existing battery with the existing supervisor. Restart re-runs
the battery from the start; task-level resume (skip already-scored tasks) is the
remaining optimization noted in the closeout, not required for wedge-survival.

Example:
    python tools/agi/run_dnu_supervised.py -- --smoke --model-tier primary
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BEACON = "data/runtime/liveness_heartbeat.json"
DEFAULT_BATTERY = [sys.executable, str(ROOT / "tools" / "agi" / "run_dnu_agi_proof_battery.py")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heartbeat", default=os.environ.get("AURA_LIVENESS_HEARTBEAT_FILE", DEFAULT_BEACON))
    parser.add_argument("--stale-ceiling", type=float, default=180.0)
    parser.add_argument("--grace", type=float, default=420.0, help="boot grace (battery boot is heavy)")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--battery-cmd", default="", help="override battery command (for testing)")
    parser.add_argument("battery_args", nargs=argparse.REMAINDER,
                        help="args after `--` are passed through to the battery")
    args = parser.parse_args(argv)

    from tools.wedge_recovery_supervisor import WedgeRecoverySupervisor

    passthrough = [a for a in args.battery_args if a != "--"]
    base_cmd = shlex.split(args.battery_cmd) if args.battery_cmd else DEFAULT_BATTERY

    def _spawn() -> Any:
        from core.runtime.subprocess_gateway import get_subprocess_gateway

        env = dict(os.environ)
        env.setdefault("AURA_LIVENESS_HEARTBEAT_FILE", args.heartbeat)
        env.setdefault("AURA_LIVENESS_SENTINEL", "1")  # ensure the runtime writes the beacon
        return get_subprocess_gateway().spawn(
            [*base_cmd, *passthrough],
            env=env,
            start_new_session=True,
            cwd=str(ROOT),
            offline_tooling=True,
            source="certification_tooling:dnu_supervised",
            accelerator_capability="none",
        )

    supervisor = WedgeRecoverySupervisor(
        spawn=_spawn,
        heartbeat_path=args.heartbeat,
        stale_ceiling_s=args.stale_ceiling,
        grace_s=args.grace,
        poll_interval_s=args.interval,
        max_restarts=args.max_restarts,
        on_restart=lambda n: print(f" DNU battery runtime WEDGED — supervised restart {n}", flush=True),
    )
    outcome = supervisor.run()
    print(json.dumps(outcome.to_dict()))
    return 0 if outcome.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
