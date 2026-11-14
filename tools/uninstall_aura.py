#!/usr/bin/env python3
"""Uninstall Aura from this Mac: what an install put there, and her own state only when asked.

    python tools/uninstall_aura.py --dry-run          # say what would go, change nothing
    python tools/uninstall_aura.py                    # the app and her launch agents
    python tools/uninstall_aura.py --purge-state      # and everything she has lived through

An install leaves three things. Her app, /Applications/Aura.app (scripts/install_to_applications.py,
tools/app_bundle_freshness.py). Her launch agents, ~/Library/LaunchAgents/com.aura.*.plist (scripts/install_service.sh,
tools/install_supervisor.sh), each booted out of launchd before it goes. Her state, under ~/.aura: memories, models,
logs. The first two go by default. Her state goes only with --purge-state, because it is not an install: it is what
she has lived through, and a reinstall that cannot find it is a different being.

Nothing is deleted outright. Each thing is moved to the Trash, where it can be put back. It refuses while she is running,
since a runtime that loses its app mid-session is a crash, not an uninstall. The source checkout and its virtualenv
stay: they are where a person chose to clone her, and removing them is theirs to do.

Every path is a flag, so the whole of it can be proved in a sandbox (tests/test_uninstall_aura.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LAUNCH_AGENT_PREFIX = "com.aura."


@dataclass
class Plan:
    app: Path | None = None
    launch_agents: list[Path] = field(default_factory=list)
    state: Path | None = None

    def items(self) -> list[Path]:
        return [path for path in (self.app, *self.launch_agents, self.state) if path is not None]


def plan(*, applications: Path, home: Path, state_root: Path, purge_state: bool) -> Plan:
    """What an uninstall would remove here."""
    app = applications / "Aura.app"
    agents_dir = home / "Library" / "LaunchAgents"
    agents = sorted(agents_dir.glob(f"{LAUNCH_AGENT_PREFIX}*.plist")) if agents_dir.is_dir() else []
    return Plan(app=app if app.exists() else None, launch_agents=agents,
                state=state_root if purge_state and state_root.exists() else None)


def she_is_running(port: int) -> bool:
    """Whether something answers on her port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def to_trash(path: Path, trash: Path) -> Path:
    """Move ``path`` into ``trash`` under a name that cannot collide; where it went."""
    trash.mkdir(parents=True, exist_ok=True)
    landing = trash / path.name
    if landing.exists():
        landing = trash / f"{path.name} {time.strftime('%Y-%m-%d %H.%M.%S')}"
    shutil.move(str(path), str(landing))
    return landing


def boot_out(plist: Path, launchctl: str) -> str:
    """Take a launch agent out of launchd before its file goes; launchctl's answer."""
    try:
        done = subprocess.run([launchctl, "bootout", f"gui/{os.getuid()}", str(plist)], capture_output=True,
                              text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"not booted out: {exc}"
    # 3 and 113 are launchctl's "no such service": an agent that was never loaded has nothing to boot out.
    return "booted out" if done.returncode == 0 else f"not loaded ({done.returncode})"


def uninstall(found: Plan, *, trash: Path, launchctl: str) -> list[dict[str, str]]:
    done = []
    for plist in found.launch_agents:
        done.append({"path": str(plist), "launchd": boot_out(plist, launchctl),
                     "moved_to": str(to_trash(plist, trash))})
    for path in (found.app, found.state):
        if path is not None:
            done.append({"path": str(path), "moved_to": str(to_trash(path, trash))})
    return done


def main(argv: list[str] | None = None) -> int:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="say what would go, change nothing")
    parser.add_argument("--purge-state", action="store_true", help="also move her state root to the Trash")
    parser.add_argument("--applications", type=Path, default=Path("/Applications"))
    parser.add_argument("--home", type=Path, default=home)
    parser.add_argument("--state-root", type=Path, default=home / ".aura")
    parser.add_argument("--trash", type=Path, default=home / ".Trash")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--launchctl", default="/bin/launchctl")
    args = parser.parse_args(argv)

    found = plan(applications=args.applications, home=args.home, state_root=args.state_root,
                 purge_state=args.purge_state)
    report: dict[str, object] = {"would_remove" if args.dry_run else "removed": [str(p) for p in found.items()],
                                 "kept": [] if args.purge_state else [str(args.state_root)]}
    if args.dry_run or not found.items():
        print(json.dumps(report, indent=2))
        return 0
    if she_is_running(args.port):
        print(f"Aura is running on port {args.port}. Quit her first; nothing was removed.", file=sys.stderr)
        return 2
    report["removed"] = uninstall(found, trash=args.trash, launchctl=args.launchctl)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
