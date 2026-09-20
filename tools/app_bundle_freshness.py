#!/usr/bin/env python
"""Report — and optionally repair — Aura.app bundle staleness (CP216).

The boot banner refuses launch when the installed bundle's recorded
``commit_sha`` differs from the current checkout. That check is correct and
worth keeping, but a manual rebuild goes stale again on the very next
commit, so the operator sees the same banner repeatedly and the remedy
feels like it did not work.

This makes freshness a queryable state and a single idempotent command:

    tools/app_bundle_freshness.py check     # exit 0 fresh, 1 stale
    tools/app_bundle_freshness.py ensure    # rebuild only when stale

``ensure`` is a no-op when the bundle already matches HEAD, so it is safe
to run at the end of any work session or from a hook.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRESHNESS_SCHEMA = "aura.app_bundle_freshness.v1"
DEFAULT_BUNDLE = Path("/Applications/Aura.app")
PROVENANCE_RELATIVE = "Contents/Resources/aura-launch-provenance.json"


def head_commit() -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def bundle_state(bundle: Path) -> dict:
    """What the installed bundle claims about its own provenance."""
    provenance = bundle / PROVENANCE_RELATIVE
    if not bundle.exists():
        return {"installed": False, "reason": "bundle_absent"}
    if not provenance.is_file():
        return {"installed": True, "reason": "provenance_absent"}
    try:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"installed": True, "reason": f"provenance_unreadable:{exc}"}
    return {
        "installed": True,
        "commit_sha": str(payload.get("commit_sha") or ""),
        "branch": payload.get("branch"),
        "source_dirty": bool(payload.get("source_dirty")),
        "generated_at_unix": payload.get("generated_at_unix"),
    }


def evaluate(bundle: Path) -> dict:
    state = bundle_state(bundle)
    head = head_commit()
    installed_commit = state.get("commit_sha", "")
    fresh = bool(installed_commit) and installed_commit == head
    return {
        "schema": FRESHNESS_SCHEMA,
        "bundle": str(bundle),
        "head_commit": head,
        "bundle_commit": installed_commit or None,
        "fresh": fresh,
        "reason": state.get("reason"),
        "bundle_state": state,
    }


def rebuild(bundle: Path) -> int:
    script = REPO_ROOT / "scripts" / "bundle_app.sh"
    if not script.is_file():
        print(f"bundle script missing: {script}", file=sys.stderr)
        return 1
    environment = {"AURA_INSTALL_PATH": str(bundle)}
    print(f"rebuilding {bundle} from HEAD…", flush=True)
    completed = subprocess.run(
        ["/bin/bash", str(script)],
        cwd=REPO_ROOT,
        env={**dict(__import__("os").environ), **environment},
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:])
        sys.stderr.write(completed.stderr[-4000:])
        return completed.returncode
    for line in completed.stdout.strip().splitlines()[-3:]:
        print(f"  {line}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "ensure"))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle).expanduser()
    verdict = evaluate(bundle)
    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        head = verdict["head_commit"][:10]
        built = (verdict["bundle_commit"] or "none")[:10]
        status = "FRESH" if verdict["fresh"] else "STALE"
        print(f"{status}: bundle={built} head={head} ({verdict['bundle']})")
        if verdict["reason"]:
            print(f"  reason: {verdict['reason']}")

    if verdict["fresh"]:
        return 0
    if args.command == "check":
        return 1
    code = rebuild(bundle)
    if code != 0:
        return code
    after = evaluate(bundle)
    if not after["fresh"]:
        print(
            "rebuild completed but the bundle still does not match HEAD",
            file=sys.stderr,
        )
        return 1
    print(f"FRESH: bundle now at {after['bundle_commit'][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
