#!/usr/bin/env python3
"""What still blocks the 27B recovery launch, and the command once nothing does.

Everything the migration prepared on CPU converges here. This runs every gate
that can run without the model, collects what each one refuses, and emits one
receipt naming the blockers with the action that clears each. When the list is
empty it prints the launch command, and it prints nothing launchable while any
blocker stands -- a command that is only correct under conditions the reader has
to remember is how a campaign gets started against a drifted tree.

Blockers are separated by who can clear them, because they behave differently:
an environmental one (memory, lane ownership) clears itself when the host
quiets down, while a package one (source drift, a stale bundle) needs somebody
to do something.

    python tools/report_27b_launch_readiness.py --json OUT
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import preflight_27b_recurrence_campaign as preflight  # noqa: E402

READINESS_SCHEMA: Final = "aura.rlc.27b_launch_readiness.v1"

BUNDLE = Path("artifacts/migration/27b/campaign_bundle.json")
PACKAGE = Path("artifacts/migration/27b/recovery/package.json")
CAMPAIGN_CONFIG = Path("artifacts/migration/27b/recovery/campaign.json")

#: Blockers that clear when the host quiets down, versus blockers somebody has
#: to act on. The distinction decides whether "wait" is a valid response.
ENVIRONMENTAL: Final = frozenset(
    {
        "insufficient_ram",
        "insufficient_disk",
        "host_under_memory_pressure",
        "ram_unmeasured",
        "disk_unmeasured",
        "model_lane_already_owned",
        "lane_ownership_unmeasured",
    }
)

REMEDIES: Final = {
    "source_drifted": "re-freeze: make rlc-27b-preflight after tools/prepare_27b_recurrence_campaign.py --out ...",
    "source_missing": "restore the file or re-freeze the bundle",
    "tissue_drifted": "re-freeze; portable tissue changed under the bundle",
    "tissue_missing": "restore the tissue asset",
    "checkpoint_config_drifted": "the checkpoint changed; re-freeze against it",
    "checkpoint_file_drifted": "the checkpoint changed; re-freeze against it",
    "checkpoint_absent": "install the checkpoint the bundle names",
    "active_model_moved": "point the runtime back, or re-freeze against the new active model",
    "attention_layout_changed": "re-freeze; the layer topology is not the one committed",
    "window_misaligned": "choose a window on the full-attention interval",
    "stale_evidence_root": "aim evidence at artifacts/migration/27b/recovery",
    "bundle_tampered": "rebuild the bundle rather than editing it",
    "schema_unrecognised": "rebuild the bundle with the current preparer",
    "insufficient_ram": "free memory, or wait for the resident model to unload",
    "insufficient_disk": "free space for the export",
    "host_under_memory_pressure": "wait for the compressor to drain",
    "ram_unmeasured": "investigate the memory guard before launching",
    "disk_unmeasured": "investigate the filesystem before launching",
    "model_lane_already_owned": "wait for the owner to release, or stop it deliberately",
    "lane_ownership_unmeasured": "investigate the lane controller before launching",
    "package_not_materialized": "make rlc-27b-package",
    "package_carries_a_verdict": "a fresh package must carry none until the run measures one",
    "campaign_config_not_materialized": (
        "materialize and validate the source-bound launchd campaign config"
    ),
    "campaign_launch_command_invalid": (
        "repair the readiness command to match the campaign controller CLI"
    ),
}


def _launch_command(config_path: Path) -> str:
    arguments = (
        "/Users/bryan/.aura/live-source/.venv/bin/python",
        "tools/run_unified_intrinsic_resident_campaign.py",
        "install",
        "--config",
        str(config_path),
    )
    return shlex.join(arguments)


def _campaign_config_findings() -> list[dict[str, str]]:
    path = REPO_ROOT / CAMPAIGN_CONFIG
    if not path.is_file() or path.is_symlink():
        return [
            {
                "kind": "campaign_config_not_materialized",
                "detail": str(CAMPAIGN_CONFIG),
            }
        ]
    return []


def _package_findings() -> list[dict[str, str]]:
    path = REPO_ROOT / PACKAGE
    if not path.exists():
        return [{"kind": "package_not_materialized", "detail": str(PACKAGE)}]
    try:
        package = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [{"kind": "package_not_materialized", "detail": str(exc)}]
    if package.get("verdict") is not None:
        return [
            {
                "kind": "package_carries_a_verdict",
                "detail": f"verdict={package['verdict']} before the run",
            }
        ]
    return []


def build() -> dict[str, Any]:
    bundle_path = REPO_ROOT / BUNDLE
    findings: list[dict[str, str]] = []
    if not bundle_path.exists():
        findings.append(
            {"kind": "schema_unrecognised", "detail": f"{BUNDLE} does not exist"}
        )
    else:
        findings.extend(preflight.check(json.loads(bundle_path.read_text())))
    findings.extend(_package_findings())
    findings.extend(_campaign_config_findings())

    for finding in findings:
        finding["remedy"] = REMEDIES.get(finding["kind"], "investigate")
        finding["class"] = (
            "environmental" if finding["kind"] in ENVIRONMENTAL else "package"
        )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""

    ready = not findings
    body = {
        "schema": READINESS_SCHEMA,
        "source_commit": commit,
        "bundle": str(BUNDLE),
        "package": str(PACKAGE),
        "ready_to_launch": ready,
        "blockers": findings,
        "blocker_counts": {
            "package": sum(1 for f in findings if f["class"] == "package"),
            "environmental": sum(
                1 for f in findings if f["class"] == "environmental"
            ),
        },
        "launch_command": _launch_command(CAMPAIGN_CONFIG) if ready else None,
        "model_active_stages": [
            "calibration",
            "training",
            "canary",
            "lesion_arms",
            "export",
        ],
        "measured_workload": {
            "basis": (
                "CP566 on the 32B: 300 decodes over 60 tasks in 4,814.53 s"
            ),
            "decode_seconds": 4814.53,
            "training_seconds": None,
            "training_note": (
                "no retained receipt records a training wall time; unmeasured "
                "rather than estimated"
            ),
        },
    }
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.json}\n")

    if report["ready_to_launch"]:
        print("READY. Launch with:\n")
        print(f"  {report['launch_command']}")
        return 0

    for finding in report["blockers"]:
        print(f"  [{finding['class']:13s}] {finding['kind']}")
        print(f"                    {finding['detail']}")
        print(f"                    -> {finding['remedy']}")
    counts = report["blocker_counts"]
    print(
        f"\n{counts['package']} package blocker(s), "
        f"{counts['environmental']} environmental. No launch command emitted."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
