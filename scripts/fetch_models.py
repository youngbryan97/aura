#!/usr/bin/env python3
"""Aura model fetcher — inventory, disk preflight, and resumable download.

Thin CLI over core.brain.llm.model_lifecycle.ModelLifecycleManager so the
download path the launcher uses and the one a human runs are the same code.

    python scripts/fetch_models.py            # download any missing models
    python scripts/fetch_models.py --status   # just print the inventory
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.model_lifecycle import ModelLifecycleManager  # noqa: E402

_GB = float(1024**3)

REASONING_SOLVER_ALIASES = {
    "qwq32b": "QwQ-32B-4bit",
    "qwq-32b": "QwQ-32B-4bit",
    "r1-qwen32b": "DeepSeek-R1-Distill-Qwen-32B-4bit",
    "deepseek-r1-qwen32b": "DeepSeek-R1-Distill-Qwen-32B-4bit",
    "r1-qwen32b-8bit": "DeepSeek-R1-Distill-Qwen-32B-8bit",
}


def _plan_with_reasoning_solver(alias: str | None) -> dict[str, str] | None:
    if not alias:
        return None
    normalized = alias.strip().lower()
    model_name = REASONING_SOLVER_ALIASES.get(normalized)
    if model_name is None:
        choices = ", ".join(sorted(REASONING_SOLVER_ALIASES))
        raise SystemExit(f"Unknown reasoning solver '{alias}'. Choices: {choices}")
    plan = ModelLifecycleManager._default_plan()
    plan["solver"] = model_name
    return plan


def _print_inventory(manager: ModelLifecycleManager) -> None:
    print("Aura model inventory")
    print("=" * 60)
    for status in manager.inventory():
        mark = "ok" if status.present else "missing"
        where = (
            status.location
            if status.present
            else f"would fetch {status.source_repo or '(no source)'}"
        )
        size = (
            f"{status.size_bytes / _GB:.1f}GB"
            if status.present
            else f"~{status.approx_download_gb:.0f}GB"
        )
        print(f"  {mark} {status.role:9s} {status.name:32s} {size:>8s}  {where}")


def _progress(event: dict) -> None:
    name = event.get("name", "")
    kind = event.get("event", "")
    if kind == "download_start":
        print(f"   {name}: downloading {event.get('repo')} (attempt {event.get('attempt')})...")
    elif kind == "download_ok":
        print(f"     ok {name} ready.")
    elif kind == "download_failed":
        print(f"     failed {name} failed: {event.get('error')}")
    elif kind == "disk_insufficient":
        print(f"   Not enough disk: need ~{event.get('required_gb')}GB, free {event.get('free_gb')}GB")
    elif kind == "all_present":
        print("  All models already present.")


def _print_env(manager: ModelLifecycleManager) -> None:
    plan = dict(getattr(manager, "_plan", {}))
    solver = plan.get("solver", "")
    if not solver:
        return
    status = manager.status_for("solver", solver)
    print("\nRuntime environment for this solver lane:")
    print(f"  export AURA_DEEP_MODEL={solver}")
    print(f"  export AURA_LLM__MLX_DEEP_MODEL_PATH={status.location}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="print inventory and exit")
    parser.add_argument("--no-disk-check", action="store_true", help="skip the disk preflight")
    parser.add_argument(
        "--reasoning-solver",
        metavar="ALIAS",
        help=(
            "replace the solver lane in this fetch plan with a local reasoning model "
            f"({', '.join(sorted(REASONING_SOLVER_ALIASES))})"
        ),
    )
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="print AURA_DEEP_MODEL/AURA_LLM__MLX_DEEP_MODEL_PATH exports for the plan",
    )
    args = parser.parse_args(argv)

    manager = ModelLifecycleManager(plan=_plan_with_reasoning_solver(args.reasoning_solver))
    _print_inventory(manager)
    if args.print_env:
        _print_env(manager)

    if args.status:
        return 0

    pf = manager.disk_preflight()
    print(
        f"\nDisk at {pf.target}: free {pf.free_bytes / _GB:.1f}GB, "
        f"need ~{pf.required_bytes / _GB:.1f}GB → {'OK' if pf.ok else 'INSUFFICIENT'}"
    )

    print("\n Fetching missing models...")
    report = manager.ensure_present(progress=_progress, check_disk=not args.no_disk_check)

    if report.get("failed"):
        print(f"\nSome downloads failed: {[f['name'] for f in report['failed']]}")
        return 1
    if report.get("skipped_disk"):
        print("\n Skipped: insufficient disk space.")
        return 1
    print("\nModel fetch complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
