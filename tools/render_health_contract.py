#!/usr/bin/env python3
"""Write docs/RUNTIME_CONTRACT.md from the runtime health contract. Use --check to compare."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOC_PATH = REPO_ROOT / "docs" / "RUNTIME_CONTRACT.md"


def render() -> str:
    from core.runtime.health_contract import (
        HEALTH_CONTRACT_VERSION,
        REQUIRED_HEALTH_PROBE_GROUPS,
        RUNTIME_CONTRACT,
        ServiceTier,
    )

    lines = [
        "# Aura runtime health contract",
        "",
        "These checks come from the [runtime health contract](../core/runtime/health_contract.py).",
        "Regenerate this page with `make contract-doc`.",
        "",
        f"Contract version: `{HEALTH_CONTRACT_VERSION}`",
        "",
    ]

    tier_meaning = {
        ServiceTier.CRITICAL: "required services",
        ServiceTier.IMPORTANT: "services monitored for degraded operation",
        ServiceTier.OPTIONAL: "optional services",
    }
    for tier in (ServiceTier.CRITICAL, ServiceTier.IMPORTANT, ServiceTier.OPTIONAL):
        entries = [r for r in RUNTIME_CONTRACT if r.tier == tier]
        if not entries:
            continue
        lines += [f"## {tier.value.upper()} — {tier_meaning[tier]}", ""]
        lines.append("| Service | Container key | Liveness check | Why it matters |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for r in entries:
            check = f"`{r.liveness_check}`" if r.liveness_check else "presence only"
            lines.append(
                f"| {r.name} | `{r.container_key}` | {check} | {r.description} |"
            )
        lines.append("")

    lines += ["## Required health probe groups", ""]
    lines.append(
        "Boot readiness additionally requires at least one passing probe from "
        "each group:"
    )
    lines.append("")
    for group, probes in REQUIRED_HEALTH_PROBE_GROUPS.items():
        probe_list = ", ".join(f"`{p}`" for p in probes)
        lines.append(f"- **{group}**: {probe_list}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the doc matches the code")
    args = parser.parse_args()

    content = render()
    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.is_file() else ""
        if current != content:
            print("DRIFT: docs/RUNTIME_CONTRACT.md does not match health_contract.py — "
                  "run: python tools/render_health_contract.py")
            return 1
        print("runtime contract doc is current")
        return 0
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
