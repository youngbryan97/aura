#!/usr/bin/env python3
"""tools/render_health_contract.py — the runtime contract, rendered from code.

"What must be alive for Aura to be considered healthy?" has exactly one
authoritative answer: core/runtime/health_contract.py. This renders that
answer into docs/RUNTIME_CONTRACT.md so a human (or a skeptical engineer)
can read the contract without reading the module — and a drift test keeps
the document honest: if the code changes and the doc doesn't, the suite
fails. Never edit the doc by hand.

Usage:
  python tools/render_health_contract.py            # writes docs/RUNTIME_CONTRACT.md
  python tools/render_health_contract.py --check    # exit 1 if the doc drifted
"""
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
        "# Aura Runtime Contract — what must be alive",
        "",
        "> GENERATED from `core/runtime/health_contract.py` by",
        "> `tools/render_health_contract.py`. Do not edit by hand — a drift",
        "> test regenerates and compares this file on every suite run.",
        "",
        f"Contract version: `{HEALTH_CONTRACT_VERSION}`",
        "",
    ]

    tier_meaning = {
        ServiceTier.CRITICAL: "Aura CANNOT function without these",
        ServiceTier.IMPORTANT: "Aura works but the experience is degraded",
        ServiceTier.OPTIONAL: "background enrichment; loss is invisible to the user",
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
