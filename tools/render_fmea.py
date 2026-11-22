#!/usr/bin/env python3
"""tools/render_fmea.py — the failure-mode registry, rendered from code.

Every known way Aura fails has exactly one authoritative enumeration:
core/runtime/fmea.py. This renders it into docs/FMEA.md so a skeptical
engineer can audit coverage without reading the module — and a drift test
keeps the document honest. Never edit the doc by hand.

Usage:
  python tools/render_fmea.py            # writes docs/FMEA.md
  python tools/render_fmea.py --check    # exit 1 if the doc drifted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOC_PATH = REPO_ROOT / "docs" / "FMEA.md"


def render() -> str:
    from core.runtime.fmea import (
        FMEA_REGISTRY,
        FMEA_VERSION,
        detection_gaps,
        mitigation_gaps,
        registry_summary,
    )

    summary = registry_summary()
    lines = [
        "# Aura FMEA — failure modes & effects registry",
        "",
        "> GENERATED from `core/runtime/fmea.py` by `tools/render_fmea.py`",
        "> (`make fmea-doc`). Do not edit by hand — a drift test regenerates",
        "> and compares this file on every suite run.",
        "",
        f"Registry version: `{FMEA_VERSION}` — "
        f"{summary['total']} modes "
        f"({summary['catastrophic']} catastrophic, {summary['critical']} critical, "
        f"{summary['major']} major, {summary['minor']} minor); "
        f"{summary['mitigation_gaps']} open mitigation gap(s), "
        f"{summary['detection_gaps']} open detection gap(s).",
        "",
        "Every entry is REAL: it either occurred live (occurrences cite when)",
        "or is a structurally-reachable state found by analysis. Gaps are",
        "explicit and pinned by an allowlist test that only shrinks.",
        "",
    ]

    for mode in FMEA_REGISTRY:
        lines.extend(
            [
                f"## {mode.id} — {mode.mode}",
                "",
                f"- **Subsystem:** {mode.subsystem}",
                f"- **Severity / blast radius:** {mode.severity} / {mode.blast_radius}",
                f"- **Cause:** {mode.cause}",
                f"- **Effect:** {mode.effect}",
                f"- **Detection:** {mode.detection}",
                f"- **Mitigation:** {mode.mitigation}",
            ]
        )
        if mode.detection_modules:
            lines.append(
                "- **Detection modules:** "
                + ", ".join(f"`{m}`" for m in mode.detection_modules)
            )
        if mode.mitigation_modules:
            lines.append(
                "- **Mitigation modules:** "
                + ", ".join(f"`{m}`" for m in mode.mitigation_modules)
            )
        if mode.occurrences:
            lines.append("- **Recorded occurrences:** " + "; ".join(mode.occurrences))
        if mode.notes:
            lines.append(f"- **Notes:** {mode.notes}")
        lines.append("")

    open_gaps = [*mitigation_gaps(), *detection_gaps()]
    lines.append("## Open gaps (the work queue)")
    lines.append("")
    if open_gaps:
        for mode in dict.fromkeys(open_gaps):
            gap_kind = []
            if mode.detection.strip().upper() == "GAP":
                gap_kind.append("detection")
            if mode.mitigation.strip().upper() == "GAP":
                gap_kind.append("mitigation")
            lines.append(f"- **{mode.id}** ({'+'.join(gap_kind)} gap): {mode.mode}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    content = render()
    if args.check:
        existing = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        if existing != content:
            print("docs/FMEA.md has drifted from core/runtime/fmea.py — run `make fmea-doc`")
            return 1
        print("docs/FMEA.md is in sync")
        return 0
    DOC_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
