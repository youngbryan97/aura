from __future__ import annotations
#!/usr/bin/env python3
"""Generate a longitudinal morphogenesis report.

Reads persisted morphogenesis registry state and recent logs, then writes a
JSON+Markdown report that helps demonstrate whether Aura's cell/tissue/organ
runtime is actually operating over time.
"""

from core.runtime.atomic_writer import atomic_write_text

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _payload(data: Any) -> Any:
    if isinstance(data, dict) and "payload" in data:
        return data["payload"]
    return data


def _candidate_state_paths(root: Path) -> list[Path]:
    paths = [
        root / "data" / "morphogenesis" / "morphogenesis_state.json",
        root / ".aura" / "data" / "morphogenesis" / "morphogenesis_state.json",
        Path.home() / ".aura" / "data" / "morphogenesis" / "morphogenesis_state.json",
    ]
    return [p for p in paths if p.exists()]


def _log_candidates(root: Path) -> list[Path]:
    out: list[Path] = []
    for base in [root / "logs", Path.home() / ".aura" / "logs"]:
        if base.exists():
            try:
                out.extend(sorted(base.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:10])
            except Exception:
                pass
    return out


def _tail(path: Path, max_chars: int = 12000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]
    except Exception:
        return ""


def analyze_registry(data: Any) -> dict[str, Any]:
    payload = _payload(data) or {}
    cells = payload.get("cells", {}) if isinstance(payload, dict) else {}
    organs = payload.get("organs", {}) if isinstance(payload, dict) else {}
    by_lifecycle: dict[str, int] = {}
    by_role: dict[str, int] = {}

    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            state = cell.get("state", {})
            manifest = cell.get("manifest", {})
            lifecycle = str(state.get("lifecycle", "unknown"))
            role = str(manifest.get("role", "unknown"))
            by_lifecycle[lifecycle] = by_lifecycle.get(lifecycle, 0) + 1
            by_role[role] = by_role.get(role, 0) + 1

    return {
        "cell_count": len(cells) if isinstance(cells, dict) else 0,
        "organ_count": len(organs) if isinstance(organs, dict) else 0,
        "by_lifecycle": by_lifecycle,
        "by_role": by_role,
        "organ_ids": sorted(list(organs.keys()))[:20] if isinstance(organs, dict) else [],
    }


def analyze_logs(root: Path) -> dict[str, Any]:
    logs = _log_candidates(root)
    text = "\n".join(_tail(p) for p in logs[:8])
    markers = {
        "morphogenetic_runtime_started": text.count("MorphogeneticRuntime started"),
        "morphogenesis_hooks": text.count("Morphogenesis hooks"),
        "organ_stabilizer": text.count("organ_stabilizer"),
        "organ_episode": text.count("morphogenesis.organ_episode"),
        "adaptive_immunity_bridge": text.count("Adaptive immunity bridge"),
        "morphogenesis_tick_failed": text.count("Morphogenesis tick failed"),
    }
    return {
        "log_count": len(logs),
        "logs": [str(p) for p in logs[:8]],
        "markers": markers,
    }


def build_report(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_paths = _candidate_state_paths(root)
    registry_reports = []
    for p in state_paths:
        data = _safe_json(p)
        registry_reports.append({"path": str(p), "analysis": analyze_registry(data)})

    report = {
        "schema": "aura.morphogenesis.longitudinal_report.v1",
        "created_at": time.time(),
        "root": str(root),
        "state_paths": [str(p) for p in state_paths],
        "registries": registry_reports,
        "logs": analyze_logs(root),
    }

    json_path = out_dir / "morphogenesis_longitudinal_report.json"
    atomic_write_text(json_path, json.dumps(report, indent=2, sort_keys=True, default=repr), encoding="utf-8")

    md = ["# Aura Morphogenesis Longitudinal Report", "", f"Created: {time.ctime(report['created_at'])}", ""]
    md.append("## Registry snapshots")
    if registry_reports:
        for item in registry_reports:
            a = item["analysis"]
            md.append(f"- `{item['path']}`: cells={a['cell_count']}, organs={a['organ_count']}, lifecycle={a['by_lifecycle']}")
    else:
        md.append("- No persisted morphogenesis registry state found.")
    md.append("")
    md.append("## Log markers")
    for k, v in report["logs"]["markers"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("A strong public demo should show non-zero runtime starts/hooks, cell activity, organ formation, and no repeated tick failures.")
    atomic_write_text((out_dir / "morphogenesis_longitudinal_report.md"), "\n".join(md), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--out", default="morphogenesis_longitudinal_report")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    report = build_report(root, out)
    print(f"Wrote morphogenesis report to {out}")
    print(json.dumps({"state_paths": report["state_paths"], "log_markers": report["logs"]["markers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
