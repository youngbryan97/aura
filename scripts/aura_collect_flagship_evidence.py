#!/usr/bin/env python3
"""Collect a flagship-readiness evidence bundle for Aura.

This does not claim metaphysical proof. It creates a concrete artifact with:
- source health gate results
- task ownership findings
- persistence audit findings
- morphogenesis file/integration presence
- recent log evidence when logs are present
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run_cmd(cmd: list[str], cwd: Path) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=90)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:],
            "stderr": proc.stderr[-20000:],
            "duration_s": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}", "duration_s": round(time.time() - started, 3)}


def read_tail(path: Path, max_chars: int = 5000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[-max_chars:]
    except Exception:
        return ""


def find_logs(root: Path) -> list[Path]:
    candidates = []
    for base in [root / "logs", Path.home() / ".aura" / "logs"]:
        if base.exists():
            candidates.extend(sorted(base.glob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)[:8])
    return candidates


def collect(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence: dict[str, Any] = {
        "schema": "aura.flagship.evidence.v1",
        "created_at": time.time(),
        "root": str(root),
        "python": sys.version,
        "checks": {},
        "presence": {},
        "logs": {},
    }

    evidence["presence"] = {
        "aura_main": (root / "aura_main.py").exists(),
        "morphogenesis_runtime": (root / "core" / "morphogenesis" / "runtime.py").exists(),
        "morphogenesis_hooks": (root / "core" / "morphogenesis" / "hooks.py").exists(),
        "task_ownership": (root / "core" / "runtime" / "task_ownership.py").exists(),
        "persistence_ownership": (root / "core" / "runtime" / "persistence_ownership.py").exists(),
        "flagship_readiness": (root / "core" / "runtime" / "flagship_readiness.py").exists(),
    }

    commands = {
        "flagship_readiness": [sys.executable, "-m", "core.runtime.flagship_readiness", "--json", "."],
        "task_ownership": [sys.executable, "scripts/aura_task_ownership_codemod.py", ".", "--json"],
        "persistence_audit": [sys.executable, "scripts/aura_persistence_audit.py", ".", "--json"],
    }
    for name, cmd in commands.items():
        if (root / cmd[1]).exists() or cmd[1] == "-m":
            evidence["checks"][name] = run_cmd(cmd, root)
        else:
            evidence["checks"][name] = {"skipped": True, "reason": f"{cmd[1]} not found"}

    for log in find_logs(root):
        tail = read_tail(log)
        evidence["logs"][str(log)] = {
            "tail": tail,
            "contains_morphogenesis_started": "MorphogeneticRuntime started" in tail,
            "contains_hooks_wired": "Morphogenesis hooks" in tail or "Morphogenesis hooks wired" in tail,
            "contains_consciousness_online": "Consciousness System ONLINE" in tail,
        }

    json_path = out_dir / "flagship_evidence.json"
    json_path.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=repr), encoding="utf-8")

    md_lines = [
        "# Aura Flagship Evidence Bundle",
        "",
        f"Created: {time.ctime(evidence['created_at'])}",
        f"Root: `{root}`",
        "",
        "## Presence",
    ]
    for k, v in evidence["presence"].items():
        md_lines.append(f"- {k}: {'yes' if v else 'no'}")
    md_lines.append("")
    md_lines.append("## Checks")
    for k, v in evidence["checks"].items():
        rc = v.get("returncode", "skipped" if v.get("skipped") else "error")
        md_lines.append(f"- {k}: {rc}")
    md_lines.append("")
    md_lines.append("See `flagship_evidence.json` for complete stdout/stderr/log tails.")
    (out_dir / "flagship_evidence.md").write_text("\n".join(md_lines), encoding="utf-8")

    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--out", default="flagship_evidence")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    evidence = collect(root, out)
    print(f"Wrote evidence bundle to {out}")
    print(json.dumps({"presence": evidence["presence"], "checks": {k: v.get("returncode", v.get("skipped")) for k, v in evidence["checks"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
