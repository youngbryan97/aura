#!/usr/bin/env python3
"""Subsystem census: one verdict per core/ directory, generated not hand-kept.

Builds on the architecture map (tools/arch_map.py) and adds what a sweep
needs to prioritize work:

- size (files/lines) and dependency degree (in/out edges)
- test coverage presence (any tests importing the subsystem)
- degradation hygiene (record_degradation sites without a recovery action)
- verdicts: ISOLATED (no deps in/out — dead or tooling), LEAF (nothing
  depends on it), HUB (high fan-in), UNTESTED (no test imports found)

Output: artifacts/architecture/census_latest.json + .md. Run:
    python tools/subsystem_census.py
"""
from __future__ import annotations

import ast
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE = ROOT / "core"
TESTS = ROOT / "tests"
OUT_DIR = ROOT / "artifacts" / "architecture"

HUB_FAN_IN = 8


def _iter_py(base: Path):
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def _subsystem_of(path: Path) -> str:
    rel = path.relative_to(CORE)
    return rel.parts[0] if len(rel.parts) > 1 else "(core root)"


def _module_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("core."):
            parts = node.module.split(".")
            if len(parts) >= 2:
                found.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("core."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        found.add(parts[1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Dynamic imports (async_safe_import, importlib) reference
            # modules as strings; without this, optionally-loaded
            # subsystems look dead.
            value = node.value
            if value.startswith("core.") and "." in value[5:]:
                found.add(value.split(".")[1])
    return found


def _degradation_stats(path: Path) -> tuple[int, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return 0, 0
    total = actionless = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name == "record_degradation":
                total += 1
                kwargs = {k.arg for k in node.keywords}
                if "action" not in kwargs and len(node.args) < 4:
                    actionless += 1
    return total, actionless


def build_census() -> dict:
    stats: dict[str, dict] = defaultdict(
        lambda: {
            "files": 0,
            "lines": 0,
            "deps_out": set(),
            "degradation_sites": 0,
            "degradation_actionless": 0,
        }
    )

    for path in _iter_py(CORE):
        sub = _subsystem_of(path)
        entry = stats[sub]
        entry["files"] += 1
        try:
            entry["lines"] += sum(1 for _ in path.open(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        entry["deps_out"] |= _module_imports(path) - {sub}
        total, actionless = _degradation_stats(path)
        entry["degradation_sites"] += total
        entry["degradation_actionless"] += actionless

    deps_in: dict[str, set[str]] = defaultdict(set)
    for sub, entry in stats.items():
        for dep in entry["deps_out"]:
            deps_in[dep].add(sub)

    tested: set[str] = set()
    for path in _iter_py(TESTS):
        for dep in _module_imports(path):
            tested.add(dep)

    # Non-core dependents (interface layer, entrypoints) keep a subsystem
    # alive even when no core module imports it.
    external_dependents: dict[str, set[str]] = defaultdict(set)
    for base, label in ((ROOT / "interface", "interface"), (ROOT / "aura_main.py", "aura_main")):
        paths = [base] if base.is_file() else list(_iter_py(base))
        for path in paths:
            for dep in _module_imports(path):
                external_dependents[dep].add(label)

    census = {}
    for sub, entry in sorted(stats.items()):
        fan_in = len(deps_in.get(sub, set())) + len(external_dependents.get(sub, set()))
        fan_out = len(entry["deps_out"])
        verdicts = []
        if fan_in == 0 and fan_out == 0:
            verdicts.append("ISOLATED")
        elif fan_in == 0:
            verdicts.append("LEAF")
        if fan_in >= HUB_FAN_IN:
            verdicts.append("HUB")
        if sub not in tested:
            verdicts.append("UNTESTED")
        census[sub] = {
            "files": entry["files"],
            "lines": entry["lines"],
            "fan_in": fan_in,
            "fan_out": fan_out,
            "tested": sub in tested,
            "degradation_sites": entry["degradation_sites"],
            "degradation_actionless": entry["degradation_actionless"],
            "verdicts": verdicts,
        }
    return {
        "schema": "aura.subsystem_census.v1",
        "generated_at_unix": time.time(),
        "subsystem_count": len(census),
        "subsystems": census,
    }


def write_markdown(census: dict, path: Path) -> None:
    rows = sorted(
        census["subsystems"].items(), key=lambda kv: kv[1]["lines"], reverse=True
    )
    lines = [
        "# Subsystem Census",
        "",
        f"Schema: `{census['schema']}` — {census['subsystem_count']} subsystems. Generated artifact; do not hand-edit.",
        "",
        "| Subsystem | Files | Lines | In | Out | Tested | Degr (actionless) | Verdicts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in rows:
        lines.append(
            f"| {name} | {e['files']} | {e['lines']} | {e['fan_in']} | {e['fan_out']} "
            f"| {'ok' if e['tested'] else 'failed'} | {e['degradation_sites']} ({e['degradation_actionless']}) "
            f"| {', '.join(e['verdicts']) or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    census = build_census()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "census_latest.json").write_text(
        json.dumps(census, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(census, OUT_DIR / "census_latest.md")
    isolated = [s for s, e in census["subsystems"].items() if "ISOLATED" in e["verdicts"]]
    untested = [s for s, e in census["subsystems"].items() if "UNTESTED" in e["verdicts"]]
    print(f"census: {census['subsystem_count']} subsystems -> {OUT_DIR}")
    print(f"ISOLATED ({len(isolated)}): {', '.join(sorted(isolated)[:12])}")
    print(f"UNTESTED ({len(untested)}): {', '.join(sorted(untested)[:12])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
