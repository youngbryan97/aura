"""What actually ships, counted rather than described.

"Every shipped capability, integration, UI control, and model" is a claim
about four different registries, and each of them can be wrong in the same
way: something declared that nothing can reach, or something reachable that
nothing declared. Both were found in this tree before — skills that could
never run, a route with no page, a model directory nobody names.

So each of the four is enumerated from the thing that decides it at runtime,
and each entry says where it is declared and what proves it reachable.

    skills        core.skills.discovery, against the live CapabilityEngine
    models        the model directory, against the active cortex manifest
    ui            interface/static, against the routes that serve it
    api           every mounted FastAPI route, by method and path

An entry is `reachable` when something outside its own declaration can get to
it, and `orphan` when it cannot. Neither is a verdict on whether it works —
that is U08's job, and this is the list U08 has to cover.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "interface/static"
ROUTES = ROOT / "interface/routes"
SERVER = ROOT / "interface/server.py"

#: A page served by a route rather than fetched from /static by name.
_FILE_RESPONSE = re.compile(r"""["'][^"']*?([A-Za-z0-9_\-]+\.html)["'?]""")
#: `@router.get("/path")` and its siblings.
_ROUTE = re.compile(
    r"@(?:router|app)\.(get|post|put|patch|delete|websocket)\(\s*[\"']([^\"']+)[\"']"
)
#: An element a person can operate: it has an id, or it declares an action.
_CONTROL = re.compile(
    r"<(button|input|select|textarea|a)\b[^>]*?"
    r"(?:id=[\"']([^\"']+)[\"']|data-action=[\"']([^\"']+)[\"'])",
    re.IGNORECASE | re.DOTALL,
)


def skills() -> list[dict[str, Any]]:
    """Every skill the catalog accepts, against the engine that runs them."""

    from core.capability_engine import CapabilityEngine
    from core.skills.discovery import build_skill_catalog

    catalog = build_skill_catalog()
    engine = CapabilityEngine()
    live = set(engine.skills)
    entries = []
    for declaration in catalog.accepted:
        meta = engine.skills.get(declaration.name)
        entries.append(
            {
                "kind": "skill",
                "name": declaration.name,
                "declared_in": str(getattr(declaration, "module", "") or ""),
                "effect_scope": str(getattr(meta, "effect_scope", "") or ""),
                "reachable": declaration.name in live,
                "reachable_by": "core.capability_engine.CapabilityEngine.skills",
            }
        )
    for name in sorted(live - {d.name for d in catalog.accepted}):
        entries.append(
            {
                "kind": "skill",
                "name": name,
                "declared_in": "",
                "effect_scope": "",
                "reachable": True,
                "reachable_by": "live registry only — the catalog does not declare it",
            }
        )
    return entries


def models() -> list[dict[str, Any]]:
    """Every model on disk, against the one the runtime is bound to."""

    from core.brain.llm.model_registry import get_active_cortex_spec, get_models_dir

    directory = get_models_dir()
    try:
        spec = get_active_cortex_spec()
    except Exception:  # noqa: BLE001 — an unreadable manifest is a finding
        spec = None
    active = str(getattr(spec, "model_id", "") or getattr(spec, "name", "") or "")
    entries = []
    if directory.is_dir():
        for path in sorted(p for p in directory.iterdir() if p.is_dir()):
            weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.gguf"))
            entries.append(
                {
                    "kind": "model",
                    "name": path.name,
                    "declared_in": str(path.relative_to(directory.parent)),
                    "bytes": sum(w.stat().st_size for w in weights),
                    "weight_files": len(weights),
                    "reachable": bool(weights) and (not active or path.name == active),
                    "reachable_by": (
                        "active cortex manifest" if path.name == active else "on disk only"
                    ),
                }
            )
    return entries


#: Everything that can put a page in front of somebody. The desktop launcher
#: is Swift and the shell is Rust, and a page opened from there is reachable
#: in exactly the way a page served by a route is — an inventory that reads
#: only the Python called four of them orphans.
_OPENERS = (
    "interface/**/*.py",
    "interface/static/**/*.js",
    "interface/static/**/*.html",
    "interface/static/**/*.json",
    "scripts/*.swift",
    "native/**/*.rs",
)


def _served_pages() -> dict[str, str]:
    """Page name to the first thing found that opens it."""

    served: dict[str, str] = {}
    for pattern in _OPENERS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file() or "node_modules" in path.parts:
                continue
            body = path.read_text("utf-8", errors="replace")
            for match in _FILE_RESPONSE.finditer(body):
                served.setdefault(
                    Path(match.group(1)).name, str(path.relative_to(ROOT))
                )
    return served


def ui() -> list[dict[str, Any]]:
    """Every page under interface/static, and the controls on it."""

    served = _served_pages()
    entries = []
    for path in sorted(STATIC.rglob("*.html")):
        body = path.read_text("utf-8", errors="replace")
        controls = sorted(
            {match.group(2) or match.group(3) for match in _CONTROL.finditer(body)}
        )
        opener = served.get(path.name, "")
        entries.append(
            {
                "kind": "ui",
                "name": str(path.relative_to(STATIC)),
                "declared_in": str(path.relative_to(ROOT)),
                "controls": len(controls),
                "control_ids": controls[:40],
                "reachable": bool(opener),
                "reachable_by": opener,
            }
        )
    return entries


def _included_routers() -> dict[tuple[str, str], str]:
    """Which router objects the server mounts, and under which prefix.

    Read from the server's own source rather than guessed from a filename:
    the modules are imported as ``from interface.routes import chat as
    chat_routes`` and mounted as ``app.include_router(chat_routes.router,
    prefix="/api")``, and a module can export more than one router.
    """

    tree = ast.parse(SERVER.read_text("utf-8", errors="replace"))
    alias_to_module: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "interface.routes"
        ):
            for name in node.names:
                module = (
                    name.name
                    if node.module == "interface.routes"
                    else (node.module or "").rsplit(".", 1)[-1]
                )
                alias_to_module[name.asname or name.name] = module

    mounted: dict[tuple[str, str], str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "include_router"):
            continue
        if not node.args:
            continue
        target = node.args[0]
        prefix = ""
        for keyword in node.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                prefix = str(keyword.value.value or "")
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            module = alias_to_module.get(target.value.id, target.value.id)
            mounted[(module, target.attr)] = prefix
        elif isinstance(target, ast.Name):
            mounted[(alias_to_module.get(target.id, target.id), "router")] = prefix
    return mounted


def api() -> list[dict[str, Any]]:
    """Every route the interface declares, and the prefix it is mounted under."""

    mounted = _included_routers()
    by_module: dict[str, dict[str, str]] = {}
    for (module, attribute), prefix in mounted.items():
        by_module.setdefault(module, {})[attribute] = prefix

    entries = []
    for path in [SERVER, *sorted(ROUTES.glob("*.py"))]:
        if not path.is_file():
            continue
        body = path.read_text("utf-8", errors="replace")
        module = path.stem
        relative = str(path.relative_to(ROOT))
        for match in re.finditer(
            r"@(router|app|[a-z_]+_router)\.(get|post|put|patch|delete|websocket)"
            r"\(\s*[\"\']([^\"\']+)[\"\']",
            body,
        ):
            holder, method, route = match.groups()
            if path == SERVER and holder == "app":
                prefix, reachable, how = "", True, "declared on the app itself"
            else:
                attribute = "router" if holder == "router" else holder
                prefix = by_module.get(module, {}).get(attribute)
                reachable = prefix is not None
                how = (
                    f"{module}.{attribute} mounted at {prefix or '/'}"
                    if reachable
                    else ""
                )
                prefix = prefix or ""
            entries.append(
                {
                    "kind": "api",
                    "name": f"{method.upper()} {prefix}{route}",
                    "declared_in": relative,
                    "reachable": reachable,
                    "reachable_by": how,
                }
            )
    return entries


def build() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for name, source in (("skill", skills), ("model", models), ("ui", ui), ("api", api)):
        try:
            entries.extend(source())
        except Exception as exc:  # noqa: BLE001 — an inventory that cannot count says so
            errors[name] = f"{type(exc).__name__}: {exc}"
    orphans = sorted(
        f"{entry['kind']}:{entry['name']}" for entry in entries if not entry["reachable"]
    )
    return {
        "schema_version": 1,
        "summary": {
            "total": len(entries),
            "by_kind": dict(sorted(Counter(e["kind"] for e in entries).items())),
            "reachable": sum(1 for e in entries if e["reachable"]),
            "orphans": orphans,
            "could_not_count": errors,
        },
        "entries": sorted(entries, key=lambda e: (e["kind"], e["name"])),
        "non_claims": [
            "Reachable means something outside the declaration can get to it.",
            "Nothing here says a capability works; that is what exercising it says.",
            "An empty orphan list is a fact about wiring, not about quality.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
