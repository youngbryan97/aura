#!/usr/bin/env python3
"""Generate Aura's true runtime architecture dependency map.

Outputs:
1. A Mermaid diagram of subsystem dependencies
2. A report of ServiceContainer.get() usage (cross-wiring audit)
3. An operational authority map for sensitive runtime surfaces
4. core/ directory structure analysis (consolidation candidates)
5. record_degradation() usage audit (log-and-limp vs fail-closed)
6. Non-runtime file identification (research/proof artifacts)
"""

import ast
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text

CORE = ROOT / "core"
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
ARCH_MAP_SCHEMA = "aura.architecture.dependency_map.v1"


@dataclass(frozen=True)
class OperationalSurface:
    name: str
    label: str
    description: str
    owners: tuple[str, ...]


@dataclass(frozen=True)
class OperationalCall:
    surface: str
    file: str
    line: int
    subsystem: str
    call: str
    source: str
    owner_path: bool


OPERATIONAL_SURFACES: tuple[OperationalSurface, ...] = (
    OperationalSurface(
        name="will_decision",
        label="UnifiedWill decisions",
        description="Calls that can ask the single will authority to approve action.",
        owners=(
            "core/will.py",
            "core/governance/will_client.py",
            "core/runtime/will_transaction.py",
            "core/executive/authority_gateway.py",
        ),
    ),
    OperationalSurface(
        name="memory_write",
        label="Memory writes",
        description="Calls that can create durable or semantically promoted memory.",
        owners=(
            "core/memory/memory_write_gateway.py",
            "core/runtime/gateways.py",
            "core/memory/memory_facade.py",
            "core/runtime/boot_probes.py",
        ),
    ),
    OperationalSurface(
        name="state_mutation",
        label="State mutation",
        description="Calls that can mutate runtime, identity, repository, or persistent state.",
        owners=(
            "core/state/state_repository.py",
            "core/runtime/state_gateway.py",
            "core/runtime/gateways.py",
            "core/orchestrator/state_repository.py",
        ),
    ),
    OperationalSurface(
        name="tool_execution",
        label="Tool execution",
        description="Calls that can execute tools, skills, shells, browsers, or external actions.",
        owners=(
            "core/capability_engine.py",
            "core/executive/authority_gateway.py",
            "core/coordinators/tool_executor.py",
            "core/agency/tool_orchestrator.py",
        ),
    ),
    OperationalSurface(
        name="patching",
        label="Self-modification and patching",
        description="Calls that can generate, validate, apply, or promote code changes.",
        owners=(
            "core/self_modification/",
            "core/self_improvement/",
            "core/learning/live_learner.py",
        ),
    ),
    OperationalSurface(
        name="llm_call",
        label="LLM inference",
        description="Calls that can spend model context or produce model-authored text/code.",
        owners=(
            "core/brain/",
            "core/phases/response_generation",
            "core/cognitive/",
            "core/capability_engine.py",
        ),
    ),
    OperationalSurface(
        name="external_io",
        label="External I/O",
        description="Calls that can touch network, subprocesses, sockets, browsers, or APIs.",
        owners=(
            "core/capability_engine.py",
            "core/executive/",
            "core/security/",
            "skills/",
        ),
    ),
)


SURFACE_BY_NAME = {surface.name: surface for surface in OPERATIONAL_SURFACES}


NETWORK_IMPORT_ROOTS = {
    "aiohttp",
    "httpx",
    "requests",
    "socket",
    "urllib",
    "websocket",
    "websockets",
}

SOCIAL_API_ROOTS = {"praw", "tweepy"}
SUBPROCESS_CALLS = {"subprocess.run", "subprocess.Popen", "subprocess.call", "subprocess.check_output"}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def get_subsystem(filepath: Path) -> str:
    """Map a file to its top-level subsystem under core/."""
    try:
        rel = filepath.relative_to(CORE)
        parts = rel.parts
        if len(parts) <= 1:
            return "core_root"
        return parts[0]
    except ValueError:
        return "other"


def analyze_imports(filepath: Path) -> list[str]:
    """Extract all core.X imports from a file."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("core."):
                parts = node.module.split(".")
                if len(parts) >= 2:
                    imports.append(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("core."):
                    parts = alias.name.split(".")
                    if len(parts) >= 2:
                        imports.append(parts[1])
    return imports


def _relative_path(filepath: Path) -> str:
    return str(filepath.relative_to(ROOT))


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    if isinstance(func, ast.Call):
        return _call_name(func.func)
    if isinstance(func, ast.Subscript):
        return _call_name(func.value)
    return ""


def _line_at(lines: list[str], lineno: int) -> str:
    if lineno <= 0 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()


def _imports_for_tree(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
                imports.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
            for alias in node.names:
                imports.add(alias.asname or alias.name)
    return imports


def _path_is_owner(rel: str, surface: OperationalSurface) -> bool:
    return any(rel == owner.rstrip("/") or rel.startswith(owner) for owner in surface.owners)


def _add_operational_call(
    calls: list[OperationalCall],
    seen: set[tuple[str, str, int, str]],
    *,
    surface_name: str,
    filepath: Path,
    line: int,
    call: str,
    source: str,
) -> None:
    rel = _relative_path(filepath)
    key = (surface_name, rel, line, call)
    if key in seen:
        return
    seen.add(key)
    surface = SURFACE_BY_NAME[surface_name]
    calls.append(
        OperationalCall(
            surface=surface_name,
            file=rel,
            line=line,
            subsystem=get_subsystem(filepath),
            call=call,
            source=source[:180],
            owner_path=_path_is_owner(rel, surface),
        )
    )


def _classify_operational_call(
    *,
    call_name: str,
    source_line: str,
    imports: set[str],
) -> set[str]:
    lowered_call = call_name.lower()
    lowered_line = source_line.lower()
    surfaces: set[str] = set()

    if call_name.endswith(".decide") and "will" in lowered_line:
        surfaces.add("will_decision")
    if call_name in {"get_will", "UnifiedWill"} and ".decide" in lowered_line:
        surfaces.add("will_decision")

    memory_terms = ("memory", "episodic_memory", "semantic_memory", "remember", "knowledge_graph")
    write_terms = ("store", "write", "append", "add", "commit", "persist", "save")
    if any(term in lowered_line for term in memory_terms) and (
        any(term in lowered_call for term in write_terms)
        or any(term in lowered_line for term in ("memory write", "add_memory"))
    ):
        surfaces.add("memory_write")

    state_terms = ("state", "identity", "repository", "registry", "snapshot")
    mutate_terms = ("set", "update", "mutate", "write", "save", "commit", "replace", "promote")
    if any(term in lowered_line for term in state_terms) and any(
        term in lowered_call for term in mutate_terms
    ):
        surfaces.add("state_mutation")

    tool_terms = ("tool", "skill", "shell", "browser", "computer", "capability")
    if (
        call_name in SUBPROCESS_CALLS
        or call_name.endswith(".safe_execute")
        or call_name.endswith(".execute_skill")
        or call_name.endswith(".execute_tool")
        or ("execute" in lowered_call and any(term in lowered_line for term in tool_terms))
    ):
        surfaces.add("tool_execution")

    patch_terms = ("patch", "self_modification", "self_modify", "apply_patch", "promotion")
    if any(term in lowered_line for term in patch_terms) and any(
        term in lowered_call for term in ("apply", "write", "promote", "validate", "commit", "modify")
    ):
        surfaces.add("patching")

    llm_terms = ("llm", "model", "brain", "inference", "router", "cognitive", "completion")
    if any(term in lowered_line for term in llm_terms) and any(
        lowered_call.endswith(suffix)
        for suffix in (".generate", ".chat", ".complete", ".call", ".think", ".route")
    ):
        surfaces.add("llm_call")

    import_roots = imports & (NETWORK_IMPORT_ROOTS | SOCIAL_API_ROOTS)
    if (
        call_name in SUBPROCESS_CALLS
        or any(call_name.startswith(f"{root}.") for root in NETWORK_IMPORT_ROOTS | SOCIAL_API_ROOTS)
        or any(root in lowered_line for root in import_roots)
    ):
        surfaces.add("external_io")

    return surfaces


def analyze_operational_surfaces(filepath: Path) -> list[OperationalCall]:
    """Map sensitive operational authority calls to concrete file/line owners."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []

    lines = content.splitlines()
    imports = _imports_for_tree(tree)
    calls: list[OperationalCall] = []
    seen: set[tuple[str, str, int, str]] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if not call_name:
            continue
        source_line = _line_at(lines, getattr(node, "lineno", 0))
        for surface_name in _classify_operational_call(
            call_name=call_name,
            source_line=source_line,
            imports=imports,
        ):
            _add_operational_call(
                calls,
                seen,
                surface_name=surface_name,
                filepath=filepath,
                line=getattr(node, "lineno", 0),
                call=call_name,
                source=source_line,
            )
    return sorted(calls, key=lambda call: (call.surface, call.file, call.line, call.call))


def analyze_service_container_usage(filepath: Path) -> list[dict]:
    """Find ServiceContainer.get() calls with their service names."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    results = []
    for i, line in enumerate(content.split("\n"), 1):
        match = re.search(r'ServiceContainer\.get\(["\'](\w+)["\']', line)
        if match:
            results.append({
                "service": match.group(1),
                "file": str(filepath.relative_to(ROOT)),
                "line": i,
            })
        match2 = re.search(r'ServiceContainer\.register[_instance]*\(["\'](\w+)["\']', line)
        if match2:
            results.append({
                "service": match2.group(1),
                "file": str(filepath.relative_to(ROOT)),
                "line": i,
                "type": "register",
            })
    return results


def analyze_degradation_usage(filepath: Path) -> list[dict]:
    """Find record_degradation() calls and check if they're log-and-limp."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    results = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if "record_degradation(" in line:
            # Check surrounding context for re-raise or return
            context_after = "\n".join(lines[i:i+5])
            is_limp = not ("raise" in context_after or "return" in context_after.split("\n")[1] if len(context_after.split("\n")) > 1 else True)
            results.append({
                "file": str(filepath.relative_to(ROOT)),
                "line": i + 1,
                "limp_on": is_limp,
                "context": line.strip(),
            })
    return results


def identify_non_runtime(filepath: Path) -> bool:
    """Check if a file is research/proof/narrative rather than runtime."""
    try:
        filepath.relative_to(ROOT)
    except ValueError:
        return False
    
    name = filepath.name.lower()
    content = filepath.read_text(encoding="utf-8", errors="ignore")[:500].lower()
    
    indicators = 0
    if "proof" in name or "narrative" in name or "scoping" in name:
        indicators += 2
    if "research" in name or "experiment" in name:
        indicators += 2
    if "validation" in content and "production" not in content:
        indicators += 1
    if "proof of concept" in content or "poc" in content:
        indicators += 2
    if filepath.stat().st_size < 500 and "class " not in content and "def " not in content:
        indicators += 1
    
    return indicators >= 2


def print_operational_authority_map(operational_calls: list[OperationalCall]) -> None:
    """Print sensitive authority surfaces with concrete caller locations."""
    by_surface: dict[str, list[OperationalCall]] = defaultdict(list)
    for call in operational_calls:
        by_surface[call.surface].append(call)

    print("\n## Operational Authority Map\n")
    print(
        f"{'Surface':<34} {'Calls':>6} {'Files':>6} {'Owner':>6} "
        f"{'Review':>7} {'Top Subsystems'}"
    )
    print("-" * 110)
    for surface in OPERATIONAL_SURFACES:
        calls = by_surface.get(surface.name, [])
        files = {call.file for call in calls}
        owner_calls = sum(1 for call in calls if call.owner_path)
        review_calls = len(calls) - owner_calls
        subsystem_counts = Counter(call.subsystem for call in calls)
        top_subsystems = ", ".join(
            f"{name}:{count}" for name, count in subsystem_counts.most_common(4)
        )
        print(
            f"{surface.label:<34} {len(calls):>6} {len(files):>6} "
            f"{owner_calls:>6} {review_calls:>7} {top_subsystems}"
        )

    print("\n### Surface Details\n")
    for surface in OPERATIONAL_SURFACES:
        calls = by_surface.get(surface.name, [])
        print(f"#### {surface.label}")
        print(f"{surface.description}")
        print(f"Owners: {', '.join(surface.owners)}")
        if not calls:
            print("No calls detected.")
            continue

        review = [call for call in calls if not call.owner_path]
        if review:
            print(f"Direct-call review candidates: {len(review)}")
            for call in review[:25]:
                print(
                    f"  - {call.file}:{call.line} [{call.subsystem}] "
                    f"{call.call} :: {call.source}"
                )
            if len(review) > 25:
                print(f"  ... {len(review) - 25} more")
        else:
            print("All detected calls are on declared owner paths.")

        owner_hits = [call for call in calls if call.owner_path]
        if owner_hits:
            print(f"Owner-path call sample: {len(owner_hits)}")
            for call in owner_hits[:8]:
                print(
                    f"  - {call.file}:{call.line} [{call.subsystem}] "
                    f"{call.call} :: {call.source}"
                )
        print()


def _mermaid_graph(
    *,
    dir_stats: dict[str, dict],
    subsystem_deps: dict[str, set[str]],
) -> str:
    lines = ["```mermaid", "graph TD"]
    sorted_subs = sorted(
        dir_stats.keys(),
        key=lambda s: dir_stats[s]["deps_in"],
        reverse=True,
    )

    for sub in sorted_subs:
        stats = dir_stats[sub]
        lines.append(f'    {sub}["{sub}<br/>{stats["files"]} files, {stats["lines"]} lines"]')

    for sub in sorted_subs:
        for dep in sorted(subsystem_deps[sub]):
            if dep in dir_stats:
                lines.append(f"    {sub} --> {dep}")

    lines.append("```")
    return "\n".join(lines)


def _counter_payload(counter: Counter, *, limit: int | None = None) -> list[dict[str, int | str]]:
    items = counter.most_common(limit)
    return [{"name": str(name), "count": int(count)} for name, count in items]


def _surface_report(operational_calls: list[OperationalCall]) -> dict[str, dict]:
    by_surface: dict[str, list[OperationalCall]] = defaultdict(list)
    for call in operational_calls:
        by_surface[call.surface].append(call)

    report: dict[str, dict] = {}
    for surface in OPERATIONAL_SURFACES:
        calls = by_surface.get(surface.name, [])
        owner_calls = [call for call in calls if call.owner_path]
        review_calls = [call for call in calls if not call.owner_path]
        subsystem_counts = Counter(call.subsystem for call in calls)
        report[surface.name] = {
            "label": surface.label,
            "description": surface.description,
            "owners": list(surface.owners),
            "call_count": len(calls),
            "file_count": len({call.file for call in calls}),
            "owner_path_call_count": len(owner_calls),
            "review_candidate_count": len(review_calls),
            "top_subsystems": _counter_payload(subsystem_counts, limit=8),
            "review_candidates": [asdict(call) for call in review_calls[:100]],
            "owner_path_sample": [asdict(call) for call in owner_calls[:25]],
        }
    return report


def build_architecture_report() -> dict:
    """Build the canonical machine-readable architecture dependency report."""
    all_files = find_python_files(CORE)
    skills_root = ROOT / "skills"
    skills_files = find_python_files(skills_root) if skills_root.exists() else []
    
    # ─── 1. SUBSYSTEM DEPENDENCY MAP ───
    subsystem_deps = defaultdict(set)
    subsystem_files = defaultdict(list)
    
    for f in all_files:
        sub = get_subsystem(f)
        subsystem_files[sub].append(f)
        for imp in analyze_imports(f):
            if imp != sub:  # Skip self-imports
                subsystem_deps[sub].add(imp)
    
    # ─── 2. SERVICE CONTAINER AUDIT ───
    sc_gets = []
    sc_registers = []
    for f in all_files + skills_files:
        for usage in analyze_service_container_usage(f):
            if usage.get("type") == "register":
                sc_registers.append(usage)
            else:
                sc_gets.append(usage)
    
    # ─── 3. DEGRADATION AUDIT ───
    degradation_calls = []
    for f in all_files + skills_files:
        degradation_calls.extend(analyze_degradation_usage(f))

    # ─── 3b. OPERATIONAL AUTHORITY SURFACES ───
    operational_calls = []
    for f in all_files + skills_files:
        operational_calls.extend(analyze_operational_surfaces(f))
    
    # ─── 4. NON-RUNTIME FILES ───
    non_runtime = []
    for f in all_files:
        if identify_non_runtime(f):
            non_runtime.append(str(f.relative_to(ROOT)))
    
    # ─── 5. CORE DIRECTORY STRUCTURE ───
    dir_stats = {}
    for sub in sorted(subsystem_files.keys()):
        files = subsystem_files[sub]
        total_lines = 0
        total_bytes = 0
        for f in files:
            total_bytes += f.stat().st_size
            try:
                total_lines += len(f.read_text(encoding="utf-8", errors="ignore").split("\n"))
            except OSError:
                pass
        dir_stats[sub] = {
            "files": len(files),
            "lines": total_lines,
            "bytes": total_bytes,
            "deps_out": len(subsystem_deps[sub]),
            "deps_in": sum(1 for other in subsystem_deps if sub in subsystem_deps[other]),
        }

    service_get_counts = Counter(u["service"] for u in sc_gets)
    service_reg_counts = Counter(u["service"] for u in sc_registers)
    missing = set(service_get_counts.keys()) - set(service_reg_counts.keys())

    limp_count = sum(1 for d in degradation_calls if d["limp_on"])
    fail_count = sum(1 for d in degradation_calls if not d["limp_on"])
    limp_by_file = Counter(d["file"] for d in degradation_calls if d["limp_on"])

    small_subs = {
        sub: stats
        for sub, stats in dir_stats.items()
        if stats["files"] <= 2 and sub != "core_root"
    }
    sorted_subs_by_lines = sorted(
        dir_stats.keys(),
        key=lambda s: dir_stats[s]["lines"],
        reverse=True,
    )

    return {
        "schema": ARCH_MAP_SCHEMA,
        "generated_at_unix": time.time(),
        "root": str(ROOT),
        "inputs": {
            "core_python_files": len(all_files),
            "skills_python_files": len(skills_files),
            "skip_dirs": sorted(SKIP_DIRS),
        },
        "totals": {
            "subsystems": len(dir_stats),
            "python_files": sum(s["files"] for s in dir_stats.values()),
            "python_lines": sum(s["lines"] for s in dir_stats.values()),
            "python_bytes": sum(s["bytes"] for s in dir_stats.values()),
        },
        "subsystems": {
            sub: {
                **dir_stats[sub],
                "dependencies": sorted(subsystem_deps[sub]),
                "source_files": [str(path.relative_to(ROOT)) for path in subsystem_files[sub]],
            }
            for sub in sorted(dir_stats)
        },
        "subsystem_order_by_lines": sorted_subs_by_lines,
        "dependency_edges": [
            {"source": sub, "target": dep}
            for sub in sorted(subsystem_deps)
            for dep in sorted(subsystem_deps[sub])
            if dep in dir_stats
        ],
        "mermaid": _mermaid_graph(dir_stats=dir_stats, subsystem_deps=subsystem_deps),
        "service_container": {
            "get_call_count": len(sc_gets),
            "register_call_count": len(sc_registers),
            "unique_services_retrieved": len(service_get_counts),
            "unique_services_registered": len(service_reg_counts),
            "missing_registrations": [
                {
                    "service": service,
                    "get_count": service_get_counts[service],
                }
                for service in sorted(missing)
            ],
            "top_fetched_services": [
                {
                    "service": svc,
                    "get_count": count,
                    "register_count": service_reg_counts.get(svc, 0),
                }
                for svc, count in service_get_counts.most_common(50)
            ],
            "gets": sorted(sc_gets, key=lambda item: (item["service"], item["file"], item["line"])),
            "registers": sorted(sc_registers, key=lambda item: (item["service"], item["file"], item["line"])),
        },
        "operational_surfaces": _surface_report(operational_calls),
        "degradation": {
            "total_calls": len(degradation_calls),
            "log_and_limp_count": limp_count,
            "fail_closed_count": fail_count,
            "top_limp_files": [
                {"file": file, "count": count}
                for file, count in limp_by_file.most_common(25)
            ],
            "calls": degradation_calls,
        },
        "non_runtime_candidates": sorted(non_runtime),
        "consolidation_candidates": {
            sub: small_subs[sub]
            for sub in sorted(small_subs)
        },
    }


def render_markdown_report(report: dict) -> str:
    """Render a reviewer-readable architecture report from the JSON contract."""
    lines: list[str] = []
    totals = report["totals"]
    lines.extend(
        [
            "# Aura Architecture Dependency Map",
            "",
            f"Schema: `{report['schema']}`",
            f"Root: `{report['root']}`",
            f"Generated: `{report['generated_at_unix']}`",
            "",
            "## Summary",
            "",
            f"- Subsystems: {totals['subsystems']}",
            f"- Python files: {totals['python_files']}",
            f"- Python lines: {totals['python_lines']}",
            f"- Dependency edges: {len(report['dependency_edges'])}",
            f"- ServiceContainer `.get()` calls: {report['service_container']['get_call_count']}",
            f"- ServiceContainer registrations: {report['service_container']['register_call_count']}",
            "",
            "## Subsystem Dependency Graph",
            "",
            report["mermaid"],
            "",
            "## Core Subsystem Stats",
            "",
            "| Subsystem | Files | Lines | Bytes | Deps Out | Deps In |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    subsystems = report["subsystems"]
    for sub in report["subsystem_order_by_lines"]:
        stats = subsystems[sub]
        lines.append(
            f"| {sub} | {stats['files']} | {stats['lines']} | {stats['bytes']} | "
            f"{stats['deps_out']} | {stats['deps_in']} |"
        )

    service = report["service_container"]
    lines.extend(
        [
            "",
            "## ServiceContainer Cross-Wiring",
            "",
            f"- Unique services retrieved: {service['unique_services_retrieved']}",
            f"- Unique services registered: {service['unique_services_registered']}",
            f"- Services retrieved without detected registration: {len(service['missing_registrations'])}",
            "",
            "### Top Fetched Services",
            "",
            "| Service | Gets | Registrations |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in service["top_fetched_services"][:20]:
        lines.append(
            f"| {item['service']} | {item['get_count']} | {item['register_count']} |"
        )

    if service["missing_registrations"]:
        lines.extend(["", "### Missing Registration Candidates", ""])
        for item in service["missing_registrations"][:50]:
            lines.append(f"- `{item['service']}` fetched {item['get_count']} time(s)")

    lines.extend(
        [
            "",
            "## Operational Authority Map",
            "",
            "| Surface | Calls | Files | Owner Calls | Review Candidates |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, surface in report["operational_surfaces"].items():
        lines.append(
            f"| {surface['label']} | {surface['call_count']} | {surface['file_count']} | "
            f"{surface['owner_path_call_count']} | {surface['review_candidate_count']} |"
        )

    for name, surface in report["operational_surfaces"].items():
        lines.extend(["", f"### {surface['label']}", "", surface["description"], ""])
        if surface["review_candidates"]:
            lines.append("Review candidates:")
            for call in surface["review_candidates"][:25]:
                lines.append(
                    f"- `{call['file']}:{call['line']}` [{call['subsystem']}] "
                    f"`{call['call']}` - {call['source']}"
                )
        else:
            lines.append("All detected calls are on declared owner paths.")

    degradation = report["degradation"]
    lines.extend(
        [
            "",
            "## Degradation Handling",
            "",
            f"- Total `record_degradation()` calls: {degradation['total_calls']}",
            f"- Log-and-limp candidates: {degradation['log_and_limp_count']}",
            f"- Nearby fail-closed candidates: {degradation['fail_closed_count']}",
            "",
        ]
    )
    if degradation["top_limp_files"]:
        lines.extend(["Top limp-on files:", ""])
        for item in degradation["top_limp_files"][:10]:
            lines.append(f"- `{item['file']}`: {item['count']}")

    if report["non_runtime_candidates"]:
        lines.extend(["", "## Non-Runtime Candidates", ""])
        for path in report["non_runtime_candidates"][:100]:
            lines.append(f"- `{path}`")

    if report["consolidation_candidates"]:
        lines.extend(["", "## Consolidation Candidates", ""])
        for sub, stats in report["consolidation_candidates"].items():
            lines.append(f"- `core/{sub}/`: {stats['files']} file(s), {stats['lines']} line(s)")

    lines.append("")
    return "\n".join(lines)


def print_report(report: dict) -> None:
    print("=" * 80)
    print("AURA ARCHITECTURE DEPENDENCY MAP")
    print("=" * 80)
    print(render_markdown_report(report))


def write_report_artifacts(report: dict, output_dir: Path) -> dict[str, str]:
    """Write canonical JSON and Markdown artifacts atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest.json"
    md_path = output_dir / "latest.md"
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    atomic_write_text(md_path, render_markdown_report(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--out-json", type=Path, default=None, help="Write JSON report to this path.")
    parser.add_argument("--out-md", type=Path, default=None, help="Write Markdown report to this path.")
    parser.add_argument(
        "--write-latest",
        action="store_true",
        help="Write artifacts/architecture/latest.json and latest.md.",
    )
    args = parser.parse_args(argv)

    report = build_architecture_report()

    if args.out_json:
        atomic_write_text(
            args.out_json,
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    if args.out_md:
        atomic_write_text(args.out_md, render_markdown_report(report), encoding="utf-8")
    if args.write_latest:
        write_report_artifacts(report, ROOT / "artifacts" / "architecture")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
