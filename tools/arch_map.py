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
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


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


def main():
    all_files = find_python_files(CORE)
    skills_files = find_python_files(ROOT / "skills")
    
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
    
    # ─── OUTPUT ───
    print("=" * 80)
    print("AURA ARCHITECTURE DEPENDENCY MAP")
    print("=" * 80)
    
    # Mermaid diagram
    print("\n## Subsystem Dependency Graph (Mermaid)\n")
    print("```mermaid")
    print("graph TD")
    
    # Sort by importance (deps_in)
    sorted_subs = sorted(dir_stats.keys(), key=lambda s: dir_stats[s]["deps_in"], reverse=True)
    
    for sub in sorted_subs:
        stats = dir_stats[sub]
        print(f'    {sub}["{sub}<br/>{stats["files"]} files, {stats["lines"]} lines"]')
    
    for sub in sorted_subs:
        for dep in sorted(subsystem_deps[sub]):
            if dep in dir_stats:
                print(f"    {sub} --> {dep}")
    
    print("```")
    
    # Directory stats
    print("\n## Core Subsystem Stats\n")
    print(f"{'Subsystem':<30} {'Files':>6} {'Lines':>8} {'Bytes':>10} {'Deps Out':>9} {'Deps In':>8}")
    print("-" * 80)
    for sub in sorted(dir_stats.keys(), key=lambda s: dir_stats[s]["lines"], reverse=True):
        s = dir_stats[sub]
        print(f"{sub:<30} {s['files']:>6} {s['lines']:>8} {s['bytes']:>10} {s['deps_out']:>9} {s['deps_in']:>8}")
    
    print(f"\nTotal subsystems: {len(dir_stats)}")
    print(f"Total files: {sum(s['files'] for s in dir_stats.values())}")
    print(f"Total lines: {sum(s['lines'] for s in dir_stats.values())}")
    
    # ServiceContainer audit
    service_get_counts = Counter(u["service"] for u in sc_gets)
    service_reg_counts = Counter(u["service"] for u in sc_registers)
    
    print("\n## ServiceContainer Cross-Wiring Audit\n")
    print(f"Total .get() calls: {len(sc_gets)}")
    print(f"Total .register() calls: {len(sc_registers)}")
    print(f"Unique services retrieved: {len(service_get_counts)}")
    print(f"Unique services registered: {len(service_reg_counts)}")
    
    # Services retrieved but never registered (potential missing registrations)
    missing = set(service_get_counts.keys()) - set(service_reg_counts.keys())
    if missing:
        print(f"\n⚠️  Services GET'd but never REGISTER'd ({len(missing)}):")
        for s in sorted(missing):
            print(f"    - {s} (get'd {service_get_counts[s]}x)")
    
    # Top cross-wired services
    print("\nTop 20 most-fetched services:")
    for svc, count in service_get_counts.most_common(20):
        reg_count = service_reg_counts.get(svc, 0)
        print(f"    {svc:<40} get={count:>3}  register={reg_count}")

    # Operational authority map
    print_operational_authority_map(operational_calls)
    
    # Degradation audit
    limp_count = sum(1 for d in degradation_calls if d["limp_on"])
    fail_count = sum(1 for d in degradation_calls if not d["limp_on"])
    print("\n## record_degradation() Audit\n")
    print(f"Total calls: {len(degradation_calls)}")
    print(f"  Log-and-limp (no raise/return after): {limp_count}")
    print(f"  Fail-closed (raise/return follows): {fail_count}")
    
    if limp_count > 0:
        print("\n  Top 10 limp-on files:")
        limp_by_file = Counter(d["file"] for d in degradation_calls if d["limp_on"])
        for fname, count in limp_by_file.most_common(10):
            print(f"    {fname}: {count}")
    
    # Non-runtime
    if non_runtime:
        print(f"\n## Non-Runtime Files ({len(non_runtime)})\n")
        for f in non_runtime:
            print(f"    {f}")
    
    # Consolidation candidates (subsystems with < 3 files)
    small_subs = {sub: stats for sub, stats in dir_stats.items() if stats["files"] <= 2 and sub != "core_root"}
    if small_subs:
        print(f"\n## Consolidation Candidates ({len(small_subs)} small subsystems with ≤2 files)\n")
        for sub in sorted(small_subs.keys()):
            s = small_subs[sub]
            print(f"    {sub}/: {s['files']} files, {s['lines']} lines")


if __name__ == "__main__":
    main()
