#!/usr/bin/env python3
"""Reject unattributed host-resource observations in production Python."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".claude",
        ".aura",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "archive",
        "artifacts",
        "build",
        "dev_archive",
        "dist",
        "logs",
        "scratch",
        "site-packages",
        "tests",
    }
)
CANONICAL_ADAPTERS = frozenset(
    {
        "core/runtime/resource_observation.py",
        "core/runtime/process_footprint.py",
        "core/runtime/resource_psutil.py",
        "core/runtime/thermal.py",
        "core/utils/memory_monitor.py",
        # This adapter converts observed child identities into termination
        # handles only on the explicitly host-backed watchdog path.
        "core/resilience/memory_watchdog.py",
        # The MLX allocator adapter. It converts the accelerator's own
        # accounting into an AllocatorDump and registers it as a provider,
        # which is the adapter role this list exists for — asking it to read
        # the accelerator through the observer would be asking the observer
        # to supply what this file is the supplier of.
        "core/runtime/memory_providers.py",
        # The thread CPU-time adapter. It reads /proc for a thread's own
        # accumulated ticks and divides by SC_CLK_TCK, which is a unit
        # conversion rather than an observation of the host.
        "core/runtime/thread_cpu.py",
    }
)
REQUIRED_OBSERVATION_SYMBOLS = frozenset(
    {
        "HostResourceObserver",
        "ObservationSource",
        "ResourceObserver",
        "ObservationProvenance",
        "SimulatedResourceObserver",
        "assert_live_pressure_observer",
        "get_resource_observer",
        "resource_observer_scope",
    }
)
REQUIRED_HERMETIC_FIXTURES = frozenset(
    {
        "_global_state_contamination_guard",
        "hermetic_resource_sandbox",
        "resource_observer",
    }
)
REQUIRED_HERMETIC_FUNCTIONS = frozenset(
    {
        "_reset_test_scoped_runtime_services",
    }
)
REQUIRED_HERMETIC_RESET_IDENTIFIERS = frozenset(
    {
        "reset_lane_admission_controller_for_test",
        "reset_model_lane_controller_for_test",
        "reset_model_registry_caches_for_test",
        "reset_receipt_store",
    }
)
REQUIRED_HERMETIC_ENV_KEYS = frozenset(
    {
        "AURA_MODEL_LANE_STATE_PATH",
        "AURA_RECEIPT_ROOT",
        "AURA_TEST_RUNTIME_ROOT",
        "AURA_TEST_STATE_GUARD",
    }
)
REQUIRED_LEAK_SANDBOX_METHODS = frozenset(
    {
        "close_and_assert_clean",
        "leaks",
        "listening_socket",
        "snapshot",
    }
)
PSUTIL_RESOURCE_CALLS = frozenset(
    {
        "boot_time",
        "cpu_count",
        "cpu_freq",
        "cpu_percent",
        "cpu_stats",
        "cpu_times",
        "disk_usage",
        "disk_io_counters",
        "getloadavg",
        "net_connections",
        "net_if_addrs",
        "net_io_counters",
        "pid_exists",
        "pids",
        "process_iter",
        "sensors_battery",
        "sensors_temperatures",
        "swap_memory",
        "virtual_memory",
    }
)
PROCESS_OBSERVATION_CALLS = frozenset(
    {
        "children",
        "cmdline",
        "connections",
        "cpu_percent",
        "cpu_times",
        "create_time",
        "cwd",
        "environ",
        "exe",
        "is_running",
        "memory_full_info",
        "memory_info",
        "memory_maps",
        "memory_percent",
        "name",
        "net_connections",
        "num_fds",
        "num_handles",
        "num_threads",
        "open_files",
        "parent",
        "parents",
        "ppid",
        "status",
        "threads",
        "username",
    }
)
ACCELERATOR_OBSERVATION_CALLS = frozenset(
    {"get_active_memory", "get_cache_memory", "get_peak_memory"}
)
OS_RESOURCE_CALLS = frozenset({"cpu_count", "getloadavg", "sysconf"})
RESOURCE_MODULE_CALLS = frozenset({"getrusage"})
PLATFORM_RESOURCE_CALLS = frozenset({"proc_listchildpids", "proc_pid_rusage"})


@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    line: int
    detail: str


def _repository_source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, child_dirs, filenames in os.walk(root):
        child_dirs[:] = [name for name in child_dirs if name not in EXCLUDED_PARTS]
        base = Path(directory)
        paths.extend(
            base / name
            for name in filenames
            if name.endswith(".py") and (base / name).is_file()
        )
    return sorted(paths)


def _qualified_name(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _assignment_targets(node: ast.AST) -> set[tuple[str, ...]]:
    if isinstance(node, (ast.Name, ast.Attribute)):
        qualified = _qualified_name(node)
        return {qualified} if qualified else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        targets: set[tuple[str, ...]] = set()
        for item in node.elts:
            targets.update(_assignment_targets(item))
        return targets
    return set()


def _is_psutil_process_call(node: ast.AST | None, aliases: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    qualified = _qualified_name(node.func)
    return len(qualified) == 2 and qualified[0] in aliases and qualified[1] == "Process"


def _findings_for_tree(tree: ast.Module, relative_path: str) -> list[AuditFinding]:
    if relative_path in CANONICAL_ADAPTERS:
        return []

    psutil_aliases: set[str] = set()
    shutil_aliases: set[str] = set()
    os_aliases: set[str] = set()
    resource_aliases: set[str] = set()
    direct_psutil_functions: dict[str, str] = {}
    direct_shutil_functions: set[str] = set()
    direct_os_functions: dict[str, str] = {}
    direct_resource_functions: dict[str, str] = {}
    process_handle_targets: set[tuple[str, ...]] = set()
    process_iteration_targets: set[tuple[str, ...]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "psutil":
                    psutil_aliases.add(alias.asname or "psutil")
                elif alias.name == "shutil":
                    shutil_aliases.add(alias.asname or "shutil")
                elif alias.name == "os":
                    os_aliases.add(alias.asname or "os")
                elif alias.name == "resource":
                    resource_aliases.add(alias.asname or "resource")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "psutil":
                for alias in node.names:
                    if alias.name in PSUTIL_RESOURCE_CALLS:
                        direct_psutil_functions[alias.asname or alias.name] = alias.name
            elif node.module == "shutil":
                for alias in node.names:
                    if alias.name == "disk_usage":
                        direct_shutil_functions.add(alias.asname or alias.name)
            elif node.module == "os":
                for alias in node.names:
                    if alias.name in OS_RESOURCE_CALLS:
                        direct_os_functions[alias.asname or alias.name] = alias.name
            elif node.module == "resource":
                for alias in node.names:
                    if alias.name in RESOURCE_MODULE_CALLS:
                        direct_resource_functions[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_psutil_process_call(node.value, psutil_aliases):
            for target in node.targets:
                process_handle_targets.update(_assignment_targets(target))
        elif isinstance(node, ast.AnnAssign) and _is_psutil_process_call(
            node.value,
            psutil_aliases,
        ):
            process_handle_targets.update(_assignment_targets(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                annotation = _qualified_name(argument.annotation) if argument.annotation else ()
                if (
                    len(annotation) == 2
                    and annotation[0] in psutil_aliases
                    and annotation[1] == "Process"
                ):
                    process_handle_targets.add((argument.arg,))
        elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, ast.Call):
            iterator = _qualified_name(node.iter.func)
            if (
                len(iterator) == 2
                and iterator[0] in psutil_aliases
                and iterator[1] == "process_iter"
            ):
                process_iteration_targets.update(_assignment_targets(node.target))

    findings: list[AuditFinding] = []
    call_function_ids = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        qualified = _qualified_name(function)
        if isinstance(function, ast.Name):
            if function.id in direct_psutil_functions:
                api = direct_psutil_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_psutil_resource_observation",
                        relative_path,
                        node.lineno,
                        f"psutil.{api} must route through ResourceObserver",
                    )
                )
            elif function.id in direct_shutil_functions:
                findings.append(
                    AuditFinding(
                        "direct_disk_observation",
                        relative_path,
                        node.lineno,
                        "shutil.disk_usage must route through ResourceObserver",
                    )
                )
            elif function.id in direct_os_functions:
                api = direct_os_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_standard_resource_observation",
                        relative_path,
                        node.lineno,
                        f"os.{api} must route through ResourceObserver",
                    )
                )
            elif function.id in direct_resource_functions:
                api = direct_resource_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_standard_resource_observation",
                        relative_path,
                        node.lineno,
                        f"resource.{api} must route through ResourceObserver",
                    )
                )
        if len(qualified) >= 2 and qualified[0] in psutil_aliases:
            api = qualified[-1]
            if api in PSUTIL_RESOURCE_CALLS:
                findings.append(
                    AuditFinding(
                        "direct_psutil_resource_observation",
                        relative_path,
                        node.lineno,
                        f"psutil.{api} must route through ResourceObserver",
                    )
                )
            elif api in PROCESS_OBSERVATION_CALLS and "Process" in qualified:
                findings.append(
                    AuditFinding(
                        "direct_process_observation",
                        relative_path,
                        node.lineno,
                        f"Process.{api} must route through ResourceObserver",
                    )
                )
        if isinstance(function, ast.Attribute) and function.attr in PROCESS_OBSERVATION_CALLS:
            receiver = function.value
            receiver_name = _qualified_name(receiver)
            rooted_process_call = _is_psutil_process_call(receiver, psutil_aliases)
            tracked_process_handle = receiver_name in (
                process_handle_targets | process_iteration_targets
            )
            if rooted_process_call or tracked_process_handle:
                findings.append(
                    AuditFinding(
                        "direct_process_observation",
                        relative_path,
                        node.lineno,
                        f"Process.{function.attr} must route through ResourceObserver",
                    )
                )
        if (
            len(qualified) >= 2
            and qualified[0] in shutil_aliases
            and qualified[-1] == "disk_usage"
        ):
            findings.append(
                AuditFinding(
                    "direct_disk_observation",
                    relative_path,
                    node.lineno,
                    "shutil.disk_usage must route through ResourceObserver",
                )
            )
        if (
            len(qualified) >= 2
            and qualified[0] in os_aliases
            and qualified[-1] in OS_RESOURCE_CALLS
        ):
            findings.append(
                AuditFinding(
                    "direct_standard_resource_observation",
                    relative_path,
                    node.lineno,
                    f"os.{qualified[-1]} must route through ResourceObserver",
                )
            )
        if (
            len(qualified) >= 2
            and qualified[0] in resource_aliases
            and qualified[-1] in RESOURCE_MODULE_CALLS
        ):
            findings.append(
                AuditFinding(
                    "direct_standard_resource_observation",
                    relative_path,
                    node.lineno,
                    f"resource.{qualified[-1]} must route through ResourceObserver",
                )
            )
        if qualified and qualified[-1] in ACCELERATOR_OBSERVATION_CALLS:
            findings.append(
                AuditFinding(
                    "direct_accelerator_observation",
                    relative_path,
                    node.lineno,
                    f"accelerator {qualified[-1]} must route through ResourceObserver",
                )
            )
        if qualified and qualified[-1] in PLATFORM_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_platform_resource_observation",
                    relative_path,
                    node.lineno,
                    f"platform {qualified[-1]} must route through a canonical adapter",
                )
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in call_function_ids:
            continue
        qualified = _qualified_name(node)
        if len(qualified) < 2:
            continue
        api = qualified[-1]
        if qualified[0] in psutil_aliases and api in PSUTIL_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_psutil_resource_reference",
                    relative_path,
                    node.lineno,
                    f"psutil.{api} reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in shutil_aliases and api == "disk_usage":
            findings.append(
                AuditFinding(
                    "direct_disk_observation_reference",
                    relative_path,
                    node.lineno,
                    "shutil.disk_usage reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in os_aliases and api in OS_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_standard_resource_reference",
                    relative_path,
                    node.lineno,
                    f"os.{api} reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in resource_aliases and api in RESOURCE_MODULE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_standard_resource_reference",
                    relative_path,
                    node.lineno,
                    f"resource.{api} reference must route through ResourceObserver",
                )
            )
    return findings


def _canonical_contract_findings(root: Path) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for relative_path in sorted(CANONICAL_ADAPTERS):
        if not (root / relative_path).is_file():
            findings.append(
                AuditFinding(
                    code="missing_canonical_adapter",
                    path=relative_path,
                    line=1,
                    detail="required canonical resource adapter is absent",
                )
            )

    contract_path = root / "core/runtime/resource_observation.py"
    if not contract_path.is_file():
        return findings
    try:
        tree = ast.parse(contract_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return findings
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for symbol in sorted(REQUIRED_OBSERVATION_SYMBOLS - defined):
        findings.append(
            AuditFinding(
                code="missing_observation_contract_symbol",
                path="core/runtime/resource_observation.py",
                line=1,
                detail=f"required top-level observer symbol is absent: {symbol}",
            )
        )
    return findings


def _autouse_fixture_names(tree: ast.Module) -> set[str]:
    fixtures: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if _qualified_name(decorator.func)[-2:] != ("pytest", "fixture"):
                continue
            if any(
                keyword.arg == "autouse"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in decorator.keywords
            ):
                fixtures.add(node.name)
    return fixtures


def _hermetic_test_contract_findings(root: Path) -> list[AuditFinding]:
    relative_path = "tests/conftest.py"
    contract_path = root / relative_path
    if not contract_path.is_file():
        return [
            AuditFinding(
                code="missing_hermetic_test_contract",
                path=relative_path,
                line=1,
                detail="required hermetic pytest contract is absent",
            )
        ]
    try:
        tree = ast.parse(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return []

    findings: list[AuditFinding] = []
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    autouse = _autouse_fixture_names(tree)
    for fixture in sorted(REQUIRED_HERMETIC_FIXTURES - autouse):
        findings.append(
            AuditFinding(
                code="hermetic_fixture_not_autouse",
                path=relative_path,
                line=1,
                detail=f"required fixture is not autouse: {fixture}",
            )
        )
    for function in sorted(REQUIRED_HERMETIC_FUNCTIONS - functions):
        findings.append(
            AuditFinding(
                code="missing_hermetic_reset_contract",
                path=relative_path,
                line=1,
                detail=f"required reset coordinator is absent: {function}",
            )
        )

    identifiers = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    for identifier in sorted(REQUIRED_HERMETIC_RESET_IDENTIFIERS - identifiers):
        findings.append(
            AuditFinding(
                code="missing_hermetic_resource_reset",
                path=relative_path,
                line=1,
                detail=f"required per-test resource reset is absent: {identifier}",
            )
        )

    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for env_key in sorted(REQUIRED_HERMETIC_ENV_KEYS - strings):
        findings.append(
            AuditFinding(
                code="missing_hermetic_environment_scope",
                path=relative_path,
                line=1,
                detail=f"required per-test environment scope is absent: {env_key}",
            )
        )

    guard_enforcing = any(
        isinstance(node, ast.Call)
        and _qualified_name(node.func)[-3:] == ("os", "environ", "setdefault")
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "AURA_TEST_STATE_GUARD"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "fail"
        for node in ast.walk(tree)
    )
    if not guard_enforcing:
        findings.append(
            AuditFinding(
                code="state_guard_not_enforcing",
                path=relative_path,
                line=1,
                detail="AURA_TEST_STATE_GUARD must default to fail",
            )
        )

    sandbox = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "HermeticResourceSandbox"
        ),
        None,
    )
    methods = (
        {
            node.name
            for node in sandbox.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if sandbox is not None
        else set()
    )
    for method in sorted(REQUIRED_LEAK_SANDBOX_METHODS - methods):
        findings.append(
            AuditFinding(
                code="missing_host_leak_assertion",
                path=relative_path,
                line=1,
                detail=f"required host leak sandbox method is absent: {method}",
            )
        )
    return findings


def run_audit(
    *, root: Path = ROOT, require_canonical_contract: bool = False
) -> dict[str, Any]:
    findings: list[AuditFinding] = []
    scanned = 0
    parse_errors: list[dict[str, Any]] = []
    for path in _repository_source_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            parse_errors.append(
                {
                    "path": relative,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        scanned += 1
        findings.extend(_findings_for_tree(tree, relative))

    if require_canonical_contract:
        findings.extend(_canonical_contract_findings(root))
        findings.extend(_hermetic_test_contract_findings(root))
    findings.sort(key=lambda item: (item.path, item.line, item.code))
    return {
        "schema": "aura.resource_observation_ownership.v1",
        "passed": not findings and not parse_errors,
        "scanned_python_files": scanned,
        "canonical_adapters": sorted(CANONICAL_ADAPTERS),
        "canonical_contract_checked": require_canonical_contract,
        "hermetic_test_contract_checked": require_canonical_contract,
        "required_observation_symbols": sorted(REQUIRED_OBSERVATION_SYMBOLS),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "parse_errors": parse_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = run_audit(
        root=root,
        require_canonical_contract=root == ROOT.resolve(),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"RESOURCE_OBSERVATION_OWNERSHIP={status} "
            f"files={report['scanned_python_files']} findings={report['finding_count']}"
        )
        for finding in report["findings"]:
            print(
                f"  {finding['path']}:{finding['line']} "
                f"{finding['code']} {finding['detail']}"
            )
        for error in report["parse_errors"]:
            print(f"  {error['path']} parse_error {error['error']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
