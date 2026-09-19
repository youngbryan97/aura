#!/usr/bin/env python3
"""Fail when Aura's process and resource lifecycle ownership contract drifts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_CALLS: dict[str, dict[str, frozenset[str]]] = {
    "core/ops/graceful_shutdown.py": {
        "GracefulShutdown.trigger_shutdown": frozenset(
            {
                "request_shutdown",
                "shutdown",
                "get_container",
                "stop",
                "publish_shutdown_verdict",
            }
        ),
    },
    # The three interpreter patches moved out of RuntimeHygieneManager into
    # the mixin it inherits from, so their contract follows them. The audit
    # reads the file that DEFINES a function; a mixin is a different file
    # with the same guarantee.
    "core/runtime/runtime_hygiene_patches.py": {
        "_WatchesWhatTheRuntimeCreates._patch_threading": frozenset(
            {"_shutdown_blocks_resource_start", "_register_thread"}
        ),
        "_WatchesWhatTheRuntimeCreates._patch_subprocess": frozenset(
            {
                "_shutdown_blocks_resource_start",
                "_register_subprocess",
                "_reap_crossed_subprocess",
            }
        ),
        "_WatchesWhatTheRuntimeCreates._patch_multiprocessing": frozenset(
            {
                "_shutdown_blocks_resource_start",
                "_register_multiprocessing_process",
                "_reap_crossed_multiprocessing",
            }
        ),
    },
    "core/runtime/runtime_hygiene.py": {
        "RuntimeHygieneManager.start": frozenset(
            {
                "install_loop_hygiene",
                "_patch_asyncio_new_event_loop",
                "_patch_threading",
                "_patch_subprocess",
                "_patch_multiprocessing",
                "_adopt_active_child_processes",
            }
        ),
        "RuntimeHygieneManager._execute_stop_sweep": frozenset(
            {
                "_cleanup_shutdown_resources",
                "_cleanup_child_processes",
                "_join_non_daemon_threads",
                "shutdown",
                "restore_loop_hygiene",
                "_restore_patches",
                "_shutdown_default_executor",
                "_native_resource_summary",
            }
        ),
        "RuntimeHygieneManager.register_shutdown_resource": frozenset(
            {"is_shutdown_requested", "_record_resource_boundary"}
        ),
        "RuntimeHygieneManager.get_root_exit_resource_report": frozenset(
            {"_native_resource_summary"}
        ),
        "RuntimeHygieneManager.close_root_exit_sockets": frozenset(
            {"_native_resource_summary"}
        ),
    },
    "core/tasks/managed_command.py": {
        "_run_async_blocking": frozenset({"Thread", "start", "join", "run"}),
        "_run_project_command_async": frozenset(
            {
                "local_internal_governed_scope",
                "spawn_async",
                "wait_for",
                "communicate",
                "kill",
            }
        ),
    },
    "core/container.py": {
        "_install_runtime_service_registry_bridge": frozenset(
            {"peek", "install_service_resolver"}
        ),
        "ServiceContainer.peek": frozenset({"_resolve_name"}),
    },
    "core/runtime/service_registry.py": {
        "get_runtime_service": frozenset({"resolver"}),
    },
    "core/state/state_repository.py": {
        "StateRepository._ensure_db": frozenset({"get_running_loop", "connect"}),
        "StateRepository.close": frozenset({"cancel", "close"}),
    },
    "core/runtime/process_identity.py": {
        "command_invokes_python_script": frozenset({"python_script_argument"}),
        "process_invokes_python_script": frozenset({"command_invokes_python_script"}),
        "select_script_process_tree": frozenset({"process_invokes_python_script"}),
    },
    "scripts/one_off/aura_cleanup.py": {
        "_verified_live_runtime_pid": frozenset(
            {"read_instance_lock_pid", "process", "process_invokes_python_script"}
        ),
        "_kill_stale_processes": frozenset(
            {
                "_verified_live_runtime_pid",
                "process_table",
                "select_script_process_tree",
                "process",
                "terminate",
                "wait_procs",
                "kill",
            }
        ),
        "_kill_stale_native_launchers": frozenset(
            {"_verified_live_runtime_pid", "process_table", "_is_native_launcher_process"}
        ),
    },
}

_REQUIRED_NONE_ASSIGNMENTS: dict[str, frozenset[str]] = {
    "core/state/state_repository.py:StateRepository.close": frozenset(
        {"_consumer_task", "_shm", "_db"}
    ),
}

_REQUIRED_QUALIFIED_CALLS: dict[str, frozenset[str]] = {
    "core/state/state_repository.py:StateRepository.close": frozenset(
        {"self._consumer_task.cancel", "self._shm.close", "self._db.close"}
    ),
}

_FORBIDDEN_CALLS: dict[str, dict[str, frozenset[str]]] = {
    "core/container.py": {
        "_install_runtime_service_registry_bridge": frozenset(
            {"get_runtime_service", "ServiceContainer.get"}
        ),
    },
    "core/runtime/service_registry.py": {
        "get_runtime_service": frozenset({"ServiceContainer.get"}),
    },
}


def _call_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Call):
            continue
        if isinstance(item.func, ast.Name):
            names.add(item.func.id)
        elif isinstance(item.func, ast.Attribute):
            names.add(item.func.attr)
            if isinstance(item.func.value, ast.Name):
                names.add(f"{item.func.value.id}.{item.func.attr}")
    return names


def _attribute_path(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_path(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _qualified_call_names(node: ast.AST) -> set[str]:
    return {
        path
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        for path in (_attribute_path(item.func),)
        if path
    }


def _qualified_functions(tree: ast.Module) -> dict[str, ast.AST]:
    functions: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                functions[qualified] = node
            elif isinstance(node, ast.ClassDef):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                visit(node.body, qualified)

    visit(tree.body)
    return functions


def _none_assigned_self_attributes(node: ast.AST) -> set[str]:
    assigned: set[str] = set()
    for item in ast.walk(node):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
            value = item.value
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
            value = item.value
        if not isinstance(value, ast.Constant) or value.value is not None:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                assigned.add(target.attr)
    return assigned


def _has_repository_owned_connection_guard(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if not isinstance(item, ast.If):
            continue
        test = item.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        left = test.left
        comparator = test.comparators[0] if test.comparators else None
        if (
            isinstance(test.ops[0], ast.Is)
            and isinstance(left, ast.Attribute)
            and isinstance(left.value, ast.Name)
            and left.value.id == "self"
            and left.attr == "_db"
            and isinstance(comparator, ast.Constant)
            and comparator.value is None
            and "connect" in _call_names(item)
        ):
            return True
    return False


def audit(root: Path) -> dict[str, Any]:
    issues: list[str] = []
    checked_functions = 0
    parsed: dict[str, dict[str, ast.AST]] = {}

    for relative, contracts in _REQUIRED_CALLS.items():
        path = root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            issues.append(f"cannot parse {relative}: {type(exc).__name__}: {exc}")
            continue
        functions = _qualified_functions(tree)
        parsed[relative] = functions
        for qualified_name, required_calls in contracts.items():
            function = functions.get(qualified_name)
            if function is None:
                issues.append(
                    f"missing lifecycle contract function: {relative}:{qualified_name}"
                )
                continue
            checked_functions += 1
            calls = _call_names(function)
            missing = sorted(required_calls - calls)
            if missing:
                issues.append(f"{relative}:{qualified_name} lost calls {missing!r}")

            forbidden = _FORBIDDEN_CALLS.get(relative, {}).get(
                qualified_name, frozenset()
            )
            present_forbidden = sorted(forbidden & calls)
            if present_forbidden:
                issues.append(
                    f"{relative}:{qualified_name} gained constructing calls "
                    f"{present_forbidden!r}"
                )

    for qualified, required_attributes in _REQUIRED_NONE_ASSIGNMENTS.items():
        relative, function_name = qualified.split(":", 1)
        function = parsed.get(relative, {}).get(function_name)
        if function is None:
            continue
        missing = sorted(required_attributes - _none_assigned_self_attributes(function))
        if missing:
            issues.append(
                f"{relative}:{function_name} no longer clears owned resources {missing!r}"
            )

    for qualified, required_calls in _REQUIRED_QUALIFIED_CALLS.items():
        relative, function_name = qualified.split(":", 1)
        function = parsed.get(relative, {}).get(function_name)
        if function is None:
            continue
        missing = sorted(required_calls - _qualified_call_names(function))
        if missing:
            issues.append(
                f"{relative}:{function_name} lost exact owner calls {missing!r}"
            )

    ensure_db = parsed.get("core/state/state_repository.py", {}).get(
        "StateRepository._ensure_db"
    )
    if ensure_db is not None and not _has_repository_owned_connection_guard(ensure_db):
        issues.append(
            "StateRepository._ensure_db no longer opens the aiosqlite worker only "
            "behind the repository-owned self._db is None guard"
        )

    managed = parsed.get("core/tasks/managed_command.py", {}).get(
        "_run_project_command_async"
    )
    if managed is not None:
        source = ast.get_source_segment(
            (root / "core/tasks/managed_command.py").read_text(encoding="utf-8"),
            managed,
        ) or ""
        if "stdout_bytes, stderr_bytes = await process.communicate()" not in source:
            issues.append(
                "managed command timeout path no longer reaps the killed child"
            )

    cleanup = parsed.get("scripts/one_off/aura_cleanup.py", {}).get(
        "_kill_stale_processes"
    )
    if cleanup is not None:
        source = ast.get_source_segment(
            (root / "scripts/one_off/aura_cleanup.py").read_text(encoding="utf-8"),
            cleanup,
        ) or ""
        if "current_create_time" not in source or "observed_create_time" not in source:
            issues.append("stale-runtime cleanup lost its PID-reuse creation-time fence")

    return {
        "schema": "aura.closeout.lifecycle_ownership_audit.v1",
        "passed": not issues,
        "checked_functions": checked_functions,
        "contract_surfaces": sorted(_REQUIRED_CALLS),
        "owned_resources_cleared": sorted(
            next(iter(_REQUIRED_NONE_ASSIGNMENTS.values()))
        ),
        "natural_interpreter_exit_required_by_test": True,
        "runtime_owner_classes": [
            "process",
            "thread",
            "task",
            "listener",
            "sentinel",
            "actor",
            "model_worker",
            "lock",
        ],
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "passed" if report["passed"] else "failed"
        print(
            f"Lifecycle ownership audit {status}: "
            f"{report['checked_functions']} contract functions"
        )
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
