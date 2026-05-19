#!/usr/bin/env python3
"""Audit record_degradation() calls to identify log-and-limp anti-patterns.

Classifies each call as:
- CRITICAL: In a path that MUST succeed (model loading, boot, response generation)
- ADVISORY: In an optional integration (background tasks, telemetry, metrics)

For CRITICAL calls, checks if the error is properly handled (raise/return/abort).
"""

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv"}

# Paths considered critical — failures here break the user experience
CRITICAL_MODULES = {
    "inference_gate",
    "response_generation",
    "unitary",
    "mlx_client",
    "mlx_worker",
    "cognitive_engine",
    "orchestrator",
    "kernel",
    "boot",
    "state_machine",
    "conversation_support",
}

# Functions considered critical
CRITICAL_FUNCTIONS = {
    "generate",
    "think",
    "route",
    "process_message",
    "handle_incoming",
    "execute",
    "run",
    "start",
    "load_model",
    "spawn_worker",
}


def find_python_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


CONTROL_FLOW_STMTS = (ast.Return, ast.Raise, ast.Break, ast.Continue)
NO_RECOVERY_ACTIONS = {
    "",
    "no recovery",
    "no recovery action specified",
    "logged",
    "log",
}


def _record_degradation_call(node: ast.AST) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "record_degradation":
        return node
    if isinstance(func, ast.Attribute) and func.attr == "record_degradation":
        return node
    return None


def _literal_string(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip()
    return ""


def _keyword_value(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _call_has_recovery_action(call: ast.Call) -> bool:
    action = _literal_string(_keyword_value(call, "action"))
    if not action and len(call.args) >= 4:
        action = _literal_string(call.args[3])
    if action.lower() not in NO_RECOVERY_ACTIONS:
        return True

    extra = _keyword_value(call, "extra")
    if isinstance(extra, ast.Dict):
        for key, value in zip(extra.keys, extra.values, strict=False):
            if _literal_string(key) == "repair_requested" and isinstance(value, ast.Constant):
                return bool(value.value)
    return False


def _build_parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return "unknown"


def _enclosing_statement(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.stmt | None:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.stmt):
            return current
    return None


def _statement_sequence(
    statement: ast.stmt,
    parents: dict[ast.AST, ast.AST],
) -> tuple[list[ast.stmt], ast.stmt] | None:
    current: ast.AST = statement
    while current in parents:
        parent = parents[current]
        for attr in ("body", "orelse", "finalbody"):
            body = getattr(parent, attr, None)
            if isinstance(body, list) and current in body:
                return body, current
        current = parent
    return None


def _has_nearby_control_flow(statement: ast.stmt, parents: dict[ast.AST, ast.AST]) -> bool:
    if isinstance(statement, CONTROL_FLOW_STMTS):
        return True
    sequence = _statement_sequence(statement, parents)
    if sequence is None:
        return False
    body, current = sequence
    index = body.index(current)
    for next_statement in body[index + 1 : index + 4]:
        if isinstance(next_statement, CONTROL_FLOW_STMTS):
            return True
    return False


def _relative_path(filepath: Path) -> str:
    try:
        return str(filepath.relative_to(ROOT))
    except ValueError:
        return str(filepath)


def analyze_file(filepath: Path) -> list[dict]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    if "record_degradation" not in content:
        return []

    try:
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        return []

    results = []
    parents = _build_parent_map(tree)

    # Determine if this is a critical module
    rel = _relative_path(filepath)
    is_critical_module = any(cm in rel.lower() for cm in CRITICAL_MODULES)

    for node in ast.walk(tree):
        call = _record_degradation_call(node)
        if call is None:
            continue

        func_name = _enclosing_function(call, parents)
        is_critical_func = any(cf in func_name.lower() for cf in CRITICAL_FUNCTIONS)
        severity = (
            "CRITICAL"
            if (is_critical_module and is_critical_func)
            else ("HIGH" if is_critical_module or is_critical_func else "ADVISORY")
        )

        statement = _enclosing_statement(call, parents)
        has_failclose = bool(statement and _has_nearby_control_flow(statement, parents))
        has_recovery_action = _call_has_recovery_action(call)
        if severity in ("CRITICAL", "HIGH") and not (has_failclose or has_recovery_action):
            source = ast.get_source_segment(content, call) or "record_degradation(...)"
            results.append(
                {
                    "file": rel,
                    "line": call.lineno,
                    "function": func_name,
                    "severity": severity,
                    "has_failclose": has_failclose,
                    "has_recovery_action": has_recovery_action,
                    "code": " ".join(source.split())[:100],
                }
            )

    return results


def main():
    all_files = find_python_files(ROOT / "core")
    all_files.extend(find_python_files(ROOT / "skills"))

    issues = []
    for f in all_files:
        issues.extend(analyze_file(f))

    critical = [i for i in issues if i["severity"] == "CRITICAL"]
    high = [i for i in issues if i["severity"] == "HIGH"]

    print(f"Total CRITICAL log-and-limp calls: {len(critical)}")
    print(f"Total HIGH log-and-limp calls: {len(high)}")

    if critical:
        print(f"\n{'=' * 80}")
        print("CRITICAL: These fail silently in user-facing code paths")
        print(f"{'=' * 80}\n")
        for issue in sorted(critical, key=lambda x: x["file"]):
            print(f"  {issue['file']}:{issue['line']} in {issue['function']}()")
            print(f"    {issue['code']}")

    if high:
        print(f"\n{'=' * 80}")
        print(f"HIGH: These fail silently in important modules ({len(high)} total)")
        print(f"{'=' * 80}\n")

        by_file = Counter(i["file"] for i in high)
        for fname, count in by_file.most_common(15):
            print(f"  {fname}: {count} calls")
            file_issues = [i for i in high if i["file"] == fname]
            for issue in file_issues[:3]:
                print(f"    L{issue['line']} {issue['function']}(): {issue['code'][:80]}")


if __name__ == "__main__":
    main()
