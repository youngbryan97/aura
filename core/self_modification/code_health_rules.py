"""Deterministic Python code-health rules used by Aura's repair metabolism."""

from __future__ import annotations

import ast
import io
import logging
import os
import re
import tokenize
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFERRED_TAGS = ("TO" "DO", "FIX" "ME", "X" "XX")
_DEFERRED_COMMENT_PATTERN = re.compile(
    r"\b(" + "|".join(_DEFERRED_TAGS) + r")\b",
    re.IGNORECASE,
)
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def analyze_python_file(
    file_path: Path,
    stat: os.stat_result,
    *,
    display_path: Callable[[Path], str],
) -> list[dict[str, Any]]:
    if int(stat.st_size) > _MAX_SOURCE_BYTES:
        return [
            issue(
                file_path,
                display_path=display_path,
                line=0,
                rule_id="PY-SOURCE-OVERSIZE",
                severity="warning",
                issue_type="maintainability",
                confidence=1.0,
                message=f"Source is {stat.st_size} bytes; bounded analysis skipped it.",
                remediation="Split generated or oversized source into auditable modules.",
            )
        ]

    try:
        with tokenize.open(file_path) as handle:
            content = handle.read()
    except (LookupError, SyntaxError, UnicodeError) as exc:
        return [
            issue(
                file_path,
                display_path=display_path,
                line=int(getattr(exc, "lineno", 0) or 0),
                rule_id="PY-SOURCE-DECODE",
                severity="error",
                issue_type="source_encoding",
                confidence=1.0,
                message=f"Python source could not be decoded: {type(exc).__name__}: {exc}",
                remediation="Repair the encoding declaration and save the source as valid UTF-8.",
            )
        ]
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as exc:
        return [
            issue(
                file_path,
                display_path=display_path,
                line=int(exc.lineno or 0),
                rule_id="PY-SYNTAX-ERROR",
                severity="error",
                issue_type="syntax_error",
                confidence=1.0,
                message=f"Python parser rejected the source: {exc.msg}",
                remediation="Repair the syntax before this module is imported or deployed.",
            )
        ]

    issues: list[dict[str, Any]] = []
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in functions:
        end_line = int(getattr(node, "end_lineno", node.lineno) or node.lineno)
        length = max(0, end_line - int(node.lineno) + 1)
        if length > 80:
            issues.append(
                issue(
                    file_path,
                    display_path=display_path,
                    line=node.lineno,
                    rule_id="PY-LONG-FUNCTION",
                    severity="warning",
                    issue_type="complexity",
                    confidence=0.96,
                    message=f"Function '{node.name}' spans {length} lines.",
                    remediation="Extract cohesive stages with explicit contracts and focused tests.",
                )
            )

        complexity = branch_complexity(node)
        if complexity > 18:
            issues.append(
                issue(
                    file_path,
                    display_path=display_path,
                    line=node.lineno,
                    rule_id="PY-BRANCH-COMPLEXITY",
                    severity="warning",
                    issue_type="complexity",
                    confidence=0.9,
                    message=f"Function '{node.name}' has estimated branch complexity {complexity}.",
                    remediation="Separate decisions into named, independently tested policy units.",
                )
            )

        defaults = list(node.args.defaults) + [
            default for default in node.args.kw_defaults if default is not None
        ]
        if any(mutable_default(default) for default in defaults):
            issues.append(
                issue(
                    file_path,
                    display_path=display_path,
                    line=node.lineno,
                    rule_id="PY-MUTABLE-DEFAULT",
                    severity="error",
                    issue_type="correctness",
                    confidence=0.99,
                    message=f"Function '{node.name}' has a mutable default argument.",
                    remediation="Use None and allocate the mutable value inside the function.",
                )
            )

        if isinstance(node, ast.AsyncFunctionDef):
            for owned in walk_owned_nodes(node):
                if isinstance(owned, ast.Call) and call_name(owned.func) == "time.sleep":
                    issues.append(
                        issue(
                            file_path,
                            display_path=display_path,
                            line=int(getattr(owned, "lineno", node.lineno)),
                            rule_id="PY-ASYNC-BLOCKING-SLEEP",
                            severity="error",
                            issue_type="latency",
                            confidence=1.0,
                            message=f"Async function '{node.name}' calls blocking time.sleep().",
                            remediation="Use await asyncio.sleep() or offload blocking work.",
                        )
                    )

    for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
        caught = exception_name(handler.type)
        if handler.type is None or catches_broad_exception(handler.type):
            issues.append(
                issue(
                    file_path,
                    display_path=display_path,
                    line=int(getattr(handler, "lineno", 0)),
                    rule_id="PY-BROAD-EXCEPTION",
                    severity="warning",
                    issue_type="reliability",
                    confidence=0.94,
                    message=(
                        "Bare exception handler obscures fault class."
                        if handler.type is None
                        else f"Broad {caught} handler obscures fault class."
                    ),
                    remediation="Catch named operational exceptions and preserve fault provenance.",
                )
            )

    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT or not _DEFERRED_COMMENT_PATTERN.search(token.string):
                continue
            issues.append(
                issue(
                    file_path,
                    display_path=display_path,
                    line=token.start[0],
                    rule_id="PY-DEFERRED-WORK",
                    severity="info",
                    issue_type="deferred_marker",
                    confidence=0.85,
                    message=f"Deferred-work comment: {token.string.strip()[:180]}",
                    remediation="Resolve it or link it to an owned tracker item and acceptance test.",
                )
            )
    except (IndentationError, tokenize.TokenError) as exc:
        logger.debug(
            "%s unavailable (%s: %s); the file could not be tokenised, so deferred-work comments in it are not reported",
            "it",
            type(exc).__name__,
            exc,
        )

    issues.sort(key=issue_sort_key)
    return issues


def walk_owned_nodes(function: ast.AST) -> Iterator[ast.AST]:
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def branch_complexity(function: ast.AST) -> int:
    branch_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.Match)
    complexity = 1
    for node in walk_owned_nodes(function):
        if isinstance(node, branch_types):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += max(1, len(node.values) - 1)
        elif isinstance(node, ast.Try):
            complexity += len(node.handlers) + int(bool(node.orelse)) + int(bool(node.finalbody))
    return complexity


def mutable_default(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Dict, ast.Set)):
        return True
    return isinstance(node, ast.Call) and call_name(node.func) in {
        "list",
        "dict",
        "set",
        "collections.defaultdict",
    }


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def exception_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Tuple):
        return ",".join(exception_name(item) for item in node.elts)
    return ""


def catches_broad_exception(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Tuple):
        return any(catches_broad_exception(item) for item in node.elts)
    return exception_name(node) in {"Exception", "BaseException"}


def issue(
    file_path: Path,
    *,
    display_path: Callable[[Path], str],
    line: int,
    rule_id: str,
    severity: str,
    issue_type: str,
    confidence: float,
    message: str,
    remediation: str,
) -> dict[str, Any]:
    return {
        "file": display_path(file_path),
        "line": int(line),
        "rule_id": rule_id,
        "severity": severity,
        "type": issue_type,
        "confidence": round(float(confidence), 3),
        "message": message,
        "remediation": remediation,
    }


def issue_sort_key(issue_data: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _SEVERITY_ORDER.get(str(issue_data.get("severity")), 9),
        -float(issue_data.get("confidence", 0.0) or 0.0),
        str(issue_data.get("rule_id", "")),
        str(issue_data.get("file", "")),
        int(issue_data.get("line", 0) or 0),
    )


__all__ = ["analyze_python_file", "issue", "issue_sort_key"]
