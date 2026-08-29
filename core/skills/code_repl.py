"""core/skills/code_repl.py — Real-Time Python REPL
====================================================
First-class BaseSkill that gives Aura a live, stateful Python REPL.
This is the exact equivalent of a code_execution/code_interpreter tool:
  - Execute arbitrary Python in a sandboxed subprocess
  - Maintain per-session variable state across turns
  - Capture stdout, stderr, return values, and generated files
  - Enforce memory/CPU/time limits via core.sandbox.runner

This closes the "code REPL" gap in tool parity.
"""

import ast
import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.config import config
from core.governance.will import ActionDomain
from core.runtime.action_executor import ActionExecutor
from core.runtime.errors import FallbackClassification, record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.CodeREPL")

_REPL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)
CODE_REPL_MODEL_RESULT_SCHEMA = "aura.code_repl.model_result.v3"
_MODEL_STDOUT_LIMIT = 1024
_MODEL_STDERR_LIMIT = 1024
_MODEL_ERROR_LIMIT = 768
_MODEL_ENGINE_LIMIT = 64
CODE_REPL_MODEL_RESULT_FIELDS = frozenset(
    {
        "schema",
        "ok",
        "status",
        "stdout",
        "stderr",
        "returncode",
        "engine",
        "summary",
        "error",
        "stdout_truncated",
        "stdout_original_chars",
        "stdout_sha256",
        "stdout_preview_sha256",
        "stderr_truncated",
        "stderr_original_chars",
        "stderr_sha256",
        "stderr_preview_sha256",
        "error_truncated",
        "error_original_chars",
        "error_sha256",
        "error_preview_sha256",
    }
)


def _bounded_model_evidence(
    value: Any,
    *,
    limit: int,
) -> tuple[str, bool, int, str]:
    text = str(value or "")
    original_chars = len(text)
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    if original_chars <= limit:
        return text, False, original_chars, digest
    marker = "\n...[TRUNCATED; VERIFY SHA256]..."
    retained = max(0, limit - len(marker))
    return text[:retained] + marker, True, original_chars, digest


def _validate_canonical_model_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    if (
        raw.get("schema") != CODE_REPL_MODEL_RESULT_SCHEMA
        or not CODE_REPL_MODEL_RESULT_FIELDS.issubset(raw)
    ):
        raise ValueError("code_repl canonical result fields are invalid")
    result = {name: raw[name] for name in CODE_REPL_MODEL_RESULT_FIELDS}
    if (
        type(result["ok"]) is not bool
        or result["status"] not in {"ok", "error", "timeout"}
        or (result["status"] == "ok") != result["ok"]
        or (
            result["returncode"] is not None
            and type(result["returncode"]) is not int
        )
        or not isinstance(result["engine"], str)
        or len(result["engine"]) > _MODEL_ENGINE_LIMIT
        or not isinstance(result["summary"], str)
        or result["summary"]
        != (
            "Code execution completed successfully."
            if result["status"] == "ok"
            else (
                "Code execution timed out."
                if result["status"] == "timeout"
                else "Code execution failed."
            )
        )
        or (result["ok"] and result["returncode"] not in {None, 0})
    ):
        raise ValueError("code_repl canonical result status is invalid")
    for name in ("stdout", "stderr", "error"):
        truncated = result[f"{name}_truncated"]
        original_chars = result[f"{name}_original_chars"]
        digest = result[f"{name}_sha256"]
        preview_digest = result[f"{name}_preview_sha256"]
        value = result[name]
        if (
            not isinstance(value, str)
            or type(truncated) is not bool
            or type(original_chars) is not int
            or original_chars < len(value)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(preview_digest, str)
            or preview_digest
            != hashlib.sha256(
                value.encode("utf-8", errors="surrogatepass")
            ).hexdigest()
            or (not truncated and original_chars != len(value))
            or (
                not truncated
                and hashlib.sha256(
                    value.encode("utf-8", errors="surrogatepass")
                ).hexdigest()
                != digest
            )
        ):
            raise ValueError(f"code_repl canonical {name} evidence is invalid")
    if result["ok"] and result["error"]:
        raise ValueError("successful code_repl result cannot contain an error")
    if not result["ok"] and not result["error"]:
        raise ValueError("failed code_repl result must contain an error")
    return result


def normalize_code_repl_model_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable evidence view shown to the model and SFT datasets."""

    if not isinstance(raw, Mapping):
        raise TypeError("code_repl result must be a mapping")
    if raw.get("schema") == CODE_REPL_MODEL_RESULT_SCHEMA:
        return _validate_canonical_model_result(raw)
    raw_status = str(raw.get("status") or "").strip().lower()
    returncode = raw.get("returncode")
    returncode = returncode if type(returncode) is int else None
    explicit_ok = raw.get("ok")
    raw_error = str(raw.get("error") or "").strip()
    stdout_raw = str(raw.get("stdout") or "")
    stderr_raw = str(raw.get("stderr") or "")
    failure_status = raw_status in {
        "aborted",
        "blocked",
        "cancelled",
        "deferred",
        "denied",
        "error",
        "failed",
        "failure",
        "partial",
        "rejected",
        "timeout",
    }
    success_status = raw_status in {"ok", "success", "completed"}
    unknown_status = bool(raw_status) and not failure_status and not success_status
    failure_evidence = (
        explicit_ok is False
        or failure_status
        or unknown_status
        or returncode is not None
        and returncode != 0
        or bool(raw_error)
    )
    success_evidence = (
        explicit_ok is True
        or success_status
        or returncode == 0
    )
    ok = bool(success_evidence and not failure_evidence)
    status = "ok" if ok else ("timeout" if raw_status == "timeout" else "error")

    error_raw = raw_error
    if not ok and not error_raw:
        if stderr_raw.strip():
            error_raw = stderr_raw.strip()
        else:
            supplemental = [
                str(raw.get(name) or "").strip()
                for name in ("repr", "traceback", "reason", "detail")
            ]
            error_raw = "\n".join(
                text for index, text in enumerate(supplemental)
                if text and text not in supplemental[:index]
            )
    if not ok and not error_raw:
        error_raw = f"code_repl execution failed ({status})"
    if ok:
        error_raw = ""

    stdout, stdout_truncated, stdout_chars, stdout_sha256 = (
        _bounded_model_evidence(stdout_raw, limit=_MODEL_STDOUT_LIMIT)
    )
    stderr, stderr_truncated, stderr_chars, stderr_sha256 = (
        _bounded_model_evidence(stderr_raw, limit=_MODEL_STDERR_LIMIT)
    )
    error, error_truncated, error_chars, error_sha256 = (
        _bounded_model_evidence(error_raw, limit=_MODEL_ERROR_LIMIT)
    )
    engine = str(raw.get("engine") or "unknown")[:_MODEL_ENGINE_LIMIT]
    summary = (
        "Code execution completed successfully."
        if ok
        else (
            "Code execution timed out."
            if status == "timeout"
            else "Code execution failed."
        )
    )
    return {
        "schema": CODE_REPL_MODEL_RESULT_SCHEMA,
        "ok": ok,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": returncode,
        "engine": engine,
        "summary": summary,
        "error": error,
        "stdout_truncated": stdout_truncated,
        "stdout_original_chars": stdout_chars,
        "stdout_sha256": stdout_sha256,
        "stdout_preview_sha256": hashlib.sha256(
            stdout.encode("utf-8", errors="surrogatepass")
        ).hexdigest(),
        "stderr_truncated": stderr_truncated,
        "stderr_original_chars": stderr_chars,
        "stderr_sha256": stderr_sha256,
        "stderr_preview_sha256": hashlib.sha256(
            stderr.encode("utf-8", errors="surrogatepass")
        ).hexdigest(),
        "error_truncated": error_truncated,
        "error_original_chars": error_chars,
        "error_sha256": error_sha256,
        "error_preview_sha256": hashlib.sha256(
            error.encode("utf-8", errors="surrogatepass")
        ).hexdigest(),
    }


def serialize_code_repl_model_result(
    raw: Mapping[str, Any],
    *,
    limit: int = 4000,
) -> str:
    """Return schema-preserving v3 JSON within the final escaped budget."""

    if type(limit) is not int or limit <= 0:
        raise ValueError("code_repl model result limit must be positive")
    result = normalize_code_repl_model_result(raw)

    def render() -> str:
        return json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    serialized = render()
    irreducible = dict(result)
    for field in ("stdout", "stderr", "error"):
        minimum_chars = 1 if field == "error" and not result["ok"] else 0
        irreducible[field] = result[field][:minimum_chars]
        irreducible[f"{field}_truncated"] = bool(
            result[f"{field}_truncated"]
            or len(result[field]) > minimum_chars
        )
        irreducible[f"{field}_preview_sha256"] = hashlib.sha256(
            irreducible[field].encode("utf-8", errors="surrogatepass")
        ).hexdigest()
    if len(
        json.dumps(
            irreducible,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) > limit:
        raise ValueError(
            "code_repl model result limit cannot contain canonical schema"
        )
    while len(serialized) > limit:
        fields = [
            name
            for name in ("stdout", "stderr", "error")
            if len(result[name])
            > (1 if name == "error" and not result["ok"] else 0)
        ]
        if not fields:
            raise ValueError(
                "code_repl model result limit cannot contain canonical schema"
            )
        field = max(
            fields,
            key=lambda name: (
                len(json.dumps(result[name], ensure_ascii=True)),
                -("stdout", "stderr", "error").index(name),
            ),
        )
        original_preview = result[field]
        candidate = dict(result)
        minimum_chars = 1 if field == "error" and not result["ok"] else 0
        candidate[field] = original_preview[:minimum_chars]
        candidate[f"{field}_truncated"] = True
        candidate[f"{field}_preview_sha256"] = hashlib.sha256(
            candidate[field].encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        empty_serialized = json.dumps(
            candidate,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(empty_serialized) > limit:
            result.update(candidate)
            serialized = empty_serialized
            continue
        low = minimum_chars
        high = len(original_preview)
        best = original_preview[:minimum_chars]
        while low <= high:
            middle = (low + high) // 2
            candidate[field] = original_preview[:middle]
            candidate[f"{field}_preview_sha256"] = hashlib.sha256(
                candidate[field].encode("utf-8", errors="surrogatepass")
            ).hexdigest()
            rendered = json.dumps(
                candidate,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(rendered) <= limit:
                best = candidate[field]
                low = middle + 1
            else:
                high = middle - 1
        result[field] = best
        result[f"{field}_truncated"] = True
        result[f"{field}_preview_sha256"] = hashlib.sha256(
            best.encode("utf-8", errors="surrogatepass")
        ).hexdigest()
        next_serialized = render()
        if len(next_serialized) >= len(serialized):
            raise ValueError("code_repl model result fitting made no progress")
        serialized = next_serialized
    return serialized


def _record_repl_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    record_degradation(
        "code_repl",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=False,
        extra=extra,
    )


#: What the restricted runner removes. A snippet that needs one of these is
#: outside what that strategy can run, whatever the program itself is doing.
_STRIPPED_BUILTINS = (
    "__import__ not found",
    "name 'open' is not defined",
    "name 'eval' is not defined",
    "name 'exec' is not defined",
    "name 'compile' is not defined",
    "name 'globals' is not defined",
    "name 'locals' is not defined",
    "name '__import__' is not defined",
)


def _is_a_sandbox_limit(result: dict[str, Any]) -> bool:
    """Whether this failure is the sandbox refusing rather than the code failing."""
    if result.get("ok"):
        return False
    said = " ".join(
        str(result.get(key) or "")
        for key in ("stderr", "error", "traceback", "summary")
    )
    return any(marker in said for marker in _STRIPPED_BUILTINS)


def _resolved(path: str) -> Callable[[], str]:
    """The resolution, as work to hand to a thread."""

    def _do() -> str:
        return str(Path(path).expanduser().resolve())

    return _do


def _a_library_the_person_named(context: dict[str, Any] | None) -> str:
    """A directory of Python named in the request, or "".

    Read from the same place every other path in a request is read from, so a
    directory that resolves here is one that resolved for the file reader too.
    """

    text = " ".join(
        str((context or {}).get(key) or "")
        for key in ("objective", "message", "prompt", "user_message")
    ).strip()
    if not text:
        return ""
    try:
        from core.language.named_paths import named_paths
    except ImportError:
        return ""
    try:
        named = list(named_paths(text) or ())
    except (OSError, TypeError, ValueError):
        return ""
    for candidate in named:
        allowed, _why = _library_path_is_allowed(str(candidate))
        if allowed:
            return str(candidate)
        # A file the person named sits in a directory that may be the library.
        try:
            parent = Path(str(candidate)).parent
        except (OSError, ValueError):
            continue
        allowed, _why = _library_path_is_allowed(str(parent))
        if allowed:
            return str(parent)
    return ""


def _library_path_is_allowed(path: str) -> tuple[bool, str]:
    """Whether this directory may be put on the import path.

    A real directory, holding Python, and not a place that would import the
    runtime's own code into a sandbox that is meant to be separate from it.
    """

    from pathlib import Path as _Path

    try:
        target = _Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"unreadable ({type(exc).__name__})"
    if not target.is_dir():
        return False, "not a directory"
    try:
        modules = [item for item in target.iterdir() if item.suffix == ".py"]
    except OSError as exc:
        return False, f"unlistable ({type(exc).__name__})"
    if not modules:
        return False, "holds no Python modules"
    here = _Path(__file__).resolve().parents[2]
    if target == here or here in target.parents:
        return False, "inside the runtime's own source"
    return True, ""



def _without_a_path_preamble_for(code: str, library_root: str) -> str:
    """Drop the two lines that ask for something the sandbox already did.

    Importing from a directory means putting it on sys.path — everywhere in
    Python except here, where the runner was given the library and made it
    importable before the code ran. A model that has written Python before
    writes the preamble anyway, because that is what the idiom is, and the
    sandbox refuses ``sys`` for good reasons and takes the whole turn with it.

    LIVE 2026-08-29: "'sys' is not part of the library this sandbox was given;
    available: ledgerkit". The library was right there under the name the code
    went on to import, and the request died on the line before.

    Only the redundancy goes. Imports here are served from the library the
    runner was handed, never from ``sys.path``, so a ``sys.path`` call cannot
    change what this code can import whatever directory it names — it is a
    no-op by construction, not merely a duplicate of work already done. That
    matters because the model does not always pass ``library_path``, and the
    line it dies on is the same line either way.

    ``import sys`` goes with it when nothing else in the code uses the name.
    Any other use of ``sys`` is untouched and still refused, which is what
    refusing it is for.
    """

    if "sys" not in code:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    # Anywhere in the file, not only at the top of it.
    #
    # LIVE 2026-08-29: the turn still died on "'sys' is not part of the library
    # this sandbox was given" after this ran, because the model had written the
    # preamble inside a try block — which is what anyone writes when an import
    # might fail, and is exactly the shape a careful model reaches for.
    # Scanning tree.body saw a try statement and nothing inside it.
    redundant: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        called = node.value.func
        if (
            isinstance(called, ast.Attribute)
            and called.attr in ("insert", "append")
            and isinstance(called.value, ast.Attribute)
            and called.value.attr == "path"
            and isinstance(called.value.value, ast.Name)
            and called.value.value.id == "sys"
        ):
            redundant.append(node)
    if not redundant:
        return code

    dropped = {id(node) for node in redundant}
    still_used = any(
        isinstance(node, ast.Name)
        and node.id == "sys"
        and not any(node in ast.walk(gone) for gone in redundant)
        for node in ast.walk(tree)
    )
    if not still_used:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and [a.name for a in node.names] == ["sys"]:
                dropped.add(id(node))
                redundant.append(node)

    # Replaced in place rather than unparsed: every other line keeps its
    # number, so a traceback still points at what the model wrote.
    #
    # ``pass`` rather than a blank line, because the statement may be the only
    # one in its block. "if ok: sys.path.insert(...)" blanked to nothing is an
    # if with no body, and code that was going to fail on a banned import now
    # fails to parse instead — a worse answer than the one it replaced. Same
    # indentation, so the block it belonged to still owns it.
    lines = code.splitlines()
    for node in redundant:
        last = getattr(node, "end_lineno", node.lineno) or node.lineno
        for number in range(node.lineno, last + 1):
            if not 1 <= number <= len(lines):
                continue
            original = lines[number - 1]
            indent = original[: len(original) - len(original.lstrip())]
            lines[number - 1] = f"{indent}pass" if number == node.lineno else ""
    logger.info(
        "code_repl: dropped %d sys.path line(s) that cannot affect this "
        "sandbox%s.",
        len(redundant),
        f" — imports come from {library_root}" if library_root else "",
    )
    return "\n".join(lines)


def _without_a_path_preamble_for(code: str, library_root: str) -> str:
    """Drop the two lines that ask for something the sandbox already did.

    Importing from a directory means putting it on sys.path — everywhere in
    Python except here, where the runner was given the library and made it
    importable before the code ran. A model that has written Python before
    writes the preamble anyway, because that is what the idiom is, and the
    sandbox refuses ``sys`` for good reasons and takes the whole turn with it.

    LIVE 2026-08-29: "'sys' is not part of the library this sandbox was given;
    available: ledgerkit". The library was right there under the name the code
    went on to import, and the request died on the line before.

    Only the redundancy goes. A ``sys.path`` call naming the directory the
    sandbox already loaded is a no-op, and ``import sys`` goes with it when
    nothing else in the code uses the name. Any other use of ``sys`` is
    untouched and still refused, which is what refusing it is for.
    """

    if "sys" not in code or not library_root:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    try:
        root = str(Path(library_root).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        root = str(library_root)

    def _names_the_library_directory(node: ast.Call) -> bool:
        for argument in node.args:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            try:
                named = str(Path(argument.value).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                named = argument.value
            if named == root:
                return True
        return False

    redundant: list[ast.stmt] = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        called = node.value.func
        if (
            isinstance(called, ast.Attribute)
            and called.attr in ("insert", "append")
            and isinstance(called.value, ast.Attribute)
            and called.value.attr == "path"
            and isinstance(called.value.value, ast.Name)
            and called.value.value.id == "sys"
            and _names_the_library_directory(node.value)
        ):
            redundant.append(node)
    if not redundant:
        return code

    dropped = {id(node) for node in redundant}
    still_used = any(
        isinstance(node, ast.Name)
        and node.id == "sys"
        and not any(node in ast.walk(gone) for gone in redundant)
        for node in ast.walk(tree)
    )
    if not still_used:
        for node in tree.body:
            if isinstance(node, ast.Import) and [a.name for a in node.names] == ["sys"]:
                dropped.add(id(node))
                redundant.append(node)

    # Blanked rather than unparsed: every other line keeps its number, so a
    # traceback still points at what the model wrote.
    lines = code.splitlines()
    for node in redundant:
        last = getattr(node, "end_lineno", node.lineno) or node.lineno
        for number in range(node.lineno, last + 1):
            if 1 <= number <= len(lines):
                lines[number - 1] = ""
    logger.info(
        "code_repl: dropped %d redundant sys.path line(s) — the sandbox had "
        "already made %s importable.",
        len(redundant),
        library_root,
    )
    return "\n".join(lines)


def _without_a_path_preamble_for(code: str, library_root: str) -> str:
    """Drop the two lines that ask for something the sandbox already did.

    Importing from a directory means putting it on sys.path — everywhere in
    Python except here, where the runner was given the library and made it
    importable before the code ran. A model that has written Python before
    writes the preamble anyway, because that is what the idiom is, and the
    sandbox refuses ``sys`` for good reasons and takes the whole turn with it.

    LIVE 2026-08-29: "'sys' is not part of the library this sandbox was given;
    available: ledgerkit". The library was right there under the name the code
    went on to import, and the request died on the line before.

    Only the redundancy goes. A ``sys.path`` call naming the directory the
    sandbox already loaded is a no-op, and ``import sys`` goes with it when
    nothing else in the code uses the name. Any other use of ``sys`` is
    untouched and still refused, which is what refusing it is for.
    """

    if "sys" not in code or not library_root:
        return code
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code

    try:
        root = str(Path(library_root).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        root = str(library_root)

    def _names_the_library_directory(node: ast.Call) -> bool:
        for argument in node.args:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            try:
                named = str(Path(argument.value).expanduser().resolve())
            except (OSError, RuntimeError, ValueError):
                named = argument.value
            if named == root:
                return True
        return False

    redundant: list[ast.stmt] = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        called = node.value.func
        if (
            isinstance(called, ast.Attribute)
            and called.attr in ("insert", "append")
            and isinstance(called.value, ast.Attribute)
            and called.value.attr == "path"
            and isinstance(called.value.value, ast.Name)
            and called.value.value.id == "sys"
            and _names_the_library_directory(node.value)
        ):
            redundant.append(node)
    if not redundant:
        return code

    dropped = {id(node) for node in redundant}
    still_used = any(
        isinstance(node, ast.Name)
        and node.id == "sys"
        and not any(node in ast.walk(gone) for gone in redundant)
        for node in ast.walk(tree)
    )
    if not still_used:
        for node in tree.body:
            if isinstance(node, ast.Import) and [a.name for a in node.names] == ["sys"]:
                dropped.add(id(node))
                redundant.append(node)

    # Blanked rather than unparsed: every other line keeps its number, so a
    # traceback still points at what the model wrote.
    lines = code.splitlines()
    for node in redundant:
        last = getattr(node, "end_lineno", node.lineno) or node.lineno
        for number in range(node.lineno, last + 1):
            if 1 <= number <= len(lines):
                lines[number - 1] = ""
    logger.info(
        "code_repl: dropped %d redundant sys.path line(s) — the sandbox had "
        "already made %s importable.",
        len(redundant),
        library_root,
    )
    return "\n".join(lines)


def _what_will_not_work_in_this_code(code: str, library_root: str) -> list:
    """What reading this code decides, before any of it is run."""

    try:
        from core.sandbox.static_check import what_will_not_work

        return what_will_not_work(code, library_root)
    except Exception as _exc:  # noqa: BLE001 - a check that fails checks nothing
        logger.debug(
            "Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc
        )
        return []


def _describe_what_will_not_work(findings: list) -> str:
    from core.sandbox.static_check import describe_findings

    return describe_findings(findings)


def _the_code_inside_any_fence(code: str) -> str:
    """The Python inside a markdown fence, or the text unchanged.

    Only consulted where a fence is actually present: the extractor also
    strips trailing whitespace, and code that was never fenced should come
    through byte for byte.
    """

    try:
        from core.brain.llm.code_generator import extract_python_code
    except ImportError:
        return code
    try:
        return str(extract_python_code(code) or code)
    except (AttributeError, TypeError, ValueError):
        return code

#: The largest computation budget this skill will accept for one execution.
_LONGEST_ACCEPTED_TIMEOUT_S = 120


class CodeREPLInput(BaseModel):
    code: str = Field(..., description="Python code to execute in the REPL.")
    session_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Optional session ID for maintaining state across turns.",
    )
    timeout: int = Field(
        30,
        ge=1,
        le=_LONGEST_ACCEPTED_TIMEOUT_S,
        description="Maximum execution time in seconds.",
    )
    capture_files: bool = Field(
        True,
        description="Whether to capture any files generated in the working directory.",
    )
    library_path: str | None = Field(
        None,
        max_length=1024,
        description=(
            "A directory of Python modules to make importable, when the person "
            "named a library to use. The code can then import it directly."
        ),
    )


class CodeREPLSkill(BaseSkill):
    name = "code_repl"
    description = (
        "Execute Python code in a real-time, sandboxed REPL. "
        "Supports multi-turn sessions with persistent state, file generation, "
        "and full stdout/stderr capture. Use for calculations, data processing, "
        "prototyping, and any computational task."
    )
    input_model = CodeREPLInput
    #: The dispatcher must outlast the sandbox, or the sandbox's own handling
    #: of a slow or runaway script never gets to be the answer.
    #:
    #: LIVE 2026-08-29: "Tool Result: code_repl in 120004ms", twice, on a
    #: sandbox allowed 180 seconds of wall clock for its 30-second computation
    #: budget. The dispatcher killed it at its own flat 120 and the tool
    #: returned nothing — not the output, not the timeout, not the traceback.
    #: The model saw an outcome of "unknown" and tried the same thing again.
    #:
    #: Derived from the same function the runner uses, for the largest budget
    #: this skill accepts, so the two cannot disagree again. The turn's own
    #: ceiling remains the outer bound, which is the clock that should end a
    #: turn.
    #: Literal because the catalog discovers this statically — a computed
    #: value makes the skill undiscoverable. Held to the runner's formula by
    #: test_the_dispatcher_outlasts_the_sandbox.
    timeout_seconds = 720.0
    metabolic_cost = 2
    effect_scope = "sandboxed_compute"

    # Session state: maps session_id -> serialized namespace dict
    _sessions: dict[str, dict[str, Any]]
    _session_dirs: dict[str, Path]

    def __init__(self) -> None:
        super().__init__()
        self._sessions = {}
        self._session_dirs = {}
        self._output_dir = Path(config.paths.data_dir) / "repl_sessions"

    async def _get_session_dir(self, session_id: str) -> Path:
        """Get or create a working directory for a session."""
        if session_id not in self._session_dirs:
            session_dir = self._output_dir / session_id
            await get_file_write_gateway().ensure_directory_async(
                session_dir,
                source="skills.code_repl.session",
            )
            self._session_dirs[session_id] = session_dir
        return self._session_dirs[session_id]

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.sha256(
            f"{time.time()}-{os.getpid()}".encode()
        ).hexdigest()[:12]

    async def execute(
        self, params: CodeREPLInput, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute Python code in a sandboxed REPL."""
        if isinstance(params, dict):
            try:
                params = CodeREPLInput(**params)
            except _REPL_RECOVERABLE_ERRORS as exc:
                _record_repl_degradation(
                    exc,
                    action="rejected invalid REPL input before code execution",
                )
                return {"ok": False, "error": f"Invalid input: {exc}"}

        code = params.code.strip()
        if not code:
            return {"ok": False, "error": "No code provided."}

        # A model writing code writes a fenced block, because that is how code
        # is written everywhere it has ever seen it. Sent straight to the
        # interpreter the fence is a syntax error, and the turn spends an
        # attempt learning that.
        #
        # The reconstruction lab already solved this: extract_python_code
        # takes the fenced body, and takes it even when the closing fence is
        # missing because the generation ran out of room. Same extractor here
        # rather than a second one that will drift from it.
        unfenced = _the_code_inside_any_fence(code)
        if "```" in code and unfenced and unfenced != code:
            logger.info(
                "code_repl: unwrapped a fenced block (%d chars -> %d).",
                len(code),
                len(unfenced),
            )
            code = unfenced

        # A library the person named, made importable.
        #
        # ``sys`` is banned in the sandbox and rightly so — it is the
        # interpreter itself — but sys.path is the only way to import a module
        # from a directory, so "read the docs at this path, then use it" was
        # impossible by construction: every attempt came back "banned import or
        # call" for doing the one thing the request asked for.
        #
        # Naming the directory as an argument makes it checkable in the way
        # every other path this runtime accepts is checkable, instead of
        # arriving inside a program nobody has read.
        library = str(getattr(params, "library_path", "") or "").strip()
        if not library:
            # The directory the person named, which the runtime already read.
            #
            # A parameter only helps when the caller fills it. The path was in
            # the request, this runtime resolves paths in requests for other
            # reasons already, and "use the library at <path>" should not
            # depend on the model remembering a field it has never seen before.
            library = _a_library_the_person_named(context)
            if library:
                logger.info(
                    "code_repl: importing from the directory named in the "
                    "request: %s",
                    library,
                )
        if library:
            allowed, why = _library_path_is_allowed(library)
            if not allowed:
                return {"ok": False, "error": f"library_path refused: {why}"}
            # Resolved off the event loop: a path lookup is a filesystem
            # call, and this one runs inside an async handler.
            library = await asyncio.to_thread(_resolved(library))

        # What is wrong with this code before an attempt is spent on it.
        #
        # A model writing code invents: a method that reads right, a keyword
        # the function never took, a helper it meant to define. Running it to
        # find out costs a full generation from a resident 27B and comes back
        # as a traceback, which says what broke rather than what exists.
        #
        # Syntax, undefined names and calls into the named library are all
        # decidable by reading, in milliseconds, and the answer can carry the
        # real API instead of an AttributeError. Nothing found here is a claim
        # that the code is right — only that nothing was decidably wrong.
        code = _without_a_path_preamble_for(code, library)
        if library:
            code = _without_a_path_preamble_for(code, library)
        if library:
            code = _without_a_path_preamble_for(code, library)
        if library:
            code = _without_a_path_preamble_for(code, library)
        will_not_work = await asyncio.to_thread(
            _what_will_not_work_in_this_code, code, library
        )
        if will_not_work:
            logger.info(
                "code_repl: refused before running — %d thing(s) decidable by "
                "reading%s.",
                len(will_not_work),
                f" (library {library})" if library else "",
            )
            return {"ok": False, "error": _describe_what_will_not_work(will_not_work)}

        session_id = params.session_id or self._generate_session_id()
        session_dir = await self._get_session_dir(session_id)
        timeout_s = params.timeout

        # List files before execution to detect new ones
        pre_files = set()
        if params.capture_files:
            try:
                pre_files = set(session_dir.iterdir())
            except OSError as _exc:
                logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

        # Strategy 1: Use core.sandbox.runner (preferred — full isolation)
        result = await self._execute_via_sandbox_runner(
            code, timeout_s, session_dir, library_root=library
        )
        # A strategy that CANNOT run this code has not run it.
        #
        # The restricted runner strips __import__, open, eval and the rest, so
        # any snippet that imports anything comes back
        # ImportError('__import__ not found'). That is the sandbox refusing,
        # not the program being wrong, and returning it as the result stopped
        # the chain at its most restricted link.
        #
        # LIVE, 2026-08-27: "read these docs, then actually use the library"
        # got through routing, offering, leasing, permission and the executive
        # — five separate fixes — reached this skill, and died here, on a
        # description that calls itself "the exact equivalent of a
        # code_execution/code_interpreter tool".
        if result is not None and _is_a_sandbox_limit(result):
            logger.info(
                "code_repl: the restricted runner cannot run this snippet (%s); "
                "falling through to the next strategy.",
                str(result.get("stderr") or result.get("error") or "")[:120],
            )
            result = None

        if result is None:
            # Strategy 2: Use SandboxOperator (fallback)
            result = await self._execute_via_sandbox_operator(
                code, timeout_s
            )

        if result is None:
            # Strategy 3: Governed subprocess (last resort)
            result = await self._execute_via_subprocess(
                code, timeout_s, session_dir
            )

        if result is None:
            result = {
                "ok": False,
                "error": "No execution backend available.",
            }

        result = normalize_code_repl_model_result(result)

        # Detect newly generated files
        new_files: list[str] = []
        if params.capture_files:
            try:
                post_files = set(session_dir.iterdir())
                for f in post_files - pre_files:
                    if f.is_file():
                        new_files.append(str(f))
            except OSError as _exc:
                logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

        # Ground affect signals into Heartstone
        self._ground_affect(result.get("ok", False), result.get("stderr", ""))

        result["session_id"] = session_id
        result["working_directory"] = str(session_dir)
        if new_files:
            result["generated_files"] = new_files

        return result

    async def _execute_via_sandbox_runner(
        self, code: str, timeout_s: int, cwd: Path, library_root: str = ""
    ) -> dict[str, Any] | None:
        """Execute via core.sandbox.runner.run_untrusted (full isolation)."""
        try:
            from core.sandbox.runner import run_untrusted

            # Note: The restricted sandbox strips __import__ from builtins,
            # so we cannot prepend 'import os; os.chdir(...)' here.
            # The sandbox runs in a temporary directory by default.
            raw = await asyncio.to_thread(
                run_untrusted,
                code,
                timeout=timeout_s,
                mem_bytes=512 * 1024 * 1024,
                library_root=library_root,
            )

            if not isinstance(raw, dict):
                return {"ok": False, "error": f"Unexpected runner result: {raw}"}

            status = raw.get("status", "ok")
            stdout = raw.get("stdout", "")
            stderr = raw.get("stderr", "")
            returncode = raw.get("returncode")

            ok = status == "ok" and returncode == 0

            return {
                "ok": ok,
                "stdout": stdout,
                "stderr": stderr,
                "status": status,
                "returncode": returncode,
                "engine": "sandbox_runner",
                "repr": raw.get("repr", ""),
                "traceback": raw.get("traceback", ""),
                "summary": (
                    "Code executed successfully."
                    if ok
                    else f"Execution failed ({status})."
                ),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="fell back from sandbox_runner to alternative execution backend",
                extra={"engine": "sandbox_runner"},
            )
            logger.debug("sandbox_runner unavailable: %s", exc)
            return None

    async def _execute_via_sandbox_operator(
        self, code: str, timeout_s: int
    ) -> dict[str, Any] | None:
        """Execute via SandboxOperator (affect-grounded fallback)."""
        try:
            from core.actuators.sandbox_operator import SandboxOperator

            operator = SandboxOperator()
            raw = await asyncio.to_thread(
                operator.execute_synthesized_tool,
                code,
                float(timeout_s),
            )

            return {
                "ok": raw.get("success", False),
                "stdout": raw.get("stdout", ""),
                "stderr": raw.get("stderr", ""),
                "returncode": raw.get("exit_code"),
                "engine": "sandbox_operator",
                "summary": (
                    "Code executed via SandboxOperator."
                    if raw.get("success")
                    else "Execution failed."
                ),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="fell back from sandbox_operator to subprocess execution",
                extra={"engine": "sandbox_operator"},
            )
            logger.debug("SandboxOperator unavailable: %s", exc)
            return None

    async def _execute_via_subprocess(
        self, code: str, timeout_s: int, cwd: Path
    ) -> dict[str, Any] | None:
        """Execute via the canonical ActionExecutor subprocess pathway."""
        import sys

        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".py", dir=str(cwd))
            os.close(fd)
            await get_file_write_gateway().write_text_async(
                temp_path,
                code,
                encoding="utf-8",
                source="core.skills.code_repl.temp_script",
            )

            # Execute via ActionExecutor
            res = await ActionExecutor.execute(
                domain=ActionDomain.TOOL_EXECUTION,
                action_name="code_repl.run_script",
                params={
                    "argv": [sys.executable, temp_path],
                    "cwd": str(cwd),
                    "timeout": float(timeout_s),
                },
                source="code_repl",
            )

            # Map ActionExecutor result to expected REPL format
            return {
                "ok": res.get("ok", False),
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "returncode": res.get("exit_code", -1),
                "engine": "subprocess",
                "summary": res.get("error", "Code executed via ActionExecutor."),
            }

        except _REPL_RECOVERABLE_ERRORS as exc:
            _record_repl_degradation(
                exc,
                action="reported execution failure after all backends exhausted",
                extra={"engine": "subprocess"},
            )
            return {"ok": False, "error": f"Subprocess failed: {exc}", "engine": "subprocess"}
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError as _exc:
                    logger.debug("Suppressed %s in core.skills.code_repl: %s", type(_exc).__name__, _exc)

    def _ground_affect(self, success: bool, stderr: str) -> None:
        """Ground execution results into Heartstone Values."""
        try:
            from core.affect.heartstone_values import get_heartstone_values

            hv = get_heartstone_values()
            if success:
                hv.on_sandbox_success()
            else:
                hv.on_sandbox_failure(-1, stderr[:500])
        except _REPL_RECOVERABLE_ERRORS as exc:
            logger.debug("Affect grounding skipped: %s", exc)
