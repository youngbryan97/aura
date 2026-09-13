from __future__ import annotations

import ast

from tools.production_surface_lint import (
    EXEMPT_FILES,
    AstLinter,
    hardcoded_local_path_findings,
)


def _findings(source: str) -> list[str]:
    visitor = AstLinter("core/example.py")
    visitor.visit(ast.parse(source))
    return [finding.kind for finding in visitor.findings]


def _findings_for_path(source: str, rel_path: str) -> list[str]:
    visitor = AstLinter(rel_path)
    visitor.visit(ast.parse(source))
    return [finding.kind for finding in visitor.findings]


def test_lint_blocks_asyncio_subprocess_exec() -> None:
    kinds = _findings(
        """
import asyncio

async def main():
    await asyncio.create_subprocess_exec("python", "-V")
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_raw_asyncio_create_task_in_production_file() -> None:
    kinds = _findings(
        """
import asyncio

async def main(coro):
    asyncio.create_task(coro)
"""
    )

    assert "raw_async_task" in kinds


def test_lint_allows_raw_asyncio_create_task_only_in_canonical_task_owner() -> None:
    kinds = _findings_for_path(
        """
import asyncio

def create(coro):
    return asyncio.create_task(coro)
""",
        "core/runtime/task_ownership.py",
    )

    assert "raw_async_task" not in kinds


def test_lint_blocks_event_loop_create_task_in_production_file() -> None:
    kinds = _findings(
        """
async def main(loop, coro):
    loop.create_task(coro)
"""
    )

    assert "raw_async_task" in kinds


def test_lint_allows_canonical_tracker_create_task() -> None:
    kinds = _findings(
        """
async def main(task_tracker, coro):
    task_tracker.create_task(coro, name="owned")
"""
    )

    assert "raw_async_task" not in kinds


def test_lint_blocks_asyncio_subprocess_shell() -> None:
    kinds = _findings(
        """
import asyncio

async def main():
    await asyncio.create_subprocess_shell("python -V")
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_subprocess_callable_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
import subprocess

async def main():
    await asyncio.to_thread(subprocess.run, ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_network_callable_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
import requests

async def main():
    await asyncio.to_thread(requests.get, "https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_network_callable_import_alias_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
from requests import get as http_get

async def main():
    await asyncio.to_thread(http_get, "https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_urllib_request_module_alias() -> None:
    kinds = _findings(
        """
import urllib.request as ureq

def main():
    return ureq.urlopen("https://example.com")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_urllib_urlretrieve_alias() -> None:
    kinds = _findings(
        """
from urllib.request import urlretrieve

def main():
    urlretrieve("https://example.com/model.onnx", "model.onnx")
"""
    )

    assert "unapproved_direct_network" in kinds


def test_lint_blocks_subprocess_import_alias_wrapped_in_to_thread() -> None:
    kinds = _findings(
        """
import asyncio
from subprocess import run as proc_run

async def main():
    await asyncio.to_thread(proc_run, ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_os_spawn_family() -> None:
    kinds = _findings(
        """
import os

def main():
    return os.posix_spawnp("python", ["python", "-V"], os.environ.copy())
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_os_spawn_alias() -> None:
    kinds = _findings(
        """
from os import spawnvp

def main():
    return spawnvp(0, "python", ["python", "-V"])
"""
    )

    assert "unapproved_direct_subprocess" in kinds


def test_lint_blocks_path_constructor_write_text() -> None:
    kinds = _findings(
        """
from pathlib import Path

def save():
    Path("runtime.txt").write_text("unsafe")
"""
    )

    assert "unapproved_direct_file_write" in kinds


def test_lint_allows_wave_open_on_bytesio_buffer() -> None:
    kinds = _findings(
        """
import io
import wave

def synthesize():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
"""
    )

    assert "unapproved_direct_file_write" not in kinds


def test_lint_allows_file_write_gateway_factory_write_text() -> None:
    kinds = _findings(
        """
from core.runtime.file_write_gateway import get_file_write_gateway

def save(path):
    get_file_write_gateway().write_text(path, "safe", source="test")
"""
    )

    assert "unapproved_direct_file_write" not in kinds


def test_lint_has_no_audited_production_exemptions() -> None:
    assert EXEMPT_FILES == {}


def test_lint_allows_broad_exception_returned_as_failure_evidence() -> None:
    kinds = _findings(
        '''
def probe(operation):
    try:
        return operation()
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
'''
    )

    assert "swallowed_broad_exception" not in kinds


def test_lint_blocks_broad_exception_collapsed_to_boolean() -> None:
    kinds = _findings(
        '''
def probe(operation):
    try:
        return operation()
    except Exception:
        return False
'''
    )

    assert "swallowed_broad_exception" in kinds


def test_hardcoded_path_lint_ignores_comments_and_docstrings() -> None:
    tree = ast.parse(
        '''
"""Reject examples such as /Users/attacker/project and /tmp/attacker/git."""

def validate(path):
    """A caller may submit /home/example/state; it must not be trusted."""
    # Never assume /Users/bryan exists on another host.
    return path
'''
    )

    assert hardcoded_local_path_findings(tree, "core/example.py") == []


def test_hardcoded_path_lint_blocks_runtime_string_constants() -> None:
    tree = ast.parse(
        '''
STATE_ROOT = "/Users/bryan/.aura/state"

def temporary_path(name):
    return f"/tmp/aura/{name}"
'''
    )

    findings = hardcoded_local_path_findings(tree, "core/example.py")
    assert [finding.line for finding in findings] == [2, 5]
    assert {finding.kind for finding in findings} == {"hardcoded_local_path"}


def test_lint_blocks_builtin_dynamic_code_execution() -> None:
    kinds = _findings(
        """
def run(source):
    code = compile(source, "<dynamic>", "exec")
    exec(code, {})
"""
    )

    assert kinds.count("raw_dynamic_code") == 2


def test_lint_blocks_builtins_dynamic_code_execution() -> None:
    kinds = _findings(
        """
import builtins

def run(source):
    return builtins.compile(source, "<dynamic>", "exec")
"""
    )

    assert "raw_dynamic_code" in kinds


def test_lint_allows_non_dynamic_compile_method_calls() -> None:
    kinds = _findings(
        """
class Compiler:
    def compile(self, value):
        return value

def run(compiler, observation):
    return compiler.compile(observation)
"""
    )

    assert "raw_dynamic_code" not in kinds


def test_lint_allows_dynamic_code_only_in_canonical_gateway() -> None:
    kinds = _findings_for_path(
        """
def run(code_object, globals_dict):
    exec(code_object, globals_dict)
""",
        "core/runtime/dynamic_execution_gateway.py",
    )

    assert "raw_dynamic_code" not in kinds


def test_a_bare_call_to_a_function_this_module_defines_is_not_a_write():
    """`write_text(...)` with no receiver names whatever the module defines.

    core/subject/archive.py defines a write_text that routes through the file
    write gateway inside a governed scope, and all three of its call sites
    were reported as unapproved direct writes. Whether that helper writes
    directly is decided where it is defined, which this same scan reads.
    """
    import ast
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
    from production_surface_lint import AstLinter

    def kinds(source: str) -> list[str]:
        tree = ast.parse(source)
        linter = AstLinter("core/probe.py")
        linter.functions_defined_here = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        linter.visit(tree)
        return [finding.kind for finding in linter.findings]

    defined_here = "def write_text(path, body):\n    pass\n\n\ndef go(p):\n    write_text(p, 'x')\n"
    assert kinds(defined_here) == []

    # The two readings it must still catch: a name this module does not
    # define, and the Path method itself.
    linter_sees_a_stranger = "def go(p):\n    write_text(p, 'x')\n"
    tree = ast.parse(linter_sees_a_stranger)
    linter = AstLinter("core/probe.py")
    linter.functions_defined_here = set()
    linter.visit(tree)
    assert [f.kind for f in linter.findings] == ["unapproved_direct_file_write"]

    assert kinds("def go(p):\n    p.write_text('x')\n") == ["unapproved_direct_file_write"]


def _lint_kinds(source: str) -> list[str]:
    import ast
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
    from production_surface_lint import AstLinter

    tree = ast.parse(source)
    linter = AstLinter("core/probe.py")
    linter._source_lines = source.splitlines()
    linter.functions_defined_here = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    linter.visit(tree)
    return [finding.kind for finding in linter.findings]


def test_a_write_into_a_directory_this_scope_made_is_not_durable_state():
    """The gateway governs what is still there afterwards.

    core/sandbox/static_check.py writes a candidate into a TemporaryDirectory
    so ruff can read it, and the file is gone before the function returns.
    """
    scratch = (
        "import tempfile\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def go(code):\n"
        "    with tempfile.TemporaryDirectory() as directory:\n"
        "        written = Path(directory) / 'candidate.py'\n"
        "        written.write_text(code)\n"
    )
    assert _lint_kinds(scratch) == []

    durable = (
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def go(code):\n"
        "    Path('/somewhere/real').write_text(code)\n"
    )
    assert _lint_kinds(durable) == ["unapproved_direct_file_write"]


def test_the_reviewed_markers_are_the_ones_the_enterprise_gate_reads():
    """One vocabulary. A line reviewed once should not need reviewing again
    in a second spelling."""
    reviewed = (
        "def go(code):\n"
        "    # This is the parser, not an interpreter.\n"
        "    compile(code, '<the code>', 'exec')  # noqa: S102\n"
    )
    assert _lint_kinds(reviewed) == []

    unreviewed = "def go(code):\n    compile(code, '<the code>', 'exec')\n"
    assert _lint_kinds(unreviewed) == ["raw_dynamic_code"]


def test_a_raw_task_that_says_why_is_not_reported():
    """Two sites demonstrate what create_task does to a trace context, and one
    is the cancellation primitive. The tracker inside either measures itself."""
    deliberate = (
        "import asyncio\n"
        "\n"
        "\n"
        "async def go():\n"
        "    # Raw task, deliberately: the context copy IS the subject here.\n"
        "    return await asyncio.create_task(work())\n"
    )
    assert _lint_kinds(deliberate) == []

    ordinary = (
        "import asyncio\n"
        "\n"
        "\n"
        "async def go():\n"
        "    return await asyncio.create_task(work())\n"
    )
    assert _lint_kinds(ordinary) == ["raw_async_task"]
