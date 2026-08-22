"""The extractor refuses the cut that broke the chat route.

`tools/extract_seam.py` moved a hundred lines out of `_api_chat_turn` and the
serving path started raising `UnboundLocalError` on the first turn that took a
particular branch. The block assigned `hard_final_quality_failed` inside an
`if`, the tool asked whether the name was bound anywhere earlier in the
function, found an assignment in a branch that had not necessarily run, and
called the cut safe. Extracted, the helper reached its `return` with the name
never assigned.

The question was wrong, not the answer. What matters is whether the BLOCK
binds the name on every path through itself. These check the corrected
analysis, and the last one runs an extraction end to end and compares the two
modules' behaviour rather than their text.
"""
from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEAM_TOOLS = ROOT / "tools" / "find_extraction_seam.py"
EXTRACTOR = ROOT / "tools" / "extract_seam.py"


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tools():
    return _load(SEAM_TOOLS, "_aura_seam_tools_test")


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


# ─────────────────────────────────────────────── must-bind


def test_a_branch_without_an_else_binds_nothing_for_certain(tools):
    names = tools.must_bind(_statements("if flag:\n    result = 1\n"))
    assert "result" not in names


def test_both_branches_binding_it_is_certain(tools):
    names = tools.must_bind(_statements("if flag:\n    result = 1\nelse:\n    result = 2\n"))
    assert "result" in names


def test_a_branch_that_returns_does_not_weaken_the_other(tools):
    names = tools.must_bind(
        _statements("if flag:\n    return None\nelse:\n    result = 2\n")
    )
    assert "result" in names


def test_a_loop_body_binds_nothing_for_certain(tools):
    names = tools.must_bind(_statements("for item in things:\n    result = item\n"))
    assert "result" not in names


def test_a_try_body_binds_nothing_for_certain(tools):
    """The body can stop at any statement, and the handler may not assign."""
    names = tools.must_bind(
        _statements("try:\n    result = risky()\nexcept ValueError:\n    pass\n")
    )
    assert "result" not in names


def test_a_finally_block_is_certain(tools):
    names = tools.must_bind(
        _statements("try:\n    risky()\nfinally:\n    result = 1\n")
    )
    assert "result" in names


# ─────────────────────────────────────────────── free variables


def test_a_tuple_assignment_reads_before_it_binds(tools):
    """The shape that made the old line-number analysis wrong.

    The targets sit above the call that reads them, so by line number the
    names look local; at runtime the value is evaluated first.
    """
    free, bound = tools.free_variables(
        _statements(
            "(\n"
            "    first,\n"
            "    second,\n"
            ") = combine(first=first, second=second)\n"
        )
    )
    assert {"first", "second"} <= free
    assert {"first", "second"} <= bound


def test_a_nested_function_does_not_export_its_parameters(tools):
    free, _bound = tools.free_variables(
        _statements("def inner(value):\n    return value + outer\n")
    )
    assert "outer" in free
    assert "value" not in free


# ─────────────────────────────────────────────── the extractor


def _run_extractor(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, str(EXTRACTOR), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


SAMPLE = '''"""Sample."""


def serve(flag, text):
    status = "start"
    if flag:
        marker = "seen"
        text = text.upper()
    if marker if flag else True:
        pass
    return status, text, marker
'''

SAFE_SAMPLE = '''"""Sample."""


def serve(flag, text):
    status = "start"
    if flag:
        text = text.upper()
        status = "upper"
    else:
        text = text.lower()
        status = "lower"
    return status, text
'''


def test_a_conditionally_bound_escape_is_refused(tmp_path, monkeypatch):
    module = tmp_path / "sample.py"
    module.write_text(SAMPLE, encoding="utf-8")
    monkeypatch.chdir(ROOT)

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 6, 8, "_helper", is_async=False, apply=False
    )
    assert code == 1, "the tool accepted a cut that cannot preserve behaviour"


def test_an_extraction_preserves_behaviour(tmp_path):
    """Run both versions and compare the answers, not the text."""
    module = tmp_path / "sample_safe.py"
    module.write_text(SAFE_SAMPLE, encoding="utf-8")

    before = _load(module, "_aura_sample_before")
    original = [before.serve(True, "Ab"), before.serve(False, "Ab")]

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 6, 11, "_branch", is_async=False, apply=True
    )
    assert code == 0, "the tool refused a cut that is safe"

    del sys.modules["_aura_sample_before"]
    after = _load(module, "_aura_sample_after")
    assert [after.serve(True, "Ab"), after.serve(False, "Ab")] == original
    assert "_branch" in module.read_text("utf-8")


def test_a_range_that_starts_inside_a_statement_is_refused(tmp_path):
    module = tmp_path / "sample_partial.py"
    module.write_text(SAFE_SAMPLE, encoding="utf-8")

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 7, 11, "_partial", is_async=False, apply=False
    )
    assert code == 2


# ─────────────────────────────────────────────── blocks that return early

EARLY_SAMPLE = '''"""Sample."""

from typing import Any


def serve(flag, text, status):
    text = text.strip()
    if flag:
        status = "refused"
        return status, text
    text = text.upper()
    status = "served"
    return status, text
'''


def test_a_block_that_returns_early_is_moved_into_a_nested_function(tmp_path):
    """The transform that unlocks the rest of the debt.

    A block that returns out of its middle cannot become the tail of a
    function: the caller has to tell "the block returned this" from "the block
    finished". A nested function makes its own `return` the signal and leaves
    the block text untouched.
    """
    module = tmp_path / "sample_early.py"
    module.write_text(EARLY_SAMPLE, encoding="utf-8")

    before = _load(module, "_aura_early_before")
    expected = [
        before.serve(True, " ab ", "start"),
        before.serve(False, " ab ", "start"),
    ]

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 8, 10, "_refusal", is_async=False, apply=True
    )
    assert code == 0, "the tool refused a cut it can make safely"

    del sys.modules["_aura_early_before"]
    after = _load(module, "_aura_early_after")
    assert [
        after.serve(True, " ab ", "start"),
        after.serve(False, " ab ", "start"),
    ] == expected

    body = module.read_text("utf-8")
    assert "_SEAM_FELL_THROUGH = object()" in body
    assert "nonlocal" in body


ESCAPING_EARLY_SAMPLE = '''"""Sample."""

from typing import Any


def serve(flag):
    if flag:
        note = "seen"
        return note
    note = "unseen"
    return note
'''


def test_an_early_return_whose_escape_is_not_an_input_is_refused(tmp_path):
    """`note` leaves the block without entering it.

    On the early path the helper would hand back a name it never set, which is
    the class of defect that put UnboundLocalError on the serving path.
    """
    module = tmp_path / "sample_escaping.py"
    module.write_text(ESCAPING_EARLY_SAMPLE, encoding="utf-8")

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 7, 9, "_note", is_async=False, apply=False
    )
    assert code == 1


CONDITIONAL_INPUT_SAMPLE = '''"""Sample."""

from typing import Any


def serve(flag):
    if flag:
        contract = {"ok": True}
    if flag:
        note = contract["ok"]
        return note
    return None
'''


def test_an_input_the_caller_may_not_hold_is_refused(tmp_path):
    """The second regression, as a test.

    `contract` is assigned inside one guard and read inside the next. The
    block's read was safe because the guard was the same; as an argument it is
    evaluated on every call, including the one that took neither branch. The
    original raised nothing and the extracted version raised
    UnboundLocalError before the helper body ran.
    """
    module = tmp_path / "sample_conditional_input.py"
    module.write_text(CONDITIONAL_INPUT_SAMPLE, encoding="utf-8")

    extractor = _load(EXTRACTOR, "_aura_extract_seam_test")
    code = extractor.extract(
        module, "serve", 9, 11, "_read_contract", is_async=False, apply=False
    )
    assert code == 1
