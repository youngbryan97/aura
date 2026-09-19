"""The seam analyser must be right about the one seam we actually cut.

Its whole value is that someone trusts it before moving 400 lines of the
hardest code in the repository. So it is checked against the extraction that
was already performed and validated by hand: the chat preflight, whose contract
was six values in, six out, exactly one early return and seven awaits.

The negative results matter as much as the positive ones. "This function has no
clean seam" is the answer that stops a cut that would silently drop a branch,
so the blocked cases are asserted too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.find_extraction_seam import Seam, analyse

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


def test_it_finds_the_seam_inside_the_function_we_already_cut():
    """_run_chat_preflight is one big try; its body should read as one seam.

    As a SHARE of the function, not as a line count. This asked for more
    than 300 lines and the seam is 278 of a 345-line function — the
    function got smaller, which is the work this tool exists to support,
    and the assertion was going to fail on every cut that succeeded.
    """
    import ast

    path = ROOT / "interface/routes/chat_preflight.py"
    seams = analyse(path, "_run_chat_preflight")
    assert seams, "no seam found in a function that is a single try block"
    biggest = max(seams, key=lambda s: s.lines)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_run_chat_preflight"
    )
    whole = (function.end_lineno or function.lineno) - function.lineno + 1
    assert biggest.lines > whole / 2, (biggest.lines, whole)


def test_multiple_early_returns_block_a_seam():
    seam = Seam(lineno=1, end_lineno=100, kind="Try", returns=[5, 20, 40])
    assert seam.safe is False
    assert any("early returns" in b for b in seam.blockers)


def test_a_wide_interface_blocks_a_seam():
    """A block reading twenty enclosing locals is a paragraph, not a unit."""
    seam = Seam(lineno=1, end_lineno=100, kind="Try", reads=[f"v{i}" for i in range(20)])
    assert seam.safe is False
    assert any("not a unit" in b for b in seam.blockers)


def test_a_generator_body_is_never_a_seam():
    seam = Seam(lineno=1, end_lineno=100, kind="Try", yields=1)
    assert seam.safe is False


def test_a_narrow_single_return_block_is_safe():
    seam = Seam(
        lineno=1, end_lineno=400, kind="Try",
        reads=["body", "request"], escapes=["status"], returns=[7], awaits=3,
    )
    assert seam.safe is True
    assert seam.blockers == []


def test_conditional_escapes_are_counted_separately():
    """The trap: defaulting a conditionally-bound name changes behaviour."""
    seam = Seam(
        lineno=1, end_lineno=200, kind="Try",
        escapes=["a", "b"], conditional_escapes=["b"],
    )
    assert seam.conditional_escapes == ["b"]


def test_the_biggest_function_reports_its_core_as_unextractable():
    """_latent_episode's bulk reads ~69 enclosing names and has 5 returns.

    Pinned because the honest answer for that function is that this method does
    not reach it — claiming otherwise would invite exactly the cut that drops a
    branch.
    """
    path = ROOT / "core/brain/llm/latent_cortex/engine.py"
    if not path.is_file():
        pytest.skip("latent cortex engine not present")
    seams = analyse(path, "_latent_episode")
    huge = [s for s in seams if s.lines > 1000]
    assert huge, "expected a very large block in the biggest function"
    assert all(not s.safe for s in huge), (
        "a 1000+ line block with many returns was reported safe to cut"
    )


def test_analysis_is_read_only():
    import inspect

    import tools.find_extraction_seam as module

    source = inspect.getsource(module)
    assert "write_text" not in source, "the analyser must not modify source"
