"""A model writing code writes a fenced block.

That is how code is written everywhere it has ever seen it. Sent straight to
the interpreter the fence is a syntax error, and the turn spends an attempt
learning that — on a machine where an attempt is a full generation from a 27B.

The reconstruction lab already solved this. Its extractor takes the fenced
body, and takes it even when the closing fence is missing because the
generation ran out of room. The chat path uses the same one rather than a
second that will drift from it.
"""

from __future__ import annotations

from core.skills.code_repl import _the_code_inside_any_fence


def test_a_fenced_block_becomes_the_code() -> None:
    assert _the_code_inside_any_fence("```python\nprint(6*7)\n```") == "print(6*7)"


def test_an_unclosed_fence_is_still_recovered() -> None:
    """Truncation is ordinary; throwing the answer away for it is not."""

    got = _the_code_inside_any_fence("```python\nfrom ledgerkit import Ledger\nL = Ledger('a')")
    assert "from ledgerkit import Ledger" in got
    assert "```" not in got


def test_plain_code_reaches_the_sandbox_byte_for_byte() -> None:
    """The extractor also trims, so it is consulted only where a fence is."""

    import inspect

    from core.skills import code_repl

    source = inspect.getsource(code_repl)
    marker = "unfenced = _the_code_inside_any_fence(code)"
    assert marker in source
    guard = source[source.index(marker) : source.index(marker) + 220]
    assert '"```" in code' in guard, guard


def test_the_sandbox_runs_what_comes_out() -> None:
    """End to end: fenced in, answer out."""

    from core.sandbox.runner import run_untrusted

    out = run_untrusted(
        _the_code_inside_any_fence("```python\nprint(6 * 7)\n```"),
        timeout=6,
        mem_bytes=256 * 1024 * 1024,
    )
    assert out["status"] == "ok", out
    assert out["stdout"].strip() == "42"


def test_a_fence_reaching_the_sandbox_unwrapped_would_fail() -> None:
    """Why this matters, stated as the failure it prevents."""

    from core.sandbox.runner import run_untrusted

    out = run_untrusted(
        "```python\nprint(6 * 7)\n```", timeout=6, mem_bytes=256 * 1024 * 1024
    )
    assert out["status"] != "ok"
