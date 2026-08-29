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


class TestThePathPreambleTheSandboxAlreadyDid:
    """Importing from a directory means sys.path — except here, where it is done."""

    LIBRARY = "/tmp/ledgerkit"

    def _drop(self, code: str) -> str:
        from core.skills.code_repl import _without_a_path_preamble_for

        return _without_a_path_preamble_for(code, self.LIBRARY)

    def test_the_preamble_the_runner_already_performed_is_dropped(self) -> None:
        left = self._drop(
            "import sys\n"
            f"sys.path.insert(0, '{self.LIBRARY}')\n"
            "from ledgerkit import Ledger\n"
        )
        assert "sys" not in left
        assert "from ledgerkit import Ledger" in left

    def test_append_is_the_same_no_op(self) -> None:
        assert "sys" not in self._drop(
            f"import sys\nsys.path.append('{self.LIBRARY}')\nimport ledgerkit\n"
        )

    def test_a_real_use_of_sys_is_left_to_be_refused(self) -> None:
        left = self._drop(
            "import sys\n"
            f"sys.path.insert(0, '{self.LIBRARY}')\n"
            "print(sys.version)\n"
        )
        assert "import sys" in left, "the name is still needed and still banned"

    def test_any_directory_at_all_since_the_call_can_do_nothing_here(self) -> None:
        """Imports come from the library the runner was handed, not sys.path.

        The model does not always pass library_path, and the line it dies on
        is the same line either way — so the test is whether the call could
        have an effect, not whether it names the directory we happen to know.
        """

        assert "sys" not in self._drop(
            "import sys\nsys.path.insert(0, '/somewhere/else')\nimport ledgerkit\n"
        )

    def test_it_works_with_no_library_named_at_all(self) -> None:
        from core.skills.code_repl import _without_a_path_preamble_for

        left = _without_a_path_preamble_for(
            "import sys\nsys.path.insert(0, '/anywhere')\nfrom ledgerkit import Ledger\n",
            "",
        )
        assert "sys" not in left
        assert "from ledgerkit import Ledger" in left

    def test_line_numbers_survive_so_a_traceback_still_points_at_the_code(self) -> None:
        left = self._drop(
            "import sys\n"
            f"sys.path.insert(0, '{self.LIBRARY}')\n"
            "from ledgerkit import Ledger\n"
            "raise ValueError('here')\n"
        )
        assert left.splitlines()[3] == "raise ValueError('here')"

    def test_code_that_never_mentions_sys_is_returned_unchanged(self) -> None:
        code = "from ledgerkit import Ledger\nprint(Ledger('a'))\n"
        assert self._drop(code) == code
