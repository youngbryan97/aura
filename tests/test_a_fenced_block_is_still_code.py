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

import ast

import pytest

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


class TestThePreambleIsFoundWhereverItIsWritten:
    """A careful model puts the import in a try, which is where it was hiding.

    LIVE 2026-08-29: the turn still died on "'sys' is not part of the library
    this sandbox was given" after the removal had shipped, because the scan
    read only the top level of the file and the model had written the preamble
    inside a try block — which is what anyone writes when an import might fail.
    """

    LIBRARY = "/tmp/ledgerkit"

    def _drop(self, code: str) -> str:
        from core.skills.code_repl import _without_a_path_preamble_for

        return _without_a_path_preamble_for(code, self.LIBRARY)

    @pytest.mark.parametrize(
        ("where", "code"),
        [
            (
                "a try block",
                "try:\n"
                "    import sys\n"
                "    sys.path.insert(0, '/tmp/ledgerkit')\n"
                "    from ledgerkit import Ledger\n"
                "except ImportError as exc:\n"
                "    print(exc)\n",
            ),
            (
                "an if",
                "import sys\n"
                "if True:\n"
                "    sys.path.insert(0, '/tmp/ledgerkit')\n"
                "from ledgerkit import Ledger\n",
            ),
            (
                "a function",
                "import sys\n"
                "def setup():\n"
                "    sys.path.append('/tmp/ledgerkit')\n"
                "setup()\n"
                "from ledgerkit import Ledger\n",
            ),
            (
                "a call across several lines",
                "import sys\n"
                "sys.path.insert(\n"
                "    0,\n"
                "    '/tmp/ledgerkit',\n"
                ")\n"
                "from ledgerkit import Ledger\n",
            ),
        ],
    )
    def test_it_is_found_and_the_code_still_parses(self, where: str, code: str) -> None:
        left = self._drop(code)
        assert "sys" not in left, f"the preamble survived in {where}"
        ast.parse(left)  # blanking the only statement in a block would break this
        assert "from ledgerkit import Ledger" in left

    def test_the_block_that_held_it_keeps_a_body(self) -> None:
        """pass, not a blank line: an if with no body is a syntax error."""

        left = self._drop(
            "import sys\nif True:\n    sys.path.insert(0, '/tmp/ledgerkit')\nprint(1)\n"
        )
        assert "    pass" in left


def test_the_dispatcher_outlasts_the_sandbox() -> None:
    """Two clocks on one execution, and the outer one gave up first.

    LIVE 2026-08-29: "Tool Result: code_repl in 120004ms", twice, on a sandbox
    allowed 180 seconds of wall clock for its 30-second computation budget. The
    dispatcher killed it at its own flat 120 and the tool returned nothing —
    not the output, not the timeout, not the traceback. The model saw an
    outcome of "unknown" and tried the same thing again.

    The declaration has to stay a literal, because the skill catalog discovers
    it by reading the source and a computed value makes the skill invisible. So
    the agreement is held here instead.
    """

    from core.sandbox.runner import wall_clock_allowance
    from core.skills.code_repl import _LONGEST_ACCEPTED_TIMEOUT_S, CodeREPLSkill

    assert CodeREPLSkill.timeout_seconds >= wall_clock_allowance(
        _LONGEST_ACCEPTED_TIMEOUT_S
    ), "the dispatcher gives up before the sandbox does"


def test_the_accepted_budget_and_the_declared_wait_are_one_number() -> None:
    from core.sandbox.runner import wall_clock_allowance
    from core.skills.code_repl import _LONGEST_ACCEPTED_TIMEOUT_S, CodeREPLSkill

    assert CodeREPLSkill.timeout_seconds == wall_clock_allowance(
        _LONGEST_ACCEPTED_TIMEOUT_S
    )
    assert CodeREPLSkill.input_model.model_fields["timeout"].metadata
