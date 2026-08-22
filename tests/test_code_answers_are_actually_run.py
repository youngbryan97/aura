"""Aura can run Python. An answer's own examples should be run, not read.

The code verifier executed a candidate only when it carried module-level
asserts. A doctest is the most common way a Python answer states what its code
returns, and it is the same kind of executable self-claim — so an answer whose
own examples were WRONG still scored as verified, because "compiles clean" was
the whole check.

Two traps in the harness, both of which made a wrong example pass silently:
doctest.testmod() with no argument tests the module named __main__, which is
whatever happens to be running the code rather than the candidate; and
DocTestFinder skips any object whose __module__ does not match the module it
was handed. Either one yields zero examples attempted and zero failed.

Separately, a code question asked the ordinary way never reached the verifier
at all: "how would you reverse a string in python?" matched "how would you"
and classified as PLANNING, because the code pattern knew the words "function"
and "class" but no language names.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

from core.brain.reasoning_amplifier_v2 import classify_task_type, is_amplifiable
from core.brain.verifiers.code_engine import (
    doctest_harness,
    has_doctest_examples,
    has_module_level_asserts,
)

_CORRECT = 'def rev(s):\n    """Reverse.\n\n    >>> rev("abc")\n    \'cba\'\n    """\n    return s[::-1]\n'
_WRONG = 'def rev(s):\n    """Reverse.\n\n    >>> rev("abc")\n    \'abc\'\n    """\n    return s[::-1]\n'
_NO_EXAMPLES = 'def rev(s):\n    """Reverse a string."""\n    return s[::-1]\n'


def _runs_clean(code: str) -> bool:
    """Run the harness the way the sandbox does: as a script."""
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "candidate.py"
        path.write_text(doctest_harness(code), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=60
        )
    return done.returncode == 0


def test_a_doctest_is_recognised_as_an_executable_claim() -> None:
    assert has_doctest_examples(_CORRECT)
    assert not has_doctest_examples(_NO_EXAMPLES)
    assert not has_module_level_asserts(_CORRECT)


def test_code_that_matches_its_examples_passes() -> None:
    assert _runs_clean(_CORRECT)


def test_code_that_contradicts_its_own_example_fails() -> None:
    """The whole point: the answer said 'abc' and the code returns 'cba'."""
    assert not _runs_clean(_WRONG)


def test_the_harness_finds_examples_under_exec_too() -> None:
    """testmod() would test the runner instead, and report zero examples."""
    with pytest.raises(AssertionError):
        # Both calls carry the marker: the gate reads it per line, and the
        # compile sits on its own line here.
        exec(  # noqa: S102 - exercising the harness is the test
            compile(  # noqa: S102 - the harness's own output, in a test
                doctest_harness(_WRONG), "<candidate>", "exec"
            ),
            {"__name__": "__main__"},
        )


def test_a_block_with_no_examples_is_not_failed_for_it() -> None:
    assert _runs_clean(_NO_EXAMPLES)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("how would you reverse a string in python?", "code"),
        ("write a python function that returns the nth fibonacci number", "code"),
        ("show me a regex that matches an email", "code"),
        ("what is 7919 * 6367?", "math"),
        ("where is the retry logic implemented in this codebase", "repo_audit"),
        ("who wrote Hamlet?", "factual"),
        ("when did the berlin wall fall?", "factual"),
        ("what's a good plan for my week?", "planning"),
    ],
)
def test_a_question_reaches_the_verifier_that_can_check_it(
    question: str, expected: str
) -> None:
    assert classify_task_type(question) == expected


def test_a_code_question_asked_naturally_is_amplified() -> None:
    assert is_amplifiable("how would you reverse a string in python?") == "code"
