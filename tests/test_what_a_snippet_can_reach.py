"""Rate the snippet, not who typed it.

`run_code` was rated critical for any stateful run and high otherwise — a
statement about authorship, not about consequence. Every snippet then needed a
confirmation the turn has no way to ask for, so "read these docs and actually
use the library" could not be answered at all, while the same sandbox ran the
same import in 40ms with no effect on anything.

The code is a string before it runs, so its reach is a question with an answer.
These hold the answer, and hold the line for code that really does reach out.
"""

from __future__ import annotations

import pytest

from core.executive.execution_policy import classify_execution_risk
from core.skills.snippet_reach import reach_of

_USES_A_LOCAL_LIBRARY = (
    'import sys\n'
    'sys.path.insert(0, "/tmp/ledgerkit")\n'
    "from ledgerkit import Ledger\n"
    'led = Ledger("t")\n'
    'led.post("2026-08-27", "Accounts Receivable", "Revenue", 25000)\n'
    "print(led.trial_balance())\n"
)


def test_using_a_local_library_reaches_nothing() -> None:
    reach = reach_of(_USES_A_LOCAL_LIBRARY)
    assert reach.only_computes, reach.why()


def test_plain_arithmetic_reaches_nothing() -> None:
    assert reach_of("print(sum(range(10)))").only_computes


@pytest.mark.parametrize(
    ("code", "because"),
    [
        ('open("/tmp/x", "w").write("hi")', "it can write files"),
        ("import requests\nrequests.get('http://x')", "it can reach the network"),
        ("import subprocess\nsubprocess.run(['ls'])", "it can start a process"),
        ('__import__("os").system("ls")', "it resolves names at runtime"),
        ("import os\nos.remove('/tmp/x')", "it can write files"),
        ("import socket\ns = socket.socket()", "it can reach the network"),
        ("from pathlib import Path\nPath('/tmp/x').write_text('h')", "it can write files"),
    ],
)
def test_code_that_reaches_out_is_seen(code: str, because: str) -> None:
    reach = reach_of(code)
    assert not reach.only_computes
    assert because in reach.why()


def test_a_snippet_that_does_not_parse_is_not_pure() -> None:
    """Unreadable is unknown, and unknown is never treated as safe."""
    reach = reach_of("def (:")
    assert not reach.parses
    assert not reach.only_computes


def test_a_runtime_resolved_name_is_never_pure() -> None:
    """eval, exec and a computed getattr make the source say nothing."""
    for code in ('eval("1+1")', 'exec("x=1")', "getattr(o, name)()"):
        assert not reach_of(code).only_computes, code


def test_a_pure_snippet_does_not_need_a_confirmation() -> None:
    """The live blocker: medium runs, high stops to ask."""
    risk = classify_execution_risk(
        "run_code", {"code": _USES_A_LOCAL_LIBRARY, "stateful": True}, effect_scope="sandboxed_compute"
    )
    assert risk == "medium"


def test_statefulness_does_not_change_a_pure_snippet() -> None:
    """What a previous snippet left cannot act; only a new one can, and it is rated."""
    for stateful in (True, False):
        assert (
            classify_execution_risk(
                "run_code",
                {"code": _USES_A_LOCAL_LIBRARY, "stateful": stateful},
                effect_scope="sandboxed_compute",
            )
            == "medium"
        )


@pytest.mark.parametrize(
    "code",
    [
        "import subprocess\nsubprocess.run(['ls'])",
        'open("/tmp/x", "w").write("hi")',
        "import requests\nrequests.get('http://x')",
    ],
)
def test_code_that_acts_still_stops_to_ask(code: str) -> None:
    risk = classify_execution_risk(
        "run_code", {"code": code, "stateful": False}, effect_scope="sandboxed_compute"
    )
    assert risk in {"high", "critical"}, f"{code!r} was rated {risk}"


def test_no_snippet_at_all_is_the_worst_case() -> None:
    """Before there is code, there is nothing to read, so nothing is assumed."""
    assert classify_execution_risk("run_code", {}, effect_scope="sandboxed_compute") == "critical"


def test_the_tool_is_still_offered_before_the_snippet_exists() -> None:
    """Its risk depends on an argument that does not exist yet.

    Withholding it on the worst case means it can never be called, which is how
    the live turn ran out of tools and answered "I couldn't get to an answer".
    """
    from core.brain.inference_gate import _tools_within_reach

    kept, withheld = _tools_within_reach(
        {"run_code": 1}, frozenset({"status", "pure_compute", "read_only", "sandboxed_compute"})
    )
    assert "run_code" in kept
    assert not withheld
