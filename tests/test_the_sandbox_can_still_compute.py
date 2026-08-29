"""Naming a library must add to what can be imported, not replace Python.

LIVE 2026-08-29, two turns apart on one request: "'sys' is not part of the
library this sandbox was given", then "'json' is not part of the library this
sandbox was given". The second is a ledger script asking for the module every
ledger script asks for. Without a named library there was no __import__ at all,
so a sandbox described as being for "calculations, data processing,
prototyping" could not import math — and the model could only find that out by
spending a turn on it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from core.sandbox.runner import run_untrusted

pytestmark = pytest.mark.unit

LIBRARY = '''
class Ledger:
    def __init__(self, name):
        self.name = name
        self.entries = []

    def post(self, date, debit, credit, amount_cents):
        self.entries.append((date, debit, credit, amount_cents))

    def trial_balance(self):
        totals = {}
        for _date, debit, credit, amount in self.entries:
            totals[debit] = totals.get(debit, 0) + amount
            totals[credit] = totals.get(credit, 0) - amount
        return totals
'''


@pytest.fixture
def library(tmp_path: Path) -> str:
    (tmp_path / "ledgerkit.py").write_text(textwrap.dedent(LIBRARY))
    return str(tmp_path)


def _run(code: str, library_root: str = "") -> dict:
    return run_untrusted(textwrap.dedent(code), timeout=10, library_root=library_root)


def test_arithmetic_and_data_modules_import() -> None:
    result = _run(
        """
        import json
        import math
        print(json.dumps({"root": round(math.sqrt(2), 3)}))
        """
    )
    assert result.get("status") == "ok", result
    assert '"root": 1.414' in result["stdout"]


def test_a_named_library_composes_with_them(library: str) -> None:
    """The case the live turn needed: a library and json in one script."""

    result = _run(
        """
        import json
        from ledgerkit import Ledger

        book = Ledger("acme")
        book.post("2026-03-01", debit="AR", credit="Revenue", amount_cents=25000)
        print(json.dumps(book.trial_balance()))
        """,
        library,
    )
    assert result.get("status") == "ok", result
    assert result["stdout"].strip() == '{"AR": 25000, "Revenue": -25000}'


@pytest.mark.parametrize("banned", ["os", "sys", "subprocess", "socket", "pathlib", "io"])
def test_nothing_that_reaches_outside_the_process_is_importable(banned: str) -> None:
    """io is absent too, because io.open is a file open by another name."""

    result = _run(f"import {banned}\nprint({banned})\n")
    assert result.get("status") != "ok", f"{banned} was importable"


def test_the_refusal_says_what_is_available() -> None:
    """The alternative was the model finding out one turn at a time."""

    result = _run("import requests\n")
    said = " ".join(str(result.get(k) or "") for k in ("error", "stderr", "traceback"))
    assert "json" in said and "math" in said
    assert "no filesystem, network or process access" in said


def test_the_sandbox_still_refuses_the_dangerous_builtins() -> None:
    for attempt in ("open('/etc/hosts')", "eval('1+1')", "exec('x=1')", "globals()"):
        result = _run(f"print({attempt})\n")
        assert result.get("status") != "ok", attempt
