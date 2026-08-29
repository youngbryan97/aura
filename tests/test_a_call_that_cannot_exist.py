"""A call the library does not define is refused by reading, not by running.

The check earns its place only if it is decidable and quiet: it must name an
invented call exactly, and it must say nothing about code that is merely
unusual. A checker that flags working code is worse than none, because the
turn then argues with it instead of the problem.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from core.sandbox.api_check import check_code_against_library

LIBRARY = '''
"""A small double-entry ledger."""


class Unbalanced(Exception):
    pass


class Ledger:
    def __init__(self, name, opening=None):
        self.name = name
        self.entries = []

    def post(self, date, debit, credit, amount_cents):
        self.entries.append((date, debit, credit, amount_cents))

    def balance(self, account):
        return 0

    def trial_balance(self):
        return {}
'''


@pytest.fixture
def library(tmp_path: Path) -> str:
    (tmp_path / "ledgerkit.py").write_text(textwrap.dedent(LIBRARY))
    return str(tmp_path)


def _problems(code: str, library: str) -> list[str]:
    return [f.describe() for f in check_code_against_library(textwrap.dedent(code), library)]


def test_code_that_uses_the_real_api_is_left_alone(library: str) -> None:
    assert not _problems(
        """
        from ledgerkit import Ledger
        book = Ledger('acme')
        book.post('2026-03-01', debit='AR', credit='Revenue', amount_cents=25000)
        print(book.trial_balance())
        """,
        library,
    )


def test_a_method_the_class_does_not_have_is_named_with_the_ones_it_does(
    library: str,
) -> None:
    (problem,) = _problems(
        """
        from ledgerkit import Ledger
        book = Ledger('acme')
        book.add_entry('AR', 25000)
        """,
        library,
    )
    assert "book.add_entry" in problem
    # The answer, not just the refusal: the next attempt needs the real names.
    assert "post" in problem and "trial_balance" in problem


def test_a_name_the_module_does_not_export(library: str) -> None:
    (problem,) = _problems("from ledgerkit import LedgerBook", library)
    assert "LedgerBook" in problem and "Ledger" in problem


def test_a_keyword_the_function_never_took(library: str) -> None:
    (problem,) = _problems(
        """
        from ledgerkit import Ledger
        book = Ledger('acme')
        book.post('2026-03-01', debit='AR', credit='Revenue', amount=25000)
        """,
        library,
    )
    assert "amount=" in problem and "amount_cents" in problem


def test_more_positional_arguments_than_the_signature_takes(library: str) -> None:
    (problem,) = _problems(
        """
        from ledgerkit import Ledger
        book = Ledger('acme')
        book.balance('AR', 'Revenue', 'Cash')
        """,
        library,
    )
    assert "3 positional" in problem and "at most 1" in problem


def test_the_module_reached_through_import_is_checked_the_same_way(
    library: str,
) -> None:
    assert not _problems(
        """
        import ledgerkit
        book = ledgerkit.Ledger('acme')
        print(book.trial_balance())
        """,
        library,
    )
    (problem,) = _problems("import ledgerkit\nledgerkit.open_book('acme')", library)
    assert "ledgerkit.open_book" in problem


def test_a_name_that_is_assigned_twice_is_not_guessed_at(library: str) -> None:
    """The second assignment may be anything, and a wrong guess reads as a lie."""

    assert not _problems(
        """
        from ledgerkit import Ledger
        book = Ledger('acme')
        book = {'entries': []}
        book.setdefault('entries', [])
        """,
        library,
    )


def test_a_class_that_builds_its_own_attributes_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "dyn.py").write_text(
        textwrap.dedent(
            """
            class Bag:
                def __getattr__(self, name):
                    return lambda *a, **k: name
            """
        )
    )
    assert not _problems(
        """
        from dyn import Bag
        bag = Bag()
        bag.anything_at_all(1, 2, 3)
        """,
        str(tmp_path),
    )


def test_a_library_that_will_not_import_is_still_checked(tmp_path: Path) -> None:
    """Read, never imported — so a missing dependency costs nothing."""

    (tmp_path / "needs.py").write_text(
        textwrap.dedent(
            """
            import a_package_that_is_not_installed_anywhere


            class Thing:
                def run(self):
                    return 1
            """
        )
    )
    (problem,) = _problems("from needs import Thing\nThing().walk()", str(tmp_path))
    assert "Thing()" in problem
    assert "walk" in problem and "run" in problem


def test_the_library_is_never_imported(tmp_path: Path) -> None:
    """Importing it to inspect it would run untrusted code in this process."""

    marker = tmp_path / "ran.txt"
    (tmp_path / "loud.py").write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            Path({str(marker)!r}).write_text('the library executed')


            class Thing:
                def run(self):
                    return 1
            """
        )
    )
    _problems("from loud import Thing\nThing().run()", str(tmp_path))
    assert not marker.exists(), "checking the library executed it"


def test_no_library_and_broken_code_report_nothing(library: str) -> None:
    assert not _problems("from ledgerkit import Ledger", "/nowhere/at/all")
    assert not _problems("def (:", library)
    assert not _problems("", library)


def test_calls_into_another_library_are_not_this_checks_business(library: str) -> None:
    assert not _problems(
        """
        import json
        import ledgerkit
        print(json.dumps(ledgerkit.Ledger('acme').trial_balance()))
        """,
        library,
    )
