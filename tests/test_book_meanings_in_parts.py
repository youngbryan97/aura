"""A book's meanings built in parts join back into the whole book."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from core.learning.procedures_from_solved_examples import ProcedureBook, Signature

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "give_book_meanings.py"
_spec = importlib.util.spec_from_file_location("give_book_meanings", _TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def test_parts_cover_every_kind_once_and_merge_back():
    kinds = ["bbeh_a", "dm_a", "dm_b", "dm_c", "dm_d", "lcb", "trip"]
    book = ProcedureBook({k: Signature({k: 1.0}, 0.5) for k in kinds}, {k: [f"# {k}"] for k in kinds},
                         {k: 3 for k in kinds}, posing={"trip": "Worked.\n\n"})
    parts = [tool.book_of(book, tool.part_of(sorted(kinds), i, 3)) for i in range(3)]
    assert sorted(k for part in parts for k in part.signatures) == sorted(kinds)
    # Kinds of one source sit together when sorted; every part gets some of them.
    assert all(any(k.startswith("dm_") for k in part.signatures) for part in parts)
    whole = ProcedureBook.merged(parts)
    assert whole.to_json() == book.to_json()


def test_a_part_outside_the_count_is_refused():
    import pytest

    with pytest.raises(ValueError):
        tool.part_of(["a", "b"], 3, 3)
