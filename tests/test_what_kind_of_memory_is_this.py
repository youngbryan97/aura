"""A taxonomy nobody uses is not a taxonomy."""
from __future__ import annotations

import pytest

from core.memory.what_kind_of_memory_is_this import (
    ARecord,
    MemoryKind,
    how_the_kinds_stand,
    kinds_written_in_the_tree,
    strings_nothing_maps,
    the_spellings,
    what_kind,
)


def test_the_declared_enum_had_one_user_and_it_was_itself() -> None:
    """The finding this module exists for, kept checkable."""
    import pathlib

    users = [
        path.name
        for path in pathlib.Path("core").rglob("*.py")
        if "MemoryType" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert "base.py" in users
    assert len(users) <= 2, (
        f"MemoryType is now imported by {users}; if it is genuinely canonical, "
        "this module should defer to it rather than sit beside it"
    )


@pytest.mark.parametrize(
    ("written", "means"),
    [
        ("episode", MemoryKind.EPISODIC),
        ("episodic", MemoryKind.EPISODIC),
        ("skill", MemoryKind.SKILL),
        ("skills", MemoryKind.SKILL),
        ("goal", MemoryKind.GOAL),
        ("goals", MemoryKind.GOAL),
        ("unanswered", MemoryKind.QUESTION),
        ("reflection", MemoryKind.PRIVATE_THOUGHT),
        ("EPISODE", MemoryKind.EPISODIC),
        ("private thought", MemoryKind.PRIVATE_THOUGHT),
        ("private-thought", MemoryKind.PRIVATE_THOUGHT),
    ],
)
def test_the_spellings_that_meant_the_same_thing_now_do(written, means) -> None:
    assert what_kind(written) is means


def test_an_unplaceable_string_is_none_rather_than_a_guess() -> None:
    """Filed under the wrong kind is worse than filed under none."""
    assert what_kind("something_new_entirely") is None
    assert what_kind("") is None
    assert what_kind(None) is None
    assert what_kind("x", default=MemoryKind.EPISODIC) is MemoryKind.EPISODIC


def test_every_kind_string_written_in_the_tree_maps_somewhere() -> None:
    """The ratchet. A new spelling shows up here, not as a missing memory."""
    strays = strings_nothing_maps()
    assert strays == (), (
        f"{len(strays)} kind string(s) nothing maps: {list(strays)}. Add the "
        "spelling, or make the store write a canonical kind."
    )


def test_the_tree_really_does_write_more_spellings_than_kinds() -> None:
    written = kinds_written_in_the_tree()
    assert len(written) > len(MemoryKind), (
        "the whole point is that stores disagree about spelling"
    )


def test_a_record_reads_a_row_whichever_field_the_store_named_it() -> None:
    for field_name in ("kind", "memory_type", "type", "category"):
        row = {field_name: "episode", "content": "she opened the game", "t": 12.0}
        record = ARecord.from_dict(row)
        assert record is not None, field_name
        assert record.kind is MemoryKind.EPISODIC
        assert record.said == "she opened the game"
        assert record.at == 12.0


def test_a_row_with_no_placeable_kind_is_none_not_a_default_record() -> None:
    assert ARecord.from_dict({"kind": "who knows", "content": "x"}) is None
    assert ARecord.from_dict({"content": "x"}) is None


def test_what_a_store_keeps_beyond_the_envelope_survives() -> None:
    record = ARecord.from_dict(
        {"kind": "fact", "text": "the sky", "timestamp": 3.0,
         "source": "a book", "confidence": 0.8, "embedding_id": 12}
    )
    assert record is not None
    assert record.kind is MemoryKind.SEMANTIC
    assert record.from_where == "a book"
    assert record.carries == {"confidence": 0.8, "embedding_id": 12}


def test_the_report_names_a_kind_nothing_writes() -> None:
    seen = how_the_kinds_stand()
    assert set(seen["kinds"]) == {str(k) for k in MemoryKind}
    assert seen["strings_nothing_maps"] == []
    assert seen["spellings_known"] >= seen["strings_written"]
    assert isinstance(seen["kinds_nothing_writes"], list)


def test_every_known_spelling_maps_to_a_real_kind() -> None:
    valid = {str(k) for k in MemoryKind}
    for word, kind in the_spellings().items():
        assert kind in valid, f"{word} maps to {kind}, which is not a kind"
