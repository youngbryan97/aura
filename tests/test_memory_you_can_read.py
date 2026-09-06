"""A readable, versioned projection of memory, tied to what happened.

Letta keeps agent memory as a git-backed filesystem, and the review said what
that buys: memory evolution is unusually inspectable and reversible. The
closure asked for a canonical subset projected into a versioned MemoryFS with
commit ids linked to event-spine receipts, keeping the databases as indexes
rather than the sole human-readable truth.
"""
from __future__ import annotations

import pytest

from core.memory.memory_you_can_read import MemoryYouCanRead, what_is_projected


@pytest.fixture
def memory(tmp_path) -> MemoryYouCanRead:
    return MemoryYouCanRead(tmp_path / "readable")


# -------------------------------------------------------------- projecting


def test_what_is_projected_is_a_subset_and_not_everything():
    """A projection of the whole of memory is a second copy of memory."""
    projected = what_is_projected()
    assert "identity" in projected
    assert 0 < len(projected) < 20


def test_something_outside_the_subset_is_refused(memory):
    with pytest.raises(KeyError, match="is not projected"):
        memory.write("everything_she_ever_thought", {"a": 1})


def test_what_was_written_can_be_read_back(memory):
    memory.write("identity", {"name": "Aura"})
    assert "Aura" in memory.read("identity")


def test_reading_something_never_written_is_empty_and_not_an_error(memory):
    assert memory.read("commitments") == ""


# ------------------------------------------------------------- readable


def test_a_dict_becomes_markdown_a_person_reads(memory):
    memory.write("identity", {"name": "Aura", "core_values": ["honesty", "care"]})
    body = memory.read("identity")
    assert "## name" in body
    assert "- honesty" in body
    assert "{" not in body, "a dict rendered as JSON is a second database"


def test_a_list_becomes_bullets(memory):
    memory.write("commitments", ["answer honestly", "say when unsure"])
    assert "- answer honestly" in memory.read("commitments")


def test_a_string_is_written_as_it_is(memory):
    memory.write("meanings", "a gap is the difference between neighbours")
    assert "difference between neighbours" in memory.read("meanings")


# -------------------------------------------------------------- versions


def test_a_change_is_a_commit(memory):
    first = memory.write("identity", {"name": "Aura"}, why="first")
    second = memory.write("identity", {"name": "Aura II"}, why="she renamed herself")

    assert first is not None and second is not None
    assert first.digest != second.digest
    assert second.why == "she renamed herself"
    assert len(memory.history("identity")) == 2


def test_writing_the_same_thing_again_is_not_a_commit(memory):
    """A history where every save is a commit cannot say when it last changed."""
    memory.write("identity", {"name": "Aura"})
    assert memory.write("identity", {"name": "Aura"}) is None
    assert len(memory.history("identity")) == 1


def test_a_change_back_to_an_earlier_value_is_a_commit(memory):
    memory.write("identity", {"name": "Aura"})
    memory.write("identity", {"name": "Aura II"})
    assert memory.write("identity", {"name": "Aura"}) is not None
    assert len(memory.history("identity")) == 3


def test_the_history_of_one_file_is_not_the_history_of_another(memory):
    memory.write("identity", {"name": "Aura"})
    memory.write("commitments", ["be honest"])
    assert len(memory.history("identity")) == 1
    assert len(memory.history()) == 2


# ----------------------------------------------- tied to what happened


def test_a_commit_carries_where_the_event_log_had_reached(memory):
    """A digest says what changed. It does not say what was happening."""
    from core.runtime.event_spine import get_spine

    spine = get_spine()
    spine.emit("something_happened", {"a": 1})
    where = spine.log.head

    commit = memory.write("identity", {"name": "Aura"})
    assert commit.after_event >= where


def test_what_changed_after_a_point_is_a_lookup(memory):
    from core.runtime.event_spine import get_spine

    spine = get_spine()
    memory.write("identity", {"name": "before"})
    spine.emit("the_thing_that_changed_her_mind", {})
    mark = spine.log.head
    memory.write("identity", {"name": "after"}, why="because of that")

    after = memory.what_changed_after(mark - 1)
    assert [one["why"] for one in after] == ["because of that"]


def test_a_commit_with_no_spine_says_zero_rather_than_pretending(memory, monkeypatch):
    import core.memory.memory_you_can_read as module

    monkeypatch.setattr(module, "_where_the_spine_is", lambda: 0)
    commit = memory.write("identity", {"name": "Aura"})
    assert commit.after_event == 0


# ---------------------------------------------------------------- reading


def test_the_report_names_what_has_been_projected(memory):
    memory.write("identity", {"name": "Aura"})
    report = memory.report()
    assert report["files"] == ["identity"]
    assert report["commits"] == 1
    assert report["latest"]["name"] == "identity"


def test_the_projection_is_not_the_authority(memory):
    """A file here being deleted loses nothing, which is why it is safe to write."""
    memory.write("identity", {"name": "Aura"})
    (memory.root / "identity.md").unlink()
    assert memory.read("identity") == ""
    assert memory.history("identity"), "the history is not the file"
