"""What core.memory guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "Every kind-string a durable store writes maps onto one canonical "
        "memory kind, so a reader asking for a kind gets all of it.",
        "checked_by": "tests/test_what_kind_of_memory_is_this.py::"
        "test_every_kind_string_written_in_the_tree_maps_somewhere",
        "if_it_fails": "strings_nothing_maps() names the spelling; memories "
        "written under it are missing from every query for their kind",
    },
    {
        "it": "A memory whose kind cannot be placed is refused rather than filed "
        "under a guess.",
        "checked_by": "tests/test_what_kind_of_memory_is_this.py::"
        "test_an_unplaceable_string_is_none_rather_than_a_guess",
        "if_it_fails": "ARecord.from_dict returns None; a wrong kind would be "
        "returned by the wrong query and missed by the right one",
    },
    {
        "it": "What a readable memory projects is a subset, and anything outside "
        "that subset is refused rather than quietly included.",
        "checked_by": "tests/test_memory_you_can_read.py::"
        "test_something_outside_the_subset_is_refused",
        "if_it_fails": "a projection carries more than it declared, and a reader "
        "trusting the declaration reads what nobody meant to publish",
    },
    {
        "it": "Two memories that disagree about an identity field produce a named "
        "conflict rather than one silently winning.",
        "checked_by": "tests/test_when_two_memories_disagree.py::"
        "test_both_sides_changing_one_field_differently_is_a_conflict",
        "if_it_fails": "AConflict lists the field and both values; a merge that "
        "picked one would make the loser unrecoverable",
    },
)
