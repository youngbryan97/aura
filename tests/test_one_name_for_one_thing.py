"""Two stores holding a fact about the same person can tell that they do.

OpenCog interns every atom in one space, so two references to a concept are
the same object. Aura keeps knowledge in several places that each name things
their own way: the associative entity memory content-addresses a kind and a
name; the perception belief state builds `label:glyph:position`; the metagraph
uses node identity. None is wrong for what it does. What was missing is the
join — a reference across two of them was a string comparison, and a duplicate
was invisible.
"""

from __future__ import annotations

import pytest

from core.knowledge.who_this_is import (
    also_known_as,
    an_id_for,
    duplicates_among,
    forget_the_equivalences,
    normalise,
    the_equivalences,
    the_same,
    what_it_was_called,
)


@pytest.fixture(autouse=True)
def _fresh():
    forget_the_equivalences()
    yield
    forget_the_equivalences()


def test_two_spellings_of_one_name_are_one_id():
    assert an_id_for("person", "Bryan") == an_id_for("person", "  bryan  ")
    assert an_id_for("person", "Bryan") != an_id_for("person", "Brian")
    assert an_id_for("person", "Bryan") != an_id_for("place", "Bryan")


def test_normalising_does_not_get_clever():
    """A normaliser that strips punctuation merges things that are not the same."""

    assert normalise(" A  B ") == "a b"
    assert normalise("St. John") != normalise("St John")
    assert normalise("C++") != normalise("C")


def test_an_equivalence_says_why_and_merges_the_same_way_round_either_way():
    one = an_id_for("person", "Bryan")
    other = an_id_for("person", "B. Young")
    first = also_known_as(one, other, because="he signed both")
    forget_the_equivalences()
    an_id_for("person", "Bryan")
    an_id_for("person", "B. Young")
    again = also_known_as(other, one, because="the other way round")
    assert first == again, "the canonical form depended on the order"
    assert first in {one, other} and first == min(one, other)


def test_everything_resolves_to_the_canonical_one_through_a_chain():
    a = an_id_for("person", "one")
    b = an_id_for("person", "two")
    c = an_id_for("person", "three")
    also_known_as(a, b, because="a note")
    also_known_as(b, c, because="another note")
    assert the_same(a) == the_same(b) == the_same(c)
    assert len(the_equivalences()) == 2
    assert all(one.because for one in the_equivalences())


def test_what_a_thing_has_been_called_survives_the_merge():
    a = an_id_for("person", "Bryan")
    b = an_id_for("person", "B. Young")
    kept = also_known_as(a, b, because="one person")
    assert set(what_it_was_called(kept)) >= {"Bryan", "B. Young"}


def test_a_duplicate_across_two_stores_is_findable():
    """A duplicate inside one store is that store's business. Across two it was nobody's."""

    found = duplicates_among(
        {
            "entity_memory": [("person", "Bryan", "ent_one")],
            "belief_state": [("person", "bryan", "ent_two")],
            "metagraph": [("person", "Someone Else", "ent_three")],
        }
    )
    assert len(found) == 1, found
    assert found[0]["name"] == "bryan"
    assert sorted(found[0]["ids"]) == ["ent_one", "ent_two"]
    assert found[0]["the_canonical_one"] == an_id_for("person", "Bryan")


def test_a_duplicate_stops_being_one_once_it_is_declared():
    also_known_as("ent_one", "ent_two", because="the same person")
    found = duplicates_among(
        {
            "entity_memory": [("person", "Bryan", "ent_one")],
            "belief_state": [("person", "bryan", "ent_two")],
        }
    )
    assert not found, found


def test_the_entity_memory_mints_through_the_shared_service():
    """Adopted rather than replaced: every id it ever stored is still that id."""

    import hashlib

    from core.memory.associative_entity_memory import (
        EntityKind,
        entity_id_for,
        normalize_name,
    )

    seed = f"{EntityKind.PERSON.value}|{normalize_name('Bryan')}".encode()
    as_it_always_was = "ent_" + hashlib.blake2b(seed, digest_size=10).hexdigest()
    assert entity_id_for(EntityKind.PERSON, "Bryan") == as_it_always_was
    assert entity_id_for(EntityKind.PERSON, "Bryan") == an_id_for(
        EntityKind.PERSON.value, "Bryan"
    )
