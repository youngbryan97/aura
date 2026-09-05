"""Finding the analogue nobody pointed at.

structure_mapping can tell whether two domains share a shape, and has to be
handed the pair — which is the easy half, because somebody who already suspects
queues and traffic are the same thing has done the interesting part. The
transfer worth having is meeting a new situation and finding that something
already understood has the same shape, with nobody having tagged the two.

Two things had to be true for that and only one was. The mapper matched
relations on the predicate string, so two domains that shared a structure and
no words scored exactly zero — while the module's own description said "share
no words at all". And nothing looked anything up by shape.
"""

from __future__ import annotations

import pytest

from core.cognition.structure_mapping import Graph, Relation, map_structures
from core.cognition.transfer_search import DomainIndex, Signature

QUEUE = Graph(
    "queue_at_the_bakery",
    (
        Relation("waits_behind", ("b", "a")),
        Relation("waits_behind", ("c", "b")),
        Relation("served_first", ("a",)),
    ),
)
TRAFFIC = Graph(
    "traffic_jam",
    (
        Relation("follows", ("car2", "car1")),
        Relation("follows", ("car3", "car2")),
        Relation("exits_first", ("car1",)),
    ),
)
ORG = Graph(
    "org_chart",
    (
        Relation("reports_to", ("dev", "lead")),
        Relation("reports_to", ("lead", "vp")),
        Relation("signs_off", ("vp",)),
    ),
)
RECIPE = Graph(
    "recipe",
    (
        Relation("needs", ("cake", "flour")),
        Relation("needs", ("cake", "egg")),
        Relation("needs", ("icing", "sugar")),
        Relation("is_step", ("mix",)),
    ),
)


@pytest.fixture
def index():
    known = DomainIndex()
    known.extend([QUEUE, ORG, RECIPE])
    return known


# ── the mapper had to stop requiring shared words ────────────────────────


def test_two_domains_sharing_a_shape_and_no_words_align():
    """This scored exactly zero before: predicates were matched as strings."""
    alignment = map_structures(QUEUE, TRAFFIC)
    assert alignment is not None
    assert alignment.score == pytest.approx(1.0)
    assert alignment.predicate_mapping["waits_behind"] == "follows"
    assert alignment.predicate_mapping["served_first"] == "exits_first"


def test_shares_no_vocabulary_means_relations_too_not_just_objects():
    """Aligning objects while requiring the relation words to match letter for
    letter is cross-naming transfer, not cross-domain."""
    alignment = map_structures(QUEUE, TRAFFIC)
    assert alignment.shares_no_vocabulary is True

    same_words = Graph(
        "another_queue",
        (
            Relation("waits_behind", ("y", "x")),
            Relation("waits_behind", ("z", "y")),
            Relation("served_first", ("x",)),
        ),
    )
    easy = map_structures(QUEUE, same_words)
    assert easy.score == pytest.approx(1.0)
    assert easy.shares_no_vocabulary is False


def test_a_relation_cannot_be_read_as_one_of_a_different_arity():
    """What keeps this from being a search over every possible renaming."""
    alignment = map_structures(QUEUE, TRAFFIC)
    for source, target in alignment.predicate_mapping.items():
        source_arity = {len(r.args) for r in QUEUE.relations if r.predicate == source}
        target_arity = {len(r.args) for r in TRAFFIC.relations if r.predicate == target}
        assert source_arity & target_arity


# ── the signature is naming-invariant ────────────────────────────────────


def test_renaming_everything_leaves_the_signature_alone():
    renamed = Graph(
        "queue_renamed",
        (
            Relation("zzz", ("q", "p")),
            Relation("zzz", ("r", "q")),
            Relation("yyy", ("p",)),
        ),
    )
    assert Signature.of(QUEUE) == Signature.of(renamed)


def test_a_different_shape_has_a_different_signature():
    assert Signature.of(QUEUE) != Signature.of(RECIPE)
    assert Signature.of(QUEUE).distance(Signature.of(RECIPE)) > 0.0


def test_signature_distance_is_zero_for_the_same_shape():
    assert Signature.of(QUEUE).distance(Signature.of(TRAFFIC)) == pytest.approx(0.0)


# ── retrieval, with nobody naming the pair ───────────────────────────────


def test_it_finds_the_analogue_without_being_told_which_to_compare(index):
    best = index.best_analogue(TRAFFIC)
    assert best is not None
    assert best.target in {"queue_at_the_bakery", "org_chart"}
    assert best.crosses_vocabularies is True
    assert best.alignment.score == pytest.approx(1.0)


def test_a_domain_with_a_different_shape_is_not_reported_as_a_transfer(index):
    found = {t.target: t for t in index.find_analogues(TRAFFIC)}
    assert found["recipe"].holds is False
    assert found["recipe"].separation < 0.2


def test_retrieval_ranks_the_same_shape_first(index):
    ranked = index.candidates(TRAFFIC)
    assert ranked[0][0] == pytest.approx(0.0)
    assert "recipe" not in [name for distance, name in ranked if distance == 0.0]


def test_a_transfer_must_beat_its_shuffled_null(index):
    for transfer in index.find_analogues(TRAFFIC):
        if transfer.holds:
            assert transfer.separation >= 0.2
            assert transfer.alignment.score > transfer.null_mean


def test_no_analogue_is_a_real_answer():
    """A domain unlike anything known returns nothing, not a near miss."""
    lonely = DomainIndex()
    lonely.extend([RECIPE])
    assert lonely.best_analogue(TRAFFIC) is None


def test_an_empty_index_finds_nothing():
    assert DomainIndex().find_analogues(TRAFFIC) == ()


def test_a_domain_does_not_retrieve_itself(index):
    index.add(TRAFFIC)
    assert "traffic_jam" not in [name for _distance, name in index.candidates(TRAFFIC)]


def test_the_order_does_not_depend_on_insertion(index):
    other = DomainIndex()
    other.extend([RECIPE, ORG, QUEUE])
    assert [n for _d, n in index.candidates(TRAFFIC)] == [
        n for _d, n in other.candidates(TRAFFIC)
    ]


def test_retrieval_narrows_before_the_factorial_search_runs(index):
    """The mapping is factorial in the object count, so the filter has to bite."""
    assert len(index.candidates(TRAFFIC, top_k=2)) == 2
