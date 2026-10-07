"""Training breadth: wordings settled late in the phrase, and chains long enough to tell first from latest."""

from __future__ import annotations

from collections import Counter

from core.learning.semantic_g04_transfer_corpus import build_g04_transfer_corpus, novelty_signatures
from core.learning.semantic_training_breadth_corpus import BREADTH_WORDINGS, build_training_breadth_corpus


def test_it_alternates_chains_of_three_to_five_with_two_step_wordings() -> None:
    rows = build_training_breadth_corpus(seed=3, requests=60)
    assert len(rows) == 60 and len({row.source_text for row in rows}) == 60
    assert all(row.split == "train" for row in rows)
    kinds = Counter(row.construction_id.split(":")[0] for row in rows)
    assert kinds == {"breadth_chain": 30, "breadth_wording": 30}
    depths = Counter(len(row.instructions) for row in rows if row.construction_id.startswith("breadth_chain"))
    assert set(depths) == {3, 4, 5}
    for row in rows:
        assert isinstance(row.program.run(row.inputs), int)


def test_no_breadth_request_uses_a_held_out_word() -> None:
    held_out = novelty_signatures("vocabulary", version=2)
    for row in build_training_breadth_corpus(seed=4, requests=200):
        text = row.source_text.lower()
        assert not any(word in text for word in held_out), row.source_text


def test_every_wording_is_drawn_and_names_its_operation() -> None:
    rows = build_training_breadth_corpus(seed=5, requests=600)
    drawn = Counter()
    for row in rows:
        if row.construction_id.startswith("breadth_wording"):
            names = row.construction_id.split(":")[1].split("+")
            for name, item in zip(names, row.instructions, strict=True):
                assert name in BREADTH_WORDINGS[item.instruction.op]
                drawn[name] += 1
    assert set(drawn) == {name for wordings in BREADTH_WORDINGS.values() for name in wordings}


def test_the_held_out_test_corpus_is_unchanged_by_the_split_parameter() -> None:
    rows = build_g04_transfer_corpus(seed=20261006, tasks_per_stratum=4)
    assert all(row.split == "test" and row.construction_id.startswith("g04_") for r in rows.values() for row in r)
