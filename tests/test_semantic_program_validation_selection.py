from types import SimpleNamespace

import pytest

from core.learning.semantic_program_compositional_campaign import (
    select_compositional_program_candidate,
)


def item(identity, split="validation"):
    return SimpleNamespace(
        split=split, hidden_states=None, public_inputs=(),
        ir=SimpleNamespace(
            source_text_sha256=identity, source_token_ids=(1,),
            model_basis_receipt_sha256="basis", to_program=lambda: identity,
        ),
    )


class Candidate:
    def __init__(self, correct):
        self.correct = set(correct)
        self.receipt_sha256 = "candidate:" + ",".join(sorted(correct))
        self.calls = []

    def decode(self, **kwargs):
        assert set(kwargs) == {
            "source_token_ids", "hidden_states", "public_inputs",
            "source_text_sha256", "model_basis_sha256",
        }
        identity = kwargs["source_text_sha256"]
        self.calls.append(identity)
        return SimpleNamespace(ir=SimpleNamespace(
            to_program=lambda: identity if identity in self.correct else "wrong"
        ))


def test_selects_program_gain_without_using_test_or_training():
    parent, better = Candidate({"a"}), Candidate({"a", "b"})
    report = select_compositional_program_candidate(
        {"parent": parent, "better": better},
        [item("train", "train"), item("a"), item("b"), item("test", "test")],
        incumbent="parent",
    )
    assert report["selected"] == "better"
    assert parent.calls == better.calls == ["a", "b"]
    assert report["serving_authority"] is False
    assert report["candidates"]["better"]["gains"] == 1


def test_regression_cannot_be_hidden_by_more_total_correct_programs():
    report = select_compositional_program_candidate(
        {"parent": Candidate({"a"}), "candidate": Candidate({"b", "c"})},
        [item("a"), item("b"), item("c")], incumbent="parent",
    )
    assert report["selected"] == "parent"
    assert report["candidates"]["candidate"]["regressions"] == 1


def test_tie_keeps_incumbent():
    report = select_compositional_program_candidate(
        {"parent": Candidate({"a"}), "candidate": Candidate({"a"})},
        [item("a")], incumbent="parent",
    )
    assert report["selected"] == "parent"


@pytest.mark.parametrize("examples", [[], [item("a"), item("a")],
                                         [item("a"), item("a", "train")]])
def test_empty_duplicate_or_overlapping_validation_is_rejected(examples):
    with pytest.raises(ValueError):
        select_compositional_program_candidate(
            {"parent": Candidate({"a"})}, examples, incumbent="parent"
        )
