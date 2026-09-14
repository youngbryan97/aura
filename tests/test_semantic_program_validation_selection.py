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


def test_equivalent_argument_exchange_is_visible_but_does_not_relax_selection():
    from core.learning.procedure_induction import Instruction, Program

    gold = Program(2, (Instruction("mul", (0, 1)),))
    swapped = Program(2, (Instruction("mul", (1, 0)),))
    example = item("a")
    example.ir.to_program = lambda: gold

    def candidate(program):
        return SimpleNamespace(
            receipt_sha256="candidate",
            decode=lambda **kwargs: SimpleNamespace(ir=SimpleNamespace(to_program=lambda: program)),
        )

    report = select_compositional_program_candidate(
        {"parent": candidate(gold), "swapped": candidate(swapped)}, [example], incumbent="parent",
    )
    assert report["selected"] == "parent"
    assert report["equivalence_used_for_selection"] is False
    assert report["candidates"]["swapped"]["regressions"] == 1
    assert report["candidates"]["swapped"]["equivalent_regressions"] == 0
    assert report["candidates"]["swapped"]["program_equivalent"] == 1


@pytest.mark.parametrize("examples", [[], [item("a"), item("a")],
                                         [item("a"), item("a", "train")]])
def test_empty_duplicate_or_overlapping_validation_is_rejected(examples):
    with pytest.raises(ValueError):
        select_compositional_program_candidate(
            {"parent": Candidate({"a"})}, examples, incumbent="parent"
        )


def test_interrupted_validation_resumes_without_changing_report(tmp_path):
    path = tmp_path / "validation.json"
    examples = [item("a"), item("b")]
    candidate = Candidate({"a", "b"})
    original = candidate.decode

    def interrupt(**kwargs):
        if kwargs["source_text_sha256"] == "b":
            raise RuntimeError("interrupted")
        return original(**kwargs)

    candidate.decode = interrupt
    with pytest.raises(RuntimeError, match="interrupted"):
        select_compositional_program_candidate({"parent": candidate}, examples,
                                               incumbent="parent", checkpoint_path=path)
    resumed = Candidate({"a", "b"})
    events = []
    report = select_compositional_program_candidate({"parent": resumed}, examples,
        incumbent="parent", checkpoint_path=path, progress=events.append)
    assert resumed.calls == ["b"]
    assert events[0]["cached"] is True
    assert events[-1]["completed"] == 2
    fresh = select_compositional_program_candidate({"parent": Candidate({"a", "b"})},
                                                   examples, incumbent="parent")
    assert report == fresh


@pytest.mark.parametrize("change", ["model", "hidden", "tokens", "inputs", "gold", "basis"])
def test_checkpoint_cannot_cross_observation_or_model_changes(tmp_path, change):
    import numpy as np

    path = tmp_path / "validation.json"
    example = item("a")
    example.hidden_states = np.zeros((1, 2), dtype=np.float32)
    select_compositional_program_candidate({"parent": Candidate({"a"})}, [example],
                                           incumbent="parent", checkpoint_path=path)
    model = Candidate({"a"})
    if change == "model":
        model.receipt_sha256 = "changed"
    elif change == "hidden":
        example.hidden_states[0, 0] = 1
    elif change == "tokens":
        example.ir.source_token_ids = (2,)
    elif change == "inputs":
        example.public_inputs = (7,)
    elif change == "gold":
        example.ir.to_program = lambda: "changed"
    else:
        example.ir.model_basis_receipt_sha256 = "changed"
    with pytest.raises(ValueError, match="identity"):
        select_compositional_program_candidate({"parent": model}, [example],
                                               incumbent="parent", checkpoint_path=path)
    assert model.calls == []


def test_checkpoint_corruption_is_not_a_score(tmp_path):
    import json

    path = tmp_path / "validation.json"
    select_compositional_program_candidate({"parent": Candidate({"a"})}, [item("a")],
                                           incumbent="parent", checkpoint_path=path)
    body = json.loads(path.read_text())
    body["rows"]["parent"]["a"] = [False, False]
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="checksum"):
        select_compositional_program_candidate({"parent": Candidate({"a"})}, [item("a")],
                                               incumbent="parent", checkpoint_path=path)
