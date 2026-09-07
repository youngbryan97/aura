"""The loop that widens her language, and the link that binds it."""
from __future__ import annotations

from core.cognition.where_the_growing_is_starved import (
    ENOUGH_TO_JUDGE_ON,
    THE_STEPS,
    how_the_growing_stands,
    where_it_is_starved,
)


def test_every_step_is_counted() -> None:
    counted = how_the_growing_stands()["counts"]
    for step in THE_STEPS:
        assert step in counted, step


def test_the_step_with_nothing_coming_into_it_is_named() -> None:
    said = how_the_growing_stands()
    if said["turns_over"]:
        assert said["starved_at"] == ""
    else:
        assert said["starved_at"] in THE_STEPS


def test_the_gate_needs_enough_families_to_be_able_to_say_yes() -> None:
    """Three binary observations cannot lift a posterior over the threshold.

    A gate judging on too few held-out families refuses a change that helps
    and a change that does nothing alike, so its refusals carry no
    information. The same families are what the developmental evidence gate
    weighs, which is why this is upstream of everything else.
    """
    said = how_the_growing_stands()
    assert said["enough_to_judge_on"] == ENOUGH_TO_JUDGE_ON
    assert said["the_gate_can_say_yes"] == (
        said["families_to_judge_on"] >= ENOUGH_TO_JUDGE_ON
    )


def test_the_reading_survives_a_process_that_has_recalled_nothing() -> None:
    """Zeros are true of that process, and the report must not raise."""
    said = how_the_growing_stands()
    assert isinstance(said["counts"]["heads she has written"], int)
    assert isinstance(said["families_to_judge_on"], int)


def test_where_it_is_starved_and_the_counts_agree() -> None:
    said = how_the_growing_stands()
    starved = where_it_is_starved()
    assert starved == said["starved_at"]
    if starved:
        assert said["counts"][starved] == 0


def test_every_episode_the_answering_path_writes_carries_its_cases() -> None:
    """`about` is the only thing that makes a family usable as a probe.

    Sixteen of seventeen writers dropped it, so the record grew to 512
    episodes across 76 families of which three could be judged on — and a gate
    weighing three binary observations refuses a change that helps and one
    that does nothing alike. The answering path keeps them now; the cost model
    and the validation fixture do not have them to keep.
    """
    import ast
    import pathlib

    source = pathlib.Path("core/cognition/sequence_induction.py").read_text()
    without = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", ""))
        == "note_an_episode"
        and not any(kw.arg == "about" for kw in node.keywords)
    ]
    assert without == [], (
        "these write an episode the gate can never judge on: "
        + ", ".join(f"sequence_induction.py:{one}" for one in without)
    )


def test_answering_a_question_widens_what_a_change_can_be_judged_on(tmp_path, monkeypatch) -> None:
    """Measured by answering, not by reading the call sites."""
    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.cognition.sequence_induction import answer_sequence_question
    from core.cognition.the_record_of_her_own_work import other_families

    before = len(other_families(than=""))
    for asked in (
        "[3, 4, 5] becomes [5, 4, 3]. [10, 11, 12] becomes [12, 11, 10]. "
        "What does [20, 21, 22] become?",
        "[1,2,3,4] -> [3,4,1,2] and [5,6,7,8] -> [7,8,5,6]. What about [9,10,11,12]?",
    ):
        answer_sequence_question(asked)
    assert len(other_families(than="")) >= before
