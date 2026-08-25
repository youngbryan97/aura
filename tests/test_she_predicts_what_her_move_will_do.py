"""A move she cannot be wrong about teaches her nothing.

Both of Aura's goal loops used to take the decision as an injected callable,
so the judgement lived outside her and nothing was ever predicted. These
tests hold the organ that replaced it: a choice arrives with an expectation,
the expectation is checked mechanically against the next observation, and a
broken one becomes evidence the next choice reads.
"""
from __future__ import annotations

import pytest

from core.agency.deliberate_action import (
    ActionOption,
    Attempt,
    Deliberation,
    Expectation,
    Verdict,
    choose_named,
    confidence_from_history,
    confirm,
    deliberate,
)


def _option(name: str, **kw) -> ActionOption:
    return ActionOption(name=name, **kw)


def _thinks(reply: str):
    async def think(objective, evidence):
        think.seen = (objective, list(evidence))
        return reply

    think.seen = None
    return think


class _Spine:
    def __init__(self):
        self.recorded = []
        self.resolved = []

    def record(self, episode):
        self.recorded.append(episode)
        return "ep_1"

    def resolve(self, episode_id, outcome):
        self.resolved.append((episode_id, outcome))


class _Graph:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.written = []

    def query_consequences(self, action, params=None):
        return self.rows.get(action, [])

    def record_outcome(self, action, context, outcome, success):
        self.written.append((action, context, outcome, success))


@pytest.mark.asyncio
async def test_a_choice_arrives_with_a_prediction():
    up = _option("up", expectation=Expectation(changed=True, describes="the board to shift"))
    result = await deliberate(
        "reach 4096",
        "a board of tiles",
        [up, _option("down")],
        think=_thinks("Sliding up keeps the big tile in the corner, so: up"),
        spine=_Spine(),
        graph=_Graph(),
    )
    assert result.reached
    assert result.chosen.name == "up"
    assert result.chosen.expectation.describes == "the board to shift"
    assert "the board to shift" in result.narrate()


@pytest.mark.asyncio
async def test_language_out_of_reach_does_not_stop_her_deciding():
    """She loses her words, not her judgement.

    This test used to assert the opposite — that an unreachable model ends
    the decision. That broke the invariant in pre_linguistic.py, and live it
    meant a pursuit made no move at all while the resident model reloaded.
    """

    async def unreachable(objective, evidence):
        raise RuntimeError("no model")

    result = await deliberate(
        "reach 4096",
        "a board",
        [_option("up"), _option("down")],
        think=unreachable,
        spine=_Spine(),
        graph=_Graph(),
        lived=False,
    )
    assert result.reached
    assert result.chosen is not None
    assert result.spoke is False, "a choice made without language must say so"


@pytest.mark.asyncio
async def test_an_answer_that_names_no_move_still_leaves_a_decision_to_make():
    """Language wandering off is not a reason to stand still.

    The evidence that decides without language is there either way, so it
    decides — and the rationale carries both what she said and why the move
    was picked, rather than quietly replacing one with the other.
    """
    result = await deliberate(
        "reach 4096",
        "a board",
        [_option("up"), _option("down")],
        think=_thinks("I would rather think about this for a while."),
        spine=_Spine(),
        graph=_Graph(),
        lived=False,
    )
    assert result.reached
    assert result.chosen is not None
    assert "has not been tried yet" in result.rationale


def test_the_move_she_settles_on_is_the_one_she_named_last():
    options = [_option("up"), _option("down"), _option("left")]
    reply = "Up looks tempting and down is safe, but the corner argues for left."
    assert choose_named(reply, options).name == "left"


@pytest.mark.asyncio
async def test_the_decision_rests_on_measured_facts_not_on_urging():
    think = _thinks("up")
    await deliberate(
        "reach 4096",
        "highest tile is 8",
        [_option("up", detail="slide every tile up")],
        think=think,
        spine=_Spine(),
        graph=_Graph(),
    )
    _objective, evidence = think.seen
    assert "Goal: reach 4096" in evidence
    assert "What is visible now: highest tile is 8" in evidence
    assert any("slide every tile up" in line for line in evidence)


@pytest.mark.asyncio
async def test_what_went_wrong_last_time_is_read_before_the_next_move():
    graph = _Graph({"up": [{"outcome": "nothing moved", "success": False}]})
    think = _thinks("down")
    result = await deliberate(
        "reach 4096", "a board", [_option("up"), _option("down")], think=think, spine=_Spine(), graph=graph
    )
    _objective, evidence = think.seen
    assert any("nothing moved" in line for line in evidence)
    assert result.recalled


@pytest.mark.asyncio
async def test_a_broken_expectation_becomes_evidence_for_the_next_move():
    up = _option("up", expectation=Expectation(changed=True, describes="the board to shift"))
    first = await deliberate(
        "reach 4096", "board A", [up, _option("down")], think=_thinks("up"), spine=_Spine(), graph=_Graph()
    )
    attempt = confirm(first, "board A", "board A", spine=_Spine(), graph=_Graph())
    assert not attempt.verdict.held
    assert attempt.verdict.stalled

    think = _thinks("down")
    await deliberate(
        "reach 4096",
        "board A",
        [up, _option("down")],
        think=think,
        history=[attempt],
        spine=_Spine(),
        graph=_Graph(),
    )
    _objective, evidence = think.seen
    assert any("nothing changed" in line for line in evidence)


def test_an_expectation_is_graded_by_measurement_not_by_the_faculty_that_made_it():
    expectation = Expectation(changed=True, contains=("4096",), absent=("cookies",))
    held = expectation.check("2048 board with cookies", "board showing 4096")
    assert held.held
    broken = expectation.check("2048 board with cookies", "2048 board with cookies")
    assert not broken.held
    assert broken.stalled and broken.missing == ("4096",) and broken.lingering == ("cookies",)


def test_no_record_of_a_move_means_no_opinion_about_it():
    assert confidence_from_history([]) == pytest.approx(0.5)
    worked = ["up worked before: tiles merged"] * 4
    assert confidence_from_history(worked) > 0.5
    failed = ["up did not work before: nothing moved"] * 4
    assert confidence_from_history(failed) < 0.5


@pytest.mark.asyncio
async def test_the_decision_is_recorded_when_made_and_resolved_when_seen():
    spine = _Spine()
    graph = _Graph()
    up = _option("up", expectation=Expectation(changed=True, describes="the board to shift"))
    result = await deliberate(
        "reach 4096", "board A", [up, _option("down")], think=_thinks("up"), spine=spine, graph=graph
    )
    assert result.episode_id == "ep_1"
    episode = spine.recorded[0]
    assert episode.decision == "up"
    assert tuple(episode.options) == ("up", "down")
    assert episode.decider == "agency.deliberate_action"

    confirm(result, "board A", "board B", spine=spine, graph=graph)
    _episode_id, outcome = spine.resolved[0]
    assert str(outcome.kind) == "success"
    assert graph.written and graph.written[0][3] is True


@pytest.mark.asyncio
async def test_nothing_to_do_is_said_plainly():
    result = await deliberate("reach 4096", "a board", [], think=_thinks("up"), spine=_Spine(), graph=_Graph())
    assert not result.reached
    assert result.reason == "nothing is available to do"


def test_an_attempt_reads_back_as_a_fact_about_the_world():
    attempt = Attempt(
        option="up",
        expected="the board to shift",
        verdict=Verdict(held=False, observed_change=False, stalled=True),
    )
    assert attempt.as_evidence() == "up was expected to the board to shift, but nothing changed."


@pytest.mark.asyncio
async def test_she_can_say_why_before_the_move_lands():
    result = await deliberate(
        "reach 4096",
        "a board",
        [_option("up", detail="slide up", expectation=Expectation(describes="the big tile to stay in the corner"))],
        think=_thinks("Keeping the corner matters most. Go up."),
        spine=_Spine(),
        graph=_Graph(),
    )
    spoken = result.narrate()
    assert "slide up" in spoken
    assert "Go up" in spoken
    assert "the big tile to stay in the corner" in spoken


@pytest.mark.asyncio
async def test_a_rehearsal_is_marked_so_her_real_history_refuses_it():
    """A test run must not be able to teach her anything.

    The experience spine refuses anything but lived experience into a live
    store, and that guarantee only holds if the producer says which it is.
    """
    from core.ontogeny.experience import Provenance

    spine = _Spine()
    await deliberate(
        "reach 4096",
        "a board",
        [_option("up")],
        think=_thinks("up"),
        spine=spine,
        graph=_Graph(),
        lived=False,
    )
    assert spine.recorded[0].provenance is Provenance.TEST

    lived_spine = _Spine()
    await deliberate(
        "reach 4096", "a board", [_option("up")], think=_thinks("up"), spine=lived_spine, graph=_Graph()
    )
    assert lived_spine.recorded[0].provenance is Provenance.LIVE


def test_the_live_store_physically_refuses_a_rehearsal():
    """Not a convention — the refusal is structural, and this proves it."""
    from core.ontogeny.experience import Episode, ExperienceSpine, Provenance

    spine = ExperienceSpine(db_path="/tmp/claude-501/x/rehearsal_check.db", autoflush=False)
    spine._store_kind = "live"
    rehearsal = Episode(
        control_point="agency.next_move",
        features={},
        decision="up",
        provenance=Provenance.TEST,
    )
    assert spine.record(rehearsal) is None


def test_a_move_can_be_chosen_with_no_language_anywhere_in_it():
    """The resident model is her language organ, not her decision organ.

    core/cognition/pre_linguistic.py holds this as a design invariant:
    actions can be dispatched even when the LLM is unavailable.
    """
    from core.agency.deliberate_action import choose_without_language

    options = [_option("up"), _option("down"), _option("left")]
    chosen, why = choose_without_language(options, history=(), recalled=())
    assert chosen is not None
    assert "not been tried" in why


def test_without_language_she_avoids_the_move_that_just_did_nothing():
    from core.agency.deliberate_action import choose_without_language

    options = [_option("up"), _option("down")]
    stalled = Attempt(option="up", expected="the board to shift", verdict=Verdict(held=False, observed_change=False, stalled=True))
    chosen, _why = choose_without_language(options, history=[stalled], recalled=())
    assert chosen.name == "down"


def test_without_language_she_prefers_what_has_worked_here():
    from core.agency.deliberate_action import choose_without_language

    options = [_option("up"), _option("down")]
    recalled = ["down worked before: tiles merged"] * 3
    chosen, why = choose_without_language(options, history=(), recalled=recalled)
    assert chosen.name == "down"
    assert "worked here before" in why


def test_without_language_she_does_not_press_the_same_key_forever():
    """Among equals, the one left alone longest — so a run keeps moving."""
    from core.agency.deliberate_action import choose_without_language

    options = [_option("up"), _option("down"), _option("left")]
    history = [
        Attempt(option="up", expected="x", verdict=Verdict(held=True, observed_change=True)),
        Attempt(option="down", expected="x", verdict=Verdict(held=True, observed_change=True)),
    ]
    chosen, _why = choose_without_language(options, history=history, recalled=())
    assert chosen.name == "left"


@pytest.mark.asyncio
async def test_she_acts_and_says_why_while_the_model_is_reloading():
    async def unreachable(objective, evidence):
        raise RuntimeError("worker_not_alive")

    result = await deliberate(
        "reach 4096",
        "a board",
        [_option("up", detail="slide up", expectation=Expectation(describes="the board to shift"))],
        think=unreachable,
        spine=_Spine(),
        graph=_Graph(),
        lived=False,
    )
    assert result.reached, "she stopped because she could not talk"
    assert result.spoke is False
    spoken = result.narrate()
    assert "slide up" in spoken
    assert "without words" in spoken
    assert "the board to shift" in spoken


def test_a_name_that_contains_another_name_is_not_mistaken_for_it():
    """"slow down" contains "down".

    Ranking a mention by where it starts picks the shorter name and turns a
    decision about her own pacing into an arrow key.
    """
    options = [_option(name) for name in ("up", "down", "left", "right", "slow down", "say less")]
    assert choose_named("slow down", options).name == "slow down"
    assert choose_named("down", options).name == "down"
    assert choose_named("say less", options).name == "say less"


def test_the_conclusion_still_wins_over_an_earlier_mention():
    options = [_option(name) for name in ("up", "down", "left", "slow down")]
    assert choose_named("Up is tempting and down is safe, but the corner argues for left.", options).name == "left"


def test_the_move_she_decided_on_beats_a_word_that_appears_later():
    """LIVE: "I'm going to press right because the left column is full" was
    read as a decision to press left, and she announced a move she had not
    made.

    "Last one named" is right for a reply that works through the options and
    settles, and wrong for one that names something else afterwards.
    """
    options = [_option(name) for name in ("up", "down", "left", "right")]
    assert choose_named("I'm going to press right because the left column is full", options).name == "right"
    assert choose_named("press up to keep the corner, avoiding right", options).name == "up"


def test_a_reply_that_settles_at_the_end_still_settles_there():
    options = [_option(name) for name in ("up", "down", "left", "right")]
    assert choose_named(
        "Up looks tempting and down is safe, but the corner argues for left.", options
    ).name == "left"


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("I'm pressing right again because the tiles are on the up side", "right"),
        ("I'm not fully certain, but i am pressing the up key", "up"),
        ("sliding left to consolidate before the right side fills", "left"),
        ("going down to keep the top row clear", "down"),
    ],
)
def test_the_verb_is_read_in_the_shapes_people_write_it(reply, expected):
    """A pattern that only matched the bare form fell through and picked a noun."""
    options = [_option(name) for name in ("up", "down", "left", "right")]
    assert choose_named(reply, options).name == expected


def test_a_bare_answer_is_still_read():
    options = [_option(name) for name in ("up", "down", "left", "right")]
    assert choose_named("down", options).name == "down"
