"""A promise she is working on knows how far along it is.

A request that takes a while becomes a commitment, and a commitment is what
she carries into every later turn of conversation: asked how it is going, the
record she answers from is that one. It was made at 0% when the work began and
stayed at 0% until the work ended, because nothing doing the work ever told
it anything. She could be three rungs from the finish of a game and her own
record of the promise said she had not started.

So the work reports into the promise it belongs to. Which promise that is
travels with the work, so nothing doing it has to know it was promised to
anybody. How far along is measured in what the rungs behind her cost: the
moves spent, against the moves still to go on the pattern her own record
shows.
"""

from __future__ import annotations

import asyncio

import pytest

from core.agency import commitment_engine as promises
from core.agency.how_it_is_going import HowItIsGoing


@pytest.fixture
def engine(tmp_path, monkeypatch):
    made = promises.CommitmentEngine.__new__(promises.CommitmentEngine)
    promises.CommitmentEngine.__init__(made)
    monkeypatch.setattr(made, "_save", lambda: None)
    monkeypatch.setattr(promises, "get_commitment_engine", lambda: made)
    return made


def test_work_that_belongs_to_a_promise_reports_into_it(engine):
    promised = engine.commit(description="play until the 2048 tile", outcome="the 2048 tile")

    async def the_work():
        return promises.report_progress(0.4, "256 so far, working on 512")

    async def the_task():
        slot = promises.a_place_for_its_commitment()
        work = asyncio.ensure_future(the_work())
        # The promise is made after the work has begun, which is how a task
        # that turns out to be long is given one.
        slot["id"] = promised.id
        return await work

    assert asyncio.run(the_task()) is True
    assert engine._commitments[promised.id].progress == pytest.approx(0.4)
    assert "256 so far" in engine._commitments[promised.id].notes[-1]


def test_work_that_belongs_to_no_promise_reports_nowhere(engine):
    async def the_work():
        return promises.report_progress(0.4, "256 so far")

    assert asyncio.run(the_work()) is False


def test_progress_is_not_a_finish(engine):
    """Only the work's own verdict fulfils a promise; a projection never does."""
    promised = engine.commit(description="play until the 2048 tile", outcome="the 2048 tile")

    async def the_task():
        slot = promises.a_place_for_its_commitment()
        slot["id"] = promised.id
        return promises.report_progress(1.0, "projected")

    asyncio.run(the_task())
    assert engine._commitments[promised.id].status == promises.CommitmentStatus.ACTIVE
    assert engine._commitments[promised.id].progress < 1.0


def test_what_she_carries_into_conversation_says_how_far_along():
    promised = promises.Commitment(
        id="c1",
        commitment_type=promises.CommitmentType.AUTONOMOUS,
        description="play until the 2048 tile",
        outcome="the 2048 tile",
        deadline=10**12,
        progress=0.35,
        notes=["[Update] 256 so far, working on 512"],
    )
    brief = promised.to_brief()
    assert "35%" in brief
    assert "256 so far, working on 512" in brief


def test_how_far_along_is_measured_in_what_the_rungs_cost():
    going = HowItIsGoing(toward=2048.0)
    # Doubling rungs, each costing twice the one before.
    for reached, at_move in ((8, 10), (16, 30), (32, 70), (64, 150)):
        going.noticed(reached, at_move)
    share = going.share_done(at_move=150)
    # 150 moves spent; five rungs to go at 160, 320, 640, 1280, 2560.
    assert 0.02 < share < 0.05
    # Counted in rungs it would read as more than a third of the way.
    assert share < 4 / 11


def test_with_no_pattern_it_is_how_much_of_the_finish_is_reached():
    going = HowItIsGoing(toward=500.0)
    going.noticed(100, at_move=5)
    assert going.share_done(at_move=5) == pytest.approx(0.2)


def test_with_nothing_to_aim_at_nothing_is_claimed():
    assert HowItIsGoing().share_done(at_move=40) == 0.0


def test_the_run_reports_a_rung_to_the_promise_it_belongs_to():
    from screen_pursuit_support import pursuit_source
    from source_contract import in_order

    in_order(pursuit_source(), "going.noticed(", "report_progress(", "going.share_done(")


def test_the_verifier_gives_its_work_a_place_for_the_promise():
    import inspect

    from core.agency import task_commitment_verifier as verifier

    source = inspect.getsource(verifier)
    assert source.count("a_place_for_its_commitment()") >= 2
    assert 'slot["id"] = commitment_id' in source
