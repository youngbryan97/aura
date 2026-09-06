"""The turn lifecycle, attacked at the semantics.

A blind maturity comparison scored Letta's runtime highest and named the
reason: a discriminated turn state machine, an immutable lease per active
turn, mutation APIs that verify the lease is still the owner, and a
cancellation that is not instantaneous idleness. It also named the tests that
earned it the score, and this file is those tests:

* stale leases after reset;
* double finish;
* overlapping ownership;
* external cleanup arriving before the cancelled owner;
* owner finish occurring before external cleanup;
* atomic takeover from a command.
"""
from __future__ import annotations

import threading

import pytest

from core.runtime.whose_turn_it_is import (
    ALease,
    NotTheOwner,
    TheTurn,
    TurnStatus,
    the_turn,
)


@pytest.fixture
def turn() -> TheTurn:
    return TheTurn()


# ------------------------------------------------------------ the states


def test_nothing_owns_it_to_begin_with(turn):
    assert turn.status is TurnStatus.IDLE
    assert turn.lease is None


def test_beginning_a_turn_makes_it_active_and_hands_out_one_lease(turn):
    lease = turn.begin(origin="user")
    assert turn.status is TurnStatus.ACTIVE
    assert turn.owns_it(lease)
    assert lease.turn_id
    assert lease.origin == "user"


def test_a_finished_turn_leaves_nothing_owning_it(turn):
    lease = turn.begin()
    assert turn.finish(lease) is TurnStatus.IDLE
    assert turn.status is TurnStatus.IDLE
    assert not turn.owns_it(lease)


# ------------------------------------------------------ overlapping owners


def test_a_second_turn_cannot_begin_while_one_is_active(turn):
    turn.begin()
    with pytest.raises(NotTheOwner, match="while the runtime is active"):
        turn.begin()


def test_a_second_turn_cannot_begin_while_one_is_cancelling(turn):
    """The window this whole module exists to close."""
    lease = turn.begin()
    turn.cancel("stop")
    assert turn.status is TurnStatus.CANCELLING
    with pytest.raises(NotTheOwner, match="cancelling"):
        turn.begin()
    turn.finish(lease)
    turn.cleanup_reported(lease)
    assert turn.begin()


def test_only_one_of_many_threads_gets_the_turn(turn):
    got: list[ALease] = []
    refused: list[BaseException] = []
    ready = threading.Barrier(8)

    def race():
        ready.wait()
        try:
            got.append(turn.begin())
        except NotTheOwner as exc:
            refused.append(exc)

    threads = [threading.Thread(target=race) for _ in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    assert len(got) == 1
    assert len(refused) == 7


# ---------------------------------------------------------- stale leases


def test_a_lease_from_before_a_reset_is_stale(turn):
    lease = turn.begin()
    turn.reset("the runtime restarted")
    assert not turn.owns_it(lease)
    assert lease.stopping.stopped
    with pytest.raises(NotTheOwner, match="nothing is"):
        turn.finish(lease)


def test_a_lease_from_the_previous_turn_cannot_finish_the_current_one(turn):
    first = turn.begin()
    turn.finish(first)
    second = turn.begin()

    with pytest.raises(NotTheOwner):
        turn.finish(first)
    assert turn.owns_it(second)
    assert turn.status is TurnStatus.ACTIVE


def test_a_forged_lease_with_the_right_id_is_still_not_the_owner():
    """An id can be copied. A lease is identity, not a string."""
    turn = TheTurn()
    real = turn.begin(turn_id="the-same-id")
    forged = ALease(turn_id="the-same-id", stopping=real.stopping)

    assert not turn.owns_it(forged)
    with pytest.raises(NotTheOwner):
        turn.finish(forged)


# ---------------------------------------------------------- double finish


def test_finishing_twice_raises_rather_than_ending_somebody_else(turn):
    lease = turn.begin()
    turn.finish(lease)
    with pytest.raises(NotTheOwner):
        turn.finish(lease)


def test_finishing_twice_across_a_new_turn_does_not_end_the_new_turn(turn):
    first = turn.begin()
    turn.finish(first)
    second = turn.begin()
    with pytest.raises(NotTheOwner):
        turn.finish(first)
    assert turn.status is TurnStatus.ACTIVE
    assert turn.owns_it(second)


# ------------------------------------------------- cancelling is not idle


def test_cancelling_tells_the_owner_and_does_not_end_the_turn(turn):
    lease = turn.begin()
    assert turn.cancel("the user pressed stop") is True
    assert turn.status is TurnStatus.CANCELLING
    assert lease.stopping.stopped
    assert lease.stopping.why == "the user pressed stop"


def test_cancelling_nothing_says_so(turn):
    assert turn.cancel("stop") is False


def test_cancelling_twice_is_not_two_cancellations(turn):
    turn.begin()
    assert turn.cancel("first") is True
    assert turn.cancel("second") is False


def test_the_owner_finishing_first_still_waits_for_the_cleanup(turn):
    lease = turn.begin()
    turn.cancel("stop")
    assert turn.finish(lease) is TurnStatus.CANCELLING
    assert turn.report()["waiting_on"] == ["the cleanup to report"]
    assert turn.cleanup_reported(lease) is TurnStatus.IDLE


def test_the_cleanup_reporting_first_still_waits_for_the_owner(turn):
    lease = turn.begin()
    turn.cancel("stop")
    assert turn.cleanup_reported(lease) is TurnStatus.CANCELLING
    assert turn.report()["waiting_on"] == ["the owner to finish"]
    assert turn.finish(lease) is TurnStatus.IDLE


def test_a_cleanup_report_for_a_gone_turn_cannot_end_the_one_that_replaced_it(turn):
    """Late cleanup is the case this takes a lease for."""
    first = turn.begin()
    turn.cancel("stop")
    turn.finish(first)
    turn.cleanup_reported(first)
    second = turn.begin()

    with pytest.raises(NotTheOwner):
        turn.cleanup_reported(first)
    assert turn.status is TurnStatus.ACTIVE
    assert turn.owns_it(second)


def test_a_cleanup_report_on_a_turn_that_was_not_cancelled_changes_nothing(turn):
    lease = turn.begin()
    assert turn.cleanup_reported(lease) is TurnStatus.ACTIVE
    assert turn.status is TurnStatus.ACTIVE


# --------------------------------------------------------- atomic takeover


def test_a_command_takes_over_and_the_previous_owner_is_told(turn):
    first = turn.begin(origin="user")
    second = turn.take_over(origin="command")

    assert turn.owns_it(second)
    assert not turn.owns_it(first)
    assert first.stopping.stopped
    assert first.stopping.why == "taken over by a command"
    assert turn.status is TurnStatus.ACTIVE


def test_a_takeover_leaves_no_window_where_nothing_owns_the_runtime(turn):
    """Either the command owns it or the previous turn still does.

    A stop-then-start would leave the runtime unowned in between, and that is
    where a third caller starts a turn on top of a teardown.
    """
    turn.begin(origin="user")
    seen: list[TurnStatus] = []

    def watch():
        for _ in range(2000):
            seen.append(turn.status)

    watcher = threading.Thread(target=watch)
    watcher.start()
    for _ in range(50):
        turn.take_over(origin="command")
    watcher.join()

    assert TurnStatus.IDLE not in seen


def test_a_takeover_works_from_idle_too(turn):
    lease = turn.take_over(origin="command")
    assert turn.owns_it(lease)
    assert turn.status is TurnStatus.ACTIVE


def test_the_stale_owner_of_a_taken_over_turn_cannot_finish_it(turn):
    first = turn.begin(origin="user")
    turn.take_over(origin="command")
    with pytest.raises(NotTheOwner):
        turn.finish(first)


# --------------------------------------------------------------- reading


def test_the_report_names_what_is_still_settling(turn):
    lease = turn.begin(origin="user", turn_id="t-1")
    turn.cancel("stop")
    report = turn.report()
    assert report["status"] == "cancelling"
    assert report["turn_id"] == "t-1"
    assert report["why_cancelled"] == "stop"
    assert set(report["waiting_on"]) == {
        "the owner to finish",
        "the cleanup to report",
    }
    turn.finish(lease)
    turn.cleanup_reported(lease)
    assert turn.report()["waiting_on"] == []


def test_the_process_has_exactly_one_turn_state():
    assert the_turn() is the_turn()


def test_the_report_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["whose_turn_it_is"]
    assert set(block) >= {"status", "turn_id", "waiting_on"}


# ------------------------------------------------------------ the wiring


def test_a_live_conversation_turn_owns_the_runtime_while_it_runs():
    """The scope that owns evidence for one turn owns the turn."""
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    the_turn().reset("test")
    with bind_turn_evidence_custody(session_id="s", turn_id="t-live"):
        report = the_turn().report()
        assert report["status"] == "active"
        assert report["turn_id"] == "t-live"
        assert report["origin"] == "conversation"
    assert the_turn().status is TurnStatus.IDLE


def test_a_superseded_turn_finds_out_through_its_lease():
    """Not by writing into the turn that replaced it."""
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    the_turn().reset("test")
    with bind_turn_evidence_custody(session_id="s", turn_id="t-first"):
        superseded = the_turn().lease
        with bind_turn_evidence_custody(session_id="s", turn_id="t-second"):
            assert the_turn().report()["turn_id"] == "t-second"
            assert superseded.stopping.stopped
            assert superseded.stopping.why == "taken over by a command"
        # The inner turn finished; the outer one is stale and says so on exit
        # rather than ending a turn it no longer owns.
    assert the_turn().status is TurnStatus.IDLE


def test_a_turn_that_raises_still_hands_the_runtime_back():
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody

    the_turn().reset("test")
    with pytest.raises(ZeroDivisionError):
        with bind_turn_evidence_custody(session_id="s", turn_id="t-bad"):
            raise ZeroDivisionError
    assert the_turn().status is TurnStatus.IDLE
