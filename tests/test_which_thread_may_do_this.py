"""What the loop's thread may not do, and what it did anyway.

Home Assistant took the top maturity score partly for thread-affinity
enforcement. Aura has been bitten by exactly this: an on-loop fsync once froze
the live event loop for twenty minutes, and the fix was a rule written in a
guide. A rule in a guide is not enforcement — nothing could tell you it had
been broken until the loop stopped answering.
"""
from __future__ import annotations

import threading

import pytest

from core.runtime.which_thread_may_do_this import (
    AKindOfWork,
    OnTheWrongThread,
    forget_everything,
    how_it_has_gone,
    note_the_loop_thread,
    strictly,
    the_loop_thread,
    this_is,
    we_are_on_the_loop,
)


@pytest.fixture(autouse=True)
def _clean():
    forget_everything()
    yield
    forget_everything()


# ------------------------------------------------------------ which thread


def test_nothing_is_the_loop_thread_until_something_says_so():
    assert the_loop_thread() is None
    assert we_are_on_the_loop() is False


def test_the_loop_thread_says_who_it_is():
    ident = note_the_loop_thread()
    assert ident == threading.get_ident()
    assert we_are_on_the_loop() is True


def test_another_thread_is_not_the_loop():
    note_the_loop_thread()
    seen = []
    other = threading.Thread(target=lambda: seen.append(we_are_on_the_loop()))
    other.start()
    other.join()
    assert seen == [False]


def test_not_knowing_the_loop_thread_reads_as_not_on_it():
    """The safe answer: assuming yes lets a blocking call through unmeasured."""
    assert we_are_on_the_loop() is False


# ------------------------------------------------------------- the rules


def test_a_blocking_call_on_the_loop_is_recorded():
    note_the_loop_thread()
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
        pass

    went = how_it_has_gone()
    assert went["rules_broken"] == 1
    assert went["broken"]["an fsync"]["times"] == 1
    assert went["broken"]["an fsync"]["kind"] == "never on the loop"


def test_a_blocking_call_off_the_loop_is_not_recorded():
    note_the_loop_thread()

    def elsewhere():
        with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
            pass

    other = threading.Thread(target=elsewhere)
    other.start()
    other.join()
    assert how_it_has_gone()["rules_broken"] == 0


def test_loop_work_on_a_worker_thread_is_the_mirror_image_defect():
    note_the_loop_thread()

    def elsewhere():
        with this_is(AKindOfWork.ON_THE_LOOP, "touching loop state"):
            pass

    other = threading.Thread(target=elsewhere)
    other.start()
    other.join()
    assert "touching loop state" in how_it_has_gone()["broken"]


def test_work_that_may_run_anywhere_is_never_wrong():
    note_the_loop_thread()
    with this_is(AKindOfWork.ANYWHERE, "reading a dict"):
        pass
    assert how_it_has_gone()["rules_broken"] == 0


def test_the_time_spent_on_the_loop_is_what_gets_added_up():
    """A millisecond is untidy; twenty minutes is the runtime being down."""
    import time

    note_the_loop_thread()
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "a slow parse"):
        time.sleep(0.02)

    assert how_it_has_gone()["seconds_on_the_loop"] >= 0.02


def test_the_same_rule_broken_twice_is_counted_twice():
    note_the_loop_thread()
    for _ in range(3):
        with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
            pass
    assert how_it_has_gone()["broken"]["an fsync"]["times"] == 3


def test_the_record_names_where_it_happened_not_the_machinery():
    """Without skipping contextlib every site reads as contextlib.py."""
    note_the_loop_thread()
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
        pass
    where = how_it_has_gone()["broken"]["an fsync"]["where"]
    assert where
    assert all("contextlib.py" not in one for one in where)
    assert any("test_which_thread_may_do_this.py" in one for one in where)


# ----------------------------------------------------------- raising loudly


def test_strictly_raises_instead_of_recording():
    note_the_loop_thread()
    with pytest.raises(OnTheWrongThread, match="never on the loop"):
        with strictly():
            with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
                pass


def test_strictly_is_per_thread_and_ends_with_the_block():
    note_the_loop_thread()
    with strictly():
        pass
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
        pass
    assert how_it_has_gone()["rules_broken"] == 1


def test_the_work_still_runs_when_the_rule_is_only_recorded():
    """A rule that crashes the runtime gets turned off; one that counts gets fixed."""
    note_the_loop_thread()
    ran = []
    with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
        ran.append(True)
    assert ran == [True]


def test_an_exception_inside_the_block_still_records_the_rule():
    note_the_loop_thread()
    with pytest.raises(ZeroDivisionError):
        with this_is(AKindOfWork.NEVER_ON_THE_LOOP, "an fsync"):
            raise ZeroDivisionError
    assert how_it_has_gone()["rules_broken"] == 1


def test_the_record_is_in_the_health_report():
    from core.runtime.health_contract import runtime_health_report

    block = runtime_health_report()["integrity"]["which_thread_may_do_this"]
    assert set(block) >= {"loop_thread", "rules_broken", "seconds_on_the_loop"}


# ------------------------------------------------------------ the wiring


def test_a_synchronous_gateway_write_on_the_loop_is_recorded(tmp_path):
    """The rule was in a guide and a ratchet caught sync writes inside async def.

    Neither can see a sync writer called FROM async through two intermediate
    frames. This can, because it asks which thread it is actually on.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    note_the_loop_thread()
    with local_internal_governed_scope("test.thread_affinity", domain="state_mutation"):
        get_file_write_gateway().write_text(
            tmp_path / "on_the_loop.txt", "x", source="a_test"
        )

    broken = how_it_has_gone()["broken"]
    assert any(name.startswith("write_text:") for name in broken)


def test_the_same_write_off_the_loop_is_not_recorded(tmp_path):
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    note_the_loop_thread()

    def elsewhere():
        with local_internal_governed_scope(
            "test.thread_affinity", domain="state_mutation"
        ):
            get_file_write_gateway().write_text(
                tmp_path / "off_the_loop.txt", "x", source="a_test"
            )

    other = threading.Thread(target=elsewhere)
    other.start()
    other.join()
    assert how_it_has_gone()["rules_broken"] == 0


def test_the_boot_path_tells_this_module_which_thread_the_loop_is():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "runtime" / "foundations.py"
    ).read_text("utf-8")
    assert "note_the_loop_thread()" in source
