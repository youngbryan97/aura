"""`close()` returned while the flusher still held the database open.

The experience spine counts open handles precisely so a shutdown can wait for
them, and its own docstring says shutdown wants that wait. Close never took
it: it set the stop flag, flushed, and returned, so the flusher thread could
still be inside `_using_the_store`. A hermetic teardown saw three handles left
on experience.db, its WAL and its shared-memory file, and reported the leak
against whichever test happened to run last.
"""

from __future__ import annotations

import threading
import time

from core.ontogeny.experience import ExperienceSpine


def _spine(tmp_path) -> ExperienceSpine:
    return ExperienceSpine(db_path=tmp_path / "experience.db")


def test_close_leaves_nothing_holding_the_store(tmp_path):
    spine = _spine(tmp_path)
    spine.close()
    assert not spine.a_write_is_in_flight()
    assert spine.wait_until_quiet(timeout=0.0) is True


def test_close_waits_for_a_holder_that_is_still_inside(tmp_path):
    """The wait is the point: a close that does not take it returns early."""
    spine = _spine(tmp_path)
    released = threading.Event()
    inside = threading.Event()

    def hold() -> None:
        with spine._using_the_store():
            inside.set()
            released.wait(5.0)

    holder = threading.Thread(target=hold, name="test-holder", daemon=True)
    holder.start()
    assert inside.wait(5.0), "the holder never opened the store"
    assert spine.a_write_is_in_flight()

    def let_go() -> None:
        time.sleep(0.2)
        released.set()

    threading.Thread(target=let_go, daemon=True).start()
    began = time.monotonic()
    spine.close()
    assert not spine.a_write_is_in_flight(), "close returned with the store still held"
    assert time.monotonic() - began >= 0.15, "close did not wait for the holder"
    holder.join(timeout=5.0)


def test_the_flusher_thread_is_gone_after_close(tmp_path):
    spine = _spine(tmp_path)
    spine.close()
    alive = [one for one in threading.enumerate() if one.name == "ontogeny-experience-flush"]
    assert alive == [], f"the flusher outlived close: {alive}"
