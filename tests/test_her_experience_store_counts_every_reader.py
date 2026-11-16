"""Every connection to her experience store is counted, so a wait for quiet waits for all of them (Master Todo Q09).

Her ontogeny service read episodes' provenance through a read-only connection of its own, which the spine's count did
not see: wait_until_quiet could say the store was quiet while that reader held it open, and the hermetic guard,
measuring after it, reported the store's three files as a leak of whichever test was in teardown.
"""
from __future__ import annotations

import threading

import pytest

from core.ontogeny.experience import ExperienceSpine

pytestmark = pytest.mark.unit


def test_a_read_only_reader_is_waited_for(tmp_path):
    spine = ExperienceSpine(db_path=tmp_path / "experience.db")
    try:
        holding, release = threading.Event(), threading.Event()

        def read():
            with spine.reading_the_store() as conn:
                conn.execute("SELECT 1").fetchone()
                holding.set()
                release.wait(5)

        reader = threading.Thread(target=read)
        reader.start()
        assert holding.wait(5)
        assert spine.a_write_is_in_flight()
        assert not spine.wait_until_quiet(0.2)
        release.set()
        reader.join(5)
        assert spine.wait_until_quiet(1.0)
    finally:
        spine.close()
