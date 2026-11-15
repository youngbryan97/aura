"""A thread owner's stop() has to leave the thread stopped.

Found by sweeping for the shape of the experience-store leak (a close that
returned while its flusher still held the database): five classes stopped a
background thread and did not wait for it, and two of them could not stop it
at all once restarted.

* SubstrateSyncThread and SubstrateInjectionThread looped on one shared
  ``_running`` flag. stop() set it False and a quick start() set it True
  again before the old loop looked, so the old thread never left.
* OntogenyCore.stop() saved its state while the maintenance pass that saves
  the same state might still be running, and OutcomeSweeper.stop() returned
  before a sweep in flight had written its resolutions.
* EmbeddingEngine.close() could be followed by a loader thread started
  before it, which loaded the encoder and took a lane lease on a closed engine.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from core.consciousness.affective_steering import SubstrateSyncThread
from core.consciousness.latent_bridge import SubstrateInjectionThread


def _wait_for(predicate, seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_a_restarted_sync_thread_leaves_one_loop_not_two():
    engine = SimpleNamespace(_state_control_lock=threading.Lock())
    sync = SubstrateSyncThread([], engine)
    sync.start()
    first = sync._thread
    sync.stop()
    sync.start()                      # before the old loop has looked
    second = sync._thread
    # The two thread objects this test made, not every thread with the name:
    # counting by name reads whatever other tests left running.
    assert second is not first
    assert _wait_for(lambda: not first.is_alive()), "the first loop never left"
    assert second.is_alive()
    sync.stop()
    assert _wait_for(lambda: not sync._thread.is_alive())


def test_a_restarted_readout_publisher_leaves_one_loop_not_two():
    pub = SubstrateInjectionThread([], channel=object())
    pub.start()
    first = pub._thread
    pub.stop()
    pub.start()
    second = pub._thread
    assert second is not first
    assert _wait_for(lambda: not first.is_alive()), "the first loop never left"
    assert second.is_alive()
    pub.stop()
    assert _wait_for(lambda: not pub._thread.is_alive())


def test_clearing_the_running_flag_still_ends_either_loop():
    """`_running` is the status readers see, and callers stop a loop by it.

    Once each run had its own event, a loop that watched only the event
    ignored the flag. A test that stopped the sync loop from inside a hook
    that way never saw it stop, and its worker grew past thirteen gigabytes.
    """
    engine = SimpleNamespace(_state_control_lock=threading.Lock())
    for owner in (SubstrateSyncThread([], engine), SubstrateInjectionThread([], channel=object())):
        owner.start()
        thread = owner._thread
        owner._running = False
        assert _wait_for(lambda: not thread.is_alive()), f"{type(owner).__name__} ignored its flag"


def test_the_sweeper_waits_for_a_pass_in_flight():
    from core.ontogeny.resolution import OutcomeSweeper

    sweeper = OutcomeSweeper.__new__(OutcomeSweeper)
    sweeper._interval = 5.0
    sweeper._stopped = threading.Event()
    inside, release = threading.Event(), threading.Event()

    def a_long_pass():
        inside.set()
        release.wait(5.0)

    sweeper._thread = threading.Thread(target=a_long_pass, daemon=True)
    sweeper._thread.start()
    assert inside.wait(2.0)
    threading.Timer(0.2, release.set).start()
    began = time.monotonic()
    sweeper.stop()
    assert not sweeper._thread.is_alive(), "stop() returned with the pass still running"
    assert time.monotonic() - began >= 0.15


def test_an_embedding_load_that_starts_after_close_does_not_load():
    from core.memory.vector_memory_engine import EmbeddingEngine

    engine = EmbeddingEngine.__new__(EmbeddingEngine)
    engine._initialized = False
    engine._closing = True
    engine._model = None
    engine._lane_lease = None
    engine._initialize_locked()          # what a late loader thread runs
    assert engine._model is None
    assert engine._lane_lease is None


def test_the_ontogeny_core_waits_for_maintenance_before_saving():
    """The state is saved once the maintenance pass that also saves it has finished, not beside it."""
    import threading
    from types import SimpleNamespace

    from core.ontogeny.service import OntogenyCore

    happened: list[str] = []
    in_the_pass = threading.Event()
    finish_it = threading.Event()

    def a_maintenance_pass():
        in_the_pass.set()
        finish_it.wait(5.0)
        happened.append("maintenance finished")

    core = OntogenyCore.__new__(OntogenyCore)
    core._stopped = threading.Event()
    core._sweeper = None
    core._maintenance = threading.Thread(target=a_maintenance_pass, daemon=True)
    core._state = SimpleNamespace(save=lambda: happened.append("saved"))
    core._spine = SimpleNamespace(off_resolve=lambda _callback: None, flush=lambda: None)
    core._note_resolution = lambda *_args: None
    core._maintenance.start()
    assert in_the_pass.wait(5.0)
    threading.Timer(0.1, finish_it.set).start()
    core.stop()
    assert happened == ["maintenance finished", "saved"]
