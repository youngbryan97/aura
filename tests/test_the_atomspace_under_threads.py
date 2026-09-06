"""Two threads discovering absence together, and the lock lockdep can see.

The blind comparison singled out the AtomTable: it acknowledges that two
threads can find an atom absent at the same moment, rechecks, and repairs the
losing installation. It called that exactly the concurrency reasoning Aura's
resource handoff had not completed.

Aura's AtomSpace takes one lock across the whole insert, so the recheck is
structural rather than optimistic — there is no window to lose. These tests
hold that property down, and one of them holds down the other half: the lock
is a checked one, so lockdep can see it. A raw lock is not a smaller version
of a checked one; it is invisible to the thing that finds deadlocks.
"""
from __future__ import annotations

import threading

from core.knowledge.atomspace import AtomSpace, Link, Node, TruthValue


def test_many_threads_adding_one_atom_leave_exactly_one():
    space = AtomSpace()
    atom = Node("CONCEPT", "Bryan")
    ready = threading.Barrier(16)

    def push():
        ready.wait()
        for _ in range(20):
            space.add(atom)

    threads = [threading.Thread(target=push) for _ in range(16)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    assert len(space) == 1
    assert space.get_tv(atom) is not None


def test_the_losing_writer_does_not_lose_its_evidence():
    """The repair the comparison named: whoever loses the race is not dropped."""
    space = AtomSpace()
    atom = Node("CONCEPT", "shared")
    ready = threading.Barrier(8)

    def assert_it(who: int):
        ready.wait()
        space.add(atom, TruthValue(0.8, 2.0), source=f"witness-{who}")

    threads = [threading.Thread(target=assert_it, args=(n,)) for n in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    assert space.evidence_sources(atom) == frozenset(
        f"witness-{n}" for n in range(8)
    )


def test_a_link_and_its_children_arrive_whole_under_contention():
    """A half-installed link is a dangling reference by another name."""
    space = AtomSpace()
    ready = threading.Barrier(8)

    def push(n: int):
        ready.wait()
        for i in range(10):
            a, b = Node("CONCEPT", f"a{i}"), Node("CONCEPT", f"b{i}")
            space.add(Link("RELATES", (a, b)))

    threads = [threading.Thread(target=push, args=(n,)) for n in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()

    for i in range(10):
        a, b = Node("CONCEPT", f"a{i}"), Node("CONCEPT", f"b{i}")
        link = Link("RELATES", (a, b))
        assert space.get_tv(link) is not None
        assert space.get_tv(a) is not None
        assert space.get_tv(b) is not None


def test_the_atomspace_lock_is_one_lockdep_can_see():
    """It was a bare threading.RLock, which lockdep cannot wrap."""
    from core.runtime.lockdep import CheckedLock

    assert isinstance(AtomSpace()._lock, CheckedLock)


def test_the_number_of_locks_lockdep_cannot_see_only_goes_down():
    """The ratchet. Hundreds exist; this stops there being more."""
    import json
    from pathlib import Path

    from tools.lint_raw_locks import raw_locks

    root = Path(__file__).resolve().parents[1]
    allowed = json.loads(
        (root / "config" / "raw_lock_baseline.json").read_text("utf-8")
    )["count"]
    found = raw_locks(root)
    assert len(found) <= allowed, (
        f"{len(found)} raw locks under core, baseline {allowed}; "
        f"newest: {found[-5:]}"
    )
