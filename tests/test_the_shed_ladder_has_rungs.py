"""The OOM ladder fills as the runtime does, and its rungs free real bytes.

The ladder had one rung in the whole tree. Discovery was a single sweep of
the already-instantiated services in boot wave one — when the container is
nearly empty by design — so everything built afterwards, which is
everything, was never offered. The MLX client found this and worked round
it for itself by registering from its own __init__; the offer now happens
where any singleton becomes real, so no organ has to know.
"""

from __future__ import annotations

from collections import deque

import pytest

from core.runtime.oom_policy import (
    get_oom_policy,
    offer_organ,
    reset_oom_policy_for_test,
)
from core.runtime.sheddable import CacheHolder, retained_bytes


@pytest.fixture(autouse=True)
def _clean_policy():
    reset_oom_policy_for_test()
    yield
    reset_oom_policy_for_test()


class _Organ(CacheHolder):
    sheddable_caches = ("_history", "_index")
    oom_rationale = "recent readings, refilled by the next tick"

    def __init__(self) -> None:
        self._history: deque[str] = deque(maxlen=2000)
        self._index: dict[str, list[int]] = {}
        self._ledger = ["durable", "not sheddable"]

    def fill(self) -> None:
        for step in range(500):
            self._history.append(f"reading-{step}-" + "x" * 64)
            self._index[f"k{step}"] = list(range(8))


def test_an_organ_with_no_shed_hook_is_not_offered():
    class Plain:
        pass

    assert offer_organ("plain", Plain()) is False
    assert get_oom_policy().report()["registered_organs"] == 0


def test_an_organ_that_can_shed_is_offered_and_becomes_a_rung():
    organ = _Organ()
    organ.fill()
    assert offer_organ("readings", organ) is True
    report = get_oom_policy().report()
    assert report["sheddable_organs"] == 1
    row = next(r for r in get_oom_policy().scoring_table() if r["organ"] == "readings")
    assert row["sheddable"] is True


def test_the_footprint_is_measured_rather_than_declared():
    organ = _Organ()
    empty = organ.memory_footprint_bytes()
    organ.fill()
    assert organ.memory_footprint_bytes() > empty * 4


def test_a_shed_reports_what_it_actually_freed():
    organ = _Organ()
    organ.fill()
    held = organ.memory_footprint_bytes()
    freed = organ.shed_memory()
    assert freed > 0
    assert freed <= held
    assert organ.memory_footprint_bytes() < held
    # Nothing durable is touched: it was never named.
    assert organ._ledger == ["durable", "not sheddable"]


def test_a_second_shed_frees_nothing_and_says_so():
    # A rung that keeps reporting progress it did not make is how a shed
    # loop spins instead of escalating.
    organ = _Organ()
    organ.fill()
    organ.shed_memory()
    assert organ.shed_memory() == 0


def test_retained_bytes_counts_a_shared_object_once():
    shared = ["x" * 1000]
    twice = [shared, shared]
    once = [shared]
    assert retained_bytes(twice) < 2 * retained_bytes(once)


def test_the_container_offers_every_singleton_it_builds():
    import inspect

    from core import container as container_module

    source = inspect.getsource(container_module)
    # Both places a singleton instance is bound must offer it.
    assert source.count("_offer_to_the_shed_order(resolved_name, instance)") == 2
    bindings = source.count("desc.initialized = True")
    assert bindings >= 2, "a new singleton binding must offer to the shed order too"
