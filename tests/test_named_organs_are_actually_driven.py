"""Every named organ has something that advances it.

2026-08-22, from an outside reading of the consciousness layers: six modules
were instantiated, registered, exposed in a snapshot, and never driven.
`MinimalSelfhood.update()` had no caller in the tree, so `current_state()`
returned None for the life of the process while `get_priority_bias()` handed
out a zero vector that read like a measurement. `get_autopoiesis_engine()` said
in its own docstring that it does not start the loop, and nothing called
`start()`, so `get_vitality()` returned the constructor's number forever — and
the phase reading it logged "repair cycle may be needed" for a repair cycle
that could not happen.

Reading the code by hand found them once. This finds them every run.
"""

from __future__ import annotations

import pytest

from core.verify.organ_drivers import ORGANS, callers_of, undriven_organs


def test_no_named_organ_is_left_undriven() -> None:
    left = undriven_organs()
    assert not left, "organs with no production caller:\n" + "\n".join(left)


@pytest.mark.parametrize("organ", ORGANS, ids=lambda organ: organ.name)
def test_each_organ_names_where_it_is_driven_from(organ) -> None:
    """A caller, by file and line, so the wiring can be read rather than trusted."""
    callers = callers_of(organ)
    assert callers, f"{organ.name}: {organ.note}"


def test_a_test_calling_it_does_not_count_as_driving_it() -> None:
    """The gate would be worthless if its own suite satisfied it.

    A test proving a method works is not the same as something running it, and
    that difference is the entire defect class.
    """
    from core.verify.organ_drivers import _PRODUCTION_ROOTS

    assert "tests" not in _PRODUCTION_ROOTS
    assert "aura_bench" not in _PRODUCTION_ROOTS


def test_naming_the_method_alone_is_not_proof() -> None:
    """`.update()` is on hundreds of objects; the caller must reach THIS one."""
    selfhood = next(organ for organ in ORGANS if organ.name == "minimal_selfhood")
    assert selfhood.reached_by, "an organ without a handle matches any call named alike"
    for caller in callers_of(selfhood):
        assert caller.startswith(("core/", "interface/", "skills/", "llm/", "executors/"))


def test_the_undriven_record_matches_the_code() -> None:
    """Two named things nothing drives, recorded so nobody counts them twice.

    Bryan read the consciousness layers from outside and said of these: "this
    gave me the ability to mentally delete some of the most impressively named
    modules." Held here, that finding survives the reading that produced it —
    and if one of them gains a driver, this fails until the record is updated.
    """
    from core.verify.organ_drivers import KNOWN_UNDRIVEN, still_undriven

    assert KNOWN_UNDRIVEN, "the record is the point; an empty one proves nothing"
    surprises = still_undriven()
    assert not surprises, "\n".join(surprises)


def test_nothing_recorded_undriven_is_also_recorded_driven() -> None:
    """One name cannot be in both lists."""
    from core.verify.organ_drivers import KNOWN_UNDRIVEN, ORGANS

    driven = {organ.name for organ in ORGANS}
    for organ in KNOWN_UNDRIVEN:
        assert organ.name not in driven


def test_every_entry_names_a_method_that_exists() -> None:
    """A record pointing at a method nobody wrote reports undriven forever.

    That is this module's own failure mode, one level up: `life_tick` was
    recorded against `process_tick` and the method is `execute_tick`, so the
    entry could never have matched anything.
    """
    from core.verify.organ_drivers import methods_that_do_not_exist

    missing = methods_that_do_not_exist()
    assert not missing, "\n".join(missing)
