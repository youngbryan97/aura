"""A second branch the schedule cannot integrate is a second branch wasted.

Cross-branch exchange writes its agreement-weighted consensus into the
communication slot. That slot sits ahead of the immutable context prefix and
every private reasoning slot, so it is the one place in the latent workspace
where influence moves backwards through position — everywhere else a causal
mask sends it one way, on every recurrent pass, however many passes run.

The schedule decides whether that matters. An exchange on the final step still
reaches the answer, because the mailbox persists into the K/V the decode
attends to, and reaches no further thinking, because no window pass remains to
carry it forward. Before the depth floor and the interval clamp, every
multi-branch allocation this allocator could produce landed there.
"""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.branch_exchange import (
    eligible_exchange_steps,
    exchange_steps_a_return_can_travel,
)

GRID = [(stakes, doubt) for stakes in (0.1, 0.3, 0.5, 0.7, 0.9) for doubt in (0.1, 0.5, 0.9)]


def test_one_branch_is_never_eligible() -> None:
    assert (
        eligible_exchange_steps(
            n_branches=1, max_steps=12, isolation_steps=2, exchange_interval=4
        )
        == ()
    )


def test_isolation_holds_the_first_exchange_back() -> None:
    """``exchange`` refuses while isolation is unsealed and counts the refusal.

    Isolation seals only once every branch has taken ``isolation_steps``, so
    step 4 is eligible here and step 8 is not held back by anything.
    """
    assert eligible_exchange_steps(
        n_branches=2, max_steps=12, isolation_steps=5, exchange_interval=4
    ) == (8, 12)


def test_a_depth_equal_to_the_interval_leaves_the_return_on_the_last_step() -> None:
    eligible = eligible_exchange_steps(
        n_branches=2, max_steps=4, isolation_steps=2, exchange_interval=4
    )
    travels = exchange_steps_a_return_can_travel(
        n_branches=2, max_steps=4, isolation_steps=2, exchange_interval=4
    )

    assert eligible == (4,)
    assert travels == ()


def test_pulling_the_interval_in_gives_the_same_depth_somewhere_to_go() -> None:
    assert exchange_steps_a_return_can_travel(
        n_branches=2, max_steps=4, isolation_steps=2, exchange_interval=3
    ) == (3,)


def test_a_depth_at_the_isolation_cannot_be_fixed_by_any_interval() -> None:
    """Depth is the binding constraint below ``isolation_steps + 1``.

    No interval helps: the first step an exchange may take is the step the
    episode ends on. This is the case the allocator's depth floor exists for,
    and the reason the clamp leaves the interval alone instead of pretending.
    """
    for interval in range(1, 6):
        assert (
            exchange_steps_a_return_can_travel(
                n_branches=2, max_steps=2, isolation_steps=2, exchange_interval=interval
            )
            == ()
        )


@pytest.mark.parametrize("foreground", [False, True])
def test_every_multi_branch_allocation_leaves_a_return_that_can_travel(
    foreground: bool,
) -> None:
    """The claim, asked of the allocator rather than of a hand-written config.

    A second branch costs a second trajectory through the recurrent window. If
    the schedule leaves its consensus on the final step, that spend reaches the
    answer surface and nothing else, and this fails. It failed on 14 of 14
    multi-branch allocations before the depth floor in ``adaptive_compute`` and
    the interval clamp in ``allocate``.
    """
    from core.brain.latent_cortex_service import LatentCortexService

    service = LatentCortexService()
    checked = 0
    for stakes, uncertainty in GRID:
        config, _budget = service.allocate(
            stakes=stakes,
            uncertainty=uncertainty,
            foreground_request=foreground,
            model_parameter_count=27_000_000_000 if foreground else 0,
        )
        if int(config["n_branches"]) < 2:
            continue
        checked += 1
        assert exchange_steps_a_return_can_travel(
            n_branches=int(config["n_branches"]),
            max_steps=int(config["max_steps"]),
            isolation_steps=int(config["isolation_steps"]),
            exchange_interval=int(config["exchange_interval"]),
        ), (
            f"stakes={stakes} uncertainty={uncertainty} foreground={foreground} "
            f"allocated {config['n_branches']} branches over {config['max_steps']} "
            f"steps with interval {config['exchange_interval']} and isolation "
            f"{config['isolation_steps']}"
        )
    assert checked, "no allocation in the grid granted a second branch"


def test_the_depth_floor_only_moves_a_depth_that_was_too_short() -> None:
    """The floor is a floor. A deep allocation keeps the depth it was given."""
    from core.brain.llm.latent_cortex.adaptive_compute import build_adaptive_compute_plan

    deep = build_adaptive_compute_plan(
        objective="Work out whether the two proofs agree, and say where they differ.",
        stakes=0.9,
        uncertainty=0.9,
        body_pressure=0.0,
        deadline_s=600.0,
        resource_snapshot={"observation_source": "test", "memory_percent": 30.0},
        foreground_request=False,
        model_parameter_count=0,
        requested_decode_tokens=512,
        isolation_steps=2,
    )
    recurrence = deep["routing"]["recurrence"]

    assert deep["routing"]["branches"]["target"] > 1
    # The floor would have set 3. A calm, unhurried, high-demand allocation is
    # deeper than that on the ladder's own terms, so the floor must not be what
    # decided this number.
    assert recurrence["max_steps"] > 2 + 1
    assert recurrence["min_steps"] <= recurrence["max_steps"]
