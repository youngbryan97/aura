"""A generator nothing can seed is a floor under every paired measurement.

Two arms of an intervention are subtracted from each other, and that
subtraction is a measurement only if the arms would have been identical without
the displacement. A `random.Random()` of its own breaks that twice: `seed()`
does not reach it, so a campaign cannot be reproduced; and a snapshot of the
process generators cannot put it back, so two arms started from one state draw
different numbers and the difference lands in the floor.

The synaptic cleft held one. Release through it is probabilistic, every
interiority faculty publishes through it, and the layer runs on every turn.
"""

from __future__ import annotations

import random

from tools.audit_unseeded_randomness import findings


def test_nothing_holds_a_generator_the_process_cannot_reach() -> None:
    loose = findings()
    assert not loose, "unseeded generators: " + "; ".join(
        f"{item['file']}:{item['line']}" for item in loose
    )


def test_the_laboratory_hands_out_a_generator_a_snapshot_can_rewind() -> None:
    """Outside a laboratory `seeded` returned `random.Random()`.

    Every caller that reached for a reproducible generator got an
    irreproducible one, and the helper's own name said otherwise.
    """
    from core.runtime.the_laboratory import seeded, seeded_generator

    saved = random.getstate()
    first = [seeded("probe").random() for _ in range(4)]
    random.setstate(saved)
    assert [seeded("probe").random() for _ in range(4)] == first, (
        "a restored process generator does not reproduce what `seeded` hands out"
    )

    import numpy as np

    state = np.random.get_state()
    drawn = [float(seeded_generator("probe").random()) for _ in range(4)]
    np.random.set_state(state)
    assert [float(seeded_generator("probe").random()) for _ in range(4)] == drawn, (
        "a restored numpy global does not reproduce what `seeded_generator` hands out"
    )


def test_a_laboratory_still_fixes_the_seed_it_declares() -> None:
    from core.runtime.the_laboratory import seeded, under_the_laboratory

    with under_the_laboratory(seed=4):
        first = seeded("probe").random()
    with under_the_laboratory(seed=4):
        assert seeded("probe").random() == first
    with under_the_laboratory(seed=5):
        assert seeded("probe").random() != first


def test_two_subsystems_asking_for_the_seed_do_not_draw_the_same_numbers() -> None:
    """Salting is what stops a shared seed becoming a shared stream.

    Two subsystems drawing identical sequences is a defect that looks like a
    coincidence, which is the worst kind to find later.
    """
    from core.runtime.the_laboratory import seeded, under_the_laboratory

    with under_the_laboratory(seed=11):
        left = [seeded("one").random() for _ in range(3)]
        right = [seeded("two").random() for _ in range(3)]
    assert left != right
