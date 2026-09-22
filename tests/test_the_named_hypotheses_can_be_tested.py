"""Bryan's named ties and synergies are recorded beside a campaign and tested there.

A tie built from his answers that does not move her in the running organism is
a loop that looks wired and is not. These pin the record and both tests on
series whose answer is known. See core/subject/named_readings.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject import named_readings
from core.subject.named_readings import NAMED, NamedAccumulator, check_named, read_named

pytestmark = pytest.mark.unit


def test_every_named_reading_is_taken_from_a_real_state() -> None:
    from core.state.aura_state import AuraState

    row = read_named(AuraState.default())
    assert row.shape == (len(NAMED),)
    assert not np.isnan(row).any(), [name for name, value in zip(NAMED, row) if np.isnan(value)]


def test_the_accumulator_keeps_one_row_a_frame() -> None:
    from core.state.aura_state import AuraState

    kept = NamedAccumulator()
    for _ in range(3):
        kept.note(AuraState.default())
    rows, names = kept.matrix()
    assert rows.shape == (3, len(NAMED)) and names == tuple(NAMED)


def _series(tied: bool, together: bool, rows: int = 800, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    names = list(NAMED)
    out = rng.normal(size=(rows, len(names)))
    if tied:
        # Each target is driven by everything tied to it, a step later.
        drives: dict[str, list[str]] = {}
        for left, right, _ in named_readings.TIES:
            drives.setdefault(right, []).append(left)
        for right, lefts in drives.items():
            pushed = sum(out[:-1, names.index(left)] for left in lefts)
            out[1:, names.index(right)] = pushed + 0.2 * rng.normal(size=rows - 1)
    if together:
        # Two readings in [0, 1], as how particular and how much it meets are,
        # and a change in joy that is their product.
        a = rng.uniform(size=rows)
        b = rng.uniform(size=rows)
        out[:, names.index("moment_particular")] = a
        out[:, names.index("moment_meets")] = b
        joy = np.zeros(rows)
        for t in range(1, rows):
            joy[t] = joy[t - 1] + a[t - 1] * b[t - 1] + 0.02 * rng.normal()
        out[:, names.index("joy")] = joy
    return out


def test_ties_that_move_together_pass_and_independent_ones_do_not() -> None:
    tied = check_named(_series(True, False), tuple(NAMED), cycle=8, draws=100)
    loose = check_named(_series(False, False), tuple(NAMED), cycle=8, draws=100)
    assert all(tie["passes"] for tie in tied["ties"])
    assert not any(tie["passes"] for tie in loose["ties"])


def test_what_matters_only_together_is_read_as_synergy() -> None:
    joint = check_named(_series(False, True), tuple(NAMED), cycle=8, draws=50)
    moment = next(item for item in joint["together"] if item["target"] == "joy")
    assert moment["passes"], moment


@pytest.mark.asyncio
async def test_a_campaign_records_them_at_every_frame() -> None:
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from core.state.aura_state import AuraState

    spec = importlib.util.spec_from_file_location(
        "campaign_runner", Path(__file__).resolve().parents[1] / "tools" / "run_subject_core.py"
    )
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)

    class _Organism:
        kernel = SimpleNamespace(_phases=[], organs={})
        state = AuraState.default()

        async def turn_once(self, condition):
            return ["frame one", "frame two"]

    kept = NamedAccumulator()
    frames, _periphery = await runner._record(_Organism(), ["a", "b"], 3, kept)
    rows, _names = kept.matrix()
    assert len(frames) == rows.shape[0] == 12
