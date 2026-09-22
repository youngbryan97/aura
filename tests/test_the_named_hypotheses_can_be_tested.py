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


# ── controls for the tie statistic, added with the amendment of 22 September ──


def _walks(rows: int = 3000, seed: int = 0, *, drift: float = 0.0) -> np.ndarray:
    """Every reading an independent random walk: drifting, and tied to nothing."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(rows, len(NAMED))) + drift
    return np.cumsum(steps, axis=0)


@pytest.mark.parametrize("seed", range(6))
def test_independent_walks_do_not_pass_as_ties(seed: int) -> None:
    """On levels these correlate at 0.9 and more by drift alone; on changes they do not."""
    report = check_named(_walks(seed=seed), tuple(NAMED), cycle=8, draws=100)
    assert not any(tie["passes"] for tie in report["ties"]), report["ties"]


def test_independent_trends_do_not_pass_as_ties() -> None:
    report = check_named(_walks(seed=11, drift=0.05), tuple(NAMED), cycle=8, draws=100)
    assert not any(tie["passes"] for tie in report["ties"]), report["ties"]


def test_a_reading_that_follows_another_s_changes_passes() -> None:
    rng = np.random.default_rng(4)
    names = list(NAMED)
    matrix = _walks(seed=4)
    # A target named by two ties follows both, as the synthetic series above does.
    drives: dict[str, list[str]] = {}
    for left, right, _ in named_readings.TIES:
        drives.setdefault(right, []).append(left)
    for right, lefts in drives.items():
        source = sum(np.diff(matrix[:, names.index(left)], prepend=0.0) for left in lefts)
        follows = np.zeros(len(source))
        follows[1:] = source[:-1] + 0.5 * rng.normal(size=len(source) - 1)
        matrix[:, names.index(right)] = np.cumsum(follows)
    report = check_named(matrix, tuple(NAMED), cycle=8, draws=100)
    assert all(tie["passes"] for tie in report["ties"]), report["ties"]


def test_a_reading_that_never_moved_is_not_measured_rather_than_failed() -> None:
    matrix = _walks(seed=2)
    matrix[:, list(NAMED).index("unknown_person")] = 0.0
    report = check_named(matrix, tuple(NAMED), cycle=8, draws=50)
    tie = next(t for t in report["ties"] if t["pair"][0] == "unknown_person")
    assert tie["measured"] is False and tie["passes"] is None
    assert "unknown_person never moved" in tie["why"]


def _content(kinds: list[str], distance) -> dict:
    classes = [{"name": f"c{i}", "kind": kind} for i, kind in enumerate(kinds)]
    internal = {f"{i}-{j}": distance(kinds[i], kinds[j]) for i in range(len(kinds)) for j in range(i + 1, len(kinds))}
    return {"classes": classes, "internal": internal, "internal_floor": {"shared": 0.01}}


_KINDS = ["cared_for", "positive_interaction", "interaction", "extended_dialogue",
          "disconnection", "error", "internal_error", "self_correction", "inner_conflict"]


def _family(kind: str) -> str:
    return next(name for name, members in named_readings.FAMILIES.items() if kind in members)


def test_warmth_that_holds_together_passes_h1() -> None:
    near = lambda a, b: 0.2 if _family(a) == _family(b) == "warmth" else 1.0  # noqa: E731
    verdict = named_readings.check_families(_content(_KINDS, near))
    assert verdict["measured"] and verdict["passes"] and verdict["margins_clear_the_floor"]


def test_warmth_washed_out_fails_h1() -> None:
    """Warmth kinds as far from each other as from loss: the grouping did not survive."""
    flat = lambda a, b: 0.5  # noqa: E731
    verdict = named_readings.check_families(_content(_KINDS, flat))
    assert verdict["measured"] and not verdict["passes"]


def test_a_run_without_the_loss_kind_does_not_measure_h1() -> None:
    kinds = [k for k in _KINDS if k != "disconnection"]
    verdict = named_readings.check_families(_content(kinds, lambda a, b: 0.3))
    assert verdict["measured"] is False and "loss" in verdict["why"]


def test_h4_holds_when_the_drives_only_null_is_not_one_component() -> None:
    assert named_readings.check_drives_only({"drives_only": {"one_component": False}})["passes"] is True
    assert named_readings.check_drives_only({"drives_only": {"one_component": True}})["passes"] is False
    unread = named_readings.check_drives_only({"ring": {"one_component": True}})
    assert unread["measured"] is False and unread["passes"] is None
