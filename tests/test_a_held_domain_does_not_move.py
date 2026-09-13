"""A domain held at the cut does not move, and releasing the clamp releases only it.

The lesion holds one side of the cheapest partition still while the other side
runs, so that no information crosses. A probe of the old clamp found the held
side moving anyway: the winner columns by more than a whole unit, the
substrate's affect by 0.4, the reservoir, the objective and the origin. The
clamp wrote the state back after each phase, and the layers and the substrate
stepped between that write-back and the reading; and the workspace, substrate,
self-model and world-model columns are read off organs it never held.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.clamp import (
    CLAMPED_FIELDS,
    RESERVOIR_FIELDS,
    Clamp,
    clamped,
    organs_read_by,
    still,
)
from core.subject.state import _SCHEMAS, DOMAINS, domain_slices, feature_names


def test_every_state_source_a_reader_reads_is_held() -> None:
    uncovered: dict[str, list[str]] = {}
    for domain in DOMAINS:
        for source in _SCHEMAS[domain].sources:
            if source.startswith(("organ:", "ontogeny.")):
                continue
            path = source.split("[", 1)[0]
            if not any(path == held or path.startswith(held + ".") for held in CLAMPED_FIELDS[domain]):
                uncovered.setdefault(domain, []).append(path)
    assert not uncovered, uncovered
    assert {"steps", "era", "last_novelty", "last_displacement"} <= set(RESERVOIR_FIELDS)


def test_each_organ_feeds_one_domain_so_holding_it_holds_nothing_else() -> None:
    readers: dict[str, set[str]] = {}
    for domain in DOMAINS:
        for organ in organs_read_by((domain,)):
            readers.setdefault(organ, set()).add(domain)
    assert readers, "no domain is read off an organ, so this checks nothing"
    shared = {organ: sorted(domains) for organ, domains in readers.items() if len(domains) > 1}
    assert not shared, shared


class _Organ:
    def __init__(self) -> None:
        self.valence = 0.1
        self.history = [0.1]


def _runtime(**organs: object) -> SimpleNamespace:
    state = SimpleNamespace(
        cognition=SimpleNamespace(current_mode="calm", phenomenal_state=None),
        phi_estimate=0.2,
        loop_cycle=3,
    )
    return SimpleNamespace(state=state, organs=SimpleNamespace(**organs), ontogeny=None, after_phase=None)


def test_a_field_held_inside_a_dict_is_held() -> None:
    """The workspace's modifiers and the body's physiology are dicts."""
    state = SimpleNamespace(
        cognition=SimpleNamespace(modifiers={"temperature_mod": 1.0}, current_mode="calm", phenomenal_state=None),
        affect=SimpleNamespace(physiology={"heart_rate": 72.0}),
        phi_estimate=0.2,
        loop_cycle=3,
    )
    runtime = SimpleNamespace(state=state, organs=SimpleNamespace(), ontogeny=None, after_phase=None)
    clamp = Clamp(runtime, ("A", "G"))
    state.cognition.modifiers["temperature_mod"] = 1.4
    state.affect.physiology["heart_rate"] = 90.0
    clamp.apply()
    assert state.cognition.modifiers["temperature_mod"] == 1.0
    assert state.affect.physiology["heart_rate"] == 72.0


def test_an_organ_a_held_domain_reads_is_held_where_it_is() -> None:
    substrate = _Organ()
    runtime = _runtime(substrate=substrate, workspace=None)
    history = substrate.history
    clamp = Clamp(runtime, ("C",))
    substrate.valence = 0.9
    substrate.history.append(0.9)
    clamp.apply()
    assert runtime.organs.substrate is substrate
    assert substrate.valence == 0.1
    assert substrate.history is history and history == [0.1]


def test_an_organ_of_a_free_domain_is_left_alone() -> None:
    substrate, workspace = _Organ(), _Organ()
    runtime = _runtime(substrate=substrate, workspace=workspace)
    clamp = Clamp(runtime, ("C",))
    workspace.valence = 0.7
    clamp.apply()
    assert workspace.valence == 0.7


def test_releasing_the_clamp_releases_exactly_it() -> None:
    substrate = _Organ()
    runtime = _runtime(substrate=substrate, workspace=None)

    def previous() -> None:
        return None

    runtime.after_phase = previous
    with clamped(runtime, ("C",)) as clamp:
        assert runtime.after_phase == clamp.apply
    assert runtime.after_phase is previous


def test_still_names_every_held_column_that_moved() -> None:
    slices = domain_slices()
    width = max(piece.stop for piece in slices.values())
    first, second = np.zeros(width), np.zeros(width)
    second[slices["G"].start + 2] = 0.5
    second[slices["S"].start] = 0.5
    rows = [SimpleNamespace(vector=lambda v=first: v), SimpleNamespace(vector=lambda v=second: v)]
    assert still(rows, ("G",)) == {feature_names("G")[2]: 0.5}
    assert still(rows, ("P",)) == {}


@pytest.mark.slow
@pytest.mark.parametrize("held", [("P", "I", "A", "G", "C"), ("S", "M", "W", "D", "N")])
def test_a_held_side_does_not_move_while_the_other_side_lives(held: tuple[str, ...]) -> None:
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )

    async def run() -> tuple[dict[str, float], dict[str, float]]:
        with TemporaryDirectory() as tmp:
            runtime = build_runtime(Path(tmp) / "runtime", seed=11)
            await start_organism(runtime)
            await quiesce_organism(runtime)
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            runtime.freeze_host()
            await runtime.turn_once(CONDITIONS[0])
            with clamped(runtime, held):
                frames = []
                for condition in CONDITIONS[:3]:
                    frames.extend(await runtime.turn_once(condition))
            free = tuple(domain for domain in DOMAINS if domain not in held)
            return still(frames, held), still(frames, free)

    try:
        moved_held, moved_free = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()
    assert moved_free, "the free side did not move either, so holding still proves nothing"
    assert not moved_held, f"columns of the held side moved: {moved_held}"
