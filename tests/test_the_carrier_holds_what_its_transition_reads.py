"""K*: the carrier holds what K's own transition reads beyond K's ten domains.

On seed 23 thirteen units outside K changed K within two turns when their state
alone was swapped in from another anchor, so K was drawn too small. The carrier
now holds that state (core/subject/carrier_reads.py): it is read whole under
`held.`, sits on the core's side of the closure test, and is held still with its
domain when a cut holds that domain.

The periphery reader was also blind to every history: a deque was not read at
all, and a dict of buffers read as nothing because only numbers directly inside
a container were taken.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.closure import _contents, _numbers, closure_gain, read_periphery
from core.subject.recording import Recording, slices_from_columns

pytestmark = pytest.mark.unit


class Appraisal:
    def __init__(self) -> None:
        self.gain = 0.4
        self.recent = deque([0.1, 0.2, 0.3], maxlen=8)


class AffectUpdatePhase:
    """Named as the phase K* holds part of, so the real table applies."""

    def __init__(self) -> None:
        self.appraisal = Appraisal()
        self.unrelated = 7.0


def test_a_deque_and_a_dict_of_buffers_are_read():
    out: dict[str, float] = {}
    _contents({"novelty": deque([0.5, 0.25]), "pain": deque([0.3])}, "buf", out)
    assert out["buf[novelty][0]"] == 0.5 and out["buf[pain][0]"] == 0.3
    holder = SimpleNamespace(history=deque([1.0, 2.0]))
    found: dict[str, float] = {}
    _numbers(holder, "organ", found)
    assert found["organ.history#"] == 2.0 and found["organ.history[1]"] == 2.0


@dataclass
class Moment:
    level: float


def test_a_history_of_records_is_read_through_its_latest():
    out: dict[str, float] = {}
    _contents(deque([Moment(0.1), Moment(0.9)]), "moments", out)
    assert out["moments[-1].level"] == 0.9
    organs: dict[str, float] = {}
    _contents([SimpleNamespace(level=0.1), SimpleNamespace(level=0.9)], "phases", organs)
    assert not any(name.startswith("phases[-1]") for name in organs), "a list of organs is not a history"


def test_what_k_star_holds_is_read_under_held_and_left_out_of_the_periphery():
    kernel = SimpleNamespace(_phases=[AffectUpdatePhase()])
    read = read_periphery(kernel)
    assert read["held.AffectUpdatePhase.appraisal.gain"] == 0.4
    assert read["held.AffectUpdatePhase.appraisal.recent#"] == 3.0
    assert not any(name.startswith("AffectUpdatePhase.appraisal") for name in read)
    assert read["AffectUpdatePhase.unrelated"] == 7.0


def _world(rows: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = np.zeros((rows, 2))
    broker = np.zeros(rows)
    for t in range(rows - 1):
        broker[t + 1] = 0.6 * broker[t] + rng.normal(0, 0.5)
        k[t + 1] = 0.7 * k[t] + 0.8 * broker[t] + rng.normal(0, 0.2, 2)
    columns = ("A.0", "A.1")
    rec = Recording(
        x=k, conditions=tuple("talk" if t % 2 else "rest" for t in range(rows)), tags=("",) * rows,
        times=np.arange(rows, dtype=np.float64), env=np.zeros((rows, 0)), env_names=(), columns=columns,
        slices=slices_from_columns(columns), notes={},
    )
    return rec, broker


def test_a_cause_the_carrier_holds_is_part_of_the_core_not_a_leak():
    rec, broker = _world()
    noise = np.random.default_rng(3).normal(0, 1, (rec.frames, 2))
    outside = closure_gain(rec, np.column_stack([broker, noise]), ("out.broker", "out.a", "out.b"), seed=1, draws=8)
    held = closure_gain(rec, np.column_stack([broker, noise]), ("held.unit.broker", "out.a", "out.b"), seed=1,
                        draws=8, margin=0.0196)
    assert outside.verdict == "OPEN"
    assert held.verdict != "OPEN"
    assert held.periphery_width == 2


def test_a_cut_that_holds_a_domain_holds_what_k_star_keeps_of_it():
    from core.subject.clamp import Clamp

    phase = AffectUpdatePhase()
    runtime = SimpleNamespace(state=SimpleNamespace(), organs=None, kernel=SimpleNamespace(_phases=[phase]))
    clamp = Clamp(runtime, ["A"])
    phase.appraisal.gain = 0.9
    phase.appraisal.recent.append(0.8)
    clamp.apply()
    assert phase.appraisal.gain == 0.4
    assert list(phase.appraisal.recent) == [0.1, 0.2, 0.3]
    free = Clamp(runtime, ["C"])
    phase.appraisal.gain = 0.9
    free.apply()
    assert phase.appraisal.gain == 0.9


def test_a_held_attribute_is_read_for_its_own_numbers_not_another_units():
    """Her heartbeat holds other services as fields; read through, their state became K*'s."""

    class Neighbour:
        def __init__(self) -> None:
            self.level = 0.75

    neighbour = Neighbour()
    phase = AffectUpdatePhase()
    phase.appraisal.friend = neighbour
    kernel = SimpleNamespace(_phases=[phase, neighbour])
    from core.subject.carrier_reads import read_held

    read = read_held(kernel)
    assert read["held.AffectUpdatePhase.appraisal.gain"] == 0.4
    assert not any("friend" in name for name in read)


def test_her_organs_are_held_whole_and_left_out_of_the_periphery():
    """An organ's internal state decides how the readout K reads will move, so the carrier holds it whole."""
    from core.subject.state import Organs

    workspace = SimpleNamespace(ignition_level=0.3, _inhibition=0.7, _history=deque([0.1, 0.2]))
    kernel = SimpleNamespace(_phases=[], organs={"workspace": workspace})
    read = read_periphery(kernel, Organs(workspace=workspace))
    assert read["held.organ:workspace._inhibition"] == 0.7
    assert read["held.organ:workspace._history#"] == 2.0
    assert not any(name.startswith("organ.workspace") for name in read)


def test_the_screen_puts_a_column_that_predicts_the_core_first():
    from core.subject.closure import _screened

    rng = np.random.default_rng(5)
    rows = 900
    cause = rng.normal(size=rows)
    k = np.zeros((rows, 3))
    for t in range(rows - 1):
        k[t + 1] = 0.6 * k[t] + np.array([0.9, 0.0, 0.4]) * cause[t] + rng.normal(0, 0.2, 3)
    own, change = k[:-1], k[1:] - k[:-1]
    outside = np.column_stack([rng.normal(size=(rows - 1, 30)), cause[:-1]])
    picked = _screened(own, change, outside, slice(0, 500), slice(500, 700), limit=3)
    assert picked[0] == 30
