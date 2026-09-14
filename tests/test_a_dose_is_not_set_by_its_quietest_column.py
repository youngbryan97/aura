"""A v25 dose is matched to the column a push typically moves.

run_003 calibrated affect, the self and deliberation to the 1e-4 clip because
one channel in each barely moves during ordinary work, and the largest
standardised movement over a domain's columns was that channel's. The graph
came back with no edges. These pin the calibration on a domain whose answer is
known: five columns with a spread of 0.1 and one with a spread of 5e-05, all
moved by the dose itself.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORDINARY = 0.1
QUIET = 5e-5


@pytest.fixture(scope="module")
def tool() -> Any:
    name = "run_subject_core_v25_dose_under_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "run_subject_core_v25.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


class _Runtime:
    """Six columns of one domain, which a dose moves by exactly the dose."""

    def __init__(self, writable: bool = True) -> None:
        self.vector = np.zeros(6)
        self.state = SimpleNamespace(runtime=self)
        self.ontogeny = None
        self.organs = None
        self.writable = writable

    def snapshot(self) -> np.ndarray:
        return self.vector.copy()

    def restore(self, snapshot: np.ndarray) -> None:
        self.vector = snapshot.copy()

    async def turn_once(self, condition: Any, perturb_at: int | None = None, perturb: Any = None) -> list[Any]:
        if perturb is not None:
            await perturb(self)
        frozen = self.vector.copy()
        return [SimpleNamespace(vector=lambda: frozen)]


@pytest.fixture
def writers(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.subject.state as state_module

    def perturb(state: Any, domain: str, delta: float, *, ontogeny: Any = None) -> bool:
        if state.runtime.writable:
            state.runtime.vector = state.runtime.vector + delta
        return state.runtime.writable

    async def perturb_organs(organs: Any, domain: str, delta: float, *, state: Any = None) -> bool:
        return False

    monkeypatch.setattr(state_module, "perturb", perturb)
    monkeypatch.setattr(state_module, "perturb_organs", perturb_organs)


def _calibrate(tool: Any, runtime: _Runtime) -> float:
    scale = np.array([ORDINARY] * 5 + [QUIET])
    doses = asyncio.run(
        tool._dose_matched(
            runtime,
            [SimpleNamespace(name="calm")],
            scale,
            {"A": slice(0, 6)},
            domains=["A"],
            rounds=4,
            trials=2,
        )
    )
    return doses["A"]


def test_the_dose_moves_the_typical_column_one_of_its_own_sds(tool: Any, writers: None) -> None:
    assert _calibrate(tool, _Runtime()) == pytest.approx(ORDINARY, rel=1e-6)


def test_the_quiet_column_does_not_set_the_domains_dose(tool: Any, writers: None) -> None:
    dose = _calibrate(tool, _Runtime())
    assert dose > 1000 * QUIET, "the dose that would bring the quiet column to one SD moves nothing else"


def test_a_domain_the_writer_cannot_move_keeps_its_starting_dose(tool: Any, writers: None) -> None:
    assert _calibrate(tool, _Runtime(writable=False)) == pytest.approx(0.15)
