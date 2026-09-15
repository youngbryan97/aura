"""A v2 run recorded before the persistence amendment is rescored beside its report, never in it.

The third amendment to docs/ISC_V2_PREREGISTRATION.md says a v2 run recorded
before the new persistence line existed is read by rescoring its own saved
recording, with the reading marked as added after the run. These pin the tool
that does it: the run's report is left byte for byte as it was, the reading
goes in a file beside it and says it was rescored, and a run that recorded the
reading itself is left alone.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from core.subject.recording import Recording
from core.subject.state import DOMAINS

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
ROWS = 400
WIDTH = 3


def _tool():
    spec = importlib.util.spec_from_file_location("rescore", REPO / "tools" / "subject_core_rescore_persistence.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(directory: Path, report: dict) -> Path:
    rng = np.random.default_rng(5)
    state = np.zeros((ROWS, WIDTH * len(DOMAINS)))
    for row in range(1, ROWS):
        state[row] = 0.95 * state[row - 1] + rng.normal(size=state.shape[1])
    Recording(
        x=state,
        conditions=tuple("x" for _ in range(ROWS)),
        tags=tuple("ontogeny" for _ in range(ROWS)),
        times=np.arange(ROWS, dtype=np.float64),
        env=rng.normal(size=(ROWS, 2)),
        env_names=("e0", "e1"),
        columns=tuple(f"{key}.{i}" for key in DOMAINS for i in range(WIDTH)),
        slices={key: slice(i * WIDTH, (i + 1) * WIDTH) for i, key in enumerate(DOMAINS)},
        notes={},
    ).save(directory)
    (directory / "subject_core_report.json").write_text(json.dumps(report), encoding="utf-8")
    return directory


def test_the_report_is_untouched_and_the_reading_goes_beside_it(tmp_path: Path) -> None:
    run = _run(tmp_path, {"campaign": {"seed": 7}, "synergy_v2": [], "nulls": {"conjunction": {}}})
    before = (run / "subject_core_report.json").read_bytes()
    result = _tool().rescore(run)
    assert (run / "subject_core_report.json").read_bytes() == before
    assert result["persistence_v2"]["rescored_after_the_run"] is True
    assert result["persistence_v2"]["seed"] == 7
    assert "without_memory" in result["persistence_v2"]
    keys = [row["criterion"] for row in result["v2_criteria"]]
    assert "intrinsic_persistence" in keys


def test_a_run_that_recorded_the_reading_itself_is_left_alone(tmp_path: Path) -> None:
    run = _run(tmp_path, {"persistence_v2": {"passes": True}, "synergy_v2": [], "nulls": {}})
    assert "skipped" in _tool().rescore(run)
