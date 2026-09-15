"""The one campaign recorded before ISC-v3, rescored beside its reports.

docs/ISC_V3_PREREGISTRATION.md lets one campaign be read under v3 after the
fact: the ISC-v2 campaign recording at 1a9ebe561, which had written no report
when v3 was committed. These pin that the rescoring refuses any other run, that
it will not read a run whose v2 reading it cannot reproduce, that its instrument
check registers a coupling the line is built to see, and that the scorecard
reads a sidecar's v2 and v3 lines while keeping the v1 verdict the run recorded.
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


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _slow_background(seed: int, rows: int = 2400, width: int = 3) -> Recording:
    rng = np.random.default_rng(seed)
    x = np.zeros((rows, width * len(DOMAINS)))
    noise = rng.normal(size=x.shape)
    for row in range(1, rows):
        x[row] = 0.99 * x[row - 1] + noise[row]
    return Recording(
        x=x, conditions=tuple("x" for _ in range(rows)), tags=tuple("" for _ in range(rows)),
        times=np.arange(rows, dtype=np.float64), env=rng.normal(size=(rows, 2)), env_names=("e0", "e1"),
        columns=tuple(f"{key}.{index}" for key in DOMAINS for index in range(width)),
        slices={key: slice(index * width, (index + 1) * width) for index, key in enumerate(DOMAINS)},
        notes={},
    )


def test_a_run_of_any_other_campaign_is_refused(tmp_path: Path) -> None:
    tool = _tool("subject_core_rescore_synergy_v3")
    report = {"campaign": {"fingerprint": "not-the-campaign", "frozen": {"seed": 7}}, "synergy_v2": [{"passes": False}]}
    (tmp_path / "subject_core_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert "refused" in tool.rescore(tmp_path)
    assert tool.RESCORED_CAMPAIGN == "39493ebf6330169637c46feb8febc62f"


def test_a_recomputed_v2_reading_that_differs_from_the_recorded_one_is_named() -> None:
    tool = _tool("subject_core_rescore_synergy_v3")
    again = [{"sources": ["A", "S"], "target": "G", "passes": False, "synergy_fraction": 0.2}]
    assert tool._aura_mismatches(again, [dict(again[0])]) == []
    assert tool._aura_mismatches(again, [{**again[0], "synergy_fraction": 0.3}])
    assert tool._aura_mismatches(again, [{**again[0], "passes": True}])


def test_the_instrument_check_registers_the_coupling_the_line_is_built_to_see() -> None:
    from core.subject.synergy import synergy

    tool = _tool("subject_core_rescore_synergy_v3")
    source_a, source_b, target = DOMAINS[2], DOMAINS[5], DOMAINS[3]
    background = _slow_background(800)
    assert not synergy(background, source_a, source_b, target, seed=0, of="change").passes_v3
    pushed = tool.coupled(background, source_a, source_b, target, 2.0)
    assert synergy(pushed, source_a, source_b, target, seed=0, of="change").passes_v3


def test_the_scorecard_reads_a_sidecars_v2_and_v3_lines_and_keeps_the_recorded_v1_verdict(tmp_path: Path) -> None:
    card = _tool("subject_core_scorecard")
    v1 = [{"criterion": "synergy", "passed": False}]
    (tmp_path / "subject_core_report.json").write_text(
        json.dumps({"verdict": {"criteria": v1, "v2_criteria": []}, "nulls": {"detail": {"old": {}}}}), encoding="utf-8"
    )
    sidecar = {
        "reproduced": True,
        "v2_criteria": [{"criterion": "synergy", "passed": False}],
        "v3_criteria": [{"criterion": "synergy", "passed": True}],
        "nulls_detail": {"new": {"kind": "surrogate", "phi_do": 0.02}},
    }
    (tmp_path / "isc_v3_rescored.json").write_text(json.dumps(sidecar), encoding="utf-8")
    loaded = card.load(tmp_path)
    assert loaded["verdict"]["criteria"] == v1
    assert loaded["verdict"]["v3_criteria"] == sidecar["v3_criteria"]
    assert loaded["verdict"]["rescored_after_the_run"] is True
    assert loaded["nulls"]["detail"] == sidecar["nulls_detail"]
    (tmp_path / "isc_v3_rescored.json").write_text(json.dumps({**sidecar, "reproduced": False}), encoding="utf-8")
    assert "v3_criteria" not in card.load(tmp_path)["verdict"]
