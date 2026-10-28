"""A predictor may omit a target that starts moving after training; name it."""
import json

import numpy as np
import pytest

from core.subject.recording import Recording, slices_from_columns
from tools.diagnose_closure_candidate import main


@pytest.mark.parametrize("later", ["variable", "constant_increment"])
def test_saved_recording_diagnostic_reports_future_activation_without_certifying(tmp_path, later):
    rng = np.random.default_rng(1919)
    rows = 300
    late = np.zeros(rows)
    if later == "variable":
        late[130:] = rng.normal(size=rows - 130)
    else:
        late[100:] = np.arange(1, rows - 100 + 1)
    recording = Recording(
        x=np.column_stack([rng.normal(size=rows), late]),
        conditions=("ordinary",) * rows, tags=("ontogeny",) * rows,
        times=np.arange(rows, dtype=float), env=np.zeros((rows, 0)), env_names=(),
        columns=("P.signal", "P.late"), slices=slices_from_columns(("P.signal", "P.late")), notes={},
    )
    run = tmp_path / "recording"
    recording.save(run)
    np.savez(run / "closure_inputs.npz", names=np.array(["outside.signal"]),
             periphery=rng.normal(size=(rows, 1)))
    out = tmp_path / "diagnostic.json"
    assert main(["--run", str(run), "--seed", "1919", "--folds", "0",
                 "--feature-limit", "8", "--surrogates", "1", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["official_verdict"] is None
    assert result["feature_limit"] == 8.
    assert result["folds"][0]["unscored_but_moving_in_test"] == ["P.late"]
    assert result["folds"][0]["target_coverage_complete"] is False
    assert len(result["source_sha256"]) == 5
    assert len(result["input_sha256"]) == 3
    assert result["complete"] is True
    assert result["folds"][0]["surrogate_floor_measured"] is True
    assert len(result["folds"][0]["surrogate_gains"]) == 1
