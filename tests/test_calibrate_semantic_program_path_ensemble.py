from __future__ import annotations

from pathlib import Path

import pytest

from tools.calibrate_semantic_program_path_ensemble import (
    _validated_output_directory,
)


def test_calibration_output_cannot_modify_the_immutable_feature_bundle(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    with pytest.raises(ValueError, match="immutable feature bundle"):
        _validated_output_directory(bundle / "result", bundle=bundle)

    assert _validated_output_directory(tmp_path / "result", bundle=bundle) == (
        tmp_path / "result"
    )
