"""The campaign, the carrier sweep and the content run read one K.

Only the campaign could read each sketched array along axes fixed before
recording, so a decisive reading at that setting would have scored the 22
campaign lines on one K and the carrier's cuts and the content run on another.
Each stage now takes the setting, and a shard worker reads the coordinator's
axes rather than fitting its own: one organism, one K.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from core.subject import calibrated_axes

REPO = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_axes_fixed_in_one_process_are_read_in_another(tmp_path):
    rng = np.random.default_rng(3)
    calibrated_axes.clear()
    calibrated_axes.calibrate({"organ:substrate.x": [rng.normal(size=12) for _ in range(40)]})
    centre, vectors = calibrated_axes.axes_for("organ:substrate.x")
    assert calibrated_axes.save(tmp_path / "axes.npz") == 1
    calibrated_axes.clear()
    assert calibrated_axes.axes_for("organ:substrate.x") is None
    assert calibrated_axes.load(tmp_path / "axes.npz") == 1
    loaded_centre, loaded_vectors = calibrated_axes.axes_for("organ:substrate.x")
    assert np.allclose(loaded_centre, centre) and np.allclose(loaded_vectors, vectors)
    calibrated_axes.clear()


def test_the_launcher_hands_every_process_the_setting():
    import argparse

    launcher = _tool("run_subject_core_v25_sharded")
    args = argparse.Namespace(workers=2, hours=1.0, calibrated_axes=True, share_anchors=True)
    plan = launcher.commands(args, Path("/tmp/launch"))
    assert len(plan) == 3
    assert all("--calibrated-axes" in argv for _name, argv in plan)


def test_the_sweep_and_the_content_run_take_the_setting():
    for name in ("run_subject_core_v25", "run_subject_core_content"):
        source = (REPO / "tools" / f"{name}.py").read_text()
        assert '"--calibrated-axes", action="store_true"' in source
        assert "calibration_pass" in source
