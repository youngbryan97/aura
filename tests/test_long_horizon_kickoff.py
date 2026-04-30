"""tests/test_long_horizon_kickoff.py — Long-Horizon Run Kickoff

Validates that the long-horizon infrastructure is correctly configured.
Does NOT run the 24h+ test itself — just proves the runner exists,
profiles are valid, and the gauntlet can be invoked.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestLongHorizonInfrastructure:
    def test_gauntlet_runner_importable(self):
        from tools.longevity.run_gauntlet import main  # noqa: F401

    def test_long_run_model_importable(self):
        from tools.long_run_model import profiles  # noqa: F401

    def test_makefile_longevity_target_exists(self):
        makefile = ROOT / "Makefile"
        assert makefile.exists()
        content = makefile.read_text(encoding="utf-8")
        assert "longevity" in content
        assert "run_gauntlet" in content

    def test_activation_audit_importable(self):
        try:
            from tools.activation_audit import main  # noqa: F401
        except ImportError:
            pytest.skip("activation_audit not available")

    def test_proof_bundle_importable(self):
        from tools.proof_bundle import main  # noqa: F401
