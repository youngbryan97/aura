"""The load requirement must come from the model, not from a constant.

Measured 2026-07-25 on the live host: the resident 32B is **17.2GB on disk**
and the flat gate demanded **24.0GB available** — a 6.8GB margin over true
need. The host sits around 20.4GB available with the fallback ladder resident,
so the cortex was refused a load it could have completed, and starved through
six deaths in a single run.

A constant cannot be right for every model. Weights × 1.20 + 1GB covers KV
cache and activations for a normal context with room to spare, and the flat
default stays the CEILING so this can only relax toward the real footprint —
never tighten past a deliberate operator setting.
"""
from __future__ import annotations

import pytest

from core.brain.llm.mlx_client import (
    _measured_model_footprint_gb,
    _model_load_min_available_gb,
)

pytestmark = pytest.mark.unit


def _fake_measure(monkeypatch, gb):
    """Stand in for the on-disk measurement — writing 17GB per test is absurd."""
    import core.brain.llm.mlx_client as mc

    monkeypatch.setattr(mc, "_measured_model_footprint_gb", lambda p: gb)


class TestMeasurement:
    def test_an_unreadable_path_measures_nothing(self):
        assert _measured_model_footprint_gb("/no/such/model") is None
        assert _measured_model_footprint_gb(None) is None
        assert _measured_model_footprint_gb("") is None

    def test_an_implausible_size_measures_nothing(self, tmp_path):
        """A near-empty directory must not wave a 20GB load through."""
        root = tmp_path / "Aura-32B-cortex"
        root.mkdir()
        (root / "config.json").write_text("{}")
        assert _measured_model_footprint_gb(root) is None


class TestTheDerivedRequirement:
    def test_it_relaxes_toward_the_real_footprint(self, monkeypatch):
        """The live case verbatim: 17.2GB model, flat gate 24GB."""
        monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)
        _fake_measure(monkeypatch, 17.2)
        derived = _model_load_min_available_gb("/models/Aura-32B-cortex")
        assert derived == pytest.approx(17.2 * 1.20 + 1.0), (
            "weights x 1.20 + 1GB is the stated headroom rule"
        )
        assert derived < 24.0, "a constant above true need starves the lane"

    def test_it_never_exceeds_the_flat_default(self, monkeypatch):
        """Only ever relaxes — a bigger model cannot raise the bar."""
        monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)
        _fake_measure(monkeypatch, 60.0)
        assert _model_load_min_available_gb("/models/Aura-32B-cortex") <= 24.0

    def test_a_small_measured_checkpoint_is_held_to_its_own_size(self, monkeypatch):
        """The 16GB floor is for a checkpoint that could not be measured.

        Until 19 September (94945a573) it was applied to a measured one as
        well, and an 8GB brainstem was held to twice its footprint: 222 spawns
        refused on one boot. A directory too small to be a model reads as
        unmeasured, not as tiny (the near-empty case above).
        """
        monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)
        _fake_measure(monkeypatch, 2.0)
        assert _model_load_min_available_gb("/models/Aura-32B-cortex") == pytest.approx(2.0 * 1.20 + 1.0)

    def test_an_unmeasurable_model_keeps_the_conservative_default(self, monkeypatch):
        monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)
        value = _model_load_min_available_gb("/models/Aura-32B-cortex-missing")
        assert value in (22.0, 24.0), (
            "no measurement means no relaxation; keep the safe constant"
        )

    def test_an_explicit_operator_setting_still_wins(self, monkeypatch):
        monkeypatch.setenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", "30")
        _fake_measure(monkeypatch, 17.2)
        assert _model_load_min_available_gb("/models/Aura-32B-cortex") == 30.0

    def test_other_tiers_are_untouched(self, monkeypatch):
        monkeypatch.delenv("AURA_MLX_LOAD_MIN_AVAILABLE_GB", raising=False)
        assert _model_load_min_available_gb("/models/Qwen2.5-1.5B") == 8.0

    def test_auto_is_no_setting_and_the_requirement_is_derived(self, monkeypatch):
        """Both launchers set "auto" since 24 November 2026; until then they set
        24, a number, and the derived requirement never applied live."""
        monkeypatch.setenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", "auto")
        _fake_measure(monkeypatch, 17.2)
        assert _model_load_min_available_gb("/models/Aura-32B-cortex") == pytest.approx(17.2 * 1.20 + 1.0)


class TestTheModelsSheRuns:
    """Her cortex and brainstem as they stood on 24 November 2026, measured on
    disk: each is held to its own size, not to the 32B's."""

    @pytest.mark.parametrize(
        ("name", "on_disk_gb", "needs_gb"),
        [
            ("/models/Aura-Qwen3.8-27B-persona-crsm", 14.1, 17.92),
            ("/models/Ternary-Bonsai-2.5-27B-mlx-2bit", 8.27, 10.92),
        ],
    )
    def test_each_is_held_to_its_own_size(self, monkeypatch, name, on_disk_gb, needs_gb):
        import core.brain.llm.mlx_client as mc

        monkeypatch.setenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", "auto")
        monkeypatch.setattr(mc, "_model_matches_class", lambda path, tokens: "32b" in tokens)
        _fake_measure(monkeypatch, on_disk_gb)
        assert _model_load_min_available_gb(name) == pytest.approx(needs_gb, abs=0.01)
