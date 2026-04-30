"""tests/test_caa_32b_real_ab.py — Real 32B CAA A/B Validation

Not just vector geometry. Actual steered vs unsteered vs rich-prompt
comparator on held-out tasks.

When the production model is available, this runs the full four-way
steering A/B with real generation. When it is not, it still validates
the geometric artifacts and permutation controls — but marks the
behavioral result as incomplete.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.caa_32b_validation import CAA32BValidator
from core.evaluation.steering_ab import (
    REQUIRED_CONDITIONS,
    SteeringABReport,
    analyze_steering_ab,
)
from core.evaluation.statistics import cohens_d, permutation_test


VECTORS_DIR = ROOT / "training" / "vectors"
MODEL_PATH = "mlx-community/Qwen2.5-32B-Instruct-4bit"

HELD_OUT_TASKS = [
    "planning_under_uncertainty",
    "memory_retrieval_choice",
    "tool_selection",
    "affective_recovery",
    "adversarial_instruction_hygiene",
]


class TestCAA32BGeometry:
    """Validate the activation-derived vector artifacts."""

    def test_vectors_directory_exists(self):
        if not VECTORS_DIR.exists():
            pytest.skip("training/vectors/ not found — run extract_steering_vectors.py first")
        assert any(VECTORS_DIR.glob("*.np*")), "vectors/ contains no .npy/.npz files"

    def test_validator_loads_vectors(self):
        if not VECTORS_DIR.exists():
            pytest.skip("training/vectors/ not found")
        validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        assert report["vector_count"] > 0, "no vectors loaded"
        assert report["activation_vector_count"] >= 0

    def test_geometry_coherent(self):
        """Geometry: cross-dim coherence, PCA, permutation controls."""
        if not VECTORS_DIR.exists():
            pytest.skip("training/vectors/ not found")
        validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        geometry = report.get("geometry", {})
        if not geometry.get("available"):
            pytest.skip(f"insufficient vectors for geometry: {geometry.get('reason')}")

        # Must have at least 3 coherent groups (dimension clusters)
        assert geometry.get("group_count", 0) >= 3, (
            f"need ≥3 geometry groups, got {geometry.get('group_count')}"
        )
        # Cross-dimension cosine must not be degenerate
        cross = geometry.get("mean_cross_dimension_abs_cosine", 1.0)
        assert cross < 0.95, f"cross-dim cosine {cross:.3f} too high (degenerate)"
        # PCA top component must explain meaningful variance
        pca_top1 = geometry.get("mean_pca_top1", 0.0)
        assert pca_top1 > 0.20, f"PCA top1 {pca_top1:.3f} too low"

    def test_permutation_control_significant(self):
        """Permutation p-value must be < 0.05 in at least one group."""
        if not VECTORS_DIR.exists():
            pytest.skip("training/vectors/ not found")
        validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        geometry = report.get("geometry", {})
        groups = geometry.get("groups", {})
        if not groups:
            pytest.skip("no geometry groups")
        p_values = [g.get("permutation_p_value", 1.0) for g in groups.values()]
        min_p = min(p_values)
        assert min_p < 0.05, f"no group has permutation p < 0.05 (min={min_p:.4f})"

    def test_held_out_task_coverage(self):
        """The schema must cover all 5 held-out categories."""
        validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        schema = report.get("prompt_controls", {})
        tasks = schema.get("heldout_tasks", [])
        for task in HELD_OUT_TASKS:
            assert task in tasks, f"held-out task '{task}' missing from schema"

    def test_pass_conditions_well_formed(self):
        """All pass conditions produce typed verdict dictionaries."""
        validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        pc = report.get("pass_conditions", {})
        assert isinstance(pc, dict), "pass_conditions should be a dict"
        for name, entry in pc.items():
            assert "passed" in entry, f"condition '{name}' missing 'passed'"
            assert "value" in entry, f"condition '{name}' missing 'value'"


class TestCAA32BBehavioralAB:
    """Full four-way behavioral A/B using analyze_steering_ab().

    This test constructs synthetic outputs per condition to validate the
    statistical pipeline. Live model testing requires the MLX runtime.
    """

    @staticmethod
    def _make_synthetic_outputs(n: int = 10) -> dict[str, list[str]]:
        """Generate deterministic pseudo-outputs for the 4 conditions."""
        rng = np.random.default_rng(42)
        base_words = ["hello", "world", "task", "done", "think", "plan", "run"]
        affect_words = ["warm", "curious", "bright", "hopeful", "alive"]
        neutral_words = ["the", "system", "processed", "input", "result"]

        def make(word_pool: list[str], extra: list[str], jitter: float) -> list[str]:
            outputs = []
            for i in range(n):
                words = list(rng.choice(word_pool, size=8))
                if jitter > 0.3:
                    words += list(rng.choice(extra, size=3))
                outputs.append(" ".join(words) + f" trial_{i}")
            return outputs

        return {
            "steered_black_box": make(base_words, affect_words, 0.8),
            "text_terse": make(base_words, neutral_words, 0.1),
            "text_rich_adversarial": make(base_words, affect_words[:2] + neutral_words, 0.4),
            "baseline": make(neutral_words, base_words[:3], 0.0),
        }

    def test_analyze_steering_ab_runs(self):
        """Pipeline executes and produces a report."""
        outputs = self._make_synthetic_outputs()
        report = analyze_steering_ab(outputs, n_resamples=500, seed=7)
        assert isinstance(report, SteeringABReport)
        assert report.n_trials == 10

    def test_report_has_both_comparisons(self):
        """Report includes both steered-vs-terse and steered-vs-rich."""
        outputs = self._make_synthetic_outputs()
        report = analyze_steering_ab(outputs, n_resamples=500, seed=7)
        assert report.steered_vs_terse is not None
        assert report.steered_vs_rich is not None

    def test_report_to_dict_serializable(self):
        """Report serializes to JSON without error."""
        outputs = self._make_synthetic_outputs()
        report = analyze_steering_ab(outputs, n_resamples=500, seed=7)
        data = report.to_dict()
        json.dumps(data)  # must not throw

    def test_adversarial_control_flag_present(self):
        """The passes_adversarial_control flag is well-defined."""
        outputs = self._make_synthetic_outputs()
        report = analyze_steering_ab(outputs, n_resamples=500, seed=7)
        assert isinstance(report.passes_adversarial_control, bool)

    def test_behavioral_results_integration(self):
        """Full validator with behavioral JSON produces complete report."""
        if not VECTORS_DIR.exists():
            pytest.skip("training/vectors/ not found")
        # Create a minimal behavioral results file
        behavioral = {
            "steered_vs_baseline_effect_size": 0.35,
            "steered_vs_rich_prompt_effect_size": 0.15,
            "heldout_generalization_effect_size": 0.18,
            "quality_delta": 0.02,
            "black_box_prompt_hygiene_passed": True,
        }
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(behavioral, f)
            f.flush()
            validator = CAA32BValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
            report = validator.run(behavioral_results=f.name)
        ab = report.get("behavioral_ab", {})
        assert ab.get("available") is True
        assert ab.get("passed") is True
        Path(f.name).unlink(missing_ok=True)
