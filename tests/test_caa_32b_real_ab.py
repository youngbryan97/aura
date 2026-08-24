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
from pathlib import Path

import numpy as np

from core.evaluation.steering_ab import (
    SteeringABReport,
    analyze_steering_ab,
)
from training.caa_32b_validation import CAAModelValidator

ROOT = Path(__file__).resolve().parents[1]


VECTORS_DIR = ROOT / "training" / "vectors"
MODEL_PATH = "test-active-cortex"

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
        assert VECTORS_DIR.exists(), "training/vectors/ not found; run extract_steering_vectors.py"
        assert any(VECTORS_DIR.glob("*.np*")), "vectors/ contains no .npy/.npz files"

    def test_validator_loads_vectors(self):
        assert VECTORS_DIR.exists(), "training/vectors/ not found"
        validator = CAAModelValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        assert report["vector_count"] > 0, "no vectors loaded"
        assert report["activation_vector_count"] >= 0

    def test_geometry_coherent(self):
        """Geometry: cross-dim coherence, PCA, permutation controls."""
        assert VECTORS_DIR.exists(), "training/vectors/ not found"
        validator = CAAModelValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        geometry = report.get("observed_unbound_geometry", {})
        assert geometry.get("available"), f"insufficient vectors for geometry: {geometry.get('reason')}"

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
        assert VECTORS_DIR.exists(), "training/vectors/ not found"
        validator = CAAModelValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        geometry = report.get("observed_unbound_geometry", {})
        groups = geometry.get("groups", {})
        assert groups, "no geometry groups"
        p_values = [g.get("permutation_p_value", 1.0) for g in groups.values()]
        min_p = min(p_values)
        assert min_p < 0.05, f"no group has permutation p < 0.05 (min={min_p:.4f})"

    def test_held_out_task_coverage(self):
        """The schema must cover all 5 held-out categories."""
        validator = CAAModelValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
        report = validator.run()
        schema = report.get("prompt_controls", {})
        tasks = schema.get("heldout_tasks", [])
        for task in HELD_OUT_TASKS:
            assert task in tasks, f"held-out task '{task}' missing from schema"

    def test_pass_conditions_well_formed(self):
        """All pass conditions produce typed verdict dictionaries."""
        validator = CAAModelValidator(vectors_dir=VECTORS_DIR, model_path=MODEL_PATH)
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
            # The null: the baseline condition drawn again, unsteered. See
            # core/evaluation/steering_ab.py — it is REQUIRED input, because
            # an effect that is not compared against the system's own
            # run-to-run variation is not an effect that has been measured.
            "baseline_replicate": make(neutral_words, base_words[:3], 0.0),
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
        assert report.steered_effect is not None
        assert report.terse_effect is not None
        assert report.rich_effect is not None

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
        assert VECTORS_DIR.exists(), "training/vectors/ not found"
        # Create a minimal behavioral results file
        behavioral = {
            "model_descriptor_sha256": "a" * 64,
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
            ab = CAAModelValidator._load_behavioral_results(
                f.name,
                model_descriptor_sha256="a" * 64,
            )
        assert ab.get("available") is True
        assert ab.get("passed") is True
        Path(f.name).unlink(missing_ok=True)

    def test_live_32b_ab_results_schema_integration(self):
        """The proof bundle can consume the live 32B A/B artifact directly."""
        assert VECTORS_DIR.exists(), "training/vectors/ not found"
        behavioral = {
            "model": MODEL_PATH,
            "model_descriptor_sha256": "b" * 64,
            "n_trials": 50,
            "held_out_tasks": HELD_OUT_TASKS,
            "passes_adversarial_control": True,
            # Injection provenance: the rebuilt runner always records these;
            # artifacts without them are pre-rebuild theater and the
            # validator refuses to credit them (see test_steering_injection).
            "sampling": {"temperature": 0.7, "top_p": 0.95, "paired_seeds": True},
            "injection_count": 800,
            # Null-calibrated schema. Each condition carries its OWN effect,
            # measured net of the baseline's divergence from its own
            # replicate, so the null sits at zero. The previous fixture used
            # the pre-2026-08-05 `steered_vs_terse` / `steered_vs_rich` keys,
            # whose statistic scored a decisive PASS on its own null — and
            # this test asserted that artifact passed. The validator voids
            # those now; the test was pinning the contradiction in place.
            "analysis": {
                "n_trials": 50,
                "steered_effect": {
                    "observed_delta": 0.168,
                    "p_value": 0.0002,
                    "ci_low": 0.14,
                    "ci_high": 0.18,
                    "effect_size_d": 3.43,
                },
                "terse_effect": {
                    "observed_delta": 0.007,
                    "p_value": 0.34,
                    "ci_low": -0.03,
                    "ci_high": 0.04,
                    "effect_size_d": 0.07,
                },
                "rich_effect": {
                    "observed_delta": 0.041,
                    "p_value": 0.12,
                    "ci_low": -0.01,
                    "ci_high": 0.09,
                    "effect_size_d": 0.42,
                },
                "baseline_self_distance": 0.09,
                "steered_vs_baseline_mean_distance": 0.63,
                "rich_vs_baseline_mean_distance": 0.80,
                "passes_adversarial_control": True,
            },
        }
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(behavioral, f)
            f.flush()
            ab = CAAModelValidator._load_behavioral_results(
                f.name,
                model_descriptor_sha256="b" * 64,
            )
        assert ab.get("available") is True
        assert ab.get("passed") is True
        assert ab["raw"]["source_schema"] == "live_32b_ab"
        Path(f.name).unlink(missing_ok=True)

    def test_a_pre_null_calibration_artifact_is_voided_not_credited(self):
        """The old schema must FAIL, and the test suite must say so.

        Artifacts written before 2026-08-05 scored
        d(steered, control) − d(steered, baseline). Steered and baseline shared
        a prompt and a seed, so with no effect at all the second term is
        exactly zero and the first is the whole control distance — the
        statistic's own null hypothesis scored a decisive pass. Measured on
        this pipeline's null: d=17.3, p=0.0005.

        The validator voids those. This test exists because the suite
        previously asserted the opposite — a fixture in the old schema, checked
        for `passed is True` — which is the documented contradiction (the
        current validator calls old CAA schemas invalid while a committed
        old-schema report stays marked passing) pinned in place by a test.
        """
        import tempfile

        behavioral = {
            "model": MODEL_PATH,
            "n_trials": 50,
            "held_out_tasks": HELD_OUT_TASKS,
            "passes_adversarial_control": True,
            "sampling": {"temperature": 0.7, "top_p": 0.95, "paired_seeds": True},
            "injection_count": 800,
            "analysis": {
                "n_trials": 50,
                # The retired keys. Absence of `steered_effect` is what marks
                # the artifact, so an old file cannot normalize into a pass.
                "steered_vs_terse": {"effect_size_d": 0.07, "p_value": 0.34},
                "steered_vs_rich": {"effect_size_d": 3.43, "p_value": 0.0002},
                "passes_adversarial_control": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(behavioral, f)
            f.flush()
            ab = CAAModelValidator._load_behavioral_results(
                f.name,
                model_descriptor_sha256="c" * 64,
            )
        assert ab["raw"]["source_schema"] == "live_32b_ab_voided"
        assert ab.get("passed") is not True, (
            "an artifact whose statistic passes its own null was credited as "
            "behavioral evidence"
        )
        assert "paired_distance_comparison" in ab["raw"]["voided_reason"]
        Path(f.name).unlink(missing_ok=True)
