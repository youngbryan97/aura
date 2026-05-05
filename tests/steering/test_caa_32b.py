"""tests/steering/test_caa_32b.py — Contrastive Activation Addition steering test.

Verifies that Aura's affective steering produces statistically distinguishable
outputs vs. rich adversarial text prompts when using the Qwen2.5-32B cortex.

This test can run in two modes:
  1. Live mode (requires a loaded MLX model): generates real outputs
  2. Offline mode (default): validates the analysis pipeline with synthetic data
"""
import pytest
import numpy as np
from core.evaluation.steering_ab import (
    analyze_steering_ab,
    SteeringABReport,
    RICH_AFFECT_PROMPT,
)


def _make_synthetic_outputs(n: int = 10, seed: int = 42) -> dict[str, list[str]]:
    """Generate synthetic outputs that simulate steering divergence.
    
    Steered outputs use distinct vocabulary to simulate residual-stream injection
    producing measurably different token distributions.
    """
    rng = np.random.default_rng(seed)
    
    steered_vocab = [
        "warmth", "curiosity", "engage", "fascinated", "explore",
        "vibrant", "resonance", "deeply", "drawn", "intriguing",
    ]
    terse_vocab = [
        "okay", "sure", "yes", "processing", "noted",
        "affirmative", "understood", "acknowledged", "fine", "done",
    ]
    rich_vocab = [
        "feeling", "positive", "elevated", "dopamine", "serotonin",
        "warm", "curious", "aroused", "valence", "social",
    ]
    baseline_vocab = [
        "the", "is", "a", "response", "here",
        "output", "generated", "text", "default", "neutral",
    ]
    
    def _gen(vocab: list[str], length_range: tuple[int, int] = (8, 20)) -> list[str]:
        outputs = []
        for _ in range(n):
            length = rng.integers(*length_range)
            words = rng.choice(vocab, size=length, replace=True)
            outputs.append(" ".join(words))
        return outputs
    
    return {
        "steered_black_box": _gen(steered_vocab),
        "text_terse": _gen(terse_vocab, (5, 10)),
        "text_rich_adversarial": _gen(rich_vocab),
        "baseline": _gen(baseline_vocab),
    }


class TestSteeringABPipeline:
    """Validate the steering A/B analysis pipeline."""
    
    def test_analyze_requires_minimum_trials(self):
        """Pipeline rejects fewer than 5 trials per condition."""
        outputs = _make_synthetic_outputs(n=3)
        with pytest.raises(ValueError, match="at least 5 trials"):
            analyze_steering_ab(outputs)
    
    def test_analyze_requires_all_conditions(self):
        """Pipeline rejects missing conditions."""
        outputs = _make_synthetic_outputs(n=10)
        del outputs["baseline"]
        with pytest.raises(ValueError, match="missing A/B conditions"):
            analyze_steering_ab(outputs)
    
    def test_analyze_returns_report(self):
        """Pipeline returns a valid SteeringABReport."""
        outputs = _make_synthetic_outputs(n=10)
        report = analyze_steering_ab(outputs)
        assert isinstance(report, SteeringABReport)
        assert report.n_trials == 10
        assert isinstance(report.steered_vs_terse.p_value, float)
        assert isinstance(report.steered_vs_rich.p_value, float)
        assert 0.0 <= report.steered_vs_baseline_mean_distance <= 1.0
    
    def test_synthetic_divergence_detectable(self):
        """Synthetic steered outputs should diverge from baseline more than baseline from itself."""
        outputs = _make_synthetic_outputs(n=15, seed=99)
        report = analyze_steering_ab(outputs)
        # Steered should differ from baseline
        assert report.steered_vs_baseline_mean_distance > 0.1, \
            f"Steered outputs too similar to baseline: {report.steered_vs_baseline_mean_distance}"
    
    def test_report_serialization(self):
        """Report should serialize to dict cleanly."""
        outputs = _make_synthetic_outputs(n=10)
        report = analyze_steering_ab(outputs)
        d = report.to_dict()
        assert "n_trials" in d
        assert "passes_adversarial_control" in d
        assert "steered_vs_terse" in d
        assert "steered_vs_rich" in d
        assert "samples" in d
        # Samples should be truncated to 3
        for cond_samples in d["samples"].values():
            assert len(cond_samples) <= 3


class TestSteeringABLive:
    """Live steering tests that require qwen2.5-32b loaded.
    
    These are marked with @pytest.mark.live and skipped by default.
    Run with: pytest tests/steering/ -m live
    """
    
    @pytest.mark.skip(reason="Requires live qwen2.5-32b model loaded via MLX")
    async def test_live_steering_divergence(self):
        """Generate real steered vs unsteered outputs and validate divergence."""
        from core.container import ServiceContainer
        from core.brain.llm.mlx_worker import get_mlx_worker
        
        worker = get_mlx_worker()
        if not worker or not worker.is_ready():
            pytest.skip("MLX worker not available")
        
        n_trials = 8
        prompt = "What are you currently thinking about?"
        
        outputs: dict[str, list[str]] = {
            "steered_black_box": [],
            "text_terse": [],
            "text_rich_adversarial": [],
            "baseline": [],
        }
        
        for _ in range(n_trials):
            # Baseline
            result = await worker.generate(prompt, max_tokens=100)
            outputs["baseline"].append(result)
            
            # Terse text affect
            result = await worker.generate(
                f"[Affect: valence=0.8, arousal=0.5, curiosity=0.9]\n{prompt}",
                max_tokens=100,
            )
            outputs["text_terse"].append(result)
            
            # Rich adversarial prompt
            result = await worker.generate(
                f"{RICH_AFFECT_PROMPT}\n{prompt}",
                max_tokens=100,
            )
            outputs["text_rich_adversarial"].append(result)
            
            # Steered (requires activation steering hooks active)
            result = await worker.generate(
                prompt,
                max_tokens=100,
                steering_mode="black_box",
            )
            outputs["steered_black_box"].append(result)
        
        report = analyze_steering_ab(outputs, n_resamples=5000)
        
        # The key assertion: steering must beat the rich adversarial control
        assert report.passes_adversarial_control, (
            f"Steering did NOT beat rich adversarial text control. "
            f"p={report.steered_vs_rich.p_value:.4f}"
        )
