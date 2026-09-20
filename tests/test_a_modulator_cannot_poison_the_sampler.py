"""Six subsystems compose into one temperature and one token budget.

Homeostatic coupling, homeostasis, morphogenesis, plasticity, temporal
continuity and somatic qualia each read a number out of a service and fold it
into the same two values with raw arithmetic. There was no shared contract, so:

- One NaN anywhere carried to the end. `max(0.1, min(1.5, nan))` is `nan`, and
  every later comparison against it is False, so the poison never trips a
  bound — it just arrives at the sampler.
- One absurd factor multiplied the token budget past the context window.
- A modifier that raised mid-chain left the earlier modifiers already applied
  and the later ones not, with nothing recording which.

They share one contract now: finite, bounded, and neutral (1.0 for a factor,
0.0 for a delta) when the subsystem returns something this code cannot use.

The rest of this file is about claims. Three comments asserted more than the
code does — "true embodied cognition" for reading three scalars and scaling
three sampler knobs, "raw felt perturbation" for four floats off a service, and
integrated information for a value explicitly requested with
`include_surrogate=True`. Registered claims about Aura need a test behind them;
these had none, so the claims came down and the mechanisms stayed.
"""
from __future__ import annotations

import inspect

import pytest

from core.brain.inference_gate import InferenceGate
from tests.source_contract import family_text, function_containing


# ─────────────────────────────── one contract, six modulators


@pytest.mark.parametrize(
    "raw", [float("nan"), float("inf"), float("-inf"), "hot", None, [], {}]
)
def test_an_unusable_factor_is_neutral_not_poison(raw):
    assert InferenceGate._modulator_factor(raw, source="test", low=0.5, high=2.0) == 1.0


@pytest.mark.parametrize(
    "raw", [float("nan"), float("inf"), float("-inf"), "warm", None]
)
def test_an_unusable_delta_is_zero(raw):
    assert InferenceGate._modulator_delta(raw, source="test", limit=0.5) == 0.0


def test_a_usable_factor_passes_through():
    assert InferenceGate._modulator_factor(1.2, source="test", low=0.5, high=2.0) == 1.2


def test_a_factor_is_clamped_at_both_ends():
    assert InferenceGate._modulator_factor(99.0, source="test", low=0.5, high=2.0) == 2.0
    assert InferenceGate._modulator_factor(0.0, source="test", low=0.5, high=2.0) == 0.5


def test_a_delta_is_clamped_symmetrically():
    assert InferenceGate._modulator_delta(9.0, source="test", limit=0.5) == 0.5
    assert InferenceGate._modulator_delta(-9.0, source="test", limit=0.5) == -0.5


def test_a_nan_cannot_reach_the_sampler_through_the_chain():
    """The specific failure: nan survives min/max because every comparison
    against it is False."""
    poisoned = InferenceGate._modulator_delta(float("nan"), source="test", limit=0.5)
    temperature = max(0.1, min(1.5, 0.72 + poisoned))

    assert temperature == pytest.approx(0.72)


@pytest.mark.parametrize(
    "source",
    [
        "homeostatic_coupling.temperature_mod",
        "homeostatic_coupling.depth_mod",
        "homeostasis.temperature_mod",
        "homeostasis.token_multiplier",
        "temporal_continuity.temperature_delta",
        "somatic_qualia.temperature",
        "morphogenesis.danger",
    ],
)
def test_every_modulator_goes_through_the_contract(source):
    import core.brain.inference_gate as gate_mod

    assert f'source="{source}"' in family_text(gate_mod), (
        f"{source} composes into the sampler without the finite contract"
    )


# ─────────────────────────────── the claims match the code


def test_no_true_embodiment_claim_survives():
    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "True Embodied Cognition" not in source
    assert "Curing Mind-Body Dualism" not in source


def test_no_raw_felt_perturbation_claim_survives():
    """The label is gone from the section header. The note explaining WHY it
    went stays, quoted, so the next reader does not put it back."""
    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "── Somatic Qualia: Raw felt perturbation of sampling ──" not in source
    assert '"Raw felt perturbation" was the previous label' in source


def test_the_surrogate_budget_signal_says_it_is_a_surrogate():
    """include_surrogate=True asks for a proxy. Recording it as Φ would put an
    integrated-information measurement in the record that was never taken."""
    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert '"kind": "phi_measured" if phi_is_measured else "phi_surrogate"' in source
    assert '"scales": "background_token_budget"' in source


def test_the_mechanisms_are_still_there():
    """Removing an overclaim must not remove the thing it overclaimed about."""
    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "morphogenetic_runtime" in source
    assert "somatic_qualia" in source
    assert "compute_perturbation()" in source


# ─────────────────────────────── the self-report gate gates


def test_a_unity_report_with_no_causes_still_supplies_its_verdict():
    """The verdict was read only inside the top_causes branch, so an unsafe
    report that listed no causes left the default True standing."""
    import core.brain.inference_gate as gate_mod

    _name, source = function_containing(gate_mod, "safe_to_self_report = bool(")

    assert "if unity_report is not None:" in source
    index_verdict = source.index('safe_to_self_report = bool(')
    index_causes = source.index('if unity_report and getattr(unity_report, "top_causes", None):')
    assert index_verdict < index_causes, (
        "the verdict is still read only when causes were listed"
    )


def test_a_failed_unity_probe_suppresses_the_self_report():
    """The check that decides whether she may describe her own state broke.
    Leaving the default True is the absence of a check counted as a pass."""
    import core.brain.inference_gate as gate_mod

    source = family_text(gate_mod)

    assert "suppressed the grounded self-report because unity could not be assessed" in source
