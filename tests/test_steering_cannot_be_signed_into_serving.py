"""Steering cannot be turned on by signing something.

"Make it live" has no honest path until a campaign passes, and the reason is
arithmetic rather than judgement. `_validate_steering` requires a causal
evaluation carrying verdict PASS, qualified True, at least 24 samples,
treatment strictly above the matched no-op, every lesion strictly under
treatment, no regression, and causal_effect_positive.

Both campaigns on the resident checkpoint fail it. The original vectors pass
the win counts and fail causal_effect_positive — a shuffled-layer control
reproduces most of the effect and a text instruction beats it. The
layer-specific set fails four ways at once.

These tests hold that the refusal follows from the recorded numbers, so
nobody has to take the reading on trust — including me.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RECOVERY = Path(__file__).resolve().parents[1] / "artifacts/migration/27b/recovery"


def _verdict(name: str) -> dict:
    path = RECOVERY / name
    if not path.is_file():
        pytest.skip(f"no {name} on disk")
    return json.loads(path.read_text(encoding="utf-8"))


def _would_qualify(verdict: dict) -> bool:
    """The subset of `_validate_steering` a campaign result decides."""
    treatment = verdict["treatment_successes"]
    lesions = verdict["lesion_successes"]
    return bool(
        verdict["samples_per_condition"] >= 24
        and treatment > verdict["matched_control_successes"]
        and lesions
        and all(0 <= one < treatment for one in lesions.values())
        and verdict["no_regression"] is True
        and verdict["causal_effect_positive"] is True
    )


def test_the_original_vectors_do_not_qualify():
    verdict = _verdict("campaign_verdict.json")
    assert verdict["treatment_successes"] > verdict["matched_control_successes"]
    assert verdict["no_regression"] is True
    # And still refused, on the one requirement the win counts cannot carry.
    assert verdict["causal_effect_positive"] is False
    assert not _would_qualify(verdict)


def test_the_layer_specific_vectors_do_not_qualify():
    verdict = _verdict("campaign_verdict_layer_specific.json")
    assert verdict["treatment_successes"] == verdict["matched_control_successes"]
    assert verdict["lesion_successes"]["shuffled_layers"] > verdict["treatment_successes"]
    assert verdict["no_regression"] is False
    assert not _would_qualify(verdict)


def test_the_authority_requires_every_one_of_those_fields():
    """So the check above is the authority's check and not a copy of it."""
    import inspect

    from core.learning import cortex_migration_authority as authority

    source = inspect.getsource(authority._validate_steering)
    for required in (
        '"verdict") != "PASS"',
        '"qualified") is not True',
        "sample_count < 24",
        "treatment_successes <= matched_control_successes",
        "successes < treatment_successes",
        '"no_regression") is not True',
        '"causal_effect_positive") is not True',
    ):
        assert required in source, required
