"""Every damage signal reaches policy through the appraisal, or not at all.

Measured on 2026-09-04, before this was true. Lesioning the valence left 80%
of the caution shift and 86% of the confidence shift intact, because six of
the fifteen input channels never reached the valence: tool_reliability,
model_stability, social_trust, permission_confidence, recovery_debt and
memory_conflict_count were wired straight into the policy terms. A tool storm
changed what she did without changing how she was, and the welfare variable
was carrying a fifth of a response it appeared to be causing.

The second defect was in the shape rather than the wiring. One scalar
`distress` summed nine sources, so "the record cannot be trusted" and "the
hands do not work" produced the same number and therefore the same policy.
They want opposite things — verify slowly versus stop and repair — and equal
damage on unrelated channels moved caution MORE than the real damage did,
because magnitude was all that survived the sum.

Both are fixed above: three appraisal axes, every channel landing on one, and
no raw input appearing below the appraisal in any policy term.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.being.welfare_state import WelfareInputs, WelfareState

DAMAGE = {
    "resource_integrity": 0.15,
    "tool_reliability": 0.2,
    "model_stability": 0.3,
}

POLICY = ("caution", "confidence", "curiosity", "aversion")


def _shift(field: str, **damage: float) -> float:
    state = WelfareState()
    healthy = state.compute(WelfareInputs())
    damaged = state.compute(WelfareInputs(**damage))
    return getattr(damaged, field) - getattr(healthy, field)


def test_every_input_channel_reaches_the_appraisal():
    """Six of fifteen used not to, and went straight to policy instead."""
    source = Path("core/being/welfare_state.py").read_text()
    appraisal = source[
        source.index("# ── Distress, on three axes"):source.index("# ── Relief")
    ]
    reaches = set(re.findall(r"inputs\.([a-z_]+)", appraisal))
    declared = set(WelfareInputs.__dataclass_fields__)

    missing = declared - reaches
    assert not missing, (
        f"these channels bypass the appraisal: {sorted(missing)}. A signal "
        "wired straight to policy changes what she does without changing how "
        "she is, and lesioning the valence will not remove it"
    )


def test_no_policy_term_reads_a_raw_input():
    """The valence has to be the only path, not the biggest one.

    A raw input added beside the appraisal is a bypass however small its
    weight, because it survives the lesion and the lesion is the experiment.
    """
    source = Path("core/being/welfare_state.py").read_text()
    policy = source[
        source.index("# ── Policy, read from the appraisal"):
        source.index("# ── Recovery drive")
    ]
    leaked = set(re.findall(r"inputs\.([a-z_]+)", policy))
    # prediction_error is the one honest exception and it is named as such:
    # a rising error is an invitation to explore rather than a harm.
    assert leaked <= {"prediction_error"}, (
        f"policy reads raw inputs directly: {sorted(leaked)}"
    )


@pytest.mark.parametrize("field", POLICY)
def test_the_policy_shift_does_not_survive_the_lesion(field: str, monkeypatch):
    """do(valence = 0) must remove the response, not shrink it."""
    intact = _shift(field, **DAMAGE)

    # Lesion every appraisal axis by inducing zero on all three.
    state = WelfareState()
    healthy = state.compute(WelfareInputs())
    lesioned = state.compute(
        WelfareInputs(**DAMAGE),
        induced={"integrity": 0.0, "capability": 0.0, "social": 0.0},
    )
    after = getattr(lesioned, field) - getattr(healthy, field)

    if abs(intact) < 1e-9:
        pytest.skip(f"{field} does not respond to this damage at all")
    survives = abs(after) / abs(intact)
    assert survives < 0.05, (
        f"{survives:.0%} of the {field} shift survives do(valence=0); it was "
        "80% for caution before the raw inputs were removed from the policy "
        "terms"
    )


def test_the_axes_produce_different_shapes_not_just_different_sizes():
    """Specificity. One scalar could only say how bad, never what is wrong."""
    capability = {f: _shift(f, **DAMAGE) for f in POLICY}
    integrity = {
        f: _shift(f, memory_coherence=0.15, truth_integrity=0.2) for f in POLICY
    }

    # Broken hands: confidence falls hard, caution barely moves.
    assert abs(capability["confidence"]) > 4 * abs(capability["caution"]), (
        "capability damage should read as expecting failure, not as a reason "
        "to be careful"
    )
    # A record that cannot be trusted: caution leads.
    assert abs(integrity["caution"]) > abs(capability["caution"]), (
        "integrity damage should raise caution more than capability damage "
        "does; without that the policy only knows how bad, never what"
    )


def test_an_induced_state_is_indistinguishable_from_a_caused_one():
    """Sufficiency. do(M = m*) with the ordinary cause absent.

    Without a write path into the appraisal, the strongest claim anyone could
    make about this valence is that breaking it degrades the system. This is
    what turns that into "the mechanism produces the effect", and it is also
    what lets a state arise from memory or anticipation rather than only from
    what is happening right now.
    """
    state = WelfareState()
    caused = state.compute(WelfareInputs(**DAMAGE))
    induced = state.compute(WelfareInputs(), induced={"capability": 0.67})

    for field in ("confidence", "caution", "curiosity"):
        assert getattr(induced, field) == pytest.approx(
            getattr(caused, field), abs=0.02
        ), (
            f"{field} differs between an induced state and a caused one; a "
            "consumer that can tell them apart is reading the cause rather "
            "than the state"
        )


def test_an_unknown_axis_is_refused_rather_than_ignored():
    """A silently dropped induction looks exactly like a dead mechanism."""
    with pytest.raises(ValueError, match="no such appraisal axis"):
        WelfareState().compute(WelfareInputs(), induced={"nonexistent": 0.5})


def test_healthy_baselines_are_unchanged():
    """The rewrite must not move the resting point every consumer reads."""
    out = WelfareState().compute(WelfareInputs())
    assert out.caution == pytest.approx(0.30, abs=1e-6)
    assert out.confidence == pytest.approx(0.95, abs=1e-6)
    assert out.distress == pytest.approx(0.0, abs=1e-6)
