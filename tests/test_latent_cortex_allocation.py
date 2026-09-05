"""Allocation must not overrun the caller, or claim authority it lacks.

CP126 60007362: allocate() called its output "the Will's thought allocation"
while consulting no Will, taking no scoped authority, and signing no receipt.

CP126 d607a287: the caller's deadline was applied ONLY in the foreground
>=20B branch — every other profile ignored it — and the clamp that did exist
used max(15.0, timeout - 8.0), handing back 15 seconds to a caller with 10.

CP126 8a7e39cc: the coefficients are a hand-tuned heuristic with no
calibration, presented as allostatic policy.
"""
from __future__ import annotations

import pytest

from core.brain.latent_cortex_service import get_latent_cortex_service


@pytest.fixture()
def service():
    return get_latent_cortex_service()


RESIDENT = 32_000_000_000
SMALL = 1_500_000_000


@pytest.mark.parametrize(
    "params,foreground,timeout",
    [
        (RESIDENT, True, 120.0),
        (RESIDENT, True, 10.0),
        (RESIDENT, True, 3.0),
        (SMALL, False, 12.0),
        (SMALL, True, 6.0),
        (7_000_000_000, False, 5.0),
        (0, False, 30.0),
    ],
)
def test_no_profile_outlives_the_caller_deadline(service, params, foreground, timeout):
    _config, budget = service.allocate(
        stakes=0.9,
        uncertainty=0.8,
        model_parameter_count=params,
        foreground_request=foreground,
        timeout_s=timeout,
    )

    assert budget["wall_clock_s"] <= timeout


def test_a_short_deadline_is_not_overridden_by_a_floor(service):
    """max(15.0, timeout - 8.0) used to grant 15s to a 10s caller."""
    _config, budget = service.allocate(
        stakes=0.9, uncertainty=0.9,
        model_parameter_count=RESIDENT, foreground_request=True, timeout_s=10.0,
    )

    assert budget["wall_clock_s"] < 10.0


def test_the_reserve_scales_with_the_deadline(service):
    """A constant reserve would consume most of a short budget."""
    _c, short = service.allocate(
        stakes=0.5, uncertainty=0.5, timeout_s=4.0, model_parameter_count=SMALL
    )
    _c2, long = service.allocate(
        stakes=0.5, uncertainty=0.5, timeout_s=200.0, model_parameter_count=SMALL
    )

    assert short["wall_clock_s"] > 0.0
    assert long["wall_clock_s"] > short["wall_clock_s"]


def test_no_deadline_means_no_clamp(service):
    _config, budget = service.allocate(stakes=0.5, uncertainty=0.5)

    assert budget["wall_clock_s"] > 0.0


@pytest.mark.parametrize("params", [SMALL, RESIDENT])
@pytest.mark.parametrize("stakes", [0.0, 0.9])
def test_foreground_uses_owner_time_not_a_second_watchdog(service, monkeypatch, params, stakes):
    monkeypatch.setattr(
        "core.brain.latent_cortex_service._runtime_bounded_wall_clock_s",
        lambda *args, **kwargs: 20.0,
    )
    _, budget = service.allocate(
        stakes=stakes,
        uncertainty=0.5,
        model_parameter_count=params,
        foreground_request=True,
        timeout_s=300.0,
    )
    assert budget["wall_clock_s"] == 292.0


# --- the receipt says what this is ---------------------------------------


def test_the_allocation_receipt_disclaims_will_authority(service):
    service.allocate(stakes=0.6, uncertainty=0.6, timeout_s=30.0)
    receipt = service._last_allocation

    assert receipt["authority"] == "policy_heuristic"
    assert receipt["will_decision"] is None


def test_the_receipt_admits_the_policy_is_uncalibrated(service):
    service.allocate(stakes=0.6, uncertainty=0.6, timeout_s=30.0)
    receipt = service._last_allocation

    assert receipt["calibrated"] is False
    assert "no model-specific calibration" in receipt["calibration_basis"]


def test_the_receipt_records_the_deadline_it_was_given(service):
    service.allocate(stakes=0.6, uncertainty=0.6, timeout_s=42.0)
    receipt = service._last_allocation

    assert receipt["owner_timeout_s"] == 42.0
    assert receipt["granted_wall_clock_s"] <= 42.0
    assert receipt["deadline_respected"] is True


def test_the_receipt_carries_its_inputs(service):
    service.allocate(stakes=0.7, uncertainty=0.4, timeout_s=30.0)
    receipt = service._last_allocation

    assert receipt["stakes"] == pytest.approx(0.7)
    assert receipt["uncertainty"] >= 0.0
    assert "body_pressure" in receipt
    assert receipt["schema"] == "aura.latent_cortex.allocation_receipt.v1"


def test_the_docstring_no_longer_attributes_the_will(service):
    summary = (type(service).allocate.__doc__ or "").strip().splitlines()[0]

    assert "Will" not in summary


# --- input validation is unchanged ---------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_an_unusable_timeout_is_refused(service, bad):
    with pytest.raises(ValueError):
        service.allocate(stakes=0.5, uncertainty=0.5, timeout_s=bad)
