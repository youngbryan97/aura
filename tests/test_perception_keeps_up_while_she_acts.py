"""She went nearly blind at the exact moment she started acting.

constitutive_compute_budget clamps a continuous loop to ``foreground_hz``
whenever a user-facing generation is running. That is right for background
curiosity and backwards for perception the generation DEPENDS ON:
continuous_vision passes foreground_hz=0.1, so from the first token of a reply
her sight fell to one frame every ten seconds.

Nothing failed and nothing was logged. Any task built on look-act-look — drag
something and watch it move, wait on a progress bar, verify a click landed,
react to a board that changes — was unreachable at that cadence, and from
inside the task the reason was invisible.

The acting half and the seeing half had no way to talk. A task now DECLARES
that it needs to see while it acts, and the perception loops read that. Neither
side knows the other exists, and nothing here knows about screens, browsers or
any particular task.
"""
from __future__ import annotations

import time

import pytest

import core.runtime.background_policy as background_policy
from core.runtime.perception_demand import (
    active_perception_reasons,
    claim_perception,
    perception_demand,
    perception_is_demanded,
    release_perception,
    renew_perception,
    reset_perception_demand,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_perception_demand()
    yield
    reset_perception_demand()


@pytest.fixture
def generation_running(monkeypatch):
    """The condition under which she used to go blind."""
    monkeypatch.setattr(
        background_policy,
        "_foreground_activity_reason",
        lambda: "foreground_generation_active",
    )


def _budget(**kw):
    return background_policy.constitutive_compute_budget(
        "test_loop", 2.0, min_hz=0.1, foreground_hz=0.1, **kw
    )


def test_a_loop_serving_the_foreground_is_not_throttled_by_it(generation_running):
    """The measured defect: 0.1Hz while acting, which is 10s per frame."""
    assert _budget().effective_hz == pytest.approx(0.1)
    assert _budget(serves_foreground=True).effective_hz == pytest.approx(2.0)


def test_the_reason_says_why_the_cadence_changed(generation_running):
    """A cadence nobody can explain is how this stayed hidden."""
    assert "serving_foreground" in _budget(serves_foreground=True).reason


def test_background_work_is_still_throttled(generation_running):
    """The clamp must keep doing its actual job."""
    assert _budget(serves_foreground=False).effective_hz == pytest.approx(0.1)


def test_host_protections_still_apply(monkeypatch, generation_running):
    """Only the foreground clamp is skipped, never the ones protecting the host."""
    monkeypatch.setattr(
        background_policy, "_read_compute_pressure_reason", lambda: "compute_pressure"
    )

    budget = _budget(serves_foreground=True, compute_pressure_hz=0.25)

    assert budget.effective_hz == pytest.approx(0.25)


def test_demand_is_refcounted():
    """Two tasks can want to see; the first to finish must not blind the second."""
    first = claim_perception("task one")
    second = claim_perception("task two")

    release_perception(first)
    assert perception_is_demanded()

    release_perception(second)
    assert not perception_is_demanded()


def test_a_claim_expires_so_a_crash_cannot_pin_the_cameras_open():
    claim_perception("task that dies", ttl_s=1.0)
    assert perception_is_demanded()

    time.sleep(1.05)

    assert not perception_is_demanded()


def test_a_live_task_can_renew_past_the_ttl():
    token = claim_perception("long task", ttl_s=1.0)
    time.sleep(0.6)

    assert renew_perception(token, ttl_s=10.0)
    time.sleep(0.6)
    assert perception_is_demanded()


def test_renewing_a_lapsed_claim_reports_failure():
    token = claim_perception("gone", ttl_s=1.0)
    time.sleep(1.05)

    assert renew_perception(token) is False


def test_the_scope_releases_even_when_the_body_raises():
    with pytest.raises(ValueError):
        with perception_demand("failing task"):
            raise ValueError("boom")

    assert not perception_is_demanded()


def test_reasons_are_reportable():
    """A surface can say WHY perception is being held open."""
    with perception_demand("driving a UI task"):
        assert "driving a UI task" in active_perception_reasons()


def test_releasing_twice_is_safe():
    token = claim_perception("x")
    release_perception(token)
    release_perception(token)

    assert not perception_is_demanded()


def test_the_vision_loop_actually_reads_the_demand(generation_running):
    """Wiring: the primitive is worthless if the sense never consults it."""
    from core.senses.continuous_vision import ContinuousSensoryBuffer

    blind = ContinuousSensoryBuffer._compute_budget()
    with perception_demand("acting on the screen"):
        seeing = ContinuousSensoryBuffer._compute_budget()

    assert blind.effective_hz == pytest.approx(0.1)
    assert seeing.effective_hz > blind.effective_hz * 5
    # The cadence a look-act-look loop needs at all.
    assert seeing.interval_s <= 1.0


def test_acting_on_the_world_raises_perception():
    """Every governed action passes through one place; that is where this lives."""
    from core.governance.will import ActionDomain
    from core.runtime.action_executor import _hold_perception_for

    _hold_perception_for(ActionDomain.ENVIRONMENT_ACTION, "click_at")

    assert perception_is_demanded()
    assert any("click_at" in reason for reason in active_perception_reasons())


def test_thinking_does_not_raise_perception():
    """Only actions that change the world create something to look at."""
    from core.governance.will import ActionDomain
    from core.runtime.action_executor import _hold_perception_for

    for domain in (
        ActionDomain.MEMORY_WRITE,
        ActionDomain.REFLECTION,
        ActionDomain.RESPONSE,
    ):
        _hold_perception_for(domain, "some_action")

    assert not perception_is_demanded()


def test_the_claim_outlives_the_action():
    """Look-act-LOOK: the result appears after the call returns.

    Releasing on return would drop her back to one frame per ten seconds
    exactly when the thing she did becomes visible.
    """
    from core.governance.will import ActionDomain
    from core.runtime.action_executor import (
        ACTION_PERCEPTION_WINDOW_S,
        _hold_perception_for,
    )

    _hold_perception_for(ActionDomain.ENVIRONMENT_ACTION, "type_text")

    assert perception_is_demanded()
    assert ACTION_PERCEPTION_WINDOW_S >= 2.0


def test_the_executor_calls_it_on_every_action():
    """Wiring: a skill added tomorrow must be covered without being edited."""
    import inspect

    from core.runtime.action_executor import ActionExecutor

    source = inspect.getsource(ActionExecutor.execute)
    assert "_hold_perception_for" in source
