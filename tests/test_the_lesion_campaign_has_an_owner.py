"""The campaign that measures whether a faculty matters is actually wired.

An external review said the automated lesion campaign is attached to a path
that is not normally booted, which would make Aura's best causal-science
machinery permanently idle. Measured rather than reasoned about: the job IS
registered on the default conductor and IS admitted in a default posture.

What was missing is the reading that tells a campaign which never ran from
one that ran and found nothing. Both report no measured channels and no
decorative ones, and that reads as a clean bill of health. The count of
registered channels that have never been measured is what separates them.
"""

from __future__ import annotations

import pytest

from core.runtime.autonomy_conductor import AutonomyConductor


@pytest.fixture
def a_default_conductor() -> AutonomyConductor:
    conductor = AutonomyConductor()
    conductor.register_defaults()
    return conductor


def test_the_campaign_is_one_of_the_default_jobs(a_default_conductor):
    """An owner, not a module nobody calls."""
    assert "influence_campaign" in a_default_conductor.jobs


def test_the_campaign_is_admitted_in_a_default_posture(a_default_conductor):
    """Registered and permanently deferred would be the same as absent."""
    job = a_default_conductor.jobs["influence_campaign"]
    reason = a_default_conductor._job_policy_reason(job)  # noqa: SLF001
    assert reason == "", f"the campaign would never run: {reason}"


def test_the_campaign_measures_one_channel_at_a_time(a_default_conductor):
    """Three generations a trial. Measuring six at once is an hour of the day."""
    import ast
    import inspect

    source = inspect.getsource(AutonomyConductor._job_influence_campaign)
    tree = ast.parse(source.lstrip())
    assert "channels=[channel]" in ast.unparse(tree)


def test_health_says_how_many_channels_have_never_been_measured():
    """The reading that separates "ran and found nothing" from "never ran"."""
    from core.runtime.health_contract import _runtime_integrity_block  # noqa: PLC2701

    block = _runtime_integrity_block()
    assert "influence_channels_registered" in block
    assert "influence_channels_never_measured" in block
    assert block["influence_channels_never_measured"] <= block[
        "influence_channels_registered"
    ]


def test_an_unmeasured_channel_is_named_rather_than_only_counted():
    """A count says something is wrong; a name says what to run."""
    from core.runtime.health_contract import _runtime_integrity_block  # noqa: PLC2701

    block = _runtime_integrity_block()
    if block.get("influence_channels_never_measured"):
        assert block.get("influence_channels_awaiting_evidence")
