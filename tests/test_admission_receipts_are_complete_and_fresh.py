"""An admission receipt must be whole and current to authorize a heavy lane.

CP126 (high), core/brain/inference_gate.py: "Missing admission fields are
interpreted as permission. Foreground admission results are consumed as
dictionaries and can_admit defaults to True when absent. An incomplete or
stale admission receipt therefore authorizes the expensive lane rather than
failing closed or requiring a validated schema."

The absent-field half was already closed — consumers read
``.get("can_admit", False)``. The other two words in that finding were not.

INCOMPLETE: a bare ``{"can_admit": True}`` was indistinguishable from a full
measurement. The test fixtures in this suite were literally built that way,
which is how the permissive reading kept looking safe.

STALE: a snapshot has no expiry, so a reading taken minutes ago authorized a
~20GB model load on evidence that may no longer be true. Memory pressure on
this host moves in seconds under load.

Snapshots now carry a schema and a measurement time, and ``admission_permits``
refuses an unrecognised shape, a partial receipt, and an old one — each with
a distinct reason, because "we were refused" and "we asked with a stale
receipt" need different responses.
"""
from __future__ import annotations

import time

import pytest

from core.brain.inference_gate import (
    ADMISSION_SNAPSHOT_MAX_AGE_S,
    ADMISSION_SNAPSHOT_SCHEMA,
    InferenceGate,
    admission_permits,
)


@pytest.fixture
def fresh():
    """A real snapshot, produced the way production produces them."""
    return InferenceGate._headroom_snapshot("primary")


class TestProducedSnapshotsCarryTheirProvenance:
    def test_a_real_snapshot_is_stamped(self, fresh):
        assert fresh["schema"] == ADMISSION_SNAPSHOT_SCHEMA
        assert isinstance(fresh["measured_at_monotonic"], float)

    def test_every_snapshot_exit_stamps(self):
        """Both the measured and the probe-failed paths, in both builders.

        Read across the class and the bases it is built from, not off one
        file. Two of the four builders moved to `_WatchesTheCortexComeUp` when
        the gate went under the module ceiling, and `inspect.getsource` of the
        subclass alone found two -- which is indistinguishable from two of them
        having lost their stamp.
        """
        import inspect

        source = "".join(
            inspect.getsource(klass)
            for klass in InferenceGate.__mro__
            if klass is not object
        )
        assert source.count('"schema": ADMISSION_SNAPSHOT_SCHEMA,') == 4
        assert source.count('"measured_at_monotonic": time.monotonic(),') == 4

    def test_a_healthy_snapshot_permits(self, fresh):
        if not fresh.get("can_admit"):
            pytest.skip("host is genuinely under memory pressure")
        assert admission_permits(fresh) == (True, "")


class TestIncompleteReceiptsAreRefused:
    def test_a_bare_can_admit_dict_does_not_permit(self):
        """The exact shape the old fixtures used."""
        permitted, reason = admission_permits({"can_admit": True})
        assert permitted is False
        assert "schema_unrecognised" in reason

    @pytest.mark.parametrize(
        "missing", ["can_admit", "measured", "pressure_pct", "available_gb", "tier"],
    )
    def test_a_missing_required_field_refuses(self, fresh, missing):
        partial = {k: v for k, v in fresh.items() if k != missing}
        permitted, reason = admission_permits(partial)
        assert permitted is False
        assert missing in reason

    def test_a_foreign_schema_refuses(self, fresh):
        permitted, reason = admission_permits(dict(fresh, schema="something.else"))
        assert permitted is False
        assert "schema_unrecognised" in reason

    @pytest.mark.parametrize("value", [None, "yes", 42, []])
    def test_a_non_dict_refuses(self, value):
        permitted, reason = admission_permits(value)
        assert permitted is False
        assert "not_a_dict" in reason


class TestStaleReceiptsAreRefused:
    def test_an_old_measurement_does_not_permit(self, fresh):
        stale = dict(fresh, can_admit=True,
                     measured_at_monotonic=time.monotonic() - (ADMISSION_SNAPSHOT_MAX_AGE_S + 60))
        permitted, reason = admission_permits(stale)
        assert permitted is False
        assert "stale" in reason

    def test_a_recent_measurement_permits(self, fresh):
        recent = dict(fresh, can_admit=True, measured_at_monotonic=time.monotonic() - 1.0)
        assert admission_permits(recent)[0] is True

    def test_an_unstamped_receipt_refuses(self, fresh):
        permitted, reason = admission_permits(dict(fresh, measured_at_monotonic="soon"))
        assert permitted is False
        assert "unstamped" in reason


class TestRefusalsStaySeparable:
    """A policy refusal and a broken receipt need different responses, so
    they must not collapse into one boolean."""

    def test_a_policy_refusal_reports_its_own_reason(self, fresh):
        refused = dict(fresh, can_admit=False, reason="memory_pressure:91%/2.0GB")
        permitted, reason = admission_permits(refused)
        assert permitted is False
        assert reason == "memory_pressure:91%/2.0GB"

    def test_a_broken_receipt_does_not_look_like_policy(self):
        _permitted, reason = admission_permits({"can_admit": True})
        assert "memory" not in reason
        assert "admission_snapshot_" in reason
