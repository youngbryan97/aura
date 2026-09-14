"""A plan written after the data is not a plan.

Two live results were reported as established on decisions taken after seeing
the numbers they justify: the Grassmann encoder width (best of a three-way
sweep, then installed as the default and described as "the measured optimum")
and the CAA steering pass criterion. Neither was dishonest; both are what
happens when the analysis plan and the analysis share a session.
"""

from __future__ import annotations

import json

import pytest

from core.evaluation.preregistration import (
    EvidenceStatus,
    Preregistration,
    load_preregistration,
)

pytestmark = pytest.mark.unit


def _plan(**overrides) -> Preregistration:
    base = dict(
        campaign="phi_width_replication",
        hypothesis="12-mode Grassmann integration exceeds the independent-halves control",
        parameters={"encoder_width": 12, "null_surrogates": 8},
        metrics={"integration_fraction": 0.10, "null_p_value_inverted": 0.95},
        arms=("live", "independent_halves", "memoryless"),
    )
    base.update(overrides)
    return Preregistration(**base)


class TestThePlanIsIdentifiable:
    def test_the_same_plan_hashes_the_same(self):
        assert _plan().plan_hash == _plan().plan_hash

    def test_the_timestamp_is_not_part_of_the_identity(self):
        a = _plan(registered_at="2026-01-01T00:00:00+00:00")
        b = _plan(registered_at="2026-08-05T00:00:00+00:00")
        assert a.plan_hash == b.plan_hash

    def test_changing_a_threshold_changes_the_identity(self):
        loosened = _plan(metrics={"integration_fraction": 0.001})
        assert loosened.plan_hash != _plan().plan_hash

    def test_a_plan_edited_after_registration_is_refused(self, tmp_path):
        path = _plan().write(tmp_path)
        data = json.loads(path.read_text())
        data["metrics"]["integration_fraction"] = 0.001  # loosened after the fact
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="edited after registration"):
            load_preregistration(path)

    def test_an_unedited_plan_round_trips(self, tmp_path):
        path = _plan().write(tmp_path)
        assert load_preregistration(path).plan_hash == _plan().plan_hash

    def test_the_same_plan_is_idempotent_but_changed_bytes_are_never_replaced(self, tmp_path):
        plan = _plan()
        path = plan.write(tmp_path)
        assert plan.write(tmp_path) == path

        path.write_text("tampered", encoding="utf-8")
        with pytest.raises(FileExistsError, match="replacement attempt"):
            plan.write(tmp_path)
        assert path.read_text(encoding="utf-8") == "tampered"


class TestWhatARunIsAllowedToClaim:
    def test_a_declared_metric_that_clears_its_threshold_confirms(self):
        verdict = _plan().verify_result(
            {"integration_fraction": 0.307, "null_p_value_inverted": 0.99},
            parameters_used={"encoder_width": 12, "null_surrogates": 8},
            arms_used=("live", "independent_halves", "memoryless"),
        )
        assert verdict["confirms_hypothesis"] is True
        assert all(
            f["status"] == EvidenceStatus.CONFIRMATORY for f in verdict["findings"]
        )

    def test_a_declared_metric_that_falls_short_is_a_negative_not_a_failure(self):
        verdict = _plan().verify_result(
            {"integration_fraction": 0.007, "null_p_value_inverted": 0.99},
            parameters_used={"encoder_width": 12, "null_surrogates": 8},
            arms_used=("live", "independent_halves", "memoryless"),
        )
        statuses = {f["metric"]: f["status"] for f in verdict["findings"]}
        assert statuses["integration_fraction"] == EvidenceStatus.NEGATIVE
        assert verdict["confirms_hypothesis"] is False

    def test_a_metric_that_appears_from_nowhere_is_exploratory(self):
        verdict = _plan().verify_result(
            {
                "integration_fraction": 0.307,
                "null_p_value_inverted": 0.99,
                "phi_s_at_16_modes": 0.219,
            },
            parameters_used={"encoder_width": 12, "null_surrogates": 8},
            arms_used=("live", "independent_halves", "memoryless"),
        )
        assert verdict["exploratory_metrics"] == ["phi_s_at_16_modes"]
        assert verdict["confirms_hypothesis"] is True, (
            "an extra observation does not invalidate the declared ones"
        )

    def test_a_declared_metric_never_measured_cannot_pass(self):
        verdict = _plan().verify_result(
            {"integration_fraction": 0.307},
            parameters_used={"encoder_width": 12, "null_surrogates": 8},
            arms_used=("live", "independent_halves", "memoryless"),
        )
        assert verdict["unmeasured_metrics"] == ["null_p_value_inverted"]
        assert verdict["confirms_hypothesis"] is False

    def test_running_at_a_different_width_makes_everything_exploratory(self):
        """The width sweep, judged by its own plan.

        Registering width 12 and then reporting the arm that happened to win
        is a different experiment. It stays reportable; it stops being
        confirmatory.
        """
        verdict = _plan().verify_result(
            {"integration_fraction": 0.307, "null_p_value_inverted": 0.99},
            parameters_used={"encoder_width": 16, "null_surrogates": 8},
            arms_used=("live", "independent_halves", "memoryless"),
        )
        assert verdict["parameter_drift"] == {"encoder_width": "12->16"}
        assert verdict["confirms_hypothesis"] is False
        assert set(verdict["exploratory_metrics"]) == set(_plan().metrics)


class TestThePlanCannotBeVacuous:
    def test_a_plan_declaring_nothing_confirms_nothing(self):
        empty = _plan(metrics={})
        assert empty.verify_result({"anything": 1.0})["confirms_hypothesis"] is False
