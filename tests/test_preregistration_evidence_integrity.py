"""A missing setting or impossible measurement cannot confirm a registered plan."""

import json

import pytest

from core.evaluation.preregistration import EvidenceStatus, Preregistration, load_preregistration


def plan(**kwargs):
    return Preregistration("replication", "paired gain", kwargs.pop("parameters", {"seed": 42}),
                           kwargs.pop("metrics", {"gain": 0.1}), arms=("base", "treatment"), **kwargs)


@pytest.mark.parametrize("used", [None, {}, {"unrelated": 42}])
def test_missing_parameters_do_not_confirm(used):
    verdict = plan().verify_result({"gain": 0.5}, parameters_used=used)
    assert not verdict["confirms_hypothesis"]
    assert verdict["parameter_drift"] == {"seed": "registered parameter not reported"}
    assert verdict["findings"][0]["status"] == EvidenceStatus.EXPLORATORY


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "not measured"])
def test_invalid_measurements_are_unmeasured(value):
    verdict = plan().verify_result({"gain": value}, parameters_used={"seed": 42})
    assert not verdict["confirms_hypothesis"]
    assert verdict["unmeasured_metrics"] == ["gain"]
    json.dumps(verdict, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True])
def test_invalid_thresholds_are_rejected(value):
    with pytest.raises(ValueError):
        plan(metrics={"gain": value})


def test_nested_plan_is_detached_and_immutable():
    source = {"seeds": [1, 2], "stop": {"tasks": 300}}
    registered = plan(parameters=source)
    before = registered.plan_hash
    source["seeds"].append(3)
    source["stop"]["tasks"] = 3
    assert registered.plan_hash == before
    with pytest.raises(TypeError):
        registered.parameters["stop"]["tasks"] = 3
    with pytest.raises(TypeError):
        registered.metrics["gain"] = 0
    assert registered.parameter_drift({"seeds": [1, 2], "stop": {"tasks": 300}}) == {}
    exported = registered.to_dict()
    exported["parameters"]["stop"]["tasks"] = 3
    assert registered.plan_hash == before


@pytest.mark.parametrize("value", [float("nan"), object(), {1: "key"}])
def test_unsupported_parameters_have_no_repr_identity(value):
    with pytest.raises(ValueError):
        plan(parameters={"bad": value})


def test_missing_hash_cannot_be_loaded_as_registered(tmp_path):
    document = plan().to_dict()
    document.pop("plan_hash")
    path = tmp_path / "unsealed.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="recorded plan hash"):
        load_preregistration(path)


def test_parameter_types_are_part_of_the_registered_contract():
    assert plan(parameters={"flag": True}).parameter_drift({"flag": 1})


@pytest.mark.parametrize("arms", [None, (), ("base",), ("base", "treatment", "posthoc"),
                                  ("base", "treatment", "treatment"), "base", (1,)])
def test_missing_or_changed_arms_cannot_confirm(arms):
    verdict = plan().verify_result({"gain": 0.5}, parameters_used={"seed": 42}, arms_used=arms)
    assert not verdict["confirms_hypothesis"]
    assert verdict["arm_drift"]


def test_arm_inventory_order_does_not_change_comparison():
    verdict = plan().verify_result({"gain": 0.5}, parameters_used={"seed": 42},
                                   arms_used=("treatment", "base"))
    assert verdict["confirms_hypothesis"]


def test_registered_evidence_invariant():
    from core.evaluation.preregistration import _preregistered_evidence
    assert _preregistered_evidence() == ()
