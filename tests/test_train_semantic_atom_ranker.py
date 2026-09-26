"""A local-factor fit must preserve source-only construction boundaries."""

import json
from collections import defaultdict
from types import SimpleNamespace

import pytest

from core.learning.semantic_construction_folds import utterance_construction_folds
from tools.probe_semantic_proposer_crossfit import crossfit_partition
from tools.train_semantic_atom_ranker import construction_weights, validate_atom_partition


def _partition():
    examples = [SimpleNamespace(
        ir=SimpleNamespace(source_text_sha256=f"{family}-{index}-{variant}",
                           n_inputs=inputs, instructions=("step",)),
        construction_id=f"{family}:wording-{index}", contrast_id="", split="train")
        for family, inputs in (("first", 2), ("second", 3))
        for index in range(3) for variant in range(index + 1)]
    folds = json.loads(json.dumps(utterance_construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 0, all_held=True)
    plan = {"fold": 0, "outer_fold": None, "input_order_policy": "source_token_order_v1",
            "fit_ids": sorted(item.ir.source_text_sha256 for item in fit),
            "calibration_ids": sorted(item.ir.source_text_sha256 for item in calibration),
            "held_ids": [held[0].ir.source_text_sha256]}
    return examples, folds, plan


def test_fit_excludes_entire_held_constructions_not_only_sampled_rows():
    examples, folds, plan = _partition()
    fit, calibration = validate_atom_partition(examples, plan, folds)
    held_constructions = {item.construction_id for item in examples
                          if folds["assignments"][item.ir.source_text_sha256] == 0}
    assert not held_constructions & {item.construction_id for item in fit + calibration}
    assert len(plan["held_ids"]) == 1


@pytest.mark.parametrize("field", ["fit_ids", "calibration_ids", "held_ids"])
def test_substituted_cohorts_are_rejected(field):
    examples, folds, plan = _partition()
    plan[field] = ["unbound-source"]
    with pytest.raises(ValueError, match="frozen bank"):
        validate_atom_partition(examples, plan, folds)


def test_nested_or_wrong_input_order_cannot_enter_this_fit():
    examples, folds, plan = _partition()
    for change in ({"outer_fold": 0}, {"input_order_policy": None}):
        with pytest.raises(ValueError, match="frozen bank"):
            validate_atom_partition(examples, {**plan, **change}, folds)


def test_construction_frequency_does_not_change_total_supervision_mass():
    examples, _folds, _plan = _partition()
    weights = construction_weights(examples)
    mass = defaultdict(float)
    for item in examples:
        mass[item.construction_id] += weights[item.ir.source_text_sha256]
    assert sum(weights.values()) == pytest.approx(len(examples))
    assert all(value == pytest.approx(next(iter(mass.values()))) for value in mass.values())
