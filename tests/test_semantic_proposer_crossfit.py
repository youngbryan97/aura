"""A proposer holdout excludes whole source constructions before fitting."""

import json
from types import SimpleNamespace

import pytest

from core.learning.semantic_construction_folds import construction_folds
from core.learning.semantic_program_campaign import _sha
from tools.probe_semantic_proposer_crossfit import crossfit_partition


def _examples():
    return [SimpleNamespace(
        ir=SimpleNamespace(source_text_sha256=f"source-{index:02d}-{contrast}",
                           n_inputs=2 + index % 2, instructions=("step",)),
        construction_id=f"construction-{index:02d}",
        contrast_id="", split="train")
        for index in range(12) for contrast in range(2)]


def test_crossfit_excludes_held_sources_from_fit_and_calibration():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 2)
    groups = [{item.construction_id for item in part}
              for part in (fit, calibration, held)]
    assert all(groups)
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert len(held) == len(groups[2])
    assert all(folds["assignments"][item.ir.source_text_sha256] == 2 for item in held)


def test_all_held_reuses_partition_without_omitting_contrasts():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    fit, calibration, held = crossfit_partition(examples, folds, 2, all_held=True)
    wanted = {item.ir.source_text_sha256 for item in examples
              if folds["assignments"][item.ir.source_text_sha256] == 2}
    assert {item.ir.source_text_sha256 for item in held} == wanted
    assert not wanted & {item.ir.source_text_sha256 for item in fit + calibration}


def test_diagnostic_subset_is_ordered_and_confined_to_held_fold():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    wanted = [item.ir.source_text_sha256 for item in examples
              if folds["assignments"][item.ir.source_text_sha256] == 2][:2]
    _fit, _calibration, held = crossfit_partition(
        examples, folds, 2, all_held=True, held_source_ids=tuple(reversed(wanted)))
    assert [item.ir.source_text_sha256 for item in held] == list(reversed(wanted))
    outside = next(item.ir.source_text_sha256 for item in examples
                   if folds["assignments"][item.ir.source_text_sha256] != 2)
    with pytest.raises(ValueError, match="outside"):
        crossfit_partition(examples, folds, 2, all_held=True,
                           held_source_ids=(outside,))


def test_crossfit_rejects_relabelled_frozen_fold():
    examples = _examples()
    folds = json.loads(json.dumps(construction_folds(examples)))
    source = examples[0].ir.source_text_sha256
    folds["assignments"][source] = (folds["assignments"][source] + 1) % folds["count"]
    body = {key: value for key, value in folds.items() if key != "receipt_sha256"}
    folds["receipt_sha256"] = _sha(body)
    with pytest.raises(ValueError, match="construction group crosses"):
        crossfit_partition(examples, folds, 2)
