"""Error acquisition uses training observations without losing source controls."""

from dataclasses import replace
import json

import pytest

import core.learning.semantic_graph_trial as trial
from core.learning.semantic_program_campaign import _sha
from tests.test_semantic_relation_graph_learning import model_examples


@pytest.fixture(scope="module")
def source():
    return model_examples()


@pytest.mark.parametrize("status", ["different", "decode_refused", "unknown", "unmeasured"])
def test_every_unresolved_training_row_is_selected_without_decoding_other_splits(source, monkeypatch, status):
    model, examples = source
    pool = trial.select_trial_examples(examples, split="train", count=len(examples))
    failure = pool[-1].ir.source_text_sha256
    seen = []

    def observe(model, item):
        assert item.split == "train"
        seen.append(item.ir.source_text_sha256)
        return {"source_text_sha256": item.ir.source_text_sha256, "split": item.split,
                "semantic_status": status if item.ir.source_text_sha256 == failure else "equivalent"}

    monkeypatch.setattr(trial, "_observe", observe)
    selected, receipt = trial.acquire_semantic_training_errors(model, examples, control_count=1)
    assert seen == [item.ir.source_text_sha256 for item in pool]
    assert [item.ir.source_text_sha256 for item in selected] == [seen[0], failure]
    assert receipt["training_example_ids_sha256"] == _sha(sorted((seen[0], failure)))
    assert receipt["control_sources"] == [seen[0]]
    assert receipt["training_pool_sources"] == seen
    assert receipt["validation_examples_used"] == receipt["test_examples_used"] == 0
    assert not receipt["serving_authority"] and not receipt["fresh_transfer_claim"]
    assert receipt["receipt_sha256"] == _sha({k: v for k, v in receipt.items() if k != "receipt_sha256"})


def test_source_controls_survive_when_every_training_row_passes(source, monkeypatch):
    model, examples = source
    monkeypatch.setattr(trial, "_observe", lambda model, item:
                        {"source_text_sha256": item.ir.source_text_sha256, "semantic_status": "equivalent"})
    selected, receipt = trial.acquire_semantic_training_errors(model, examples, control_count=2)
    reordered, other = trial.acquire_semantic_training_errors(model, tuple(reversed(examples)), control_count=2)
    assert [item.ir.source_text_sha256 for item in selected] == [item.ir.source_text_sha256 for item in reordered]
    assert receipt == other
    assert len(selected) == 2


def test_acquisition_never_accepts_overlapping_source_identities(source):
    model, examples = source
    with pytest.raises(ValueError, match="overlap"):
        trial.acquire_semantic_training_errors(model, (examples[0], replace(examples[0], split="test")), control_count=1)


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_acquisition_rejects_invalid_control_count(source, count):
    with pytest.raises(ValueError, match="control count"):
        trial.acquire_semantic_training_errors(*source, control_count=count)


def test_acquisition_observes_actual_decoder_and_retains_all_unknown_cases(source):
    model, examples = source
    selected, receipt = trial.acquire_semantic_training_errors(model, examples, control_count=1)
    expected = set(receipt["control_sources"]) | {
        row["source_text_sha256"] for row in receipt["observations"] if row["semantic_status"] != "equivalent"}
    assert {item.ir.source_text_sha256 for item in selected} == expected
    assert all("source_grounding_aligned" in row for row in receipt["observations"])


def test_cli_cannot_enable_acquisition_without_retention(monkeypatch, tmp_path):
    from tools.refit_semantic_argument_proposals import main

    monkeypatch.setattr("sys.argv", ["refit", "--transducer", "unused", "--source-report", "unused",
        "--bundle", "unused=unused", "--output", str(tmp_path / "candidate.json"),
        "--acquire-training-errors", "2"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_acquisition_rejects_source_changes_during_observation(source, monkeypatch):
    import core.learning.semantic_validation_checkpoint as checkpoint

    identities = iter(("before", "after"))
    monkeypatch.setattr(checkpoint, "validation_implementation_identity", lambda: next(identities))
    monkeypatch.setattr(trial, "_observe", lambda model, item:
                        {"source_text_sha256": item.ir.source_text_sha256, "semantic_status": "equivalent"})
    with pytest.raises(ValueError, match="implementation changed"):
        trial.acquire_semantic_training_errors(*source, control_count=1)


def test_refit_cli_retains_full_source_pool_and_embeds_acquisition_receipt(source, monkeypatch, tmp_path):
    import core.learning.semantic_joint_graph_learning as learning
    import core.learning.semantic_program_compositional_campaign as campaign
    import tools.refit_semantic_argument_proposals as tool

    model, examples = source
    original = tmp_path / "parent.json"
    report = tmp_path / "source.json"
    output = tmp_path / "candidate.json"
    original.write_text(json.dumps(model.to_dict()))
    report.write_text("{}")
    monkeypatch.setattr(tool, "load_source_examples", lambda *args: examples)
    monkeypatch.setattr(trial, "_observe", lambda model, item:
                        {"source_text_sha256": item.ir.source_text_sha256, "semantic_status": "equivalent"})
    seen = []

    def refit(parent, mining, **options):
        training = tuple(item for item in mining if item.split == "train")
        seen.extend(training)
        assert len(training) == 2
        assert all(item.split != "test" for item in mining)
        retained = options["source_retention_examples"]
        assert retained == tuple(item for item in examples if item.split == "train")
        assert len(retained) > len(training)
        body = {key: value for key, value in parent.training_receipt.items() if key != "receipt_sha256"}
        body["joint_graph_refit"] = {
            "training_example_ids_sha256": _sha(sorted(item.ir.source_text_sha256 for item in training))}
        return replace(parent, training_receipt={**body, "receipt_sha256": _sha(body)})

    monkeypatch.setattr(learning, "refit_compositional_joint_graphs", refit)
    def select(candidates, cohort, **options):
        assert options["scoring"] == "source_anchors_v2"
        assert cohort is examples
        assert candidates["incumbent"].receipt_sha256 == model.receipt_sha256
        assert candidates["refit"].training_receipt["training_error_acquisition"]
        return {"fixture": True, "serving_authority": False}

    monkeypatch.setattr(campaign, "select_compositional_program_candidate", select)
    monkeypatch.setattr("sys.argv", ["refit", "--transducer", str(original), "--source-report", str(report),
        "--bundle", "fixture=fixture", "--output", str(output), "--objective", "joint_graphs",
        "--retain-semantic-constraints", "--acquire-training-errors", "2",
        "--validation-output", str(tmp_path / "validation.json")])
    assert tool.main() == 0
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    exported = compositional_semantic_program_transducer_from_dict(json.loads(output.read_text()))
    receipt = exported.training_receipt["training_error_acquisition"]
    assert receipt["parent_transducer_receipt_sha256"] == model.receipt_sha256
    assert receipt["training_sources"] == [item.ir.source_text_sha256 for item in seen]
    assert receipt["observation_identity"]
    assert not receipt["serving_authority"]
