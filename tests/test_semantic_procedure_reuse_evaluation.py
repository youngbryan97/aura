"""Reuse evidence separates interpretation, lowering, and finite execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.learning.semantic_procedure_reuse_evaluation import evaluate_learned_procedure_reuse
from tests.test_semantic_program_shared_transducer import _shared_example


def example():
    return _shared_example(three_steps=False, variant=0, split="validation")


class Decoder:
    receipt_sha256 = "d" * 64

    def __init__(self, ir):
        self.ir, self.calls = ir, []

    def decode(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(ir=self.ir, refusal=None if self.ir else "unavailable")


def test_procedure_reuse_changes_values_without_passing_targets_to_consumers():
    item = example()
    decoder = Decoder(item.ir)
    receipt = evaluate_learned_procedure_reuse(decoder, (item,), split="validation")
    assert receipt["proved_equivalent"] == receipt["accepted"] == 1
    assert receipt["fresh_probes"] > 0
    assert receipt["lowering_failures"] == receipt["task_failures"] == 0
    assert set(decoder.calls[0]) == {"source_token_ids", "hidden_states", "public_inputs",
                                   "source_text_sha256", "model_basis_sha256"}
    row = receipt["rows"][0]
    assert row["procedure_receipt"]["source_tokens_retained"] is False
    assert all(tuple(probe["inputs"]) != item.public_inputs for probe in row["probes"])
    assert not receipt["serving_authority"]


def test_wrong_interpretation_can_lower_faithfully_without_passing_task_check():
    item = example()
    wrong = replace(item.ir, instructions=(replace(item.ir.instructions[0], op="sub"),
                                          *item.ir.instructions[1:]))
    receipt = evaluate_learned_procedure_reuse(Decoder(wrong), (item,), split="validation")
    assert receipt["lowering_failures"] == 0
    assert receipt["task_failures"] > 0
    assert receipt["proved_equivalent"] == 0
    assert receipt["rows"][0]["equivalence"]["status"] == "different"


def test_refused_parse_remains_in_the_denominator():
    receipt = evaluate_learned_procedure_reuse(Decoder(None), (example(),), split="validation")
    assert receipt["total"] == 1 and receipt["accepted"] == 0
    assert receipt["fresh_probes"] == 0 and receipt["proved_equivalent"] == 0


def test_actual_learned_decoder_enters_existing_registry_and_executes_new_values():
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    item = next(item for item in examples if item.split == "train")
    receipt = evaluate_learned_procedure_reuse(model, (item,), split="train", probe_count=8)
    assert receipt["accepted"] == receipt["proved_equivalent"] == 1
    assert receipt["fresh_probes"] > 0
    assert receipt["lowering_failures"] == receipt["task_failures"] == 0


@pytest.mark.parametrize("items,count", [((), 8), ((example(), example()), 8), ((example(),), 0)])
def test_invalid_cohort_or_probe_allowance_is_not_empty_success(items, count):
    with pytest.raises(ValueError):
        evaluate_learned_procedure_reuse(Decoder(None), items, split="validation", probe_count=count)
