"""Count, span, source and gradient contracts for whole-request operation learning."""

import itertools
import math
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from core.learning.semantic_native_operation_field import (
    NativeOperationField,
    OperationSetSupervision,
    operation_set_log_partition,
    operation_span_inventory,
)
from core.learning.semantic_program_ir import TokenSpan


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_interval_partition_matches_exhaustive_labelled_sets_and_gradients(steps):
    spans = operation_span_inventory(3, 2)
    values = np.arange(len(spans) * 2).reshape(-1, 2) / 10. - .4
    signatures, scores = [()], [0.]
    for count in range(1, steps + 1):
        for indices in itertools.combinations(range(len(spans)), count):
            ordered = sorted(indices, key=lambda index: spans[index].start)
            if any(spans[a].end > spans[b].start for a, b in zip(ordered, ordered[1:])):
                continue
            for labels in itertools.product(range(2), repeat=count):
                signature = tuple(zip(ordered, labels, strict=True))
                signatures.append(signature)
                scores.append(sum(values[row, column] for row, column in signature))
    weights = np.exp(scores)
    expected = np.zeros_like(values)
    for signature, probability in zip(signatures, weights / weights.sum(), strict=True):
        for row, column in signature:
            expected[row, column] += probability
    objective = lambda energies: operation_set_log_partition(energies, spans, 3, steps)
    energies = mx.array(values, dtype=mx.float32)
    assert float(objective(energies).item()) == pytest.approx(math.log(weights.sum()), abs=1e-5)
    assert np.allclose(np.array(mx.grad(objective)(energies)), expected, atol=1e-5)


def test_missing_and_duplicate_operation_decisions_receive_opposite_gradient():
    spans = (TokenSpan(0, 1), TokenSpan(1, 2))
    # One true operation and one extra placement, not a supplied frame count.
    objective = lambda value: operation_set_log_partition(value, spans, 2, 2) - value[0, 0]
    gradient = mx.grad(objective)(mx.zeros((2, 2)))
    assert gradient[0, 0].item() < 0
    assert gradient[1, 0].item() > 0 and gradient[0, 1].item() > 0


def test_supervision_rejects_unreachable_program_before_fit():
    field = NativeOperationField(4, depths=1, labels=("add", "sub"), max_span_tokens=1, max_steps=1)
    valid = OperationSetSupervision("fit", 3, (TokenSpan(0, 1),), (("sub", TokenSpan(1, 2)),))
    assert valid.indices(field)[1]
    for item in (replace(valid, targets=(("mul", TokenSpan(1, 2)),)),
                 replace(valid, targets=(("sub", TokenSpan(0, 1)),)),
                 replace(valid, targets=(("sub", TokenSpan(1, 3)),)),
                 replace(valid, targets=valid.targets * 2)):
        with pytest.raises(ValueError, match="grammar"):
            item.indices(field)


def test_operation_loss_reaches_native_states_and_learns_span_presence():
    mx.random.seed(37)
    field = NativeOperationField(4, depths=2, labels=("add", "sub"), relation_width=8,
        max_span_tokens=1, max_steps=2)
    states = mx.random.normal((3, 2, 4))
    supervision = OperationSetSupervision("fit", 3, (), (("sub", TokenSpan(1, 2)),))
    gradient = mx.grad(lambda value: field.source_loss(value, supervision))(states)
    assert mx.all(mx.isfinite(gradient)).item() and mx.sum(mx.abs(gradient)).item() > 0
    initial = field.source_loss(states, supervision).item()
    optimizer = optim.Adam(learning_rate=.02)
    for _ in range(48):
        loss, gradients = nn.value_and_grad(field, lambda owner: owner.source_loss(states, supervision))(field)
        optimizer.update(field, gradients)
        mx.eval(field.parameters(), loss)
    assert field.source_loss(states, supervision).item() < initial / 10.
    proposer = field.proposal("fit", states)
    best = next(iter(proposer(source_id="fit", source_token_ids=(1, 2, 3), input_spans=(), max_steps=2)))
    assert [(node.operation, node.span) for node in best] == [("sub", TokenSpan(1, 2))]
    assert not proposer.last_receipt["target_available_to_proposer"]
    assert not proposer.last_receipt["runtime_search_exhaustive"]


def test_public_inventory_does_not_inherit_parent_span_or_operation_winners():
    field = NativeOperationField(4, depths=1, labels=("add", "sub"), max_span_tokens=2, max_steps=2)
    proposer = field.proposal("source", mx.ones((3, 1, 4)), chart_limit=4)
    inventory = tuple(proposer(source_id="source", source_token_ids=(1, 2, 3),
        input_spans=(TokenSpan(0, 1),), max_steps=2))
    assert inventory and len(inventory) == 4
    assert proposer.last_receipt["span_count"] == 3 and proposer.last_receipt["node_count"] == 6
    assert not proposer.last_receipt["inventory_pruned_by_parent"]
    with pytest.raises(ValueError, match="source"):
        proposer(source_id="other", source_token_ids=(1, 2, 3), input_spans=(), max_steps=2)
    with pytest.raises(ValueError, match="grammar"):
        proposer(source_id="source", source_token_ids=(1, 2, 3), input_spans=(), max_steps=3)


def test_null_offset_cancels_and_contract_cannot_claim_transfer():
    mx.random.seed(4)
    field = NativeOperationField(4, depths=1, labels=("sub", "add"), relation_width=4)
    states = mx.random.normal((3, 1, 4))
    spans = operation_span_inventory(3, 1)
    before = field(states, spans)
    field.output.bias = field.output.bias + 123.
    assert mx.allclose(field(states, spans), before, atol=2e-5).item()
    contract = field.to_contract()
    assert NativeOperationField.from_contract(contract).to_contract() == contract
    with pytest.raises(ValueError, match="contract"):
        NativeOperationField.from_contract({**contract, "whole_program_calibration_proven": True})
