"""Native pilot sampling and continuation loss remain source-bound and causal."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from core.learning.semantic_native_program import NativeProgramSequence
from tools.train_semantic_native_program import (
    construction_subset,
    exact_length_batches,
    native_loss,
)


def test_source_identity_sampling_keeps_all_constructions_without_labels():
    examples = [SimpleNamespace(ir=SimpleNamespace(source_text_sha256=f"{group}-{index}"),
                                construction_id=group)
                for group in ("a", "b") for index in range(3)]
    identities = [item.ir.source_text_sha256 for item in reversed(examples)]
    assert construction_subset(examples, identities, per_construction=1) == ("a-0", "b-0")
    assert construction_subset(examples[::-1], identities[::-1], per_construction=2) == (
        "a-0", "a-1", "b-0", "b-1")
    for invalid in (["missing"], ["a-0", "a-0"]):
        with pytest.raises(ValueError, match="identities"):
            construction_subset(examples, invalid, per_construction=1)


def test_prefix_batches_preserve_every_complete_sequence_and_ignore_input_order():
    sequences = {"c": NativeProgramSequence((1, 2, 3), 1),
                 "a": NativeProgramSequence((3, 4, 5), 1),
                 "b": NativeProgramSequence((2, 4, 5, 6), 1)}
    batches = list(exact_length_batches(sequences, batch_size=2))
    assert batches == [("a", "c"), ("b",)]
    assert batches == list(exact_length_batches(dict(reversed(list(sequences.items()))), batch_size=2))
    assert {identity for batch in batches for identity in batch} == set(sequences)
    assert all(len({len(sequences[identity].tokens) for identity in batch}) == 1 for batch in batches)
    for size in (0, True, 33):
        with pytest.raises(ValueError, match="batch size"):
            list(exact_length_batches(sequences, batch_size=size))


class Suffix(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(4, 8)

    def __call__(self, hidden):
        return self.output(hidden)


def test_native_loss_masks_every_prompt_target_but_no_continuation_target():
    mx.random.seed(3)
    suffix = Suffix()
    hidden = mx.random.normal((1, 4, 4))
    sequence = NativeProgramSequence((1, 2, 3, 4, 5), 3)
    expected = nn.losses.cross_entropy(suffix(hidden)[:, 2:].astype(mx.float32),
                                      mx.array([[4, 5]]))
    assert mx.allclose(native_loss(suffix, hidden, sequence), mx.mean(expected)).item()
    assert mx.allclose(native_loss(suffix, hidden, sequence, summed=True), mx.sum(expected)).item()
    changed = NativeProgramSequence((7, 6, 1, 4, 5), 3)
    assert mx.array_equal(native_loss(suffix, hidden, sequence),
                          native_loss(suffix, hidden, changed)).item()
    _value, gradient = nn.value_and_grad(suffix, native_loss)(suffix, hidden, sequence)
    assert mx.sum(mx.abs(gradient["output"]["weight"])).item() > 0


@pytest.mark.parametrize("start", [0, 5, 6])
def test_supervision_boundary_is_never_inferred_from_a_malformed_row(start):
    with pytest.raises(ValueError, match="boundary"):
        native_loss(Suffix(), mx.ones((1, 4, 4)), NativeProgramSequence((1, 2, 3, 4, 5), start))


def test_semantic_loss_masks_format_and_private_targets_by_explicit_graph_positions():
    suffix = Suffix()
    hidden = mx.ones((1, 5, 4))
    sequence = NativeProgramSequence((1, 2, 3, 4, 5, 6), 2, (3, 5))
    logits = suffix(hidden)[:, 2::2].astype(mx.float32)
    expected = nn.losses.cross_entropy(logits, mx.array([[4, 6]]))
    assert mx.allclose(native_loss(suffix, hidden, sequence, scope="semantic_decisions"),
                       mx.mean(expected)).item()
    assert mx.allclose(native_loss(suffix, hidden, sequence, scope="semantic_decisions", summed=True),
                       mx.sum(expected)).item()
    changed = NativeProgramSequence((7, 7, 7, 4, 7, 6), 2, (3, 5))
    assert mx.array_equal(native_loss(suffix, hidden, sequence, scope="semantic_decisions"),
                          native_loss(suffix, hidden, changed, scope="semantic_decisions")).item()


@pytest.mark.parametrize("positions", [(), (1,), (6,), (3, 3), (4, 3), (True,)])
def test_invalid_semantic_decision_maps_have_no_loss(positions):
    with pytest.raises(ValueError, match="decision positions"):
        native_loss(Suffix(), mx.ones((1, 5, 4)),
                    NativeProgramSequence((1, 2, 3, 4, 5, 6), 2, positions),
                    scope="semantic_decisions")
