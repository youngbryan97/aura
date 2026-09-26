"""Native pilot sampling and continuation loss remain source-bound and causal."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from core.learning.semantic_native_program import NativeProgramSequence
from tools.train_semantic_native_program import construction_subset, native_loss


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
