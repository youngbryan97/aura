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
    native_relational_source_loss,
    native_relation_metric_loss,
    native_source_embedding,
    selected_projection_error,
    native_source_loss,
    native_supervision_sets,
    native_training_schedule,
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

    def __call__(self, hidden, *, logit_positions=None):
        selected = hidden if logit_positions is None else hidden[:, mx.array(logit_positions)]
        return self.output(selected)


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


def test_update_schedule_matches_the_original_epoch_shuffle_without_labels():
    import random
    order, expected, rng = list("abcde"), [], random.Random(37)
    for step in range(13):
        if step % len(order) == 0:
            rng.shuffle(order)
        expected.append(order[step % len(order)])
    assert native_training_schedule("abcde", steps=13, seed=37) == tuple(expected)
    assert len(set(native_training_schedule("abcde", steps=2, seed=37))) == 2
    for ids, steps in (([], 3), (["a", "a"], 3), (["a"], True), (["a"], 0)):
        with pytest.raises(ValueError, match="schedule"):
            native_training_schedule(ids, steps=steps, seed=37)


def test_source_competition_uses_the_same_candidate_scores_and_positive_likelihood():
    from core.learning.semantic_native_program import native_choice_loss
    suffix = Suffix()
    hidden = [mx.ones((1, 5, 4)), mx.ones((1, 5, 4))]
    rows = [NativeProgramSequence((1, 2, 3, 4, 5, 6), 2, (3, 5)),
            NativeProgramSequence((1, 2, 3, 5, 5, 4), 2, (3, 5))]
    scores = mx.stack([-native_loss(suffix, state, row, summed=True, scope="semantic_decisions")
                       for state, row in zip(hidden, rows, strict=True)])
    expected = -scores[0] / 2 + native_choice_loss(scores, (0,))
    assert mx.allclose(native_source_loss(suffix, hidden, rows,
        scope="semantic_decisions", objective="contrastive"), expected).item()
    loss, gradient = nn.value_and_grad(suffix, lambda tail: native_source_loss(
        tail, hidden, rows, scope="semantic_decisions", objective="contrastive"))(suffix)
    assert mx.isfinite(loss).item()
    assert mx.sum(mx.abs(gradient['output']['weight'])).item() > 0
    with pytest.raises(ValueError, match="supervision set"):
        native_source_loss(suffix, hidden, rows, scope="continuation", objective="contrastive")


def test_relational_objective_couples_two_source_forms_without_hiding_a_weak_one():
    suffix = Suffix()
    rows = [NativeProgramSequence((1, 2, 3, 4, 5, 6), 2, (3, 5)),
            NativeProgramSequence((1, 2, 3, 5, 5, 4), 2, (3, 5))]
    left = [mx.ones((1, 5, 4)) for _ in rows]
    right = [mx.full((1, 5, 4), 2.) for _ in rows]
    a = native_source_loss(suffix, left, rows, scope="semantic_decisions", objective="contrastive")
    b = native_source_loss(suffix, right, rows, scope="semantic_decisions", objective="contrastive")
    paired = native_relational_source_loss(suffix, left, rows, right, rows)
    assert mx.allclose(paired, mx.logsumexp(mx.stack((a, b))) - mx.log(2.)).item()
    assert paired.item() >= min(a.item(), b.item())
    value, gradient = nn.value_and_grad(suffix, lambda tail: native_relational_source_loss(
        tail, left, rows, right, rows))(suffix)
    assert mx.isfinite(value).item()
    assert mx.sum(mx.abs(gradient['output']['weight'])).item() > 0


def test_relation_metric_reads_pre_answer_state_and_prefers_cross_form_relation():
    class MetricSuffix(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4, bias=False)

        def normalized_states(self, hidden):
            return self.projection(hidden)

    suffix = MetricSuffix()
    suffix.projection.weight = mx.eye(4)
    sequence = NativeProgramSequence((1, 2, 3, 4), 2, (2, 3))
    a = mx.array([[[1., 0., 0., 0.], [9., 0., 0., 0.], [9., 0., 0., 0.]]])
    b = mx.array([[[1., 0., 0., 0.], [-9., 0., 0., 0.], [-9., 0., 0., 0.]]])
    c = mx.array([[[0., 1., 0., 0.], [9., 0., 0., 0.], [9., 0., 0., 0.]]])
    assert mx.allclose(native_source_embedding(suffix, a, sequence),
                       native_source_embedding(suffix, b, sequence)).item()
    good = native_relation_metric_loss(suffix, (a, sequence), (b, sequence), (c, sequence))
    reversed_loss = native_relation_metric_loss(suffix, (a, sequence), (c, sequence), (b, sequence))
    assert good.item() < reversed_loss.item()
    _value, gradient = nn.value_and_grad(suffix, lambda model: native_relation_metric_loss(
        model, (a, sequence), (b, sequence), (c, sequence)))(suffix)
    assert mx.all(mx.isfinite(gradient['projection']['weight'])).item()
    assert mx.sum(mx.abs(gradient['projection']['weight'])).item() > 0
    with pytest.raises(ValueError, match="source-only token boundary"):
        native_source_embedding(suffix, a, NativeProgramSequence((1, 2, 3, 4), 1, (2, 3)))


def test_selected_projection_gate_checks_supervised_probability_not_unrelated_logits():
    sequence = NativeProgramSequence((1, 2, 2, 3), 2, (2, 3))
    full = mx.array([[[0., 0., 0., 0.], [0., 0., 2., -10.],
                      [0., 0., -10., 2.]]], dtype=mx.bfloat16)
    selected = mx.array([[[0., 0., 2., -9.9375],
                          [0., 0., -9.9375, 2.]]], dtype=mx.bfloat16)
    error, tolerance = selected_projection_error(full, selected, sequence, (1, 2))
    assert 0 < error < tolerance
    wrong = mx.array([[[0., 0., 0., 2.], [0., 0., 2., 0.]]], dtype=mx.bfloat16)
    error, tolerance = selected_projection_error(full, wrong, sequence, (1, 2))
    assert error > tolerance
    with pytest.raises(ValueError, match="complete causal states"):
        selected_projection_error(full, selected,
                                  NativeProgramSequence((1, 2, 4, 3), 2, (2, 3)), (1, 2))


def test_native_partner_map_uses_typed_relation_not_construction_identity():
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_counterfactual_corpus import (
        cross_construction_relation_partners, cross_construction_relation_triplets,
    )

    shared = Program(2, (Instruction('sub', (0, 1)),))
    reversed_roles = Program(2, (Instruction('sub', (1, 0)),))
    def item(identity, construction, program, lineage):
        return SimpleNamespace(split='train', construction_id=construction,
            contrast_id=lineage, public_inputs=(5, 2), ir=SimpleNamespace(
                source_text_sha256=identity, to_program=lambda: program))
    rows = (item('a', 'form-a', shared, 'a'), item('b', 'form-b', shared, 'b'),
            item('c', 'form-a', reversed_roles, 'c'),
            item('d', 'form-d', shared, 'a'))
    assert cross_construction_relation_partners(rows) == {'a': 'b', 'b': 'a', 'd': 'b'}
    assert cross_construction_relation_triplets(rows) == {'a': ('b', 'c')}


def test_metric_triplets_reject_structurally_distinct_equivalent_rivals():
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_counterfactual_corpus import cross_construction_relation_triplets

    target = Program(2, (Instruction('sub', (0, 1)),))
    same_meaning = Program(2, (Instruction('add', (0, 1)),
                               Instruction('sub', (2, 1)), Instruction('sub', (3, 1))))
    different = Program(2, (Instruction('sub', (1, 0)),))

    def item(identity, construction, program):
        return SimpleNamespace(split='train', example_id=identity,
            construction_id=construction, contrast_id=identity,
            public_inputs=(5, 2), ir=SimpleNamespace(
                source_text_sha256=identity, to_program=lambda: program))

    rows = (item('a', 'form-a', target), item('b', 'form-b', target),
            item('c', 'form-a', same_meaning))
    assert cross_construction_relation_triplets(rows) == {}
    assert cross_construction_relation_triplets((*rows, item('d', 'form-a', different))) == {
        'a': ('b', 'd')}


def test_supervision_reuses_witnessed_floor_contrasts_and_never_reads_a_held_target():
    import hashlib

    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_native_program import parse_native_program
    from tests.test_semantic_native_program import Tokenizer

    source = "Subtract 2 from 5."
    identity = hashlib.sha256(source.encode()).hexdigest()
    target = Program(2, (Instruction("sub", (0, 1)),))
    item = SimpleNamespace(split="train", ir=SimpleNamespace(
        source_text_sha256=identity, to_program=lambda: target), public_inputs=(5, 2))
    class Held:
        @property
        def ir(self):
            raise AssertionError("held target was read")
    sequences, groups = native_supervision_sets(
        {identity: item, "held": Held()}, {identity: source}, Tokenizer(), (identity,),
        contrast_limit=4)
    assert 2 <= len(groups[identity]) <= 4
    assert groups[identity][0] == (identity, target.sha())
    assert set(sequences) == set(groups[identity])
    programs = [parse_native_program(Tokenizer().decode(list(row.tokens[row.continuation_start:])))
                for row in sequences.values()]
    assert target == programs[0]
    assert any(program.run((5, 2)) != target.run((5, 2)) for program in programs[1:])
    item.split = "validation"
    with pytest.raises(ValueError, match="validation or test"):
        native_supervision_sets({identity: item}, {identity: source}, Tokenizer(), (identity,))
