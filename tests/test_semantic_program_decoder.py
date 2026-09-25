"""Direct program learning shares the floor and never consumes answer labels at decode."""

from types import SimpleNamespace

import pytest
import torch

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_decoder import ProgramDecoderConfig, SemanticProgramDecoder
from core.learning.semantic_program_floor import semantic_primitive_type_signature
from core.learning.semantic_program_ir import TokenSpan


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    with torch.random.fork_rng():
        yield


def model_and_inputs():
    torch.manual_seed(73)
    model = SemanticProgramDecoder(ProgramDecoderConfig(8, width=16, max_steps=3))
    features = torch.randn(6, 8)
    spans = (TokenSpan(0, 1), TokenSpan(4, 5))
    return model, features, spans, ("integer", "integer")


def test_complete_program_loss_reaches_context_and_register_mechanisms():
    model, features, spans, kinds = model_and_inputs()
    target = Program(2, (Instruction("sub", (0, 1)), Instruction("mul", (2, 0))))
    loss = model.loss(features, spans, kinds, target)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    for name in (
        "project.weight",
        "attention_query.weight",
        "register_query.weight",
        "result.weight",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize(
    "kinds",
    [
        ("integer", "integer"),
        ("integer_sequence", "integer"),
        ("integer_sequence", "integer_sequence"),
    ],
)
def test_free_decode_only_emits_floor_typed_acyclic_programs(kinds):
    model, features, spans, _ = model_and_inputs()
    program, receipt = model.decode(features, spans, kinds)
    assert 1 <= len(program.instructions) <= model.config.max_steps
    registers = list(kinds)
    for instruction in program.instructions:
        arguments, result = semantic_primitive_type_signature(instruction.op)
        assert tuple(registers[i] for i in instruction.args) == arguments
        registers.append(result)
    assert receipt["search"] == "greedy"


def test_typed_beam_retains_scored_complete_programs_without_target():
    model, features, spans, kinds = model_and_inputs()
    greedy, _ = model.decode(features, spans, kinds)
    single = model.propose_beam(features, spans, kinds, beam_width=1, max_programs=1)
    assert single[0][1]["log_probability"] >= float(model.score(
        features, spans, kinds, greedy).detach())
    proposals = model.propose_beam(features, spans, kinds, beam_width=8, max_programs=8)
    assert 1 < len(proposals) <= 8
    assert len({program for program, _receipt in proposals}) == len(proposals)
    assert [receipt["log_probability"] for _program, receipt in proposals] == sorted(
        (receipt["log_probability"] for _program, receipt in proposals), reverse=True)
    for program, receipt in proposals:
        assert 1 <= program.depth <= model.config.max_steps
        assert receipt["search"] == "typed_beam"
        assert receipt["log_probability"] == pytest.approx(
            float(model.score(features, spans, kinds, program).detach()), abs=1e-5)
        types = list(kinds)
        for instruction in program.instructions:
            arguments, result = semantic_primitive_type_signature(instruction.op)
            assert tuple(types[index] for index in instruction.args) == arguments
            types.append(result)


def test_typed_beam_rejects_unbounded_search():
    model, features, spans, kinds = model_and_inputs()
    for width, count in ((0, 1), (33, 1), (4, 0), (4, 5), (True, 1)):
        with pytest.raises(ValueError, match="bounded positive widths"):
            model.propose_beam(features, spans, kinds, beam_width=width,
                               max_programs=count)


def test_source_fold_beam_counts_reach_after_target_blind_generation():
    from tools.compare_semantic_candidate_methods import _direct_beam_observation

    model, features, spans, kinds = model_and_inputs()
    target = Program(2, (Instruction("sub", (0, 1)),))
    alternative = Program(2, (Instruction("mul", (0, 1)),))

    class FixedProposer:
        def propose_beam(self, received_features, received_spans, received_kinds,
                         *, beam_width, max_programs):
            assert received_features is features
            assert received_spans is spans
            assert received_kinds is kinds
            assert beam_width == max_programs == 2
            return ((alternative, {"log_probability": -1.0}),
                    (target, {"log_probability": -2.0}))

    item = SimpleNamespace(
        ir=SimpleNamespace(to_program=lambda: target, input_spans=spans),
        public_inputs=(11, 3))
    result = _direct_beam_observation(
        FixedProposer(), features, spans, kinds, item,
        frozenset({alternative.sha()}), width=2)
    assert result["exact_reachable"] is True
    assert result["novel_exact_reachable"] is True
    assert result["top_exact"] is False
    assert result["new_programs"] == 1
    assert result["proposals"][1]["exact"] is True


@pytest.mark.parametrize(
    "instruction",
    [
        Instruction("sub", (0, 2)),
        Instruction("total", (0,)),
        Instruction("sub", (True, 1)),
        Instruction("sub", (-1, 0)),
        Instruction("sub", (0,)),
        Instruction("made_up", (0,)),
    ],
)
def test_teacher_cannot_smuggle_invalid_programs_through_typed_loss(instruction):
    model, features, spans, kinds = model_and_inputs()
    with pytest.raises(ValueError):
        model.loss(features, spans, kinds, Program(2, (instruction,)))


def test_serialized_weights_preserve_eval_and_training_scores():
    model, features, spans, kinds = model_and_inputs()
    target = Program(2, (Instruction("sub", (1, 0)),))
    before = model.score(features, spans, kinds, target).detach()
    restored = SemanticProgramDecoder(model.config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored.score(features, spans, kinds, target), before)
    assert restored.decode(features, spans, kinds) == model.decode(features, spans, kinds)


def test_small_complete_program_can_be_learned_without_operation_span_targets():
    model, features, spans, kinds = model_and_inputs()
    target = Program(2, (Instruction("sub", (1, 0)), Instruction("mul", (2, 1))))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(100):
            optimizer.zero_grad()
            model.loss(features, spans, kinds, target).backward()
            optimizer.step()
        assert model.decode(features, spans, kinds)[0] == target
    finally:
        torch.set_num_threads(previous_threads)


def test_input_register_permutation_follows_grounded_anchors():
    from core.learning.semantic_graph_coordinates import reanchor_program_inputs

    model, features, spans, kinds = model_and_inputs()
    normal, _ = model.decode(features, spans, kinds)
    permuted, _ = model.decode(features, spans[::-1], kinds)
    normalized = reanchor_program_inputs(
        permuted, from_spans=spans[::-1], to_spans=spans, from_inputs=(3, 3), to_inputs=(3, 3)
    )
    assert normalized == normal


def test_training_distinguishes_requests_instead_of_learning_one_constant_program():
    model, features, spans, kinds = model_and_inputs()
    other = -features
    targets = (Program(2, (Instruction("sub", (0, 1)),)), Program(2, (Instruction("mul", (1, 0)),)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(100):
            optimizer.zero_grad()
            loss = sum(
                model.loss(value, spans, kinds, target)
                for value, target in zip((features, other), targets, strict=True)
            )
            loss.backward()
            optimizer.step()
        assert tuple(model.decode(value, spans, kinds)[0] for value in (features, other)) == targets
    finally:
        torch.set_num_threads(previous_threads)


def test_variance_preserving_projection_keeps_the_unit_feature_contract():
    model = SemanticProgramDecoder(
        ProgramDecoderConfig(8, width=16, feature_scaling="unit_variance")
    )
    features = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    spans, kinds = (TokenSpan(0, 1),), ("integer",)
    assert model.decode(features, spans, kinds)[0].n_inputs == 1
    with pytest.raises(ValueError, match="unit resident"):
        model.decode(features * 2, spans, kinds)


@pytest.mark.parametrize("dimension", [32, 512, 15360])
def test_unit_vector_fan_in_scaling_has_width_independent_variance(dimension):
    torch.manual_seed(73)
    features = torch.nn.functional.normalize(torch.randn(256, dimension), dim=-1)
    projection = torch.nn.Linear(dimension, 64, bias=False)
    scaled = projection(features * dimension**0.5)
    assert 0.28 < float(scaled.detach().var()) < 0.38
