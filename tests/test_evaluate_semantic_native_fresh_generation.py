"""Native generation grades parsed public programs, not plausible answer text."""

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_relative_program import relative_native_program_surface
from tools.evaluate_semantic_native_fresh_generation import grade_generation


def test_closed_native_channel_requires_exact_typed_program():
    target = Program(2, (Instruction("sub", (0, 1)),))
    source = "Subtract the second input from the first."
    exact = grade_generation(source, '</think>\n{"inputs":2,"steps":[["sub",[0,1]]]}',
                             prompt_thinking=True, target_sha256=target.sha(),
                             inputs=(7, 2), target_value=5)
    assert exact["parse_status"] == "valid_program"
    assert exact["program_exact"] is True
    assert exact["answer_correct"] is True
    assert exact["boundary_closed"] is True

    inverted = grade_generation(source, '</think>\n{"inputs":2,"steps":[["sub",[1,0]]]}',
                                prompt_thinking=True, target_sha256=target.sha(),
                                inputs=(7, 2), target_value=5)
    assert inverted["program_exact"] is False
    assert inverted["answer_correct"] is False


def test_unclosed_reasoning_and_numeric_answers_have_no_program_authority():
    target = Program(2, (Instruction("sub", (0, 1)),))
    pending = grade_generation("question", '{"inputs":2,"steps":[["sub",[0,1]]]}',
                               prompt_thinking=True, target_sha256=target.sha(),
                               inputs=(7, 2), target_value=5)
    assert pending["parse_status"] == "thinking_incomplete"
    assert pending["program_exact"] is False
    numeric = grade_generation("question", "</think>\n5", prompt_thinking=True,
                               target_sha256=target.sha(), inputs=(7, 2), target_value=5)
    assert numeric["parse_status"] == "program_invalid"
    assert numeric["answer_correct"] is False


def test_public_relative_program_requires_its_declared_codec():
    target = Program(3, (Instruction("add", (0, 1)), Instruction("mul", (3, 2))))
    text, _ = relative_native_program_surface(target)
    args = dict(prompt_thinking=True, target_sha256=target.sha(), inputs=(4, 2, 3), target_value=18)
    wrong_codec = grade_generation("question", "</think>\n" + text, **args)
    assert wrong_codec["parse_status"] == "program_invalid"
    valid = grade_generation("question", "</think>\n" + text,
                             register_encoding="role_relative_v1", **args)
    assert valid["program_exact"] and valid["answer_correct"]
