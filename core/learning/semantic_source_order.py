"""Align supervised semantic inputs with the live source-order literal contract."""

from __future__ import annotations

from dataclasses import replace

from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample


def source_order_training_example(
    example: SemanticTransducerTrainingExample,
) -> SemanticTransducerTrainingExample:
    """Permute input identities and every dependent reference, never the source."""

    spans = example.ir.input_spans
    order = tuple(sorted(range(len(spans)), key=lambda index: (spans[index].start, spans[index].end)))
    if order == tuple(range(len(spans))):
        return example
    old_to_new = {old: new for new, old in enumerate(order)}
    instructions = tuple(replace(instruction, args=tuple(
        old_to_new[arg] if arg < len(spans) else arg for arg in instruction.args
    )) for instruction in example.ir.instructions)
    ir = replace(example.ir, input_spans=tuple(spans[index] for index in order),
                 instructions=instructions)
    definitions = example.register_definition_spans
    if definitions:
        definitions = (tuple(definitions[index] for index in order)
                       + definitions[len(spans):])
    return replace(example, ir=ir,
                   public_inputs=tuple(example.public_inputs[index] for index in order),
                   register_definition_spans=definitions)
