from dataclasses import replace

from core.learning.semantic_source_order import source_order_training_example
from tests.test_compositional_source_training import _examples


def test_source_order_rebinds_values_references_and_definitions():
    example = _examples()[0]
    spans = example.ir.input_spans
    assert len(spans) >= 2
    reversed_example = replace(example,
        ir=replace(example.ir, input_spans=tuple(reversed(spans)),
                   instructions=tuple(replace(ins, args=tuple(
                       len(spans) - 1 - arg if arg < len(spans) else arg
                       for arg in ins.args)) for ins in example.ir.instructions)),
        public_inputs=tuple(reversed(example.public_inputs)),
        register_definition_spans=(tuple(reversed(example.register_definition_spans[:len(spans)]))
                                   + example.register_definition_spans[len(spans):]))
    restored = source_order_training_example(reversed_example)
    assert restored.ir.input_spans == example.ir.input_spans
    assert restored.ir.to_program() == example.ir.to_program()
    assert (reversed_example.ir.to_program().run(reversed_example.public_inputs)
            == restored.ir.to_program().run(restored.public_inputs))
    assert restored.public_inputs == example.public_inputs
    assert restored.register_definition_spans == example.register_definition_spans
    assert restored.hidden_states is example.hidden_states
    assert source_order_training_example(restored) is restored
