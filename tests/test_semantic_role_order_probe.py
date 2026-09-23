"""Reference identifiability cannot read semantic order from annotation order."""

from types import SimpleNamespace

import numpy as np

from core.learning.semantic_program_ir import TokenSpan
from tools.probe_semantic_role_order import _role_row


def test_reference_features_follow_text_order_not_gold_argument_order():
    hidden = np.asarray(((1., 0.), (0., 1.), (1., 1.), (-1., 1.)), dtype=np.float32)
    hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
    source = (TokenSpan(2, 3), TokenSpan(1, 2))

    def item(args, spans):
        return SimpleNamespace(
            hidden_states=hidden, public_inputs=((2, 3), 2, 4),
            construction_id="source", register_definition_origin="explicit_annotation",
            register_definition_spans=(TokenSpan(0, 1), TokenSpan(0, 1),
                                       TokenSpan(1, 2), TokenSpan(0, 1),
                                       TokenSpan(3, 4)),
            ir=SimpleNamespace(source_text_sha256="a" * 64, instructions=(
                SimpleNamespace(),
                SimpleNamespace(args=args, operation_span=TokenSpan(3, 4),
                                argument_spans=spans))))

    original = _role_row(item((3, 2), source))
    reversed_annotation = _role_row(item((2, 3), source[::-1]))
    assert original["label"] != reversed_annotation["label"]
    assert original["reference_label"] == reversed_annotation["reference_label"] == 0
    np.testing.assert_array_equal(original["reference"], reversed_annotation["reference"])
