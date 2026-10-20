"""The rule that picks where to change who she is, fixed before any run: it keeps what she knows first."""
from __future__ import annotations

import pytest

from tools.where_she_holds_who_she_is import chosen

pytestmark = pytest.mark.unit

UNCHANGED = {"window": None, "push": None, "development": [3.0, 2.0], "knowledge": [4.0, 4.0]}


def test_the_lowest_development_leaning_that_keeps_what_she_knows():
    results = [UNCHANGED,
               {"window": [0, 64], "push": 2.0, "development": [-9.0, -9.0], "knowledge": [-1.0, 4.0]},  # forgets
               {"window": [16, 48], "push": None, "development": [-2.0, -1.0], "knowledge": [3.5, 3.5]},
               {"window": [16, 24], "push": 1.0, "development": [-2.0, -1.0], "knowledge": [3.2, 3.2]},  # fewer layers
               {"window": [24, 32], "push": 0.5, "development": [-4.0, -1.0], "knowledge": [2.0, 2.0]}]  # below 3/4
    assert chosen(results)["window"] == [16, 24]


def test_none_when_no_change_keeps_what_she_knows():
    assert chosen([UNCHANGED, {"window": [0, 8], "push": None, "development": [-5.0], "knowledge": [0.0, 4.0]}]) is None


def test_the_chosen_push_becomes_part_of_the_model(tmp_path, monkeypatch):
    """Each layer of the chosen window is followed by the move the test measured, at every position; the rest are as
    they were, and the model is called as before (a hybrid model asks each layer whether it is linear)."""
    import json

    import mlx.core as mx
    import mlx.nn as nn

    from tools import where_she_holds_who_she_is as tool

    class Layer(nn.Module):
        is_linear = True

        def __call__(self, h, mask=None, cache=None):
            return h + 1.0

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer() for _ in range(4)]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    unit = [mx.array([1.0, 0.0]) if i % 2 else mx.array([0.0, 1.0]) for i in range(4)]
    monkeypatch.setattr(tool, "direction", lambda inner, tokenizer: (unit, [1.0, 2.0, 3.0, 4.0], [0.0] * 4))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"chosen": {"window": [1, 3], "push": 0.5}}))
    model = Model()
    assert tool.pushed(model, None, report) == {"window": [1, 3], "push": 0.5, "chosen_in": str(report)}
    h = mx.zeros((1, 3, 2))
    for layer in model.model.layers:
        assert layer.is_linear
        h = layer(h, mask=None, cache=None)
    # +1 per layer, less 0.5 x 2 along [1, 0] after layer 1 and 0.5 x 3 along [0, 1] after layer 2.
    assert mx.allclose(h[0, -1], mx.array([3.0, 2.5])) and mx.allclose(h[0, 0], h[0, -1])
