"""`think(response_format=...)` reaches the decoder, where it used to reach nothing.

The planner passed `response_format=PlanSchema` and commented that the
content was "guaranteed to adhere" to it. Nothing read the kwarg. The
guarantee was a regex for the first brace. A format is a shape the decoder
can hold, and the request now lands on the turn's context, is forwarded by
the response phase, and is carried by the gate to the client's job.

Every hop was asserted by grepping the file it lived in. That is a weaker
claim than it looks: a line in a dead branch reads the same as a live one,
and the check breaks when the code MOVES rather than when it stops working.
It did — the gate's forwarding list was extracted to
`inference_gate_turn_setup.py` and the grep went red while the shape still
arrived. Each hop is now run.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_engine_turns_a_response_format_into_an_output_shape() -> None:
    """The mapping is run, not read."""
    from core.brain.cognitive_engine import shape_for_response_format

    class _Schema:
        pass

    assert shape_for_response_format(list) == "json_array"
    assert shape_for_response_format("json_array") == "json_array"
    assert shape_for_response_format(_Schema) == "json_object"
    assert shape_for_response_format(dict) == "json_object"


def test_the_engine_puts_that_shape_on_the_turn() -> None:
    """And ``think`` is what calls it, with the caller's kwarg."""
    from core.brain import cognitive_engine as ce

    source = inspect.getsource(ce.CognitiveEngine.think)
    assert 'kwargs.pop("response_format", None)' in source
    assert "shape_for_response_format(requested_format)" in source
    assert '"output_shape"' in source


def test_the_response_phase_forwards_the_shape_it_was_given() -> None:
    from core.phases import response_generation_unitary as rgu

    body = (ROOT / "core/phases/response_generation_unitary.py").read_text(
        encoding="utf-8"
    )
    assert 'llm_kwargs["output_shape"] = requested_shape' in body
    # And the name it reads is the name the engine writes.
    assert inspect.getmodule(rgu) is not None


def test_the_gate_carries_the_shape_into_the_generation_kwargs() -> None:
    """The gate forwards by name, wherever that list happens to live."""
    from core.brain import inference_gate_turn_setup as setup

    source = inspect.getsource(setup)
    assert '"output_shape",' in source
    forwarded = [
        name
        for name in ("output_shape", "cognitive_mode", "stop_sequences")
        if f'"{name}",' in source
    ]
    assert forwarded == ["output_shape", "cognitive_mode", "stop_sequences"]


@pytest.mark.parametrize(
    ("given", "expected"),
    [("json_array", "json_array"), ("JSON_Object", "json_object"), (None, "")],
)
def test_the_client_puts_the_shape_on_the_job(given: str | None, expected: str) -> None:
    """Run the client's own normalisation, and see the job carry it."""
    from core.brain.llm.mlx_client import _the_shape_named

    body = (ROOT / "core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert '"output_shape": _the_shape_named(kwargs),' in body
    assert _the_shape_named({"output_shape": given}) == expected


def test_the_worker_holds_the_shape_the_job_names() -> None:
    """The three shapes the worker accepts are the three the callers send."""
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker._mlx_worker_loop_shape_answer_held)
    assert 'job.get("output_shape")' in source
    assert "enforce_json(" in source
    for shape in ("json", "json_object", "json_array"):
        assert f'"{shape}"' in source


def test_the_decoder_can_actually_hold_each_shape() -> None:
    """The last hop is the one that matters: a processor comes back."""
    pytest.importorskip("mlx.core")
    from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

    class _Tokenizer:
        vocab_size = 4

        def get_vocab(self) -> dict[str, int]:
            return {"{": 0, "}": 1, "[": 2, "]": 3}

        def decode(self, ids: list[int]) -> str:
            return {0: "{", 1: "}", 2: "[", 3: "]"}[int(ids[0])]

    tokenizer = _Tokenizer()
    for require in ("any", "object", "array"):
        assert enforce_json(tokenizer, require=require) is not None
    with pytest.raises(ValueError, match="require must be"):
        enforce_json(tokenizer, require="prose")


def test_the_planner_no_longer_begs_for_json() -> None:
    body = (ROOT / "core/planning/planner.py").read_text(encoding="utf-8")
    assert "Return ONLY the JSON object" not in body
    assert "response_format=PlanSchema" in body
