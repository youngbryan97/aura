"""`think(response_format=...)` reaches the decoder, where it used to reach nothing.

The planner passed `response_format=PlanSchema` and commented that the
content was "guaranteed to adhere" to it. Nothing read the kwarg. The
guarantee was a regex for the first brace. A format is a shape the decoder
can hold, and the request now lands on the turn's context, is forwarded by
the response phase, and is carried by the gate to the client's job.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_engine_turns_a_response_format_into_an_output_shape() -> None:
    from core.brain import cognitive_engine as ce

    source = inspect.getsource(ce.CognitiveEngine.think)
    assert 'kwargs.pop("response_format", None)' in source
    assert '"output_shape"' in source


def test_the_gate_forwards_the_shape_to_the_client() -> None:
    body = (ROOT / "core/brain/inference_gate.py").read_text(encoding="utf-8")
    assert '"output_shape",' in body


def test_the_client_puts_the_shape_on_the_job() -> None:
    body = (ROOT / "core/brain/llm/mlx_client.py").read_text(encoding="utf-8")
    assert '"output_shape": str(kwargs.get("output_shape") or "").strip().lower()' in body


def test_the_worker_holds_the_shape_the_job_names() -> None:
    body = (ROOT / "core/brain/llm/mlx_worker.py").read_text(encoding="utf-8")
    assert 'job.get("output_shape")' in body
    assert "enforce_json(" in body


def test_the_planner_no_longer_begs_for_json() -> None:
    body = (ROOT / "core/planning/planner.py").read_text(encoding="utf-8")
    assert "Return ONLY the JSON object" not in body
    assert "response_format=PlanSchema" in body
