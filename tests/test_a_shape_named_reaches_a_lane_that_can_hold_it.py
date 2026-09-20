"""A turn that names an output shape is not answered by a lane without one.

LIVE, 2026-09-19 (U07). The autonomous planner asks for a JSON array and gets
prose, then falls back to a deterministic plan. The shape is held by the
decoder in the MLX worker, and the router's substrate-primary path never
reaches it: that readout head is an untrained random projection onto a
32-word proto vocabulary, so a JSON array is not something it produces badly,
it is something it cannot produce at all. Its own comment says the path stays
on for background work — which is where nearly every shaped call is made.

Two holds, and the second is why the first went unnoticed for a month:

1. A declared ``output_shape`` disqualifies the substrate-primary path.
2. When the answer carries no steps, the discarded plan is recorded. Silent,
   the fallback reads as "the model had no plan".
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.brain.llm import llm_router as router_module
from core.planning.task_decomposer import TaskDecomposer


class _Substrate:
    """Stands in for the live substrate; reaching it at all is the failure."""


def _router() -> Any:
    return router_module.IntelligentLLMRouter.__new__(
        router_module.IntelligentLLMRouter
    )


def test_a_named_shape_declines_the_substrate_before_any_readout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = []

    def _should_not_run(*args: Any, **kwargs: Any) -> Any:
        reached.append(kwargs)
        raise AssertionError("substrate readout ran for a turn that named a shape")

    monkeypatch.setattr(
        router_module.IntelligentLLMRouter,
        "_substrate_primary_enabled",
        lambda self: True,
        raising=False,
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda *a, **k: _Substrate()),
        raising=False,
    )
    monkeypatch.setattr(
        "core.brain.llm.substrate_token_generator.get_substrate_token_generator",
        _should_not_run,
        raising=False,
    )

    result = asyncio.run(
        router_module.IntelligentLLMRouter._try_substrate_primary(
            _router(),
            "decompose this objective",
            {"output_shape": "json_array", "origin": "task_decomposer"},
            is_background=True,
        )
    )
    assert result is None
    assert not reached


def test_the_decline_is_written_where_the_path_begins() -> None:
    """The gate sits before the readout, not after it."""
    source = inspect.getsource(
        router_module.IntelligentLLMRouter._try_substrate_primary
    )
    gate = source.index('kwargs.get("output_shape")')
    readout = source.index("get_substrate_token_generator")
    assert gate < readout


def test_an_unshaped_background_turn_still_reaches_the_substrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The null. A gate that declined everything would pass the test above."""
    asked: list[str] = []

    monkeypatch.setattr(
        router_module.IntelligentLLMRouter,
        "_substrate_primary_enabled",
        lambda self: True,
        raising=False,
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda *a, **k: _Substrate()),
        raising=False,
    )

    def _reached(substrate: Any) -> Any:
        asked.append("yes")
        raise RuntimeError("stop here; reaching the generator is the assertion")

    monkeypatch.setattr(
        "core.brain.llm.substrate_token_generator.get_substrate_token_generator",
        _reached,
        raising=False,
    )

    asyncio.run(
        router_module.IntelligentLLMRouter._try_substrate_primary(
            _router(),
            "an ordinary background thought",
            {"origin": "curiosity"},
            is_background=True,
        )
    )
    assert asked == ["yes"]


def test_a_plan_that_parsed_to_nothing_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "core.planning.task_decomposer.record_degradation",
        lambda subsystem, error, **kwargs: recorded.append(
            (subsystem, kwargs.get("action", ""))
        ),
        raising=False,
    )

    class _Prose:
        async def think(self, *args: Any, **kwargs: Any) -> str:
            return "world action hold grounded choose loop result repair"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Prose() if name == "llm_router" else default),
        raising=False,
    )

    decomposer = TaskDecomposer.__new__(TaskDecomposer)
    steps = asyncio.run(
        TaskDecomposer._llm_decompose(
            decomposer, "group these files by extension", "", "", {}
        )
    )
    assert steps == []
    assert recorded and recorded[0][0] == "task_decomposer.llm"
    assert "heuristic" in recorded[0][1]


def test_a_plan_that_parsed_is_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The null for the record: a parsed plan writes no degradation."""
    recorded: list[str] = []

    monkeypatch.setattr(
        "core.planning.task_decomposer.record_degradation",
        lambda subsystem, error, **kwargs: recorded.append(subsystem),
        raising=False,
    )

    class _Shaped:
        async def think(self, *args: Any, **kwargs: Any) -> str:
            return '[{"id": "t1", "action": "open_app", "params": {"name": "Notes"}}]'

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: _Shaped() if name == "llm_router" else default),
        raising=False,
    )

    decomposer = TaskDecomposer.__new__(TaskDecomposer)
    steps = asyncio.run(
        TaskDecomposer._llm_decompose(decomposer, "open Notes", "", "", {})
    )
    assert [step["action"] for step in steps] == ["open_app"]
    assert recorded == []
