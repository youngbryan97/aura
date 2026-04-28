"""Benchmark adapter interface.

A concrete adapter wraps an external benchmark (SWE-bench, WebArena,
GAIA, ...) and answers two questions:

    tasks()                 -> iterable of BenchTask descriptors
    run(task, profile, llm) -> TaskOutcome

The adapter is responsible for translating an ``AblationProfile`` into
whatever subsystem-toggling magic the underlying benchmark needs.  It
is *not* the runner's job to know which Aura services to disable for
each profile.

Adapters are pure data flows: no global state, no singletons, no
implicit container access — so the harness can run them concurrently
without contention.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Protocol


@dataclass
class BenchTask:
    """A single task in a benchmark suite."""

    task_id: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOutcome:
    """The result of running a task under one ablation profile.

    ``score`` is in [0.0, 1.0] where 1.0 is perfect.  ``metadata``
    carries adapter-specific detail (e.g. which evaluator was used,
    what the model wrote).
    """

    task_id: str
    profile_name: str
    score: float
    runtime_seconds: float
    raw_response: str = ""
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.time)


class LLMCallable(Protocol):
    """A minimal LLM contract: prompt -> response."""

    def __call__(self, prompt: str, profile_name: str) -> str: ...


class BenchAdapter(Protocol):
    """The contract every external-benchmark adapter implements."""

    name: str

    def tasks(self) -> Iterable[BenchTask]: ...

    def run(
        self,
        task: BenchTask,
        profile_name: str,
        llm: LLMCallable,
    ) -> TaskOutcome: ...
