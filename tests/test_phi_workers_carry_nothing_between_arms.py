"""The one subprocess the organism starts carries nothing from one arm to the next.

Hierarchical phi hands its partition search to a spawned process pool, because
the search is pure Python and holds the interpreter lock. A campaign runs that
pool; pytest runs threads. A worker that kept state between calls, or a result
collected after the call that submitted it, would carry what one arm computed
into the arm after the restore, and the fork test would not see it, because the
worker is another process.

Neither happens. The computation is a function of its arguments, the futures
are joined inside the call that submits them, and the object that keeps the
result lives in the parent and is forked with every other service.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import random
from pathlib import Path

import numpy as np
import pytest

from core.consciousness import hierarchical_phi as hp
from core.subject.snapshot import _UNFORKED_SERVICES

pytestmark = pytest.mark.unit

_TIMING = ("time", "ms", "elapsed", "duration", "took")


def _comparable(result):
    fields = dataclasses.asdict(result) if dataclasses.is_dataclass(result) else dict(vars(result))
    return {
        key: repr(value)
        for key, value in sorted(fields.items())
        if not any(word in key.lower() for word in _TIMING)
    }


def test_the_worker_computation_is_a_function_of_its_arguments() -> None:
    rng = np.random.default_rng(3)
    nodes = tuple(range(8))
    length = max(400, 3 * int(getattr(hp, "MIN_HISTORY", 0)))
    # A handful of recurring states, so the partition search has repeated
    # transitions to measure and cannot pass on an unmeasurable result.
    history = [int(value) for value in rng.choice([3, 12, 48, 192, 5, 80], size=length)]
    other = [int(value) for value in rng.choice([7, 9, 130, 64, 33], size=length)]

    def run(states):
        random.seed(0)
        np.random.seed(0)
        return hp.HierarchicalPhi._compute_subsystem(states, "probe", nodes)

    first = run(history)
    assert first is not None, "the probe history was unmeasurable, so this proves nothing"
    run(other)
    second = run(history)
    assert _comparable(first) == _comparable(second)


def test_the_futures_are_joined_inside_the_call_that_submits_them() -> None:
    source = Path(hp.__file__).read_text(encoding="utf-8")
    submitting = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node) or ""
            if "_executor.submit(" in segment:
                submitting.append((node.name, ".result(" in segment))
    assert submitting, "nothing submits phi work any more, so this test is out of date"
    assert all(joined for _name, joined in submitting), submitting


def test_the_worker_is_a_classmethod_and_not_bound_to_instance_state() -> None:
    assert isinstance(inspect.getattr_static(hp.HierarchicalPhi, "_compute_subsystem"), classmethod)


def test_the_object_that_keeps_the_result_is_forked_with_the_arm() -> None:
    assert "hierarchical_phi" not in _UNFORKED_SERVICES
