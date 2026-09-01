"""core/science/singletons.py — a reset that does not outlive its test.

A module-level singleton is installed once per process, so a test that replaces
one and does not put it back has replaced it for every test that runs
afterwards. The failure looks like nothing at all until some later file reads
the singleton and finds it empty — and then it looks like a defect in that
file, which is the expensive part.

This happened twice here in one day. ``health_fragments`` had an autouse fixture
that cleared every registered provider, so every expected fragment reported
absent for the rest of the session. The claim ladder and the parameter registry
had it next: emptying them left ``tools/evidence_report.py`` reading one claim
where four are installed.

So the reset is a context manager. It hands the body a fresh instance and puts
the original back on the way out, whatever happened in between.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

__all__ = ["scoped_singleton", "scoped_science_singletons"]

T = TypeVar("T")


@contextlib.contextmanager
def scoped_singleton(
    module: Any,
    attribute: str,
    factory: Callable[[], T],
    lock: threading.Lock | threading.RLock | None = None,
) -> Iterator[T]:
    """Swap a module-level singleton for the body, restore it afterwards.

    ``module`` and ``attribute`` name the binding, because the point is to put
    back the exact object that was there — a caller holding a reference to the
    original keeps working, which a re-created replacement would not give them.
    """
    guard = lock or contextlib.nullcontext()
    with guard:
        saved = getattr(module, attribute)
        fresh = factory()
        setattr(module, attribute, fresh)
    try:
        yield fresh
    finally:
        with guard:
            setattr(module, attribute, saved)


#: Every process-wide registry the science surface reads. Named here rather
#: than in each test file, so a new one is scoped by adding a line in one place
#: instead of by remembering to.
_SCIENCE_SINGLETONS: tuple[tuple[str, str, str], ...] = (
    ("core.science.claim_ladder", "_ladder", "ClaimLadder"),
    ("core.science.parameter_registry", "_registry", "ParameterRegistry"),
    ("core.science.experiment_registry", "_registry", "ExperimentRegistry"),
    ("core.science.calibration_layer", "_layer", "CalibrationLayer"),
    ("core.science.learning_audit", "_audit", "LearningAudit"),
    ("core.science.neuro_reference", "_reference", "NeuroReference"),
    ("core.science.organ_accounting", "_accounting", "OrganAccounting"),
)


@contextlib.contextmanager
def scoped_science_singletons() -> Iterator[None]:
    """Give a test its own copy of every science registry, and put them back.

    A test that records a malformed experiment, or empties the ladder, is
    writing into state ``tools/evidence_report.py`` reads. Without this the
    report sees whatever the last test left and reports a contradiction that
    belongs to nobody.
    """
    import importlib

    saved: list[tuple[Any, str, Any]] = []
    try:
        for module_name, attribute, _ in _SCIENCE_SINGLETONS:
            module = importlib.import_module(module_name)
            saved.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, None)
        yield
    finally:
        for module, attribute, original in saved:
            setattr(module, attribute, original)
