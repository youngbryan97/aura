"""What core.architecture_quality guarantees, and the test for each."""
from __future__ import annotations

from core.verify.a_promise_with_a_test import APromise

THE_PROMISES: tuple[APromise, ...] = (
    APromise(
        it="A file may grow only if another shrinks by more, so the total "
        "oversize is one number that holds rather than a per-file argument.",
        checked_by="tests/test_a_god_object_only_shrinks.py::"
        "test_uncompensated_growth_fails",
        if_it_fails="the budget line names the total and the overshoot; "
        "without it every God object grows a little and none is to blame",
    ),
    APromise(
        it="A class may never grow another method even while its file shrinks, "
        "because splitting a God class is the point and growing one is not a "
        "trade.",
        checked_by="tests/test_a_god_object_only_shrinks.py::"
        "test_a_god_class_may_never_grow_even_while_its_file_shrinks",
        if_it_fails="the gate names the class and its baseline; a file could "
        "otherwise be shrunk by moving lines while the class kept growing",
    ),
    APromise(
        it="Moving four hundred lines out of a God object into two new modules "
        "passes, so the gate does not block the work it exists to cause.",
        checked_by="tests/test_a_god_object_only_shrinks.py::"
        "test_the_decomposition_this_gate_exists_to_encourage_passes",
        if_it_fails="the decomposition fails its own gate, which is how a gate "
        "gets deleted",
    ),
)
