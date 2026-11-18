"""A fix's invariant reaches the code before the fix only when nothing of the fix comes with it."""
from __future__ import annotations

from types import ModuleType

from core.self_modification.a_fixs_own_specification import injected, module_of, specification_in

_BEFORE = """
LIMIT = 3

def size(things):
    return len(things) + 1
"""

_AFTER = """
from core.verify import invariant
import math

LIMIT = 3

def size(things):
    return len(things)

def _new_helper():
    return 1

@invariant("size counts", scope="x", owner="y")
def size_counts():
    assert size([1]) == math.floor(1.0) and LIMIT == 3

@invariant("leans on the fix", scope="x", owner="y")
def leans_on_the_fix():
    assert _new_helper() == 1

@invariant("nobody asked", scope="x", owner="y")
def nobody_asked():
    assert size([]) == 0
"""


def test_only_an_invariant_the_tests_name_and_that_reads_nothing_of_the_fix_is_carried():
    source, names = specification_in(_BEFORE, _AFTER, {"size_counts", "leans_on_the_fix"})
    assert names == ("size_counts",)
    assert "def size_counts" in source and "_new_helper" not in source and "nobody_asked" not in source


def test_the_imports_it_needs_come_with_it_and_the_repaired_function_does_not():
    source, _ = specification_in(_BEFORE, _AFTER, {"size_counts"})
    assert "from core.verify import invariant" in source and "import math" in source
    assert "def size(" not in source, "the fix itself stays behind"


def test_an_invariant_already_there_is_left_alone():
    assert specification_in(_AFTER, _AFTER, {"size_counts"}) == ("", ())


def test_source_that_does_not_parse_carries_nothing():
    assert specification_in("def broken(:", _AFTER, {"size_counts"}) == ("", ())


def test_the_carried_check_fails_on_the_code_before_the_fix_and_passes_on_the_fix(monkeypatch):
    import sys

    module = ModuleType("a_module_under_repair")
    exec(_BEFORE, module.__dict__)  # noqa: S102
    monkeypatch.setitem(sys.modules, "a_module_under_repair", module)
    source, _ = specification_in(_BEFORE, _AFTER, {"size_counts"})
    exec(injected("a_module_under_repair", source), {})  # noqa: S102

    try:
        module.size_counts()
    except AssertionError:
        pass
    else:
        raise AssertionError("the check must refuse the defect it was written for")
    module.size = lambda things: len(things)
    module.size_counts()


def test_a_path_names_its_module():
    assert module_of("core/a/b.py") == "core.a.b"
    assert module_of("core/a/__init__.py") == "core.a"
