"""Only the last definition runs, and the ones above it look fine."""
from __future__ import annotations

from pathlib import Path

from tools.lint_shadowed_definitions import scan, shadowed_in

ROOT = Path(__file__).resolve().parents[1]


def test_the_shape_that_killed_the_fix_is_detected() -> None:
    """Three copies, the newest first, which is where a fix naturally goes."""
    source = (
        "def repair(code):\n"
        "    return 'the fix that walks the whole tree'\n"
        "\n"
        "def repair(code):\n"
        "    return 'an older one'\n"
        "\n"
        "def repair(code):\n"
        "    return 'the oldest'\n"
    )
    found = shadowed_in(source)
    assert found == [("repair", [1, 4, 7])]


def test_a_method_written_twice_in_one_class_is_detected() -> None:
    source = "class A:\n    def go(self): ...\n    def go(self): ...\n"
    assert shadowed_in(source) == [("A.go", [2, 3])]


def test_a_conditional_definition_is_deliberate_and_is_not_counted() -> None:
    """Defining a name two ways under a branch is the point of that shape."""
    source = (
        "try:\n"
        "    from fast import parse\n"
        "except ImportError:\n"
        "    def parse(x): return x\n"
        "\n"
        "if True:\n"
        "    def helper(): ...\n"
        "else:\n"
        "    def helper(): ...\n"
    )
    assert shadowed_in(source) == []


def test_one_definition_is_not_a_finding() -> None:
    assert shadowed_in("def only(): ...\nclass B: pass\n") == []


def test_unparsable_source_reports_nothing_rather_than_raising() -> None:
    assert shadowed_in("def (: broken") == []


def test_the_tree_has_none() -> None:
    """The measured state after removing the three copies in code_repl."""
    findings = scan(ROOT)
    assert findings == [], "\n".join(findings)
