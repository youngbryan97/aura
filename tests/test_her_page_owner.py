"""Q06: what is done to her browser page goes through one owner, core/runtime/page_gateway.py.

37 modules called Playwright directly, 153 calls the effect-ownership gate counted as ungoverned (10 November 2026).
One of them put a subreddit's name into a script's text, so a name with a quote in it was a script of its own.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.runtime.page_gateway import PageRefused, PageScript, get_page_gateway, page_script

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
_DECLARED = page_script("(n) => n + 1")


class _Page:
    def __init__(self):
        self.ran: list[tuple] = []

    def evaluate(self, script, *args):
        self.ran.append(("evaluate", script, args))
        return "read"

    def goto(self, address, **options):
        self.ran.append(("goto", address, options))
        return "went"


def test_a_page_runs_a_declared_script_with_its_values_as_arguments():
    page = _Page()
    assert get_page_gateway().read(page, _DECLARED, 41) == "read"
    assert page.ran == [("evaluate", _DECLARED, (41,))]
    assert _DECLARED.owner == __name__


@pytest.mark.parametrize("built", [
    "(n) => n + 1",
    page_script("(n) => n + %d") % 1,
    page_script("(n) => n") + " + 1",
    f"(n) => n + {1}",
])
def test_a_script_built_at_the_call_is_refused(built):
    page = _Page()
    with pytest.raises(PageRefused, match="declared"):
        get_page_gateway().read(page, built)
    assert page.ran == []


def test_a_page_script_is_only_made_by_declaring_it():
    with pytest.raises(TypeError):
        PageScript("() => 1", owner="anyone")
    with pytest.raises(PageRefused):
        page_script("   ")


@pytest.mark.parametrize("address", ["https://example.org/", "http://localhost:8000/", "file:///tmp/a.html",
                                     "about:blank"])
def test_she_goes_to_a_web_page_a_file_of_hers_or_a_blank_page(address):
    assert get_page_gateway().go_to(_Page(), address, source="test") == "went"


@pytest.mark.parametrize("address", ["javascript:alert(1)", "JavaScript:void(0)", "chrome://settings", "/relative"])
def test_any_other_address_is_refused(address):
    page = _Page()
    with pytest.raises(PageRefused):
        get_page_gateway().go_to(page, address, source="test")
    assert page.ran == []


def test_every_effect_is_counted_by_the_source_that_asked():
    class Keys:
        def press(self, key):
            return key

    gateway = get_page_gateway()
    before = gateway.counts().get("press:test.keys", 0)
    assert gateway.press(Keys(), "Enter", source="test.keys") == "Enter"
    assert gateway.counts()["press:test.keys"] == before + 1


def _scanned() -> list[Path]:
    roots = [ROOT / "core", ROOT / "skills", ROOT / "interface"]
    return [p for root in roots for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_every_declared_script_is_the_modules_own_text():
    """page_script() takes a string literal, or stands where the module or a class is defined: at import, a module's
    constants are its own text. Inside a function anything but a literal could carry what she read or a model wrote."""
    wrong: list[str] = []
    for path in _scanned():
        source = path.read_text(encoding="utf-8")
        if "page_script(" not in source:
            continue
        tree = ast.parse(source)
        inside: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                inside.update(id(child) for child in ast.walk(node) if child is not node)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "page_script"):
                continue
            literal = len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
            if id(node) in inside and not literal:
                wrong.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert wrong == []


def test_no_page_effect_is_made_outside_her_page_owner():
    """What the gate still counts as a raw browser call is opening the system browser for the person (a desktop
    effect, the desktop action gateway's) and Selenium's own navigation: each named here, with the line's text."""
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.lint_governance import CANONICAL_PRIMITIVE_OWNERS, EffectVisitor

    owners = CANONICAL_PRIMITIVE_OWNERS["raw_browser"]
    left: list[str] = []
    for path in _scanned():
        relative = str(path.relative_to(ROOT))
        if relative in owners:
            continue
        source = path.read_text(encoding="utf-8")
        visitor = EffectVisitor(relative_path=relative)
        visitor.visit(ast.parse(source))
        lines = source.splitlines()
        left += [f"{relative}: {lines[line - 1].strip()}" for category, _callee, line in visitor.calls
                 if category == "raw_browser"]
    named = ("webbrowser.open", "driver.get(url)")
    assert [line for line in left if not any(name in line for name in named)] == []
    assert len(left) <= 5
