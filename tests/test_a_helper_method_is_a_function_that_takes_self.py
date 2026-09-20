"""A helper the sweep left as a method is moved out of the class, whole.

The size gate holds a class to its method count. The first sweep moved blocks
out of long methods into `_<method>_<slug>` METHODS: `InferenceGate` went
from 151 methods to 189 with no new behaviour. `tools/refactor/demote_helper_methods.py`
moves each such helper to module level with `self` as its first parameter
and rewrites the calls; a static helper loses its decorator and takes none.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "refactor"))

import demote_helper_methods as tool  # noqa: E402

SOURCE = '''
class Thing:
    def __init__(self):
        self.count = 0

    def work(self, n):
        first = self._work_part_1(n)
        # the second half
        second = self._work_scale(first)
        return second + self._work_tail()

    # a comment above it travels with it
    def _work_part_1(self, n):
        self.count += 1
        return n * 2

    @staticmethod
    def _work_scale(value):
        return value + 1

    def _work_tail(self):
        return self.count

    def unrelated(self):
        return "kept"
'''


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_helpers_move_out_and_the_class_still_works(tmp_path: Path) -> None:
    path = tmp_path / "thing_mod.py"
    path.write_text(SOURCE, encoding="utf-8")

    moved = tool.demote(path, "Thing", None, False)

    assert moved == 3
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert {n.name for n in cls.body if isinstance(n, ast.FunctionDef)} == {"__init__", "work", "unrelated"}
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert top == {"_work_part_1", "_work_scale", "_work_tail"}
    assert "_work_part_1(self, n)" in text and "_work_scale(first)" in text and "_work_tail(self)" in text
    assert "@staticmethod" not in text
    assert "# a comment above it travels with it\ndef _work_part_1" in text

    module = _load(path)
    thing = module.Thing()
    assert thing.work(5) == 12 and thing.unrelated() == "kept"


def test_a_method_that_is_used_as_a_member_stays(tmp_path: Path) -> None:
    path = tmp_path / "thing_mod.py"
    path.write_text(
        SOURCE + "\n    def other(self):\n        return getattr(self, '_work_tail')\n", encoding="utf-8"
    )

    moved = tool.demote(path, "Thing", None, False)

    text = path.read_text(encoding="utf-8")
    assert moved == 2 and "    def _work_tail(self):" in text


CLASS_METHODS = '''
class Registry:
    items = {"a": 1}

    @classmethod
    def lookup(cls, key):
        return cls._lookup_or_absent(key)

    def read(self, key):
        return self._lookup_or_absent(key) + Registry._lookup_or_absent(key)

    @classmethod
    def _lookup_or_absent(cls, key):
        return cls.items.get(key, -1)
'''


def test_a_class_method_keeps_its_class(tmp_path: Path) -> None:
    path = tmp_path / "registry_mod.py"
    path.write_text(CLASS_METHODS, encoding="utf-8")

    assert tool.demote(path, "Registry", None, False) == 1
    text = path.read_text(encoding="utf-8")
    assert "@classmethod\ndef _lookup_or_absent" not in text
    assert "return _lookup_or_absent(cls, key)" in text
    assert "_lookup_or_absent(type(self), key) + _lookup_or_absent(Registry, key)" in text
    module = _load(path)
    assert module.Registry.lookup("a") == 1 and module.Registry().read("zz") == -2
