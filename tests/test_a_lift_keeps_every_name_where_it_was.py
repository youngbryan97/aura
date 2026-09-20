"""A lifted function is still found where it was, and reads what is patched there.

`tools/refactor/lift_functions.py` moves module-level definitions into a
sibling module. The parent imports them straight back, so callers and
monkeypatches that name them on the parent still work; what the moved code
takes from the parent is imported at call time, so a patch on the parent
reaches the code that reads it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "refactor"))

import lift_functions as tool  # noqa: E402

PARENT = '''"""A module with a cluster worth lifting."""
from __future__ import annotations

import re
from typing import Any

logger = None
_LIMIT = 3
_WORD_RE = re.compile(r"\\w+")


def _count(text: str) -> int:
    return len(_WORD_RE.findall(text))


# the reader everyone calls
def over_limit(text: str, extra: Any = None) -> bool:
    """Whether the text has more words than the limit."""
    return _count(text) > _LIMIT


def unrelated() -> str:
    return "stays"
'''


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_parent_still_serves_the_lifted_names_and_reads_the_patch(tmp_path: Path) -> None:
    pkg = tmp_path / "liftpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    parent = pkg / "words.py"
    parent.write_text(PARENT, encoding="utf-8")

    moved_lines, count = tool.lift(parent, "words_counting", "Counting words.", ["_count", "over_limit", "_WORD_RE"])

    assert count == 3 and moved_lines > 5
    sibling = (pkg / "words_counting.py").read_text(encoding="utf-8")
    assert sibling.startswith('"""Counting words.')
    assert "import re\n" in sibling
    assert "from .words import (\n        _LIMIT,\n    )" in sibling, "the limit is read from the parent at call time"
    assert "# the reader everyone calls\ndef over_limit" in sibling
    text = parent.read_text(encoding="utf-8")
    assert "from .words_counting import (  # noqa: F401  (re-exported: they were defined here)" in text
    assert "def _count(" not in text and "def unrelated(" in text

    sys.path.insert(0, str(tmp_path))
    try:
        for name in ("liftpkg", "liftpkg.words", "liftpkg.words_counting"):
            sys.modules.pop(name, None)
        import importlib

        words = importlib.import_module("liftpkg.words")
        assert words.over_limit("one two three four") is True
        assert words._count("a b") == 2
        words._LIMIT = 10  # a patch on the parent reaches the moved code
        assert words.over_limit("one two three four") is False
    finally:
        sys.path.remove(str(tmp_path))
        for name in ("liftpkg", "liftpkg.words", "liftpkg.words_counting"):
            sys.modules.pop(name, None)
