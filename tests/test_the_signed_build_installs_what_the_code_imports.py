"""Every third-party module the runtime imports is in the release lockfile.

LIVE, found by `make deps-gate` on 2026-09-21, red since the tenth.
`snowballstemmer==3.0.1` was added to `requirements.txt` on 10 September and
never reached `requirements_lock.txt`. The release lane installs the LOCK,
with `--require-hashes`, so the signed desktop bundle did not contain it —
and `core/language/word_forms.py` imports it at module scope, from
`core/conversation/request_coverage.py` and `thread_continuity.py`, which
are on the chat path. The shipped app would have raised ImportError where
this checkout runs fine.

A digest gate catches the drift once it is noticed. This catches the thing
the drift causes.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements_lock.txt"
REQUIREMENTS = ROOT / "requirements.txt"

#: Import name -> distribution name, where they differ.
_DISTRIBUTION_OF = {
    "cv2": "opencv-python",
    "fitz": "pymupdf",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "bs4": "beautifulsoup4",
    "serial": "pyserial",
    "OpenGL": "pyopengl",
    "psutil": "psutil",
}


def _locked_distributions() -> set[str]:
    names = set()
    for line in LOCK.read_text().splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def _declared_distributions() -> set[str]:
    names = set()
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match:
            names.add(match.group(1).lower().replace("_", "-"))
    return names


def test_every_declared_requirement_is_in_the_release_lock() -> None:
    """The release lane installs the lock, so the lock is what ships."""
    missing = sorted(_declared_distributions() - _locked_distributions())
    assert not missing, (
        "declared in requirements.txt and absent from requirements_lock.txt, "
        f"so the signed bundle will not have them: {missing}"
    )


def test_the_stemmer_the_chat_path_imports_is_locked() -> None:
    """The one that was missing, named, so its return is caught by name."""
    assert "snowballstemmer" in _locked_distributions()


@pytest.mark.parametrize(
    "module",
    [
        "core/language/word_forms.py",
        "core/conversation/request_coverage.py",
        "core/conversation/thread_continuity.py",
    ],
)
def test_the_chat_path_modules_import_at_module_scope(module: str) -> None:
    """Which is why an absent distribution is a boot failure, not a branch."""
    tree = ast.parse((ROOT / module).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported |= {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    third_party = {
        name
        for name in imported
        if name
        and name not in sys.stdlib_module_names
        and not name.startswith(("core", "interface", "skills", "tests", "_"))
    }
    locked = _locked_distributions()
    for name in sorted(third_party):
        distribution = _DISTRIBUTION_OF.get(name, name).lower().replace("_", "-")
        assert distribution in locked, (
            f"{module} imports {name} at module scope and {distribution} is "
            "not in the release lock"
        )
