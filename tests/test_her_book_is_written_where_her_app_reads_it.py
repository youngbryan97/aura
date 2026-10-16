"""A book written for her lands in her launched app's data directory, not in ~/.aura/data.

Live 2026-10-16: launch_aura.sh exports AURA_ROOT as the checkout, so her app reads <checkout>/data;
every book written for her from a worktree or a run script, without AURA_ROOT, went to ~/.aura/data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool():
    path = Path(__file__).resolve().parent.parent / "tools" / "keep_procedures_for_her.py"
    spec = importlib.util.spec_from_file_location("keep_procedures_for_her", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_her_app_root_is_the_primary_checkout_even_from_a_worktree():
    root = _tool().her_app_root()
    # The primary checkout holds git's directory; a worktree holds only a file pointing at it.
    assert (root / ".git").is_dir()
