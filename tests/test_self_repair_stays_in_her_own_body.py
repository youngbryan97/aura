"""Self-repair was editing other agents' worktrees.

Live 2026-08-17: 178 autonomous repair attempts against
.claude/worktrees/codex-autonomy-deferral/core/collective/delegator.py, 54
more against a codex-wow-cp400 copy, and FileNotFoundError storms for
worktrees that had since been deleted.

Those paths pass the containment check because they are literally under the
source root — .claude/worktrees lives inside the checkout. So a traceback
through another agent's working copy was accepted as a location in her own
code.

Editing them is wrong twice over. They are someone else's in-flight work, and
they are transient: a fix applied there is discarded with the branch, so the
same "bug" is rediscovered and re-repaired forever while her actual source
keeps the defect.
"""
from __future__ import annotations

import os

import pytest

from core.self_modification.error_intelligence import (
    _SOURCE_ROOT_REALPATH,
    _is_her_own_source,
)

ROOT = _SOURCE_ROOT_REALPATH


@pytest.mark.parametrize(
    "relative",
    [
        "core/collective/delegator.py",
        "core/runtime/errors.py",
        "interface/routes/chat.py",
    ],
)
def test_her_own_modules_are_repairable(relative):
    assert _is_her_own_source(os.path.join(ROOT, relative))


@pytest.mark.parametrize(
    "relative",
    [
        ".claude/worktrees/codex-autonomy-deferral/core/collective/delegator.py",
        ".claude/worktrees/codex-wow-cp400/core/runtime/errors.py",
        ".venv/lib/python3.12/site-packages/pydantic/main.py",
        "dist/Aura.app/Contents/Resources/thing.py",
    ],
)
def test_checkouts_nested_inside_her_checkout_are_not_her_body(relative):
    """Under the root is not the same as part of her."""
    assert not _is_her_own_source(os.path.join(ROOT, relative))


@pytest.mark.parametrize(
    "outside", ["/tmp/elsewhere/core/x.py", "/usr/lib/python3/thing.py", "/"]
)
def test_paths_outside_the_checkout_are_still_excluded(outside):
    assert not _is_her_own_source(outside)


def test_a_worktree_path_is_rejected_even_when_it_exists():
    """The exclusion is about ownership, not about the file being missing.

    Rejecting only deleted worktrees would leave her editing live ones, which
    is the more damaging half — that is someone's work in progress.
    """
    existing = os.path.join(ROOT, ".claude", "worktrees")
    if not os.path.isdir(existing):
        pytest.skip("no worktrees on this host")
    for entry in os.listdir(existing)[:1]:
        candidate = os.path.join(existing, entry, "core", "config.py")
        assert not _is_her_own_source(candidate)


def test_a_malformed_path_does_not_raise():
    """This runs inside an error handler; it may not add a second error."""
    for junk in ("", "\x00", "relative/path.py"):
        assert _is_her_own_source(junk) in (True, False)
