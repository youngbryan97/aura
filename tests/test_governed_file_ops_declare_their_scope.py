"""The same missing-scope defect, three times, in one file.

Every consequential file operation in host_automation goes through
``ActionExecutor.execute(domain=FILE_WRITE, ...)``, and the Will refuses it
unless the caller has declared a governed scope. Three call sites needed one
and were written without it:

  1. ``ensure_screenshot_directory`` — screen perception was dead behind a
     warning card; take_screenshot failed before reaching screencapture on
     every ambient tick;
  2. ``screenshot_retention_delete`` — retention silently kept everything and
     the capture directory grew without bound;
  3. ``ephemeral_ocr_cleanup`` — found live 2026-08-10, minutes after the
     first fix went live and capture started working. ``~/.aura/data/
     ephemeral`` reached 10 files / 15MB in the first few minutes, one about
     every 7 seconds — roughly 770MB an hour of full-screen captures that
     nothing could ever delete.

Each was fixed when it was found. Nothing checked the class, so the third sat
there while the first two were being repaired — and it was the most damaging,
because an ephemeral capture that is never deleted is a permanent record of
everything on the person's screen. That is the harm the file_delete rule
exists to prevent, arrived at by obeying it.

To be clear about what is NOT being changed: ``file_delete`` is off by default
and listed in ``_UNGRANTABLE_MODALITIES``, and it stays that way. Aura may not
delete a person's files. A declared internal scope is how she does maintenance
on files she created herself under her own state root, which is a different
act with a different blast radius.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Where to look. Not one file: `host_automation.py` was split and its
#: FILE_WRITE calls went to `host_automation_screen.py`, so a detector
#: anchored on the original path found nothing and said so — "the detector
#: has drifted from the code and is no longer checking anything" is this
#: file's own guard reporting that it had stopped being a test.
#:
#: The host-automation family, as the directory has it. The scope this
#: file is about is the one for INTERNAL MAINTENANCE writes — the runtime's
#: own housekeeping, which the Will refuses without it. A skill that writes
#: a file because somebody asked it to is not that, and wrapping those in an
#: internal scope would be declaring the person's request to be the
#: runtime's own errand.
_ACTING_MODULES = "core/capabilities"
_FAMILY = "host_automation"


def _modules_that_could_write() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    return [
        path
        for path in sorted((here / _ACTING_MODULES).glob(f"{_FAMILY}*.py"))
        if "__pycache__" not in path.parts
    ]


#: The original single path, for the tests that read one module's structure.
_HOST_AUTOMATION = Path("core/capabilities/host_automation.py")


def _governed_file_write_calls() -> list[tuple[str, int, bool]]:
    """Every ActionExecutor FILE_WRITE call, and whether a scope encloses it.

    Structural rather than textual: a `with local_internal_governed_scope(...)`
    that merely appears nearby is not the same as one that actually contains
    the call, and the whole defect class is a call sitting outside its scope.
    """
    found: list[tuple[str, int, bool]] = []
    for module in _modules_that_could_write():
        found.extend(_calls_in(module))
    return found


def _calls_in(path: Path) -> list[tuple[str, int, bool]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    scoped_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            name = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", "")
            if name == "local_internal_governed_scope":
                scoped_ranges.append((node.lineno, node.end_lineno or node.lineno))

    found: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if getattr(func, "attr", "") != "execute":
            continue
        if getattr(getattr(func, "value", None), "id", "") != "ActionExecutor":
            continue

        action_name = ""
        is_file_write = False
        for keyword in node.keywords:
            if keyword.arg == "action_name" and isinstance(keyword.value, ast.Constant):
                action_name = str(keyword.value.value)
            if keyword.arg == "domain":
                is_file_write = "FILE_WRITE" in ast.dump(keyword.value)
        if not is_file_write:
            continue

        enclosed = any(
            start <= node.lineno <= end for start, end in scoped_ranges
        )
        found.append(
            (action_name or f"{path.name} line {node.lineno}", node.lineno, enclosed)
        )
    return found


def test_there_are_file_write_calls_to_check():
    """Guard the guard: a rename must not turn this file into a no-op."""
    assert _governed_file_write_calls(), (
        "found no ActionExecutor FILE_WRITE calls — the detector has drifted "
        "from the code and is no longer checking anything"
    )


@pytest.mark.parametrize(
    ("action_name", "line", "enclosed"),
    _governed_file_write_calls(),
    ids=lambda value: str(value) if isinstance(value, str) else None,
)
def test_every_file_write_declares_a_governed_scope(action_name, line, enclosed):
    """The class, not the three instances that happened to be found."""
    assert enclosed, (
        f"{action_name} (line {line}) calls ActionExecutor with FILE_WRITE "
        f"outside any local_internal_governed_scope. The Will refuses it, and "
        f"the refusal is silent in the sense that matters: the caller keeps "
        f"working and the effect never happens."
    )


def test_the_ephemeral_cleanup_is_the_third_instance():
    """Named explicitly, because it is the one with a privacy cost."""
    names = {name for name, _, _ in _governed_file_write_calls()}

    assert "host_automation.ephemeral_ocr_cleanup" in names
    assert "host_automation.screenshot_retention_delete" in names
    assert "host_automation.ensure_screenshot_directory" in names


def test_file_delete_remains_off_and_ungrantable():
    """What this fix must NOT have done.

    The scope is how Aura does maintenance on her own ephemeral captures. It
    is not a route to deleting a person's files, and the modality it would
    need for that stays off and stays impossible to grant.
    """
    from core.capabilities.permission_model import (
        _UNGRANTABLE_MODALITIES,
        ModalityPermissions,
    )

    assert ModalityPermissions().file_delete is False
    assert "file_delete" in _UNGRANTABLE_MODALITIES


def test_retention_is_not_gated_on_retaining_the_capture():
    """The ephemeral directory needs a backstop more than any other.

    Retention used to run only when retain_capture was true, so the ONLY thing
    bounding ~/.aura/data/ephemeral was the per-call cleanup — one point of
    failure in front of the highest-churn directory in the system, about one
    full-screen capture every 7 seconds. When that cleanup was refused for
    hours, nothing pruned and nothing noticed.
    """
    tree = ast.parse(_HOST_AUTOMATION.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body = ast.dump(node)
        if "_enforce_screenshot_retention" not in body:
            continue
        assert "retain_capture" not in ast.dump(node.test), (
            "retention must not be gated on retain_capture: that leaves the "
            "ephemeral capture directory with no backstop at all"
        )
