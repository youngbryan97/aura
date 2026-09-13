"""A timestamp reads the clock when it is taken.

`field(default_factory=time.time)` hands the dataclass the function object
`time.time` named when the module was imported. The subject-core experiment
clock replaces `time.time` process-wide so both arms of an intervention see the
same elapsed time, and every record built through one of those factories kept
stamping itself with the machine's clock instead: 637 dataclass fields, 27
parameter defaults and one module that imported the function by name. One of
them, the draft in the multiple-drafts engine, came out dated after the moment
it was aged against, took `log1p` below its domain, and failed the
conversational dynamics phase 281 times in a single campaign.
"""

from __future__ import annotations

import ast
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [ROOT / name for name in ("core", "interface", "llm", "skills") if (ROOT / name).exists()]


def _is_time_time(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "time"
        and isinstance(node.value, ast.Name)
        and node.value.id == "time"
    )


def _bindings() -> list[str]:
    found: list[str] = []
    for root in SOURCES:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported_by_name = any(
                isinstance(node, ast.ImportFrom)
                and node.module == "time"
                and any(alias.name == "time" for alias in node.names)
                for node in ast.walk(tree)
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg == "default_factory":
                    named = imported_by_name and isinstance(node.value, ast.Name) and node.value.id == "time"
                    if _is_time_time(node.value) or named:
                        found.append(f"{path.relative_to(ROOT)}:{node.value.lineno}")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
                    found.extend(
                        f"{path.relative_to(ROOT)}:{default.lineno}"
                        for default in defaults
                        if _is_time_time(default)
                    )
    return found


def test_no_record_is_stamped_with_the_clock_that_existed_at_import() -> None:
    found = _bindings()
    assert not found, found[:20]


def test_a_draft_is_stamped_with_the_installed_clock() -> None:
    from core.consciousness.multiple_drafts import Draft
    from core.subject.clock import ExperimentClock

    clock = ExperimentClock(0.03, start=time.time() - 3600.0)
    clock.install()
    try:
        draft = Draft(
            draft_id="d", stream_index=0, content="", goal="", valence=0.0, urgency=0.0, coherence=0.5
        )
        assert draft.created_at == clock.now()
    finally:
        clock.uninstall()


def test_a_draft_dated_after_now_has_no_negative_age() -> None:
    from core.consciousness.multiple_drafts import Draft

    draft = Draft(
        draft_id="d", stream_index=0, content="", goal="", valence=0.0, urgency=0.0, coherence=0.5
    )
    draft.created_at = time.time() + 60.0
    assert draft.age_secs() == 0.0
