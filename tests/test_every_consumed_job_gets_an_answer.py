"""A job the worker took off the queue gets an answer, whatever went wrong.

The parent's future is resolved by the reply the worker writes. A guard that
names exception types therefore does not decide which failures are reported —
it decides which failures are ABANDONED, because a type nobody listed leaves
the caller waiting on a reply that will never come.

LIVE, 2026-09-07: `jinja2.TemplateError("Unexpected message role.")` was not
one of the seven types the latent handler named. A file-read request reached
the worker at 15:44:48 and had no answer thirteen minutes later; the client
gave up at five, and the runtime went on reporting `Conversation: WORKING`
with `conversation_busy: true` the whole time.

The same escape was found in this file on 2026-08-19, fixed at the one site
that had shown it, and the two sites that had not were left naming types.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "core/brain/llm/mlx_worker.py"


def _worker_loop() -> ast.FunctionDef:
    tree = ast.parse(WORKER.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_mlx_worker_loop":
            return node
    raise AssertionError("_mlx_worker_loop is gone; this gate now protects nothing")


def _handlers(loop: ast.FunctionDef) -> list[ast.ExceptHandler]:
    return [node for node in ast.walk(loop) if isinstance(node, ast.ExceptHandler)]


def _the_request_loops_own_handlers(loop: ast.FunctionDef) -> list[ast.ExceptHandler]:
    """The handlers of the try the request loop is made of.

    Not `ast.walk`: that yields nested handlers too, and a nested one catching
    everything says nothing about whether the job survives the ones around it.
    """

    for node in ast.walk(loop):
        if not isinstance(node, ast.While):
            continue
        for statement in node.body:
            if isinstance(statement, ast.Try) and statement.handlers:
                return list(statement.handlers)
    raise AssertionError("the request loop is no longer a try inside a while")


def _catches_everything(handler: ast.ExceptHandler) -> bool:
    kind = handler.type
    if kind is None:
        return True
    return isinstance(kind, ast.Name) and kind.id in {"Exception", "BaseException"}


def test_the_worker_loop_is_still_where_jobs_are_answered() -> None:
    loop = _worker_loop()
    body = ast.get_source_segment(WORKER.read_text("utf-8"), loop) or ""
    assert "ipc_writer.put" in body


def test_the_last_guard_in_the_job_loop_catches_everything() -> None:
    """The outermost handler in the loop is what stops a job vanishing."""

    handlers = _the_request_loops_own_handlers(_worker_loop())
    catch_alls = [h for h in handlers if _catches_everything(h)]
    assert catch_alls, (
        "no handler in _mlx_worker_loop catches every exception, so a type "
        "nobody listed abandons the job instead of failing it"
    )


def test_a_caught_failure_still_writes_a_reply() -> None:
    """Every catch-all in the loop answers rather than only recording.

    Two shapes count as answering: writing the reply itself, or filling in the
    ``response`` the loop writes once the job is done.
    """

    source = WORKER.read_text("utf-8")
    loop = _worker_loop()
    for handler in _handlers(loop):
        if not _catches_everything(handler):
            continue
        segment = ast.get_source_segment(source, handler) or ""
        assert "ipc_writer.put" in segment or "response.update" in segment, (
            f"catch-all at line {handler.lineno} records the failure and does "
            "not answer the job"
        )


def test_the_job_loop_does_not_end_on_a_tuple_of_named_types() -> None:
    """The outermost handler of the request loop names no types.

    A tuple there is a prediction about which failures can happen, and the
    cost of predicting wrong is a job with no answer rather than a job that
    failed. jinja2.TemplateError escaped one such tuple on 2026-08-19 and
    another on 2026-09-07.
    """

    handlers = _the_request_loops_own_handlers(_worker_loop())
    assert _catches_everything(handlers[-1]), (
        "the last handler of the request loop names types: "
        f"{ast.dump(handlers[-1].type) if handlers[-1].type else 'bare'}"
    )
