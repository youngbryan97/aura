"""An incomplete state projection is not an interrupted model generation."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

pytestmark = pytest.mark.unit


def _recovery_helper(namespace):
    """Execute the production closure without starting the desktop server."""
    source = Path("interface/routes/chat.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    helper = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_try_serve_grounded_recovery"
    )
    factory = ast.parse(
        "def factory():\n"
        "    pending_exchange_id = None\n"
        "    return _try_serve_grounded_recovery\n"
    ).body[0]
    factory.body.insert(1, helper)
    module = ast.Module(body=[factory], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["factory"]()


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
async def test_recovery_continues_only_an_owned_generation(monkeypatch, owned):
    from core.conversation import response_reliability

    question = "Explain how this calculation works and how to verify it."
    trace = {
        "foreground_model_generation_consumed": owned,
        "foreground_model_generation_count": int(owned),
        "foreground_model_generation_segment_count": int(owned),
        "foreground_model_generation_transaction_count": int(owned),
        "foreground_model_generation_transaction_id": "owned-turn" if owned else "",
    }
    calls = []

    async def run_turn(message, **kwargs):
        calls.append((message, kwargs))
        return None

    monkeypatch.setattr(
        response_reliability, "assess_user_facing_reply",
        lambda *_args, **_kwargs: SimpleNamespace(reasons=("unanswered_question_part",)),
    )
    namespace = {
        "desktop_requires_cognitive_engine": True,
        "_semantic_user_message": question,
        "_live_turn_trace": trace,
        "_required_foreground_memory_snapshot": lambda: SimpleNamespace(
            refuse_heavy_local_generation=False,
        ),
        "_desktop_required_cognitive_budget": lambda **_kwargs: 120.0,
        "_DESKTOP_COGNITIVE_MIN_REQUIRED_BUDGET_S": 1.0,
        "_run_cognitive_engine_chat_turn": run_turn,
        "foreground_timeout": 120.0,
        "request_started_at": time.monotonic(),
        "preflight_context_message": question,
        "_turn_sensory_evidence": None,
        "_chat_session_id": "test-session",
        "chat_origin": "user",
        "lane": {"conversation_ready": True},
        "conversation_only_surface": False,
        "_profile_user_id": "test-principal",
        "_CHAT_RECOVERABLE_ERRORS": (RuntimeError, TypeError, ValueError),
        "time": time,
        "logger": logging.getLogger(__name__),
    }
    helper = _recovery_helper(namespace)
    await helper("Only the first part was available.")

    assert len(calls) == 1
    message, options = calls[0]
    assert message == question
    assert options["visible_user_message"] == question
    if owned:
        assert options["continuation_partial"] == "Only the first part was available."
        assert options["continuation_evidence"][
            "foreground_model_generation_transaction_id"
        ] == "owned-turn"
        assert options["continuation_evidence"]["foreground_model_generation_count"] == 1
    else:
        assert options["continuation_partial"] == ""
        assert options["continuation_reasons"] is None
        assert options["continuation_evidence"] is None
    assert trace["cognitive_engine_grounded_recovery_attempted"] is True
