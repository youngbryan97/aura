from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.llm import qualified_recurrent_ingress as ingress
from core.learning.frontier_process_supervision import frontier_process_task_battery
from interface.routes import chat_preflight


def _semantic_task():
    return frontier_process_task_battery(
        ("calibration",),
        (1,),
        1,
        seed=2026082105,
    )[0]


def _active_status(family: str) -> dict:
    return {
        "active": True,
        "receipt": {
            "allowed_families": [family],
            "allowed_surface_profiles": [],
        },
    }


def test_state_native_owner_requires_active_signed_family(monkeypatch):
    from core.brain.llm import semantic_neural_serving

    task = _semantic_task()
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None

    monkeypatch.setattr(
        semantic_neural_serving,
        "semantic_neural_default_serving_status",
        lambda: _active_status(admission.family),
    )
    profile, owner = chat_preflight._chat_evidence_profile(
        task.prompt,
        bounded_surface=False,
    )
    assert profile == chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
    assert owner == admission

    monkeypatch.setattr(
        semantic_neural_serving,
        "semantic_neural_default_serving_status",
        lambda: {"active": False, "reason": "source_drift"},
    )
    profile, owner = chat_preflight._chat_evidence_profile(
        task.prompt,
        bounded_surface=False,
    )
    assert profile == chat_preflight._CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE
    assert owner is None


@pytest.mark.asyncio
async def test_state_native_preflight_does_not_collect_unconsumed_evidence(monkeypatch):
    task = _semantic_task()
    admission = ingress.admit_qualified_recurrent_objective(task.prompt)
    assert admission is not None
    monkeypatch.setattr(
        chat_preflight,
        "_chat_evidence_profile",
        lambda *_args, **_kwargs: (
            chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT,
            admission,
        ),
    )

    body = SimpleNamespace(message=task.prompt, session_id="evidence-owner")
    result = await chat_preflight._run_chat_preflight(
        body,
        SimpleNamespace(client=SimpleNamespace(host="test")),
        task.prompt,
        "bryan",
        False,
        False,
        _chat_session_id="default",
        _grounded_recall_context="",
        raw_user_message=task.prompt,
    )

    assert body.message == task.prompt
    assert result.evidence_profile == (
        chat_preflight._CHAT_EVIDENCE_PROFILE_QUALIFIED_RECURRENT
    )
    assert result.evidence_owner_receipt == admission.receipt()
    assert result.skipped_components == (
        chat_preflight._QUALIFIED_RECURRENT_SKIPPED_PREFLIGHT_COMPONENTS
    )
    assert result.turn_sensory_evidence is None


def test_general_chat_keeps_contextual_language_evidence_profile(monkeypatch):
    from core.brain.llm import semantic_neural_serving

    monkeypatch.setattr(
        semantic_neural_serving,
        "semantic_neural_default_serving_status",
        lambda: pytest.fail("unsupported language must not inspect serving artifacts"),
    )
    profile, owner = chat_preflight._chat_evidence_profile(
        "How are you feeling about the work today?",
        bounded_surface=False,
    )
    assert profile == chat_preflight._CHAT_EVIDENCE_PROFILE_CONTEXTUAL_LANGUAGE
    assert owner is None
