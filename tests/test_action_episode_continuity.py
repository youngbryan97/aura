from __future__ import annotations

import pytest

from core.conversation.action_episode import (
    ActionEpisode,
    action_episode_from_execution,
    action_episode_grounding,
    select_action_episode,
)
from core.conversation.request_mood import RequestMood, assess_request_mood
from core.language.action_outcome import action_outcome_question
from core.runtime.turn_analysis import analyze_turn


@pytest.mark.parametrize(
    ("text", "failure"),
    [
        ("What went wrong there?", True),
        ("Do you know why it broke?", True),
        ("Could you explain why the wallpaper attempt failed?", True),
        ("How did that go?", False),
        ("What happened with the export?", False),
    ],
)
def test_action_outcome_questions_are_retrospective_chat(
    text: str,
    failure: bool,
) -> None:
    relation = action_outcome_question(text)

    assert relation.asks_about_outcome is True
    assert relation.asks_about_failure is failure
    mood = assess_request_mood(text)
    assert mood.mood is RequestMood.MENTION
    assert mood.temporal_scope == "retrospective"
    assert analyze_turn(text).intent_type == "CHAT"


@pytest.mark.parametrize(
    "text",
    [
        "The build failed overnight.",
        "Can you fix the broken export tool?",
        "Why would a browser fail in theory?",
        "Suppose the task failed; what would happen?",
    ],
)
def test_action_outcome_language_abstains_without_a_prior_outcome_question(
    text: str,
) -> None:
    assert action_outcome_question(text).asks_about_outcome is False


def test_governed_execution_becomes_a_bounded_action_episode() -> None:
    episode = action_episode_from_execution(
        "Find an image and set it as my wallpaper.",
        {
            "ok": False,
            "status": "desktop_objective_failed",
            "response": "The browser step did not complete.",
            "result": {
                "ok": False,
                "status": "desktop_task_failed",
                "error": "active browser target could not be verified",
                "steps_requested": 2,
                "steps_completed": 0,
                "receipts": [{"receipt_id": "receipt-1"}],
            },
        },
        capability="desktop_task",
    )

    assert episode is not None
    assert episode.succeeded is False
    assert episode.failure_kind == "error"
    assert episode.failure_detail == "active browser target could not be verified"
    assert episode.steps_completed == 0
    assert episode.steps_requested == 2
    assert episode.evidence_refs == ("receipt-1",)
    grounding = action_episode_grounding(episode)
    assert "objective: Find an image and set it as my wallpaper." in grounding
    assert "failure_detail: active browser target could not be verified" in grounding


def test_failure_question_selects_relevant_failed_episode_not_latest_success() -> None:
    failed_wallpaper = ActionEpisode(
        objective="Find an image online and set it as my wallpaper.",
        capability="desktop_task",
        status="desktop_objective_failed",
        succeeded=False,
        failure_kind="error",
        failure_detail="browser target unavailable",
        recorded_at=1.0,
    )
    failed_export = ActionEpisode(
        objective="Export the research synthesis as a PDF.",
        capability="desktop_task",
        status="desktop_objective_failed",
        succeeded=False,
        failure_kind="error",
        failure_detail="PDF renderer unavailable",
        recorded_at=2.0,
    )
    later_success = ActionEpisode(
        objective="Open Notes.",
        capability="desktop_task",
        status="desktop_objective_completed",
        succeeded=True,
        recorded_at=3.0,
    )

    selected = select_action_episode(
        "Why did the wallpaper attempt fail?",
        [failed_wallpaper, failed_export, later_success],
    )

    assert selected is failed_wallpaper


@pytest.mark.asyncio
async def test_exchange_metadata_reaches_recent_context_and_unified_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.conversation.unified_transcript import UnifiedTranscript
    from interface.routes import chat, chat_common, chat_memory_state, chat_preflight

    chat_common._conversation_log.clear()
    transcript = UnifiedTranscript.get_instance()
    monkeypatch.setattr(transcript, "_entries", [])
    monkeypatch.setattr(chat_preflight, "_persist_pending_conversation_user", _committed)
    monkeypatch.setattr(chat_preflight, "_persist_completed_conversation_exchange", _committed_exchange)
    monkeypatch.setattr(chat_preflight, "_conversation_memory_outbox_available", lambda: False)

    exchange_id = await chat_preflight._begin_logged_exchange(
        "Set the wallpaper.",
        session_id="action-episode-test",
    )
    episode = ActionEpisode(
        objective="Set the wallpaper.",
        capability="desktop_task",
        status="desktop_objective_failed",
        succeeded=False,
        failure_kind="error",
        failure_detail="browser target unavailable",
        recorded_at=1.0,
    )
    assert await chat_preflight._attach_logged_exchange_metadata(
        exchange_id,
        {"action_episode": episode.to_dict()},
    )
    await chat_preflight._complete_logged_exchange(
        exchange_id,
        "Set the wallpaper.",
        "I could not complete it.",
        record_experience=False,
    )

    recent = await chat_memory_state._recent_completed_conversation_exchanges(
        current_user_message="Why did it fail?",
        session_id="action-episode-test",
        limit=4,
        allow_cross_session=False,
    )
    assert recent[-1]["action_episode"]["failure_detail"] == "browser target unavailable"
    grounding = await chat._resolve_action_episode_grounding(
        "Do you know why that broke?",
        session_id="action-episode-test",
    )
    assert "failure_detail: browser target unavailable" in grounding
    aura_entries = [
        entry
        for entry in transcript.get_context_window(
            4,
            conversation_id="action-episode-test",
        )
        if entry.role == "aura"
    ]
    assert aura_entries[-1].metadata["action_episode"]["status"] == "desktop_objective_failed"


async def _committed(*_args, **_kwargs) -> str:
    return "committed"


async def _committed_exchange(**_kwargs) -> str:
    return "committed"
