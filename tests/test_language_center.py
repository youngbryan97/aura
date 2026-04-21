from unittest.mock import AsyncMock, MagicMock

import pytest

from core.inner_monologue import ThoughtPacket
from core.language_center import LanguageCenter


@pytest.mark.asyncio
async def test_language_center_dispatches_expression_as_messages():
    router = MagicMock()
    router.generate = AsyncMock(return_value="Sharp answer.")

    center = LanguageCenter()
    center._router = router

    thought = ThoughtPacket(
        stance="Here's the point.",
        primary_points=["Say the point clearly."],
        model_tier="primary",
        tone="direct",
        length_target="brief",
        llm_briefing="SYSTEM BRIEF",
    )

    result = await center.express(
        thought,
        "What do you think?",
        history=[
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
        ],
    )

    assert result == "Sharp answer."
    router.generate.assert_awaited_once()
    kwargs = router.generate.await_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "SYSTEM BRIEF"},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "What do you think?"},
    ]
    assert kwargs["prefer_tier"] == "primary"
    assert kwargs["purpose"] == "expression"
    assert kwargs["origin"] == "user"
    assert kwargs["is_background"] is False


@pytest.mark.asyncio
async def test_language_center_can_mark_autonomous_expression_as_background():
    router = MagicMock()
    router.generate = AsyncMock(return_value="Quiet reflection.")

    center = LanguageCenter()
    center._router = router

    thought = ThoughtPacket(
        stance="Reflect quietly.",
        primary_points=["Stay internal."],
        model_tier="primary",
        tone="thoughtful",
        length_target="brief",
        llm_briefing="SYSTEM BRIEF",
    )

    result = await center.express(
        thought,
        "What should I explore next?",
        origin="autonomous",
    )

    assert result == "Quiet reflection."
    kwargs = router.generate.await_args.kwargs
    assert kwargs["origin"] == "autonomous"
    assert kwargs["is_background"] is True
