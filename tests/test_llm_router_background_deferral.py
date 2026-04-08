from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier


@pytest.mark.asyncio
async def test_legacy_llm_router_defers_background_inference_when_gate_is_guarded():
    router = IntelligentLLMRouter()
    tertiary = MagicMock()
    tertiary.think = AsyncMock(return_value=(True, "7B response", {}))
    router.register_endpoint(
        LLMEndpoint(
            name="Brainstem",
            tier=LLMTier.TERTIARY,
            model_name="brainstem-7b",
            client=tertiary,
        )
    )

    fake_gate = MagicMock()
    fake_gate._background_local_deferral_reason.return_value = "cortex_startup_quiet"

    with patch("core.container.ServiceContainer.get", side_effect=lambda name, default=None: fake_gate if name == "inference_gate" else default):
        result = await router.think(
            "Idle thought",
            prefer_tier="tertiary",
            origin="system",
            is_background=True,
        )

    assert result == ""
    tertiary.think.assert_not_called()
