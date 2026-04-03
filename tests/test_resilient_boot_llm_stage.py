from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.ops.resilient_boot import ResilientBoot


class _DummyOrchestrator:
    pass


@pytest.mark.asyncio
async def test_stage_llm_prepares_client_without_warmup():
    boot = ResilientBoot(_DummyOrchestrator())
    client = MagicMock()
    client.warmup = AsyncMock()

    with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=client) as get_client:
        with patch("core.brain.llm.model_registry.get_local_backend", return_value="mlx"):
            with patch("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await boot._stage_llm()

    get_client.assert_called_once_with(model_path="/models/active")
    client.warmup.assert_not_awaited()
