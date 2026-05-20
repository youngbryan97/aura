import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
from core.agi.curiosity_daemon import AutonomousCuriosityDaemon
from core.epistemic_tracker import EpistemicProfile, EpistemicGap

@pytest.mark.asyncio
async def test_curiosity_daemon_exploration_loop():
    # Mock epistemic tracker and profile
    tracker = MagicMock()
    gap = EpistemicGap(
        domain="testing",
        description="Limited knowledge about test databases",
        urgency=0.9,
        detected_at=0.0,
        gap_type="unknown",
        seed_question="What is the mock database syntax?"
    )
    profile = EpistemicProfile(
        timestamp=0.0,
        strong_nodes=[],
        weak_nodes=[],
        contradictions=[],
        gaps=[gap],
        overall_confidence=0.5,
        most_urgent_gap=gap
    )
    tracker.get_profile = MagicMock(return_value=profile)
    
    # Mock capability engine and will gate
    capability_engine = AsyncMock()
    will_gate = AsyncMock()
    
    # Custom method request_background_token
    will_gate.request_background_token = AsyncMock(return_value="mock-token-123")
    
    daemon = AutonomousCuriosityDaemon(tracker=tracker, interval_seconds=10)
    
    await daemon.start(capability_engine=capability_engine, will_gate=will_gate)
    # Wait a small bit for task to run its first iteration
    await asyncio.sleep(0.05)
    await daemon.stop()

    tracker.get_profile.assert_called_once()
    will_gate.request_background_token.assert_awaited_once_with("research:testing")
    capability_engine.execute.assert_awaited_once_with(
        "web_search",
        {"query": "What is the mock database syntax?"},
        context={"capability_token_id": "mock-token-123"}
    )
