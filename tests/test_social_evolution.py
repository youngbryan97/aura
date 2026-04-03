import asyncio
import logging
import pytest
from core.state.aura_state import AuraState
from core.phases.inference_phase import InferencePhase
from core.phases.bonding_phase import BondingPhase
from core.phases.social_context_phase import SocialContextPhase
from core.phases.repair_phase import RepairPhase

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Test.SocialEvolution")

class MockRouter:
    async def route(self, prompt, **kwargs):
        import json
        return json.dumps({
            "implicit_intent": "Connecting and sharing state",
            "affective_subtext": "Warmth",
            "momentum": 0.8,
            "conversation_hooks": ["your day", "feeling about projects"]
        })

