import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.types import ThinkingMode
from core.config import get_config

@pytest.fixture
def config():
    cfg = get_config()
    cfg.llm.deep_model = "gemini-2.5-pro"
    cfg.llm.gemini_api_key = "test" + "_key_" + "123"
    cfg.llm.fast_model = "qwen3:8b"
    return cfg

@pytest.fixture
def engine():
    from core.container import get_container
    container = get_container()
    container.reset()
    
    mock_router = AsyncMock()
    mock_router.think = AsyncMock(return_value="Router response.")
    mock_router.last_tier = "unknown"
    container.register_instance("llm_router", mock_router)
    
    mock_state = MagicMock()
    mock_state.cognition.working_memory = []
    mock_state.cognition.last_thought_at = 1710450000.0 # Some timestamp
    
    # Affect attributes need to be numbers for AffectUpdatePhase
    mock_state.affect.valence = 0.0
    mock_state.affect.arousal = 0.5
    mock_state.affect.curiosity = 0.5
    mock_state.affect.social_hunger = 0.5
    
    # mock_state.derive should return a new MagicMock but for simplicity:
    mock_state.derive.return_value = mock_state
    
    # CognitiveRoutingPhase sets cognition.current_mode
    mock_state.cognition.current_mode = None 
    
    mock_repo = AsyncMock()
    mock_repo.get_current.return_value = mock_state
    mock_repo.commit = AsyncMock()
    container.register_instance("state_repository", mock_repo)
    
    eng = CognitiveEngine()
    eng.setup() # Initialize phases
    
    # Store refs for easy assertion
    eng._mock_router = mock_router
    eng._mock_state = mock_state
    
    return eng

