import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path
from core.senses.continuous_vision import ContinuousSensoryBuffer
from core.skills.visual_context_skill import VisualContextSkill
from core.container import ServiceContainer

