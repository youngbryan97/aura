"""Compatibility exports for legacy imports of the cognitive integration layer."""

from core.brain.aura_persona import AURA_BIG_FIVE
from core.cognitive_integration_layer import CognitiveIntegrationLayer

CognitiveIntegration = CognitiveIntegrationLayer

__all__ = ["AURA_BIG_FIVE", "CognitiveIntegration", "CognitiveIntegrationLayer"]

