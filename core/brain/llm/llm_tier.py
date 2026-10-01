"""The quality tiers a local model call is routed by.

Lifted whole out of `llm_router`, which imports them straight back, so every
caller that names them there still finds them. Fourteen of the modules that
imported the router took these and nothing else, and each depended on the
router for a vocabulary.
"""
from __future__ import annotations

from enum import StrEnum


class LLMTier(StrEnum):
    """LLM quality tiers"""

    PRIMARY = "primary"          # Local powerful, best quality
    SECONDARY = "secondary"      # Local medium, good quality
    TERTIARY = "tertiary"        # Local lightweight, basic quality
    EMERGENCY = "emergency"      # Fallback to rule-based


class LLMTierAlias:
    """Compatibility labels for local tiers; they do not denote remote APIs."""
    API_DEEP   = "api_deep"
    API_FAST   = "api_fast"
    LOCAL      = "local"
    EMERGENCY  = "emergency"
