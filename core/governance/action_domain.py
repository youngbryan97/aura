"""The kinds of action the Will decides on.

Lifted whole out of `will`, which imports it straight back, so every caller
that names it there still finds it. Twenty of the modules that imported the
Will took this enum and nothing else, and each depended on the whole
governance engine for a vocabulary.
"""
from __future__ import annotations

from enum import StrEnum


class ActionDomain(StrEnum):
    """What kind of action is being decided on."""
    RESPONSE = "response"               # sending a reply to the user
    TOOL_EXECUTION = "tool_execution"   # external tool / skill dispatch
    MEMORY_WRITE = "memory_write"       # episodic, semantic, belief mutation
    INITIATIVE = "initiative"           # autonomous goal / impulse
    STATE_MUTATION = "state_mutation"    # internal state change
    EXPRESSION = "expression"           # spontaneous output
    EXPLORATION = "exploration"         # novelty-seeking action
    STABILIZATION = "stabilization"     # rest / recovery action
    REFLECTION = "reflection"           # internal reflection / metacognition
    SEMANTIC_WEIGHT_UPDATE = "semantic_weight_update"  # plastic adapter update
    BELIEF_UPDATE = "belief_update"       # explicit belief graph update
    ENVIRONMENT_ACTION = "environment_action"  # embodied/digital environment action
    EXTERNAL_ACTION = "external_action"   # externally visible side effect
    FILE_WRITE = "file_write"             # persistent filesystem mutation
    NETWORK_CALL = "network_call"         # network or browser action
    CLOUD_CALL = "cloud_call"             # cloud/provider side effect
    CI_CD = "ci_cd"                       # CI/CD and deployment authority
    SELF_MODIFICATION = "self_modification"  # code/architecture mutation
    CLOUD_FALLBACK = "cloud_fallback"     # Falling back to cloud LLM APIs
