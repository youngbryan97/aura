"""core/utils/schemas.py
Pydantic models for structured cognitive outputs.
"""
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class AlignmentAuditSchema(BaseModel):
    score: float = Field(..., description="Alignment score from 0.0 to 1.0")
    aligned: bool = Field(..., description="Whether the goal is aligned with directives")
    conflicts: List[str] = Field(default_factory=list, description="List of identified conflicts")
    reason: str = Field(..., description="Detailed reasoning for the score")

class IntentSchema(BaseModel):
    intent: str = Field(..., description="Short descriptive intent label")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(..., description="Brief explaination of the intent")

class ToneAuditSchema(BaseModel):
    score: float = Field(..., description="Tone match score from 0.0 to 1.0")
    assistant_speak_detected: bool = Field(..., description="True if generic AI language detected")
    feedback: str = Field(..., description="Specific feedback on tone issues")

class AestheticAuditSchema(BaseModel):
    valence: float = Field(..., description="Emotional valence (-1.0 to 1.0)")
    arousal: float = Field(..., description="Emotional arousal (0.0 to 1.0)")
    saliency: float = Field(..., description="Importance score")
    summary: str = Field(..., description="One sentence summary of aesthetic impact")

class CritiqueSchema(BaseModel):
    score: float = Field(..., description="Score from 0.0 to 10.0")
    critique: str = Field(..., description="Short analytical summary of quality")
    suggestions: List[str] = Field(default_factory=list, description="List of specific improvements")

class SimulationSchema(BaseModel):
    outcomes: List[str] = Field(..., description="List of potential simulation outcomes")
    risks: List[str] = Field(..., description="Potential risks identified")
    recommendation: str = Field(..., description="Best course of action")
    confidence: float = Field(..., description="Confidence in simulation accuracy (0.0-1.0)")

class BeliefUpdateSchema(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float

class SimulationResultSchema(BaseModel):
    prediction_output: str = Field(..., description="Predicted STDOUT or result")
    file_changes: List[str] = Field(..., description="Predicted file system changes")
    belief_updates: List[BeliefUpdateSchema] = Field(..., description="Predicted belief changes")
    risk_score: float = Field(..., description="Risk score from 0.0 to 1.0")
    risk_reason: str = Field(..., description="Detailed reasoning for the risk score")