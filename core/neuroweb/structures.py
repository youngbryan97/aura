from dataclasses import dataclass
from typing import Optional


@dataclass
class Neuron:
    id: str
    type: str # 'skill', 'concept', 'motor'
    path: str # Import path or identifier

@dataclass
class Synapse:
    id: str
    intent_pattern: str
    neuron_id: str
    strength: float
    created_at: float
    status: str

@dataclass
class Intent:
    text: str
    confidence: float
