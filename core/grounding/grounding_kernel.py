"""Sensory kernel — turns raw observations into PerceptualEvidence.

Version 1 supports text via a deterministic hash-token feature
encoder.  Vision/audio encoders (CLIP, image classifiers) drop in by
adding new modality branches; the encoder contract returns a
fixed-dimensional feature vector regardless of input type.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.grounding.types import GroundingMethod, PerceptualEvidence, new_id


def hash_features(text: str, dim: int = 128) -> list:
    vec = np.zeros(dim, dtype=np.float32)
    for token in str(text).lower().split():
        h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec /= norm
    return vec.tolist()


@dataclass
class GroundingObservation:
    symbol: str
    modality: str
    raw: Any
    source: str = "user"
    label_confirmed: bool | None = None


class GroundingKernel:
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = int(feature_dim)

    def default_text_method(self) -> GroundingMethod:
        return GroundingMethod(
            method_id="method_text_hash_v1",
            name="Text hash feature encoder",
            kind="textual",
            confidence_floor=0.55,
            metadata={"feature_dim": self.feature_dim},
        )

    def encode(self, observation: GroundingObservation) -> PerceptualEvidence:
        if observation.modality == "text":
            features = hash_features(str(observation.raw), self.feature_dim)
        else:
            # Placeholder: real implementations override per modality.
            features = hash_features(str(observation.raw), self.feature_dim)
        return PerceptualEvidence(
            evidence_id=new_id("evidence"),
            modality=observation.modality,
            features=features,
            raw_ref=str(observation.raw)[:512],
            metadata={
                "source": observation.source,
                "symbol": observation.symbol,
                "label_confirmed": observation.label_confirmed,
            },
        )
