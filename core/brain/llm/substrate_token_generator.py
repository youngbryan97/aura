"""Substrate-first token generation.

This module makes the continuous substrate the first computation attempted for
lightweight generation. The transformer becomes the fallback cortex when the
substrate's own prediction error is too high for the requested prompt.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np


PROTO_TOKENS = (
    "notice", "hold", "verify", "repair", "continue", "care", "check",
    "evidence", "thread", "quiet", "active", "grounded", "memory",
    "will", "receipt", "action", "observe", "choose", "test", "learn",
    "steady", "curious", "cautious", "ready", "loop", "state", "world",
    "plan", "result", "trace", "signal", "budget",
)


@dataclass
class SubstrateGeneration:
    used_substrate: bool
    text: str
    token_ids: list[int]
    prediction_error: float
    threshold: float
    logits_checksum: str
    fallback_reason: str = ""
    state_energy: float = 0.0
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generated_at"] = self.generated_at or time.time()
        return payload


class SubstrateTokenGenerator:
    """A learned-readout style head over the live substrate state.

    In production, a tokenizer can be attached and the readout can target the
    model vocabulary. In CPU tests and emergency paths, a compact proto-token
    vocabulary gives deterministic state-dependent output without loading the
    transformer.
    """

    def __init__(
        self,
        substrate: Any,
        *,
        vocab_size: Optional[int] = None,
        seed: int = 913,
        threshold: Optional[float] = None,
    ) -> None:
        self.substrate = substrate
        self.seed = int(seed)
        self.threshold = (
            float(threshold)
            if threshold is not None
            else float(os.getenv("AURA_SUBSTRATE_PREDICTION_THRESHOLD", "0.34"))
        )
        self._vocab_size = int(vocab_size or len(PROTO_TOKENS))
        self._readout: Optional[np.ndarray] = None
        self.last_generation: Optional[SubstrateGeneration] = None

    def _state_vector(self) -> np.ndarray:
        getter = getattr(self.substrate, "get_state_vector", None)
        if callable(getter):
            state = np.asarray(getter(), dtype=np.float32).ravel()
        elif hasattr(self.substrate, "x"):
            state = np.asarray(getattr(self.substrate, "x"), dtype=np.float32).ravel()
        else:
            state = np.zeros(64, dtype=np.float32)
        if state.size == 0:
            state = np.zeros(64, dtype=np.float32)
        return np.tanh(state).astype(np.float32)

    def _ensure_readout(self, state_dim: int) -> np.ndarray:
        if self._readout is not None and self._readout.shape == (self._vocab_size, state_dim):
            return self._readout
        rng = np.random.default_rng(self.seed + state_dim * 31 + self._vocab_size)
        self._readout = (
            rng.standard_normal((self._vocab_size, state_dim)).astype(np.float32)
            / math.sqrt(max(1, state_dim))
        )
        return self._readout

    @staticmethod
    def _prompt_vector(prompt: str, *, dim: int) -> np.ndarray:
        raw = str(prompt or "").encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(raw, digest_size=32).digest()
        vec = np.zeros(dim, dtype=np.float32)
        for i, byte in enumerate(digest):
            vec[(byte + i * 13) % dim] += (1.0 if byte & 1 else -1.0) * (byte / 255.0)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    def estimate_prediction_error(self, prompt: str, *, state: Optional[np.ndarray] = None) -> float:
        s = self._state_vector() if state is None else np.asarray(state, dtype=np.float32).ravel()
        if s.size == 0:
            return 1.0
        p = self._prompt_vector(prompt, dim=s.size)
        alignment = float(np.dot(s, p) / (np.linalg.norm(s) * np.linalg.norm(p) + 1e-6))
        token_count = len(str(prompt or "").split())
        complexity = min(0.45, token_count / 80.0)
        interrogative = 0.10 if any(ch in str(prompt or "") for ch in "?\n") else 0.0
        state_energy = min(1.0, float(np.linalg.norm(s)) / math.sqrt(max(1, s.size)))
        low_energy_penalty = max(0.0, 0.18 - state_energy)
        error = 0.42 - 0.28 * max(-1.0, min(1.0, alignment)) + complexity + interrogative + low_energy_penalty
        return max(0.0, min(1.0, error))

    def logits(self, prompt: str) -> np.ndarray:
        state = self._state_vector()
        readout = self._ensure_readout(state.size)
        prompt_bias = self._prompt_vector(prompt, dim=self._vocab_size) * 0.15
        return (readout @ state) + prompt_bias

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 24,
        force: bool = False,
        threshold: Optional[float] = None,
    ) -> SubstrateGeneration:
        state = self._state_vector()
        error = self.estimate_prediction_error(prompt, state=state)
        active_threshold = float(threshold if threshold is not None else self.threshold)
        state_energy = min(1.0, float(np.linalg.norm(state)) / math.sqrt(max(1, state.size)))

        logits = self.logits(prompt)
        checksum = hashlib.blake2b(logits.astype(np.float32).tobytes(), digest_size=10).hexdigest()

        if not force and error > active_threshold:
            result = SubstrateGeneration(
                used_substrate=False,
                text="",
                token_ids=[],
                prediction_error=round(error, 6),
                threshold=active_threshold,
                logits_checksum=checksum,
                fallback_reason="prediction_error_exceeded",
                state_energy=round(state_energy, 6),
                generated_at=time.time(),
            )
            self.last_generation = result
            return result

        k = max(1, min(int(max_tokens or 24), min(12, self._vocab_size)))
        token_ids = list(np.argsort(logits)[-k:][::-1].astype(int))
        words = [PROTO_TOKENS[i % len(PROTO_TOKENS)] for i in token_ids]
        text = "Substrate path: " + " ".join(words[:8]) + "."
        result = SubstrateGeneration(
            used_substrate=True,
            text=text,
            token_ids=token_ids,
            prediction_error=round(error, 6),
            threshold=active_threshold,
            logits_checksum=checksum,
            state_energy=round(state_energy, 6),
            generated_at=time.time(),
        )
        self.last_generation = result
        return result


def get_substrate_token_generator(substrate: Any | None = None) -> SubstrateTokenGenerator:
    from core.container import ServiceContainer

    if substrate is None:
        substrate = (
            ServiceContainer.get("continuous_substrate", default=None)
            or ServiceContainer.get("liquid_state", default=None)
        )
    existing = ServiceContainer.get("substrate_token_generator", default=None)
    if existing is not None and (substrate is None or getattr(existing, "substrate", None) is substrate):
        return existing
    if substrate is None:
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate)
    try:
        ServiceContainer.register_instance("substrate_token_generator", generator, required=False)
    except Exception:
        pass
    return generator


__all__ = [
    "SubstrateGeneration",
    "SubstrateTokenGenerator",
    "get_substrate_token_generator",
]
