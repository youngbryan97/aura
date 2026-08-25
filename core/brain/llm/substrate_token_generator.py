
"""Substrate-first token generation.

This module makes the continuous substrate the first computation attempted for
lightweight generation. The transformer becomes the fallback cortex when the
substrate's own prediction error is too high for the requested prompt.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("core.brain.llm.substrate_token_generator")


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
    steering_telemetry: dict[str, Any | None] = None
    #: The vocabulary this text is drawn from. PROTO_TOKENS is 32 words and the
    #: readout mapping onto them is an UNTRAINED random projection, so its
    #: output ("Substrate path: world action hold grounded choose loop result
    #: repair.") is a state fingerprint, not language. Anything that puts text
    #: in front of a person must check this.
    vocabulary: str = "proto"

    @property
    def is_user_presentable(self) -> bool:
        """Whether this text may be shown to a person as an answer.

        MEASURED 2026-08-04: the substrate-first path is enabled for
        user-facing turns by default (AURA_SUBSTRATE_PRIMARY_USER=1), the
        threshold is 0.34, and a short prompt whose hashed vector aligns with
        the live state reaches a prediction error of 0.157 — so proto-token
        output was reachable as a live reply. It is a diagnostic of substrate
        state and reads as word salad. The trained readout lives in
        :mod:`core.brain.llm.endogenous_vocab_head` and shapes the
        transformer's own distribution rather than replacing it, so this
        answers no for the proto vocabulary and always will.
        """
        return self.used_substrate and self.vocabulary != "proto"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["generated_at"] = self.generated_at or time.time()
        # Travels with the record: whether this ran, and whether its text may
        # be shown to a person, are two different questions.
        payload["is_user_presentable"] = self.is_user_presentable
        return payload


class SubstrateTokenGenerator:
    """A readout over the live substrate state.

    The head is a random projection (``_ensure_readout`` seeds it from numpy
    and never trains it) onto a 32-word proto vocabulary, so what it produces
    is a deterministic fingerprint of substrate state — useful for telemetry,
    for tests, and for proving the substrate is doing something without loading
    the transformer. It is NOT language, and ``SubstrateGeneration``
    distinguishes the two: ``used_substrate`` says the path ran,
    ``is_user_presentable`` says whether its text may be shown to a person.

    A real substrate-first readout means a trained head over the model
    vocabulary, and that is now a separate pathway rather than a plan: see
    :mod:`core.brain.llm.endogenous_vocab_head`, which maps the named
    cognitive channels of :mod:`core.brain.llm.endogenous_state` to a bounded
    bias over the resident model's own vocabulary inside its plausible set.
    Nothing here claims to be that. This stays what it is — a fingerprint for
    telemetry and for tests, and a way to show the substrate is doing
    something without loading the transformer.
    """

    #: The vocabulary every generation from this class is drawn from. A class
    #: attribute, not a per-result one, because a caller needs to know whether
    #: the output could EVER be shown to a person before paying for it.
    VOCABULARY = "proto"

    @classmethod
    def can_be_shown_to_a_person(cls) -> bool:
        """Whether this readout's text could ever be served as an answer.

        False for the proto vocabulary, permanently. Callers on a user-facing
        turn use this to skip the readout entirely rather than computing one
        and then discarding it, which is what they were doing.
        """
        return cls.VOCABULARY != "proto"

    def __init__(
        self,
        substrate: Any,
        *,
        vocab_size: int | None = None,
        seed: int = 913,
        threshold: float | None = None,
    ) -> None:
        self.substrate = substrate
        self.seed = int(seed)
        # A non-finite/out-of-range threshold would make every error>threshold
        # comparison degenerate (NaN comparisons are always False), enabling
        # every substrate response. Clamp to a valid [0,1] gate.
        raw_threshold = (
            threshold if threshold is not None
            else os.getenv("AURA_SUBSTRATE_PREDICTION_THRESHOLD", "0.34")
        )
        self.threshold = self._finite_unit(raw_threshold, 0.34)
        # Bound vocab_size so a negative/huge value cannot fail unpredictably or
        # allocate an unbounded readout matrix.
        try:
            requested_vocab = int(vocab_size or len(PROTO_TOKENS))
        except (TypeError, ValueError):
            requested_vocab = len(PROTO_TOKENS)
        self._vocab_size = max(1, min(65536, requested_vocab))
        self._readout: np.ndarray | None = None
        self.last_generation: SubstrateGeneration | None = None

    @staticmethod
    def _finite_unit(value: Any, default: float) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(v):
            return default
        return max(0.0, min(1.0, v))

    # A single substrate state must not exceed a sane dimension — a huge or
    # malformed vector would poison the readout matmul and receipt.
    _MAX_STATE_DIM = 16384

    def _state_vector(self) -> np.ndarray:
        getter = getattr(self.substrate, "get_state_vector", None)
        if callable(getter):
            state = np.asarray(getter(), dtype=np.float32).ravel()
        elif hasattr(self.substrate, "x"):
            state = np.asarray(self.substrate.x, dtype=np.float32).ravel()
        else:
            state = np.zeros(64, dtype=np.float32)
        if state.size == 0:
            state = np.zeros(64, dtype=np.float32)
        if state.size > self._MAX_STATE_DIM:
            state = state[: self._MAX_STATE_DIM]
        # Replace non-finite elements (NaN/inf) with 0 so a corrupt substrate
        # cannot propagate NaN through tanh into every logit.
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
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

    def estimate_prediction_error(self, prompt: str, *, state: np.ndarray | None = None) -> float:
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
        threshold: float | None = None,
    ) -> SubstrateGeneration:
        state = self._state_vector()
        error = self.estimate_prediction_error(prompt, state=state)
        active_threshold = float(threshold if threshold is not None else self.threshold)
        state_energy = min(1.0, float(np.linalg.norm(state)) / math.sqrt(max(1, state.size)))

        logits = self.logits(prompt)
        checksum = hashlib.blake2b(logits.astype(np.float32).tobytes(), digest_size=10).hexdigest()

        telemetry = None
        try:
            engine = get_runtime_service("affective_steering_engine", default=None)
            if engine and hasattr(engine, "telemetry"):
                import dataclasses
                telemetry = dataclasses.asdict(engine.telemetry)
        except (ImportError, AttributeError, RuntimeError) as _exc:
            logger.debug("Suppressed %s in core.brain.llm.substrate_token_generator: %s", type(_exc).__name__, _exc)

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
                steering_telemetry=telemetry,
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
            steering_telemetry=telemetry,
        )
        self.last_generation = result
        return result


def get_substrate_token_generator(substrate: Any | None = None) -> SubstrateTokenGenerator:
    if substrate is None:
        substrate = (
            get_runtime_service("continuous_substrate", default=None)
            or get_runtime_service("liquid_state", default=None)
        )
    existing = get_runtime_service("substrate_token_generator", default=None)
    if existing is not None and (substrate is None or getattr(existing, "substrate", None) is substrate):
        return existing
    if substrate is None:
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        substrate = ContinuousSubstrate()
    generator = SubstrateTokenGenerator(substrate)
    try:
        register_runtime_service(
            "substrate_token_generator",
            generator,
            required=False,
            owner="core/brain/llm/substrate_token_generator.py",
            registered_by="get_substrate_token_generator",
        )
    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
        logger.debug("Suppressed %s in core.brain.llm.substrate_token_generator: %s", type(_exc).__name__, _exc)
    return generator


__all__ = [
    "SubstrateGeneration",
    "SubstrateTokenGenerator",
    "get_substrate_token_generator",
]
