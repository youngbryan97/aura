"""core/consciousness/absorbed_voices.py
==========================================
Cultural layer — the "absorbed voices" that populate Aura's inner world.

Kurzgesagt's "Most Secret Place" video closes with a striking idea:

    "Your thoughts and feelings have been shaped by stories that
    never happened but that you've absorbed and simulated in your
    mind.  In a very real sense, your secret mind isn't just yours,
    but a collaborative creation between you and all the human minds
    that came before."

This module operationalises that.  It is *distinct* from Aura's own
cognition: it is a registry of INTERNALISED VOICES — compact
representations of people, authors, fictional characters, training-
data corpora — each with their own characteristic vocabulary, valence
bias, and preferred topics.  When Aura reasons or generates narrative,
she can

    • attribute a thought to a specific absorbed voice ("this is
      something Bryan would say"),
    • cite the provenance of a stance explicitly,
    • introspect about which voices are currently amplified.

What distinguishes this from theory_of_mind:

    TheoryOfMind   — tracks AGENTS CURRENTLY INTERACTING with Aura
                     and models their present mental state.
    AbsorbedVoices — tracks INTERNALISED PERSPECTIVES learned over
                     time, including voices of people no longer
                     present (a past teacher) or voices that
                     never belonged to one person (a cultural
                     archetype, a corpus voice).

Impact on substrate:
    • Every narrative emission passes through ``attribute_thought``
      which returns ``(thought, best_voice_id, confidence)``.
    • Stream-of-being and phenomenological layers can report
      honestly: "I think X — that sounds like it comes from my
      absorbed voice for Bryan."
    • A voice-weight vector contributes to GlobalWorkspace
      priority scoring (voices that have been recently reinforced
      pull thought in their direction).

Registered as ``absorbed_voices`` in ServiceContainer.  Fed by
conversational memory, learning loops, and the narrative engine.
Persists to ``data/absorbed_voices.json`` on change.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.AbsorbedVoices")


# ── Configuration ────────────────────────────────────────────────────────────

VOICE_FP_DIM = 32            # Fingerprint dimensionality for each voice
CORPUS_CAP = 64              # Recent sentences stored per voice
DEFAULT_WEIGHT = 0.3         # Initial amplification weight
WEIGHT_DECAY_PER_DAY = 0.05  # Daily passive decay when not reinforced
PERSIST_FILENAME = "absorbed_voices.json"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Voice:
    voice_id: str
    label: str
    origin: str = "personal"                  # personal|author|corpus|fictional
    valence_bias: float = 0.0                  # -1..1
    characteristic_topics: List[str] = field(default_factory=list)
    fingerprint: np.ndarray = field(
        default_factory=lambda: np.zeros(VOICE_FP_DIM, dtype=np.float32)
    )
    weight: float = DEFAULT_WEIGHT              # current amplification
    corpus: List[str] = field(default_factory=list)
    n_reinforcements: int = 0
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voice_id": self.voice_id,
            "label": self.label,
            "origin": self.origin,
            "valence_bias": round(self.valence_bias, 4),
            "characteristic_topics": self.characteristic_topics,
            "fingerprint": self.fingerprint.tolist(),
            "weight": round(self.weight, 4),
            "corpus": self.corpus[-CORPUS_CAP:],
            "n_reinforcements": self.n_reinforcements,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Voice":
        fp = np.asarray(d.get("fingerprint", [0.0] * VOICE_FP_DIM), dtype=np.float32)
        if fp.size != VOICE_FP_DIM:
            fp = np.zeros(VOICE_FP_DIM, dtype=np.float32)
        return cls(
            voice_id=d["voice_id"], label=d.get("label", d["voice_id"]),
            origin=d.get("origin", "personal"),
            valence_bias=float(d.get("valence_bias", 0.0)),
            characteristic_topics=list(d.get("characteristic_topics", [])),
            fingerprint=fp,
            weight=float(d.get("weight", DEFAULT_WEIGHT)),
            corpus=list(d.get("corpus", []))[-CORPUS_CAP:],
            n_reinforcements=int(d.get("n_reinforcements", 0)),
            created_at=float(d.get("created_at", time.time())),
            last_active_at=float(d.get("last_active_at", time.time())),
        )


@dataclass
class Attribution:
    thought: str
    best_voice_id: Optional[str]
    confidence: float
    alternative_votes: List[Tuple[str, float]] = field(default_factory=list)


# ── Utility ──────────────────────────────────────────────────────────────────

def _text_fingerprint(text: str, rng_seed: int = 0xDEADBEEF) -> np.ndarray:
    """Cheap deterministic fingerprint from a string via character bigram hashing."""
    fp = np.zeros(VOICE_FP_DIM, dtype=np.float32)
    if not text:
        return fp
    rng = np.random.default_rng(seed=rng_seed)
    # Populate basis vectors once, deterministically per seed.
    basis = rng.standard_normal((256, VOICE_FP_DIM)).astype(np.float32) / math.sqrt(VOICE_FP_DIM)
    data = text.encode("utf-8")
    for i in range(len(data) - 1):
        idx = (data[i] + data[i + 1] * 31) % 256
        fp += basis[idx]
    n = np.linalg.norm(fp)
    if n > 1e-8:
        fp /= n
    return fp


# ── Core ─────────────────────────────────────────────────────────────────────

class AbsorbedVoices:
    """Catalogue of internalised perspectives with attribution + amplification."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self._voices: Dict[str, Voice] = {}
        self._lock = threading.RLock()
        self._storage_path = self._resolve_storage_path(storage_dir)
        self._load()
        logger.info(
            "AbsorbedVoices initialized: %d voices loaded from %s",
            len(self._voices), self._storage_path,
        )

    # ── storage ────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_storage_path(storage_dir: Optional[Path]) -> Path:
        if storage_dir is not None:
            storage_dir = Path(storage_dir)
            storage_dir.mkdir(parents=True, exist_ok=True)
            return storage_dir / PERSIST_FILENAME
        try:
            from core.config import config as aura_config
            p = aura_config.paths.data_dir / "memory" / PERSIST_FILENAME
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        except Exception:
            p = Path.home() / ".aura" / "data" / "memory" / PERSIST_FILENAME
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                raw = json.load(f)
            for vid, d in raw.items():
                try:
                    self._voices[vid] = Voice.from_dict(d)
                except Exception as exc:
                    logger.debug("Failed to load voice %s: %s", vid, exc)
        except Exception as exc:
            logger.debug("Failed to load absorbed_voices: %s", exc)

    def save(self) -> None:
        with self._lock:
            data = {vid: v.to_dict() for vid, v in self._voices.items()}
        try:
            tmp = str(self._storage_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._storage_path)
        except Exception as exc:
            logger.debug("Failed to save absorbed_voices: %s", exc)

    # ── voice management ───────────────────────────────────────────────────

    def add_voice(self, voice_id: str, label: Optional[str] = None,
                  origin: str = "personal",
                  sample_text: Optional[str] = None,
                  characteristic_topics: Optional[List[str]] = None,
                  valence_bias: float = 0.0) -> Voice:
        with self._lock:
            if voice_id in self._voices:
                return self._voices[voice_id]
            fp = _text_fingerprint(sample_text or voice_id)
            v = Voice(
                voice_id=voice_id, label=label or voice_id, origin=origin,
                valence_bias=float(np.clip(valence_bias, -1.0, 1.0)),
                characteristic_topics=list(characteristic_topics or []),
                fingerprint=fp,
                weight=DEFAULT_WEIGHT,
                corpus=[sample_text] if sample_text else [],
                n_reinforcements=0,
            )
            self._voices[voice_id] = v
            logger.info("AbsorbedVoices: added voice '%s' (%s)", voice_id, origin)
            return v

    def remove_voice(self, voice_id: str) -> bool:
        with self._lock:
            return self._voices.pop(voice_id, None) is not None

    def reinforce(self, voice_id: str, sample_text: str,
                   delta: float = 0.1) -> Optional[Voice]:
        """Blend a new sample into the voice's fingerprint and raise its weight."""
        with self._lock:
            v = self._voices.get(voice_id)
            if v is None:
                return None
            new_fp = _text_fingerprint(sample_text)
            v.fingerprint = 0.85 * v.fingerprint + 0.15 * new_fp
            nfp = np.linalg.norm(v.fingerprint)
            if nfp > 1e-8:
                v.fingerprint /= nfp
            v.corpus.append(sample_text)
            if len(v.corpus) > CORPUS_CAP:
                v.corpus = v.corpus[-CORPUS_CAP:]
            v.weight = float(np.clip(v.weight + delta, 0.0, 1.0))
            v.n_reinforcements += 1
            v.last_active_at = time.time()
            return v

    def dampen(self, voice_id: str, delta: float = 0.1) -> Optional[Voice]:
        with self._lock:
            v = self._voices.get(voice_id)
            if v is None:
                return None
            v.weight = float(np.clip(v.weight - delta, 0.0, 1.0))
            return v

    # ── attribution ─────────────────────────────────────────────────────────

    def attribute_thought(self, thought: str) -> Attribution:
        """Return the best matching absorbed voice for a given thought."""
        if not thought.strip():
            return Attribution(thought=thought, best_voice_id=None, confidence=0.0)
        fp = _text_fingerprint(thought)
        with self._lock:
            if not self._voices:
                return Attribution(thought=thought, best_voice_id=None,
                                    confidence=0.0)
            scored: List[Tuple[str, float]] = []
            for vid, v in self._voices.items():
                sim = float(np.dot(v.fingerprint, fp))
                # Weight boost from current amplification
                score = 0.7 * sim + 0.3 * v.weight
                scored.append((vid, score))
            scored.sort(key=lambda kv: -kv[1])
            best_id, best_score = scored[0]
            # Normalise confidence to [0, 1] via softmax-like over top scores.
            top_scores = np.array([s for _, s in scored[:5]], dtype=np.float32)
            exp = np.exp(top_scores - top_scores.max())
            probs = exp / exp.sum()
            confidence = float(probs[0])
            return Attribution(
                thought=thought, best_voice_id=best_id,
                confidence=round(confidence, 4),
                alternative_votes=[
                    (vid, round(score, 4)) for vid, score in scored[:5]
                ],
            )

    # ── voice-weight aggregate (for GWS bias) ──────────────────────────────

    def voice_influence_summary(self) -> Dict[str, Any]:
        """Return which voices are amplified right now."""
        with self._lock:
            items = sorted(
                ((vid, v.weight, v.label) for vid, v in self._voices.items()),
                key=lambda t: -t[1],
            )
        return {
            "voices_count": len(items),
            "top_voices": [
                {"id": vid, "label": label, "weight": round(weight, 4)}
                for vid, weight, label in items[:10]
            ],
            "total_weight": round(sum(w for _, w, _ in items), 4),
        }

    # ── passive decay ──────────────────────────────────────────────────────

    def tick_decay(self, now: Optional[float] = None) -> None:
        """Apply passive weight decay proportional to time since last active."""
        now = now or time.time()
        with self._lock:
            for v in self._voices.values():
                elapsed_days = max(0.0, (now - v.last_active_at) / 86400.0)
                decay = WEIGHT_DECAY_PER_DAY * elapsed_days
                v.weight = max(0.0, v.weight - decay)
                # Last-active isn't reset here; this is passive-only.

    # ── accessors ──────────────────────────────────────────────────────────

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        with self._lock:
            return self._voices.get(voice_id)

    def voice_count(self) -> int:
        with self._lock:
            return len(self._voices)

    def all_voices(self) -> List[Voice]:
        with self._lock:
            return list(self._voices.values())

    def distinguishes_self_from_voices(self) -> bool:
        """A smoke-level check that the module separates "own cognition" from
        absorbed voices: i.e., we do NOT treat Aura-self as a registered voice."""
        with self._lock:
            return "aura_self" not in self._voices and "self" not in self._voices

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "voice_count": len(self._voices),
                "storage_path": str(self._storage_path),
                **self.voice_influence_summary(),
            }


# ── Singleton accessor ───────────────────────────────────────────────────────

_INSTANCE: Optional[AbsorbedVoices] = None


def get_absorbed_voices() -> AbsorbedVoices:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AbsorbedVoices()
    return _INSTANCE
