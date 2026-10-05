"""Which capability a request means, by what it means: her own embedding of it beside each capability's.

The tool hand-off reads a request's words against what each capability
declares; a request in other words reached nothing. Asking her model to read it
was a guess with nothing to hold it to. Here every capability is a set of
meanings: the sentences of its own description, and every request it has
served (each call that came back done is kept as one). A request is embedded by
her own embedding model and goes to the capability whose meanings are nearest,
when it is near enough and clearly nearer than the next.

How near is near enough is read off the request's own scores, not set as a
similarity: the nearest capability must stand out from all of them (STANDS_OUT
deviations above their mean) and from the next (CLEAR_OF deviations ahead), so
the rule holds whatever the model's similarities run to. Measured 2026-10-05
with her own model on ten requests (repair, rebuilding, building, changing,
using, browsing) and three of small talk, from the capabilities' descriptions
alone: five were taken, four where they were meant and one not ("my own
version of excel" read as using a build; the words' own route takes that one
first), and no small talk. Where none stands out, nothing is chosen here, and
each request a capability serves narrows the next. What a capability has served makes it easier to reach next time in the
same words or others like them.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.CapabilityByMeaning")

__all__ = ["CLEAR_OF", "STANDS_OUT", "TASK", "Meanings"]

#: The most requests kept as meanings of one capability, the newest kept.
MOST_SERVED = 60

#: How far, in deviations of the request's scores, the nearest capability must stand above them all, and ahead of the next.
STANDS_OUT = 2.8
CLEAR_OF = 0.5

#: What the request side is embedded for (core/memory/embedding_model.py TASK_INSTRUCTIONS).
TASK = "capability"


def _sentences(description: str) -> list[str]:
    said = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", str(description or "")) if len(s.strip()) > 12]
    return said[:6]


class Meanings:
    """The meanings of the capabilities, as vectors, and the requests each has served."""

    def __init__(self, embed: Callable[[list[str]], np.ndarray], store: Path | None = None,
                 embed_query: Callable[[str], np.ndarray] | None = None) -> None:
        self._embed = embed
        self._embed_query = embed_query
        self._store = store
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self.served: dict[str, list[str]] = self._read()

    def _read(self) -> dict[str, list[str]]:
        try:
            held = json.loads(self._store.read_text("utf-8")) if self._store and self._store.exists() else {}
            return {str(k): [str(t) for t in v][-MOST_SERVED:] for k, v in held.items() if isinstance(v, list)}
        except (OSError, ValueError, AttributeError):
            return {}

    def _vector(self, texts: list[str]) -> np.ndarray:
        wanted = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if wanted:
            made = np.asarray(self._embed(wanted), dtype=np.float32).reshape(len(wanted), -1)
            made /= np.maximum(np.linalg.norm(made, axis=1, keepdims=True), 1e-9)
            with self._lock:
                self._vectors.update(zip(wanted, made, strict=True))
        return np.stack([self._vectors[t] for t in texts])

    def meanings_of(self, skills: Mapping[str, Any]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for name, meta in skills.items():
            said = _sentences(str(getattr(meta, "description", "") or ""))
            said += self.served.get(name, [])
            if said:
                found[name] = said
        return found

    def nearest(self, text: str, skills: Mapping[str, Any]) -> list[tuple[str, float]]:
        """Each capability and how near its nearest meaning is to ``text``, the nearest first."""
        meanings = self.meanings_of(skills)
        if not meanings or not str(text or "").strip():
            return []
        if self._embed_query is not None:
            asked = np.asarray(self._embed_query(text), dtype=np.float32)
            asked = asked / max(float(np.linalg.norm(asked)), 1e-9)
        else:
            asked = self._vector([text])[0]
        near = [(name, float(np.max(self._vector(said) @ asked))) for name, said in meanings.items()]
        return sorted(near, key=lambda pair: -pair[1])

    def which(self, text: str, skills: Mapping[str, Any]) -> str | None:
        """The capability ``text`` means, when one stands out from them all and clearly from the next; else None."""
        near = self.nearest(text, skills)
        if len(near) < 3:
            return None
        scores = np.array([score for _name, score in near])
        spread = float(scores.std()) or 1e-9
        stands, clear = (scores[0] - scores.mean()) / spread, (scores[0] - scores[1]) / spread
        logger.info("by meaning: %s stands %.2f out and %.2f clear of %s", near[0][0], stands, clear, near[1][0])
        return near[0][0] if stands >= STANDS_OUT and clear >= CLEAR_OF else None

    def learned(self, text: str, name: str) -> None:
        """``name`` served ``text``: one more of its meanings, kept."""
        text = " ".join(str(text or "").split())[:400]
        if not text:
            return
        kept = [t for t in self.served.get(name, []) if t != text] + [text]
        self.served[name] = kept[-MOST_SERVED:]
        if self._store is not None:
            try:
                self._store.parent.mkdir(parents=True, exist_ok=True)
                self._store.write_text(json.dumps(self.served, indent=1), "utf-8")
            except OSError as why:
                logger.info("what a capability served could not be kept: %s", why)
