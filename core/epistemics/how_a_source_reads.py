"""How a source's passage reads against a claim, by an entailment model: whether it states it, says otherwise, or
neither (L10).

Her check's first judge was every word and figure of the claim found in the passage. LIVE 19 October 2026 that held
"Sydney is the capital of Australia" verified by "Sydney is the capital of the state of New South Wales and the
largest city in Australia", which has every word; and it missed "elevation of 8,849 m" for "8,849 metres tall".
The smaller entailment model her runtime reads her own words with (core/self/how_her_words_stand.py) read the Sydney
passage as entailing the claim at 0.996. DeBERTa-v3-large trained on MNLI, FEVER, ANLI, LingNLI and WANLI (Laurer et
al. 2022) read it as a contradiction (0.93) and the other five cases tried right, at 0.44 s a pair on this machine's
CPU. Held on this machine only: ``local_files_only``, loaded once a process.
"""
from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.Epistemics.Reads")

__all__ = ["READER", "how_it_reads"]

#: The entailment model, by its Hugging Face name.
READER = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

_LOCK = checked_lock("how_a_source_reads.loaded")
_LOADED: dict[str, Any] = {}


def _reader() -> tuple[Any, dict[str, int]] | None:
    with _LOCK:
        if "model" in _LOADED:
            return _LOADED["model"]
        found: tuple[Any, dict[str, int]] | None = None
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(READER, local_files_only=True)
            names = getattr(getattr(model, "model", None), "config", None)
            labels = {str(name).lower(): int(index) for index, name in dict(getattr(names, "id2label", {}) or {}).items()}
            if {"entailment", "contradiction"} <= set(labels):
                found = (model, labels)
            else:
                record_degradation("how_a_source_reads", ValueError(f"labels {sorted(labels)} name no entailment"),
                                   severity="info", action="a source is judged by its words alone")
        except Exception as exc:  # noqa: BLE001 - an absent reader is judged by words alone, as before it
            record_degradation("how_a_source_reads", exc, severity="info", action="a source is judged by its words alone")
        _LOADED["model"] = found
        return found


def how_it_reads(passage: str, claim: str) -> dict[str, float] | None:
    """The probability of each label (entailment, neutral, contradiction) of the claim given the passage; None when
    there is no reader here."""
    loaded = _reader()
    said = " ".join(str(passage or "").split())
    if loaded is None or not said or not str(claim or "").strip():
        return None
    model, labels = loaded
    try:
        import numpy as np

        logits = np.asarray(model.predict([(said, str(claim))]), dtype=float)[0]
        chances = np.exp(logits - logits.max())
        chances = chances / chances.sum()
    except Exception as exc:  # noqa: BLE001
        record_degradation("how_a_source_reads", exc, severity="warning", action="a source is judged by its words alone")
        return None
    return {name: float(chances[index]) for name, index in labels.items()}
