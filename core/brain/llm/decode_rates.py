"""How fast each model writes on this host, measured per model.

The host's decode rate was one average that every generation on every model moved: her 27B cortex, her 1.5B and 7B
background lanes, the 2-bit model her curriculum loop asks. An answer was fitted to the time it was given at that
rate (core/brain/inference_gate.py, fit_the_answer_to_the_time), so a reply on her cortex was priced at whatever had
decoded last. LIVE 17 November 2026, 07:33, her first reply after a start asked for 512 tokens and was given 160, was
cut, and ended on a stand-in's "I couldn't get to an answer I'd stand behind"; ten of the twenty-seven replies to a
person in that log were given fewer tokens than their plan asked for. Prefill was already kept per model
(mlx_client._HOST_PREFILL_TPS); this keeps decode the same way.
"""
from __future__ import annotations

import os

#: Each model's decode rate, tokens a second, by the model's file name: the same smoothing as the host's.
_BY_MODEL: dict[str, float] = {}


def _key(model: object) -> str:
    return os.path.basename(str(model or "").rstrip("/")).strip().lower()


def note_decode(model: object, observed: float) -> None:
    """One measurement of ``model`` writing, ``observed`` tokens a second."""
    key = _key(model)
    if not key or observed <= 0.0:
        return
    previous = _BY_MODEL.get(key, 0.0)
    _BY_MODEL[key] = observed if previous <= 0.0 else previous * 0.8 + observed * 0.2


def decode_rate_of(model: object) -> float:
    """How fast ``model`` has been measured writing on this host, or 0.0 when it has not been."""
    return _BY_MODEL.get(_key(model), 0.0)


def forget_for_test() -> None:
    _BY_MODEL.clear()


__all__ = ["decode_rate_of", "forget_for_test", "note_decode"]
