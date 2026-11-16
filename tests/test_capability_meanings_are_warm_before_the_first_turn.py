"""The catalogue's meanings are embedded at boot, not on the first turn.

LIVE 15 November 2026: the first typed turn after a boot spent 8.6 s reading
its request by meaning, embedding every capability's description while the
person waited. The turn after it spent 0.05 s on the same step.
"""

from __future__ import annotations

import inspect

import numpy as np

from core.intent.capability_by_meaning import Meanings


class _Skill:
    def __init__(self, description: str) -> None:
        self.description = description


def test_a_warm_index_embeds_only_the_request():
    calls: list[list[str]] = []

    def embed(texts: list[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.random.default_rng(len(calls)).normal(size=(len(texts), 8))

    meanings = Meanings(embed, None, lambda _text: np.ones(8, dtype=np.float32))
    skills = {
        "open_file": _Skill("Opens a file on this machine. Reads what it holds."),
        "send_message": _Skill("Sends a message to a person."),
        "play_music": _Skill("Plays music from the library."),
    }

    warmed = meanings.warm(skills)

    assert warmed == sum(len(said) for said in meanings.meanings_of(skills).values())
    calls.clear()
    assert meanings.nearest("open my notes", skills)
    assert calls == [], "a request read after the warm embedded the catalogue again"


def test_boot_warms_the_meanings_after_the_embedding_model():
    from interface import server

    source = inspect.getsource(server)
    stage = source[source.index('"semantic_projection"'):]
    stage = stage[: stage.index(")\n", stage.index('("self_condition"'))]
    assert "prewarm_capability_meanings" in stage
