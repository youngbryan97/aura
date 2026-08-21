"""Whether a request needs the screen, learned from what actually ran.

`looks_like_desktop_objective` decides this with seventeen patterns, and it
has been wrong in both directions: a request to build a web app went to the
screen lane and came back "os_automation refused to act… completed 0/1
steps", while a goal to play a game online read as conversation and answered
about identity with the browser sitting on a blank page.

The intention log holds a hundred and ten distinct requests a person made and
the capability that succeeded for each. Measured on a held-out third of them,
that decision ranks at AUROC 0.979.

Seeded from the log rather than declared, so the examples are things that
happened rather than things somebody imagined.
"""

from __future__ import annotations

import threading

from core.language.learned_matcher import LearnedMatcher, embed_sentences

__all__ = ["actuation_surface"]

_SURFACE: LearnedMatcher | None = None
_LOCK = threading.Lock()


def actuation_surface() -> LearnedMatcher:
    """The surface, seeded once from the log of what ran.

    Topical features on purpose. This decision is about WHAT is being acted
    on — apps, windows, browsers against files, endpoints, text — and a
    topical embedder reads that well: 0.979 against the model's own states,
    which are the better feature space for decisions about mood and agency
    rather than subject.
    """
    global _SURFACE
    with _LOCK:
        if _SURFACE is not None:
            return _SURFACE
        surface = LearnedMatcher(name="desktop_actuation", features=embed_sentences)
        try:
            from core.language.label_mining import mine_desktop_actuation_labels

            positives, negatives = mine_desktop_actuation_labels()
            surface.positives = tuple(positives)
            surface.negatives = tuple(negatives)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        _SURFACE = surface
        return surface
