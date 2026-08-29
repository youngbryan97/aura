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

from core.language.learned_matcher import LearnedMatcher, embed_sentences
from core.runtime.lockdep import checked_lock

__all__ = ["actuation_surface"]

_SURFACE: LearnedMatcher | None = None
_LOCK = checked_lock("core.language.desktop_actuation")


def actuation_surface() -> LearnedMatcher:
    """The surface, seeded once from the log of what ran.

    Topical features on purpose. This decision is about WHAT is being acted
    on — apps, windows, browsers against files, endpoints, text — and a
    topical embedder reads that well: 0.979 against the model's own states,
    which are the better feature space for decisions about mood and agency
    rather than subject.
    """
    global _SURFACE
    # Built outside the lock, published under it.
    #
    # The build mines labels and embeds them, and it used to happen with the
    # lock held. Live on 2026-08-29 that was 291ms on the event loop thread
    # against a 50ms limit: "the loop could not make progress for that window".
    # Everything else waiting on the loop — a turn's tokens, a heartbeat — was
    # stopped for the duration.
    #
    # Two callers racing now both build, and one of them throws its copy away.
    # That costs one extra build once, in exchange for a lock nobody waits on,
    # and it is the reason lockdep can see this lock at all: a hold this long
    # is where an ABBA deadlock hides.
    existing = _SURFACE
    if existing is not None:
        return existing

    surface = LearnedMatcher(name="desktop_actuation", features=embed_sentences)
    try:
        from core.language.label_mining import mine_desktop_actuation_labels

        positives, negatives = mine_desktop_actuation_labels()
        surface.positives = tuple(positives)
        surface.negatives = tuple(negatives)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass

    with _LOCK:
        if _SURFACE is None:
            _SURFACE = surface
        return _SURFACE
