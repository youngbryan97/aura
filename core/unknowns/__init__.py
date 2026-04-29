"""Unknown-unknown generation — bounded failure-finding test synthesis.

The honest answer to "self-generate perfect new tests for unknown
unknowns" is that "perfect" is impossible — the generator can't know
what it doesn't know.  What it *can* do is:

  * Maintain a ``NoveltyArchive`` of prompts it has already seen.
  * Mutate seed tasks (from the F18 dynamic benchmark) and keep only
    those whose embedding is far from anything in the archive.
  * Optionally use ``EmbeddingEntropyProbe`` to find input embeddings
    that maximise the LatticeLM's output entropy — those are the
    model's blind spots in continuous-input space.

Generated tasks feed back into:
  * F9 curriculum loop's task generator (as seeds for the curriculum
    to attempt + learn from)
  * F18 holdout vault (so today's failures become tomorrow's
    regression tests)
"""
from core.unknowns.novelty_archive import NoveltyArchive, NoveltyEntry
from core.unknowns.generator import UnknownUnknownGenerator
from core.unknowns.entropy_probe import EmbeddingEntropyProbe

__all__ = [
    "EmbeddingEntropyProbe",
    "NoveltyArchive",
    "NoveltyEntry",
    "UnknownUnknownGenerator",
]
