"""Word-form equivalence for lexical evidence, not semantic correctness."""

from functools import lru_cache
from threading import local

import snowballstemmer

_state = local()


@lru_cache(maxsize=4096)
def word_form(word: str) -> str:
    """Normalize English inflection with an isolated Snowball instance."""
    stemmer = getattr(_state, "stemmer", None)
    if stemmer is None:
        stemmer = snowballstemmer.stemmer("english")
        _state.stemmer = stemmer
    return str(stemmer.stemWord(word.lower()))


def matching_word_forms(wanted: set[str], observed: set[str]) -> set[str]:
    """Return requested anchors present in another grammatical word form."""
    observed_forms = {word_form(word) for word in observed}
    return {word for word in wanted if word_form(word) in observed_forms}
