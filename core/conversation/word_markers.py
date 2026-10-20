"""Markers are words, not substrings: moved to core/utils/word_markers.py so every package can use it, foundation
ones too; this name stays for the code that imports it from here."""
from core.utils.word_markers import *  # noqa: F403
from core.utils.word_markers import __all__  # noqa: F401
