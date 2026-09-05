"""core/workspace/private_scratchpad.py
Private working memory scratchpad for temporary agent deliberations.
"""
from typing import Dict, Any


class PrivateScratchpad:
    """Manages active text buffer for sandbox thinking."""

    def __init__(self):
        self._content: str = ""

    def write_buffer(self, text: str) -> None:
        self._content = text

    def append_buffer(self, text: str) -> None:
        self._content += "\n" + text

    def read_buffer(self) -> str:
        return self._content

    def clear_buffer(self) -> None:
        self._content = ""
