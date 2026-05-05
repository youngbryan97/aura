"""Compatibility import for the canonical terminal-grid state compiler.

The environment OS owns NetHack parsing and compilation through
``core.environments.terminal_grid``. This module remains only so older import
paths keep using the same implementation instead of a second writable parser.
"""
from __future__ import annotations

from core.environments.terminal_grid.nethack_parser import NetHackStateCompiler

__all__ = ["NetHackStateCompiler"]
