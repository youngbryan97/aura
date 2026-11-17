"""A fault from her own running, made again: core.runtime.subprocess_gateway._coerce_argv raised builtins.ValueError.

Heard 1 time(s) from base_skill, first at 16 November 2026 23:31: argv entries must not be empty
Raised at core/runtime/subprocess_gateway.py:524. Written by core/self_modification/faults_she_can_reproduce.py.
"""
from core.runtime.subprocess_gateway import _coerce_argv


def test_fault_coerce_argv_bcdb0f784c():
    _coerce_argv(argv=['osascript', '-e', 'on run argv', '-e', 'display notification (item 1 of argv) with title (item 2 of argv) sound name (item 4 of argv)', '-e', 'end run', "Time to stretch! Shake out your shoulders, roll your neck, unclench those hands. You've been at it a while — give yourself 60 seconds of movement before heading back in.", 'Stretch break 🧘', '', 'Tink'])
