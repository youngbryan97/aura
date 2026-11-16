"""A fault from her own running, made again: core.self_modification.loop_drill.nth raised builtins.IndexError.

Heard 1 time(s) from loop_drill, first at 16 November 2026 15:39: list index out of range
Raised at core/self_modification/loop_drill.py:39. Written by core/self_modification/faults_she_can_reproduce.py.
"""
from core.self_modification.loop_drill import nth


def test_fault_nth_8ade365fa6():
    nth(rows=[1], n=2)
