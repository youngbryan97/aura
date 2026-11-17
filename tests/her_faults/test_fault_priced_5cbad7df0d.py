"""A fault from her own running, made again: core.self_modification.loop_drill.priced raised builtins.KeyError.

Heard 2 time(s) from loop_drill, first at 17 November 2026 02:51: 'cake'
Raised at core/self_modification/loop_drill.py:57. Written by core/self_modification/faults_she_can_reproduce.py.
"""
from core.self_modification.loop_drill import priced


def test_fault_priced_5cbad7df0d():
    priced(prices={'tea': 3}, item='cake')
