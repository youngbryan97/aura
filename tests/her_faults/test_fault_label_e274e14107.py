"""A fault from her own running, made again: core.self_modification.loop_drill.label raised builtins.AttributeError.

Heard 2 time(s) from loop_drill, first at 16 November 2026 10:27: 'NoneType' object has no attribute 'name'
Raised at core/self_modification/loop_drill.py:33. Written by core/self_modification/faults_she_can_reproduce.py.
"""
from core.self_modification.loop_drill import label


def test_fault_label_e274e14107():
    label(person=None)
