"""Autopoiesis tries only a repair it has a handler for.

LIVE, day run 4 (9 November 2026): reduce_load/global had no handler and was tried 202 times in 35 minutes. Each try
was recorded as a failed repair, her resilience took every one as a new loss, and her depletion climbed from 0.42 to
0.63 in 25 minutes with frustration at 1.00.
"""
from __future__ import annotations

from core.cognitive.autopoiesis import _ESCALATION_ORDER, AutopoiesisEngine, RepairStrategy


def test_a_strategy_with_no_handler_is_skipped_for_the_next_one_that_has_one():
    engine = AutopoiesisEngine()
    engine.register_repair_handler(RepairStrategy.RESTART_COMPONENT, "global", lambda: True)
    engine._repair_attempts["global"] = engine._ESCALATION_THRESHOLD * _ESCALATION_ORDER.index(RepairStrategy.REDUCE_LOAD)
    assert engine._choose_strategy("global") is RepairStrategy.RESTART_COMPONENT


def test_a_component_with_no_handler_at_all_is_not_repaired_but_escalated_once_in_a_while():
    engine = AutopoiesisEngine()
    escalated = []
    engine._escalate_to_human = escalated.append
    assert engine._choose_strategy("global") is None
    assert escalated == ["global"]


def test_the_gentlest_strategy_with_a_handler_comes_first():
    engine = AutopoiesisEngine()
    engine.register_repair_handler(RepairStrategy.HEAL, "memory", lambda: True)
    engine.register_repair_handler(RepairStrategy.CLEAR_CACHE, "memory", lambda: True)
    assert engine._choose_strategy("memory") is RepairStrategy.HEAL
