"""Tests for the attentional context gate.

Verifies that the gate selects blocks under budget, respects essentials,
filters duplicates, and obeys salience/priority thresholds.
"""
from __future__ import annotations

import pytest

from core.brain.llm.context_gate import (
    AttentionalContextGate,
    ContextBlock,
    ContextDeltaTracker,
    estimate_tokens,
)


def test_essential_blocks_always_included():
    gate = AttentionalContextGate()
    blocks = [
        ContextBlock(id="identity", content="I am Aura.", essential=True, max_tokens=100),
        ContextBlock(id="fluff", content="x" * 5000, priority=0.1, salience=0.1, max_tokens=50),
    ]
    selected = gate.select(blocks, token_budget=50)
    ids = {b.id for b in selected}
    assert "identity" in ids


def test_budget_respected():
    gate = AttentionalContextGate()
    blocks = [
        ContextBlock(id="a", content="A" * 100, priority=0.9, salience=0.9, max_tokens=200),
        ContextBlock(id="b", content="B" * 100, priority=0.8, salience=0.8, max_tokens=200),
        ContextBlock(id="c", content="C" * 100, priority=0.7, salience=0.7, max_tokens=200),
    ]
    selected = gate.select(blocks, token_budget=80)
    total = sum(estimate_tokens(b.content) for b in selected)
    assert total <= 80


def test_low_salience_filtered():
    gate = AttentionalContextGate()
    block = ContextBlock(id="low", content="Low priority noise", priority=0.2, salience=0.1)
    assert not gate.should_include_block(block)


def test_duplicate_filtered_unless_high_priority():
    gate = AttentionalContextGate()
    block = ContextBlock(id="dup", content="Same content", priority=0.6, salience=0.6)

    assert gate.should_include_block(block)  # first time: include
    assert not gate.should_include_block(block)  # second time: duplicate, filtered


def test_delta_tracker_initial_value_not_reported():
    tracker = ContextDeltaTracker()
    # First observation should NOT be reported as changed (no prior value)
    assert not tracker.changed("valence", 0.5)
    # Second observation with same value should also not be reported
    assert not tracker.changed("valence", 0.5)


def test_delta_tracker_detects_significant_change():
    tracker = ContextDeltaTracker()
    tracker.changed("valence", 0.3)  # set initial
    assert tracker.changed("valence", 0.6)  # delta = 0.3, threshold = 0.20


def test_delta_tracker_ignores_small_change():
    tracker = ContextDeltaTracker()
    tracker.changed("valence", 0.5)  # set initial
    assert not tracker.changed("valence", 0.55)  # delta = 0.05, below 0.20


def test_delta_tracker_critical_always_fires():
    tracker = ContextDeltaTracker()
    tracker.changed("cpu_usage", 50.0)
    # Critical threshold fires regardless of delta
    assert tracker.changed("cpu_usage", 95.0, critical=90.0)


def test_include_if_predicate():
    gate = AttentionalContextGate()
    # Block with include_if returning False should be excluded
    block = ContextBlock(
        id="gated", content="conditional", priority=0.9, salience=0.9,
        include_if=lambda: False,
    )
    assert not gate.should_include_block(block)

    # Block with include_if returning True should be included
    block2 = ContextBlock(
        id="gated2", content="conditional yes", priority=0.9, salience=0.9,
        include_if=lambda: True,
    )
    assert gate.should_include_block(block2)


def test_estimate_tokens_reasonable():
    assert estimate_tokens("") == 1
    # 100 chars → ~28 tokens
    assert 20 <= estimate_tokens("x" * 100) <= 40
    # 3500 chars → ~1000 tokens
    assert 900 <= estimate_tokens("x" * 3500) <= 1100


def test_compact_truncates():
    block = ContextBlock(id="big", content="x" * 5000, max_tokens=100)
    compacted = block.compact()
    assert len(compacted.content) < 400  # 100 tokens * 3.5 chars
    assert "[compacted]" in compacted.content
