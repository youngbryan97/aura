"""
Tests for core/consciousness/agency_comparator.py
Validates the efference copy + comparator model for sense of agency.
"""

import time
import pytest


def test_efference_emission():
    """Emitting an efference copy stores it for later comparison."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ef = ac.emit_efference(
        "executive_authority",
        {"goal_completed": 1.0, "valence_delta": 0.05},
        action_goal="test goal",
        action_source="test_source",
    )
    assert ef.layer == "executive_authority"
    assert ef.predicted_state["goal_completed"] == 1.0
    assert ef.action_goal == "test goal"
    assert ef.action_source == "test_source"
    assert ac.get_pending_count() == 1


def test_compare_good_match_high_agency():
    """When actual closely matches predicted, self_caused_fraction is high."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ac.emit_efference(
        "test",
        {"goal_completed": 1.0, "valence_delta": 0.05, "hedonic_gain": 0.02},
        action_goal="goal_a",
    )
    trace = ac.compare_and_attribute(
        None,
        {"goal_completed": 1.0, "valence_delta": 0.06, "hedonic_gain": 0.01},
        action_goal="goal_a",
    )
    # Small error -> high self-caused
    assert trace.self_caused_fraction > 0.6
    assert trace.world_caused_fraction < 0.4
    assert trace.is_high_agency
    assert "self-authored" in trace.attribution_label
    assert trace.agency_confidence > 0.5
    assert ac.get_pending_count() == 0


def test_compare_bad_match_low_agency():
    """When actual diverges from predicted, world_caused_fraction is high."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ac.emit_efference(
        "test",
        {"goal_completed": 1.0, "valence_delta": 0.05},
        action_goal="goal_b",
    )
    trace = ac.compare_and_attribute(
        None,
        {"goal_completed": 0.0, "valence_delta": -0.8},
        action_goal="goal_b",
    )
    # Large error -> high world-caused
    assert trace.self_caused_fraction < 0.5
    assert trace.world_caused_fraction > 0.5
    assert not trace.is_high_agency


def test_compare_no_efference_low_confidence():
    """When no efference copy exists, trace has low confidence."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    trace = ac.compare_and_attribute(
        None,
        {"goal_completed": 1.0},
        action_goal="nonexistent_goal",
    )
    assert trace.agency_confidence < 0.2
    assert trace.self_caused_fraction == 0.5  # Neutral
    assert len(ac.get_recent_traces()) == 1


def test_agency_score_neutral_when_empty():
    """Agency score is 0.5 (neutral) with no traces."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    assert ac.get_agency_score() == 0.5


def test_agency_score_reflects_traces():
    """Agency score increases with high-agency traces."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    # Add several high-agency traces
    for i in range(5):
        ac.emit_efference("test", {"x": 1.0}, action_goal=f"goal_{i}")
        ac.compare_and_attribute(
            None, {"x": 1.01}, action_goal=f"goal_{i}"
        )
    score = ac.get_agency_score()
    assert score > 0.6, f"Expected high agency score, got {score}"


def test_recent_traces_limit():
    """get_recent_traces respects the limit parameter."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    for i in range(10):
        ac.emit_efference("test", {"x": float(i)}, action_goal=f"g{i}")
        ac.compare_and_attribute(None, {"x": float(i)}, action_goal=f"g{i}")

    assert len(ac.get_recent_traces(5)) == 5
    assert len(ac.get_recent_traces(20)) == 10


def test_ring_buffer_maxlen():
    """Ring buffer respects maxlen."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator(max_traces=5)
    for i in range(10):
        ac.emit_efference("test", {"x": float(i)}, action_goal=f"g{i}")
        ac.compare_and_attribute(None, {"x": float(i)}, action_goal=f"g{i}")

    assert len(ac._traces) == 5


def test_context_block_empty_when_no_traces():
    """Context block is empty string when no traces exist."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    assert ac.get_context_block() == ""


def test_context_block_populated():
    """Context block contains meaningful content with traces."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ac.emit_efference("test", {"x": 1.0}, action_goal="goal_1")
    ac.compare_and_attribute(None, {"x": 1.0}, action_goal="goal_1")

    block = ac.get_context_block()
    assert "SENSE OF AGENCY" in block
    assert "Agency score" in block
    assert "Last action" in block


def test_get_status():
    """Status dict contains expected keys."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    status = ac.get_status()
    assert "agency_score" in status
    assert "total_traces" in status
    assert "pending_efferences" in status
    assert "total_emissions" in status
    assert "total_comparisons" in status
    assert "recent_attribution" in status


def test_stale_efference_pruning():
    """Stale efference copies are pruned."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ef = ac.emit_efference("test", {"x": 1.0}, action_goal="old_goal")
    # Manually backdate the efference
    ef.emitted_at = time.time() - 700  # older than 600s threshold
    ac._pending_efferences[ac._goal_key("old_goal")] = ef

    # Emit new one to trigger pruning
    ac.emit_efference("test", {"y": 1.0}, action_goal="new_goal")
    assert ac.get_pending_count() == 1  # only new_goal remains


def test_singleton():
    """get_agency_comparator returns the same instance."""
    from core.consciousness.agency_comparator import get_agency_comparator
    import core.consciousness.agency_comparator as mod

    # Reset singleton for test isolation
    mod._instance = None
    s1 = get_agency_comparator()
    s2 = get_agency_comparator()
    assert s1 is s2
    mod._instance = None  # Clean up


def test_direct_efference_compare():
    """Passing efference directly to compare_and_attribute works."""
    from core.consciousness.agency_comparator import AgencyComparator

    ac = AgencyComparator()
    ef = ac.emit_efference("test", {"a": 0.5, "b": 0.8}, action_goal="direct_test")
    trace = ac.compare_and_attribute(ef, {"a": 0.5, "b": 0.8})
    # Perfect match
    assert trace.self_caused_fraction > 0.8
    assert trace.total_error < 0.01


def test_authorship_trace_dataclass():
    """AuthorshipTrace properties work correctly."""
    from core.consciousness.agency_comparator import AuthorshipTrace

    trace = AuthorshipTrace(
        layer="test",
        predicted_state={"x": 1.0},
        actual_state={"x": 1.0},
        delta={"x": 0.0},
        total_error=0.0,
        self_caused_fraction=0.9,
        world_caused_fraction=0.1,
        agency_confidence=0.8,
    )
    assert trace.is_high_agency
    assert "strongly self-authored" in trace.attribution_label
