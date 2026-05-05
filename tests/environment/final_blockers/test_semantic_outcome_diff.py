import pytest
from core.environment.outcome.semantic_diff import SemanticDiffLearner
from core.environment.parsed_state import ParsedState

@pytest.fixture
def diff_learner():
    return SemanticDiffLearner()

def test_wall_bump_is_semantic_failure(diff_learner):
    parsed_before = ParsedState(environment_id="test", sequence_id=1, self_state={"raw_text": "...", "local_coordinates": (10, 10)})
    parsed_after = ParsedState(environment_id="test", sequence_id=2, self_state={"raw_text": "You hit a wall.", "local_coordinates": (10, 10)})
    
    events = diff_learner.compute_diff(parsed_before, parsed_after)
    
    event_names = [e.name for e in events]
    assert "blocked_by_wall" in event_names
    assert "position_unchanged" in event_names

def test_successful_movement_detected_by_position_delta(diff_learner):
    parsed_before = ParsedState(environment_id="test", sequence_id=1, self_state={"raw_text": "...", "local_coordinates": (10, 10)})
    parsed_after = ParsedState(environment_id="test", sequence_id=2, self_state={"raw_text": "...", "local_coordinates": (11, 10)})
    
    events = diff_learner.compute_diff(parsed_before, parsed_after)
    
    event_names = [e.name for e in events]
    assert "position_changed" in event_names
