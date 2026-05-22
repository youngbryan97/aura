import json

from core.epistemic_tracker import EpistemicTracker
from core.runtime.errors import get_degradation_tracker


def test_epistemic_tracker_public_save_round_trips(tmp_path):
    path = tmp_path / "epistemic_map.json"
    tracker = EpistemicTracker(db_path=path)

    tracker.update_node("Runtime epistemics", confidence_delta=0.25, depth_delta=0.4, new_source=True)

    assert tracker.save() is True
    reloaded = EpistemicTracker(db_path=path)

    node = reloaded._nodes["runtime epistemics"]
    assert node.concept == "Runtime epistemics"
    assert node.confidence == 0.75
    assert node.depth == 0.5
    assert node.source_count == 1


def test_epistemic_tracker_quarantines_invalid_persistence(tmp_path):
    get_degradation_tracker().reset()
    path = tmp_path / "epistemic_map.json"
    path.write_text("{not-json", encoding="utf-8")

    tracker = EpistemicTracker(db_path=path)

    assert tracker._nodes == {}
    assert not path.exists()
    assert list(tmp_path.glob("epistemic_map.json.corrupt.*"))
    assert get_degradation_tracker().count("epistemic_tracker", "degraded") >= 1


def test_epistemic_tracker_sanitizes_loaded_state(tmp_path):
    path = tmp_path / "epistemic_map.json"
    path.write_text(
        json.dumps(
            {
                "nodes": {
                    "unsafe": {
                        "concept": "Unsafe",
                        "confidence": 4.0,
                        "depth": -2.0,
                        "last_updated": -10,
                        "source_count": -5,
                        "contradicted": True,
                        "contradiction_with": "Other",
                    }
                },
                "gaps": [
                    {
                        "domain": "science",
                        "description": "Gap",
                        "urgency": 9.0,
                        "detected_at": -1,
                        "gap_type": "unknown",
                        "seed_question": "Question?",
                    }
                ],
                "resolved": "not-a-list",
            }
        ),
        encoding="utf-8",
    )

    tracker = EpistemicTracker(db_path=path)

    node = tracker._nodes["unsafe"]
    gap = tracker._gaps[0]
    assert node.confidence == 1.0
    assert node.depth == 0.0
    assert node.last_updated == 0.0
    assert node.source_count == 0
    assert gap.urgency == 1.0
    assert gap.detected_at == 0.0
    assert tracker._resolved_gaps == []
