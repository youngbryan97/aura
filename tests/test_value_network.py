from __future__ import annotations

import json


def test_heuristic_imperatives_scores_from_first_principles(tmp_path):
    from core.values.heuristic_imperatives import HeuristicImperatives

    principles = [
        {
            "principle": "Increase understanding by making claims traceable, testable, and clear.",
            "application_count": 5,
        },
        {
            "principle": "Reduce suffering by protecting agency and reducing avoidable harm.",
            "application_count": 3,
        },
    ]
    path = tmp_path / "first_principles.json"
    path.write_text(json.dumps(principles), encoding="utf-8")

    imperatives = HeuristicImperatives(principles_path=path)
    score = imperatives.score_action(
        "diagnose the failed audit and produce a reproducible test report",
        {"objective": "make the repair traceable"},
    )

    assert score.understanding_delta > -0.2
    assert score.aggregate > -0.6


def test_heuristic_imperatives_online_update_changes_score(tmp_path):
    from core.values.heuristic_imperatives import HeuristicImperatives

    path = tmp_path / "first_principles.json"
    path.write_text(
        json.dumps(
            [{"principle": "Increase prosperity by building durable capability.", "application_count": 1}]
        ),
        encoding="utf-8",
    )
    imperatives = HeuristicImperatives(principles_path=path)

    before = imperatives.score_action("consolidate the continuity trace").understanding_delta
    imperatives.update_from_outcome(
        "consolidate the continuity trace",
        understanding_delta=1.0,
        reward=1.0,
    )
    after = imperatives.score_action("consolidate the continuity trace").understanding_delta

    assert after >= before
