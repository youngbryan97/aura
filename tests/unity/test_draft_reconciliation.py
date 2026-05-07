from __future__ import annotations

from core.unity.draft_reconciliation import DraftReconciliationEngine


def test_conflicting_drafts_become_conflicted_memory():
    result = DraftReconciliationEngine().reconcile(
        [
            {"draft_id": "a", "content": "Bryan wants the branch pushed now", "coherence": 0.82, "valence": 0.2},
            {"draft_id": "b", "content": "Bryan explicitly asked not to push yet", "coherence": 0.78, "valence": -0.3},
        ]
    )

    assert result.memory_commit_mode in {"conflicted", "defer"}
    assert result.contradiction_score > 0.35
    assert len(result.alternatives) == 1
    assert result.alternatives[0].suppressed_reason


def test_similar_drafts_remain_clean():
    result = DraftReconciliationEngine().reconcile(
        [
            {"draft_id": "a", "content": "Patch the unity phase before response generation", "coherence": 0.9},
            {"draft_id": "b", "content": "Patch the unity phase before generating the response", "coherence": 0.88},
        ]
    )

    assert result.memory_commit_mode == "clean"
    assert result.consensus_score > 0.7
