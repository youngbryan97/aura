"""Her own status is not a subject to research.

LIVE 2026-08-17, from the neural stream:

    [SubjectiveChoice] Chose 'Deconstruct and comprehensively research: Aura is
    idle' because preference alignment 0.00 and drive alignment 0.49 produced
    final score 0.27.

"Aura is idle." is a UI activity label emitted by the cognitive state machine.
It reached the knowledge graph as a sparse node, was drawn as a research topic,
and became a durable goal — one that sets her focus and spawns a research shard
to investigate a fact she already holds, about herself, that a status field
answers exactly.

The filter already refused stale receipts, prompt scaffolds and desktop
actions. Her own state belongs in that category: something the runtime KNOWS,
so researching it is not curiosity, it is a loop.
"""

from __future__ import annotations

import pytest

from core.autonomy.research_goal_filter import (
    is_runtime_status_goal,
    is_unresearchable_goal,
    research_query_for_goal,
)


def test_the_live_case_is_refused() -> None:
    assert is_unresearchable_goal("Aura is idle.") is True
    assert research_query_for_goal("Aura is idle.") == ""


@pytest.mark.parametrize(
    "label",
    [
        "Aura is typing...",
        "Aura is searching the web...",
        "Aura is generating an image...",
        "Aura is executing a terminal command...",
        "Aura is managing files...",
        "The runtime is warming up",
        "The system is shutting down",
    ],
)
def test_ui_activity_labels_are_refused(label: str) -> None:
    """These are the ACTIVITY_MAP strings; every one could reach the graph."""
    assert is_runtime_status_goal(label) is True


# ── it must not cost her real curiosity ─────────────────────────────────────

@pytest.mark.parametrize(
    "topic",
    [
        "Quantum Neural Network Architectures",
        "Cognitive Neuroscience of Agency",
        "Idle animation techniques in game design",
        "How idle CPU states affect laptop battery life",
    ],
)
def test_real_topics_survive(topic: str) -> None:
    """A subject that merely CONTAINS a status word is still a subject."""
    assert is_runtime_status_goal(topic) is False
    assert research_query_for_goal(topic)


def test_a_status_word_mid_sentence_is_not_a_status_line() -> None:
    """The pattern anchors at the start; 'X is idle' as a clause is not this."""
    assert is_runtime_status_goal("Why the scheduler thinks Aura is idle") is False


@pytest.mark.parametrize("value", [None, "", "   ", 0])
def test_garbage_is_safe(value) -> None:
    assert is_runtime_status_goal(value) is False
