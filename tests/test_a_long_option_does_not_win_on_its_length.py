"""An option speaks to a tension by the words that tell the tension apart, not by how many words it has.

On 19 October a 300-word role instruction left in her goal store shared "the", "and" and "for" with every
tension she held, scored 1.0 on tension resolution, and was chosen 25 times in three hours.
"""
from __future__ import annotations

from types import SimpleNamespace

from core.agency.initiative_arbiter import InitiativeArbiter
from core.container import ServiceContainer

# As her store held them on 19 October: lists of benchmark problems her goal scanner found competing.
TENSIONS = [
    "Multiple goals competing for 'general': I have a service configuration that need, Find the shortest path "
    "through a 6x6 gri, I have a rule execution engine in Python",
    "Multiple goals competing for 'general': Solve this task scheduling problem with, A dynamic event has "
    "occurred in the",
    "Multiple goals competing for 'general': Evaluate vendors and make a purchase dec, Solve this budget "
    "optimization problem",
    "Stalled goal: finish the pong repair so the paddle tracks the ball",
]
# The objective as stored (goal 10ff8b63e809), created 20 August by a role table removed on 15 September.
ROLE = ("[SWARM PROTOCOL: You are 'The Architect'. Design the high-level structure to solve the problem. Focus on "
        "patterns, resilience, and scalability. Focus exclusively on your specialized perspective.]\nAnalyze this "
        "topic from your perspective and return compact JSON with keys claim, evidence_refs, confidence, flaws: "
        "Self-reflection on current runtime status: Standby (running=True, healthy=True, is_processing=False, "
        "is_throttled=False, cycle_count=11923, uptime=0.0)")


class _Tensions:
    def get_active_tensions(self):
        return [SimpleNamespace(description=d, severity=1.0) for d in TENSIONS]


def _score(goal: str) -> float:
    ServiceContainer.register_instance("tension_engine", _Tensions())
    try:
        return InitiativeArbiter()._score_tension_resolution({"goal": goal})
    finally:
        ServiceContainer.register_instance("tension_engine", None)


def test_an_option_naming_what_a_tension_is_about_outscores_a_long_one_that_shares_its_filler():
    named = _score("Repair pong so the paddle tracks the ball")
    wordy = _score(ROLE)
    assert named > 0.5
    assert wordy < named / 2


def test_words_every_tension_shares_count_for_less_than_what_sets_one_apart():
    filler = _score("Multiple goals competing for general")
    apart = _score("Find the shortest path through the grid")
    assert filler < apart
