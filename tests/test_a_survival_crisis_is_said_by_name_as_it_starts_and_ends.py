"""A survival crisis is said by name, with its readings, as it starts and as it ends.

Day run 4's fifth start, 10 November 2026: "Endogenous fitness: 1 (measured energy,...)" at warning on 59 of about 69
background ticks. 1 was CrisisReason.ENERGY_DEPLETED's value, and the line gave neither the name nor the reading.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from core.consciousness.endogenous_fitness import CrisisReason
from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase


class _Fitness:
    def __init__(self, crises):
        self.crises = list(crises)

    def current_crisis(self):
        crisis = self.crises.pop(0)
        return {"crisis": crisis.value if crisis else "", "readings": {"energy": 3.5, "vitality": 0.85},
                "measured": ["energy", "vitality"], "defaulted": []}


def test_a_standing_crisis_is_said_once_by_name_and_its_end_is_said(caplog):
    phase = SimpleNamespace(_endogenous_fitness=_Fitness(
        [CrisisReason.ENERGY_DEPLETED] * 3 + [None, CrisisReason.PHI_ZERO]))
    state = SimpleNamespace(response_modifiers={})
    with caplog.at_level(logging.INFO, logger="Aura.CognitiveIntegration"):
        for _ in range(5):
            CognitiveIntegrationPhase._run_survival_watch(phase, state)
    said = [(r.levelname, r.getMessage()) for r in caplog.records if "Endogenous fitness" in r.getMessage()]
    assert said[0][0] == "WARNING" and "ENERGY_DEPLETED began" in said[0][1] and "energy=3.5" in said[0][1]
    assert said[1][0] == "INFO" and "ENERGY_DEPLETED ended" in said[1][1]
    assert said[2][0] == "WARNING" and "PHI_ZERO began" in said[2][1]
    assert len(said) == 3, "three ticks of one crisis are one line"
