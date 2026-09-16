"""Four cognitive modes declared, two of them never written anywhere.

`CognitiveMode` declares REACTIVE, DELIBERATE, DREAMING and DORMANT. Grepping
the tree for an assignment of the last two returns nothing: they are read in
`canonical_self`, in `cognitive_routing` and in the metabolic coordinator, and
assigned by no one. Two of the four columns the subject schema reads off the
mode were flat in every recording because two of her four modes could not
occur.

The engine set the mode only on a foreground turn, so a background turn ran
under whichever mode the last foreground turn left. DREAMING is the enum's own
word for "background synthesis, no user", which is exactly what a background
turn is.

And the coordinator's test for a cool period compared a `CognitiveMode` against
two strings, which is false whatever mode she is in, so the distillation it
guards could only ever be reached through the energy arm beside it.
"""

from __future__ import annotations

from core.state.aura_state import AuraState, CognitiveMode


def test_an_enum_is_not_its_value() -> None:
    """The comparison that could not be true."""
    assert CognitiveMode.DORMANT not in ("dormant", "dreaming")
    assert CognitiveMode.DORMANT.value in ("dormant", "dreaming")


def test_the_coordinator_reads_the_mode_by_its_name() -> None:
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    state = AuraState()
    state.cognition.current_mode = CognitiveMode.DORMANT
    assert MetabolicCoordinator._cognitive_mode_value(state) == "dormant"
    state.cognition.current_mode = CognitiveMode.DREAMING
    assert MetabolicCoordinator._cognitive_mode_value(state) == "dreaming"
    assert MetabolicCoordinator._cognitive_mode_value(None) == ""


def test_low_metabolism_names_her_dormant() -> None:
    from core.coordinators.metabolic_coordinator import (
        _DORMANT_ENERGY,
        MetabolicCoordinator,
    )

    coordinator = MetabolicCoordinator.__new__(MetabolicCoordinator)
    state = AuraState()

    class _Orch:
        pass

    orch = _Orch()
    orch.state = state
    coordinator._orch = orch

    coordinator._metabolic_energy = _DORMANT_ENERGY / 2
    coordinator._name_the_resting_mode()
    assert state.cognition.current_mode is CognitiveMode.DORMANT


def test_she_comes_out_of_it_when_the_energy_does() -> None:
    from core.coordinators.metabolic_coordinator import (
        _DORMANT_ENERGY,
        MetabolicCoordinator,
    )

    coordinator = MetabolicCoordinator.__new__(MetabolicCoordinator)
    state = AuraState()
    state.cognition.current_mode = CognitiveMode.DORMANT

    class _Orch:
        pass

    orch = _Orch()
    orch.state = state
    coordinator._orch = orch

    coordinator._metabolic_energy = 1.0
    coordinator._name_the_resting_mode()
    assert state.cognition.current_mode is CognitiveMode.REACTIVE


def test_a_deliberate_turn_is_not_overwritten_by_the_metabolism() -> None:
    from core.coordinators.metabolic_coordinator import MetabolicCoordinator

    coordinator = MetabolicCoordinator.__new__(MetabolicCoordinator)
    state = AuraState()
    state.cognition.current_mode = CognitiveMode.DELIBERATE

    class _Orch:
        pass

    orch = _Orch()
    orch.state = state
    coordinator._orch = orch

    coordinator._metabolic_energy = 1.0
    coordinator._name_the_resting_mode()
    assert state.cognition.current_mode is CognitiveMode.DELIBERATE


def test_a_background_turn_is_named_dreaming() -> None:
    """Read in the engine's source, because the turn itself needs a runtime."""
    from pathlib import Path

    engine = Path("core/brain/cognitive_engine.py").read_text(encoding="utf-8")
    assert "state.cognition.current_mode = CognitiveMode.DREAMING" in engine


def test_every_declared_mode_now_has_a_writer() -> None:
    import subprocess

    for mode in ("REACTIVE", "DELIBERATE", "DREAMING", "DORMANT"):
        found = subprocess.run(
            ["grep", "-rIl", "--include=*.py", f"current_mode = CognitiveMode.{mode}", "core"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert found.stdout.strip(), f"{mode} is declared and never assigned"
