"""The dominance neuron had readers and no writer, and self-state had no channel into C.

The substrate declares three VAD neurons and `idx_dominance` is the third: in the
circumplex it is the sense of being in control of one's situation. The aesthetic
engine reads it, the substrate gates name it, two of the substrate's own
summaries publish it, and nothing wrote it.

The substrate is 25 of the 84 columns of recurrent cognition, and
`core/self/will_engine.py` drives motivation's budgets into two of its neurons —
so deliberation has a channel into recurrent cognition and self-state has none.
On whole-s7-27dc1dda9 a displacement of S moves C by 4.23, the second largest of
any domain, while S's unique information about C's next change is exactly 0.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.phases.identity_reflection import IdentityReflectionPhase


class _Substrate:
    idx_dominance = 2

    def __init__(self, start: float = 0.0) -> None:
        self.x = np.zeros(8)
        self.x[self.idx_dominance] = start
        self.marked: list[str] = []

    @property
    def sync_lock(self):
        import contextlib

        return contextlib.nullcontext()

    def mark_state_mutated_locked(self, why: str) -> None:
        self.marked.append(why)


class _Identity:
    def __init__(self, stability: float) -> None:
        self.stability = stability


class _State:
    def __init__(self, stability: float) -> None:
        self.identity = _Identity(stability)
        self.response_modifiers: dict[str, object] = {}


@pytest.fixture
def _substrate(monkeypatch: pytest.MonkeyPatch):
    substrate = _Substrate()

    class _Container:
        @staticmethod
        def get(name, default=None):
            return substrate if name in {"liquid_substrate", "conscious_substrate"} else default

    import core.container

    monkeypatch.setattr(core.container, "ServiceContainer", _Container)
    return substrate


def test_off_unless_a_run_asks_for_it(monkeypatch: pytest.MonkeyPatch, _substrate) -> None:
    monkeypatch.delenv("AURA_SELF_DOMINANCE", raising=False)
    state = _State(1.0)
    IdentityReflectionPhase._hold_together_reaches_the_substrate(state)
    assert _substrate.x[2] == 0.0
    assert "self_dominance" not in state.response_modifiers


def test_holding_together_drives_the_neuron_up(monkeypatch: pytest.MonkeyPatch, _substrate) -> None:
    monkeypatch.setenv("AURA_SELF_DOMINANCE", "1")
    state = _State(1.0)
    IdentityReflectionPhase._hold_together_reaches_the_substrate(state)
    assert _substrate.x[2] == pytest.approx(0.2)
    assert state.response_modifiers["self_dominance"] == pytest.approx(1.0)
    assert "identity_reflection.dominance" in _substrate.marked


def test_coming_apart_drives_it_down(monkeypatch: pytest.MonkeyPatch, _substrate) -> None:
    monkeypatch.setenv("AURA_SELF_DOMINANCE", "1")
    IdentityReflectionPhase._hold_together_reaches_the_substrate(_State(0.0))
    assert _substrate.x[2] == pytest.approx(-0.2)


def test_the_middle_of_her_scale_is_the_neuron_s_rest(
    monkeypatch: pytest.MonkeyPatch, _substrate
) -> None:
    """Her stability is 0 to 1 and the neuron is signed and rests at zero."""
    monkeypatch.setenv("AURA_SELF_DOMINANCE", "1")
    IdentityReflectionPhase._hold_together_reaches_the_substrate(_State(0.5))
    assert _substrate.x[2] == pytest.approx(0.0)


def test_it_blends_rather_than_assigns(monkeypatch: pytest.MonkeyPatch, _substrate) -> None:
    monkeypatch.setenv("AURA_SELF_DOMINANCE", "1")
    for _ in range(40):
        IdentityReflectionPhase._hold_together_reaches_the_substrate(_State(1.0))
    assert 0.9 < _substrate.x[2] <= 1.0


def test_a_substrate_that_is_not_there_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_SELF_DOMINANCE", "1")

    class _Empty:
        @staticmethod
        def get(name, default=None):
            return default

    import core.container

    monkeypatch.setattr(core.container, "ServiceContainer", _Empty)
    IdentityReflectionPhase._hold_together_reaches_the_substrate(_State(1.0))


def test_the_phase_pushes_before_every_early_return(monkeypatch) -> None:
    """What she is does not depend on whether her narrative was revised: the identity guard's hard stop still pushes."""
    import asyncio

    from core.state.aura_state import AuraState

    pushed: list[object] = []
    monkeypatch.setattr(IdentityReflectionPhase, "_hold_together_reaches_the_substrate",
                        staticmethod(lambda state: pushed.append(state)))
    state = AuraState.default()
    state.identity.name = "Somebody Else"
    returned = asyncio.run(IdentityReflectionPhase.__new__(IdentityReflectionPhase).execute(state))
    assert returned is state and state.identity.name == "Aura"
    assert pushed == [state]


def test_only_this_phase_marks_a_dominance_mutation() -> None:
    """It was a reader with no writer, and this is the writer.

    The substrate marks every mutation with the name of what made it, so the
    marker is the record of who writes a neuron. Grepping for an assignment
    would miss this one, which writes through an index held in a local.
    """
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    marks = subprocess.run(
        ["grep", "-rn", "dominance", str(root / "core")],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    writers = [line for line in marks if "mark_state_mutated" in line or ".dominance\"" in line]
    named = [line for line in writers if "identity_reflection.dominance" in line]
    assert named, writers
    assert all("identity_reflection" in line for line in named)
