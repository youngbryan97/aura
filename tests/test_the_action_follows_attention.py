"""What she does is decided by what she is attending to, not by a coin.

Three defects in one small method. The action was chosen by hashing the name of
the most depleted drive, and `hash()` on a string is salted per process — so
the drive that reached for one action in one run reached for another in the
next, and two runs of the battery were not comparable in what she actually did.
The drives move over days, so whichever one was lowest stayed lowest and the
same action happened every turn. And every action in the repertoire always
succeeded, so the agency ledger's efficacy and authored share sat at one for the
whole of every run and two of the self-state's liveliest columns were constants.
"""

from __future__ import annotations

from core.subject.driver import SubjectRuntime


def test_the_choice_is_not_salted_per_process() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "subject" / "driver.py").read_text()
    assert "hash(levels[0][1])" not in source, "the action is a coin flip again"


def test_every_drive_and_every_attended_source_names_a_real_action() -> None:
    for mapping in (SubjectRuntime.DRIVE_ACTIONS, SubjectRuntime.ATTENTION_ACTIONS):
        for name, action in mapping.items():
            assert action in SubjectRuntime.ACTIONS, f"{name} -> {action}"


def test_every_budget_the_state_carries_can_pick_something() -> None:
    from core.state.aura_state import AuraState

    budgets = set(AuraState.default().motivation.budgets)
    assert budgets <= set(SubjectRuntime.DRIVE_ACTIONS), budgets - set(
        SubjectRuntime.DRIVE_ACTIONS
    )


def test_attention_decides_before_the_drives_do() -> None:
    """The claim global workspace theory makes: what wins reaches the process."""
    from types import SimpleNamespace

    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime.state = SimpleNamespace(
        cognition=SimpleNamespace(attention_focus="perception: a window moved"),
        motivation=SimpleNamespace(budgets={"growth": {"level": 1.0}}),
    )
    assert runtime._chosen_action() == SubjectRuntime.ATTENTION_ACTIONS["perception"]

    # With nothing attended, the most depleted drive decides instead. The
    # levels are read from every budget the state carries, so a stand-in has to
    # name the ones the driver looks at.
    runtime.state.cognition.attention_focus = ""
    runtime.state.motivation.budgets = {
        "growth": {"level": 1.0},
        "curiosity": {"level": 90.0},
        "social": {"level": 90.0},
    }
    assert runtime._chosen_action() == SubjectRuntime.DRIVE_ACTIONS["growth"]


def test_a_feeling_that_wins_is_attention_to_herself() -> None:
    from types import SimpleNamespace

    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime.state = SimpleNamespace(
        cognition=SimpleNamespace(attention_focus="affect_joy: feeling joy"),
        motivation=SimpleNamespace(budgets={}),
    )
    assert runtime._chosen_action() == SubjectRuntime.ATTENTION_ACTIONS["self"]


def test_one_of_them_can_fail_for_a_reason_that_is_hers() -> None:
    """An action repertoire in which nothing can fail cannot teach efficacy.

    `read_room` looks in the room for the previous turn, and that room exists
    only if she chose to make one then.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "subject" / "driver.py").read_text()
    assert "read_room" in SubjectRuntime.ACTIONS
    assert "there is no room" in source
