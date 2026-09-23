"""What she does in a campaign turn: what she is used to doing here, or what weighing picks.

Habits as Bryan described them on 22 September: done on autopilot, with more
agency than a reflex, because she is used to doing them, and changeable with
conscious effort. The cue is where she is: the condition and what she is
attending to. When an act has become second nature at the cue
(core/agency/habits_are_hers.py `habit_at`) she takes it without weighing,
unless it has done worse for her than weighing and what it has cost her is
more than she is tired; then she weighs, at the effort weighing costs. Every
choice is kept at its cue, so repetition is what makes a habit and what
unmakes one.

This was a fixed table from what she attended to straight to an act, labelled
weighed. It never compared anything and she never learned it, so no habit
could form: in four rounds she logged 105 weighed acts and one automatic one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.soma.effort import note_effort

__all__ = ["choose", "ranks"]


def choose(runtime: Any) -> tuple[str, str]:
    """The act she takes this turn, and how it was chosen: forced, automatic or weighed."""
    forced = getattr(runtime, "forced_action", None)
    if forced in runtime.ACTIONS:
        return str(forced), "forced"
    attending = str(getattr(runtime.state.cognition, "attention_focus", "") or "")
    source = attending.split(":", 1)[0].strip()
    if source.startswith("affect_"):
        source = "self"
    from core.agency.habits_are_hers import get_habit_ledger
    from core.soma.fatigue import get_fatigue_ledger

    ledger = get_habit_ledger()
    cue = f"{getattr(runtime, '_condition_name', '') or 'turn'}/{source or 'unattended'}"
    habit = ledger.habit_at(cue)
    if habit in runtime.ACTIONS and not ledger.worth_the_effort(habit, get_fatigue_ledger().read().share):
        ledger.note_choice(cue, habit)
        return habit, "automatic"
    chosen = weighed(runtime, source, ledger)
    # Weighing is work, in the unit the body already prices it in: how many
    # things were weighed against each other. See core/soma/effort.py.
    note_effort("candidates", float(len(runtime.ACTIONS)))
    ledger.note_choice(cue, chosen)
    return chosen, "weighed"


def weighed(runtime: Any, source: str, ledger: Any) -> str:
    """Every reason she has ranks the acts, and the act with the most support wins.

    Four reasons: what she is attending to (the broadcast winner reaches action
    selection as one voice, which is the global-workspace claim), how far below
    the fullest the drive an act serves has fallen, what she has come to feel
    about the act (core/affect/feelings_about.py), and what a habit of it has
    cost her against weighing. Each is turned into ranks and the ranks are
    summed, a Borda count, so no reason needs a weight set by hand. Ties go to
    what she is attending to, then to the order the acts are listed in.
    """
    from core.affect.feelings_about import get_feelings_about

    acts = list(runtime.ACTIONS)
    attended = runtime.ATTENTION_ACTIONS.get(source, "")
    budgets = getattr(getattr(runtime.state, "motivation", None), "budgets", {}) or {}
    levels = {
        name: float(entry.get("level", 100.0) or 0.0)
        for name, entry in budgets.items()
        if isinstance(entry, dict)
    }
    fullest = max(levels.values(), default=0.0)

    def pull(act: str) -> float:
        return max(
            (fullest - level for name, level in levels.items() if runtime.DRIVE_ACTIONS.get(name) == act),
            default=0.0,
        )

    feelings = get_feelings_about()
    reasons = (
        [1.0 if act == attended else 0.0 for act in acts],
        [pull(act) for act in acts],
        [float(feelings.pull_for_text(act)) for act in acts],
        [-float(ledger.deficit(act)) for act in acts],
    )
    support = [0.0] * len(acts)
    for values in reasons:
        for index, rank in enumerate(ranks(values)):
            support[index] += rank
    best = max(range(len(acts)), key=lambda i: (support[i], acts[i] == attended, -i))
    return acts[best]


def ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, ties sharing the mean of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranked = [0.0] * len(values)
    position = 0
    while position < len(order):
        tied = position
        while tied + 1 < len(order) and values[order[tied + 1]] == values[order[position]]:
            tied += 1
        for index in order[position : tied + 1]:
            ranked[index] = (position + tied) / 2.0 + 1.0
        position = tied + 1
    return ranked
