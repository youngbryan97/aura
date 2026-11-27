"""Where her results against another player vary, and her games there as the judge of what her practice keeps.

A player who beats her every game teaches her nothing by its results, and neither does one she always beats. Where the
world lets her set another player's strength (how deep it looks, how long it takes a move, a level), her points against
it at each setting she has played are kept, and she plays next at the setting where her expected score is nearest even:
where a logistic curve of her points against the log of the setting crosses a half. Until her games have gone both
ways, the next setting is a step past the last toward where they would: half the setting after games all lost, twice
it after games all won.

Everything her practice judges, it judges against herself, and LIVE 26-27 November 2026 everything built from her own
search (practice against herself at every depth, her own deeper look, a deep first guess) preferred weights that a
stronger outside player beat more easily. So what practice keeps is checked here, against the outside player at the
setting where her results vary. Her games there after a keep are paired with her games there before it, and the
sequential test (core/cognition/whether_one_way_is_better.py) is asked whether the games before were better. Where it
says so, the weights she had before the keep are put back. Games in the world are few, so only a keep that cost her a
great deal can be caught: the margin is what as many pairs as the trial is given can tell.

Nothing here knows a game, a player or a setting's name: a setting is a positive number, larger for a stronger player.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = ["KeepOnTrial", "Ladder", "came_from_a_game", "open_a_trial", "the_setting_to_play"]

#: Games after a keep that its trial is given: about an evening of games in the world. A test of as many pairs tells a
#: loss of about one game in eight.
GAMES_A_TRIAL = 40


@dataclass
class Ladder:
    """Her points against one player at each setting of its strength she has played."""

    results: dict[float, list[float]] = field(default_factory=dict)

    def came(self, setting: float, points: float) -> None:
        self.results.setdefault(float(setting), []).append(float(points))

    def next_setting(self, lowest: float, highest: float | None = None) -> float:
        """The setting to play next, no lower than ``lowest`` and no higher than ``highest`` where the world's highest is
        known. With no games played and no highest known, the lowest: doubling up from it finds where her results turn
        in as many games as the doublings it takes."""
        top = math.inf if highest is None else highest
        played = {s: p for s, p in self.results.items() if p and lowest <= s <= top}
        if not played:
            return lowest if highest is None else math.sqrt(lowest * highest)
        points = [x for p in played.values() for x in p]
        if all(x == points[0] for x in points) and points[0] in (0.0, 1.0):
            last = sorted(played, key=lambda s: -len(played[s]))[0]
            stepped = last / 2 if points[0] == 0.0 else last * 2
            return min(top, max(lowest, stepped))
        level, slope = self._fitted(played)
        if slope >= 0:
            # A curve that does not fall as the player grows stronger says nothing yet of where it crosses a half: the
            # setting played most is played again.
            return max(played, key=lambda s: len(played[s]))
        return min(top, max(lowest, math.exp(-level / slope)))

    @staticmethod
    def _fitted(played: dict[float, list[float]]) -> tuple[float, float]:
        """Her chance of a point as a logistic curve of the log of the setting, fitted by maximum likelihood, held back a
        little toward a flat curve at even so that a few games do not make it a cliff."""
        xs = [math.log(s) for s, p in played.items() for _ in p]
        ys = [x for p in played.values() for x in p]
        level, slope = 0.0, 0.0
        held_back = 1.0 / max(1, len(ys))
        for _ in range(50):
            g0 = g1 = h00 = h01 = h11 = 0.0
            for x, y in zip(xs, ys):
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, level + slope * x))))
                w = p * (1 - p)
                g0 += p - y
                g1 += (p - y) * x
                h00 += w
                h01 += w * x
                h11 += w * x * x
            g0 += held_back * level
            g1 += held_back * slope
            h00 += held_back
            h11 += held_back
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-12:
                break
            step0 = (h11 * g0 - h01 * g1) / det
            step1 = (h00 * g1 - h01 * g0) / det
            level -= step0
            slope -= step1
            if max(abs(step0), abs(step1)) < 1e-9:
                break
        return level, slope

    def as_dict(self) -> dict[str, list[float]]:
        return {str(s): list(p) for s, p in self.results.items()}

    @classmethod
    def from_dict(cls, held: dict[str, Any]) -> Ladder:
        return cls(results={float(s): [float(x) for x in p] for s, p in (held or {}).items()})


@dataclass
class KeepOnTrial:
    """A keep of her practice on trial at one setting: her points there before it, her points since, and the weights
    she had before it, to put back."""

    setting: float
    before: list[float]
    weights_before: dict[str, Any]
    after: list[float] = field(default_factory=list)

    def came(self, points: float) -> str | None:
        """One more game since the keep. "put back" where the games before it were better by what the trial can tell,
        "kept" where they were not, None while the games so far show neither."""
        from core.cognition.whether_one_way_is_better import Comparison

        self.after.append(float(points))
        pairs = [(b + 1.0 - a) / 2.0 for b, a in zip(self.before[-len(self.after):], self.after)]
        if not pairs:
            return None
        spread = 0.25
        if len(pairs) > 1:
            mean = sum(pairs) / len(pairs)
            spread = math.sqrt(sum((x - mean) ** 2 for x in pairs) / (len(pairs) - 1)) or 0.25
        test = Comparison(better=0.5 + 2 * spread / math.sqrt(GAMES_A_TRIAL))
        said = None
        for pair in pairs:
            said = test.add(pair)
            if said:
                break
        if said == "better":
            return "put back"
        if said == "no better" or len(self.after) >= min(GAMES_A_TRIAL, len(self.before)):
            return "kept"
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"setting": self.setting, "before": self.before, "after": self.after,
                "weights_before": self.weights_before}

    @classmethod
    def from_dict(cls, held: dict[str, Any]) -> KeepOnTrial:
        return cls(setting=float(held["setting"]), before=[float(x) for x in held.get("before", [])],
                   weights_before=dict(held.get("weights_before") or {}),
                   after=[float(x) for x in held.get("after", [])])


#: The weakest a setting of another player's strength is taken to go: a setting is a count, and one is the least.
LOWEST = 1.0


def the_setting_to_play(rehearsal: Any) -> int:
    """The setting her next practice game is played at: a keep's trial's, while one is open, so its games compare with
    the games before it; else where her results vary. A whole number, for a setting is typed as one."""
    trial = getattr(rehearsal, "on_trial", None)
    if trial:
        return max(1, round(float(trial["setting"])))
    ladder = Ladder.from_dict(getattr(rehearsal, "results_vary", None) or {})
    return max(1, round(ladder.next_setting(LOWEST)))


def came_from_a_game(rehearsal: Any, setting: float, points: float) -> str | None:
    """A practice game's points at ``setting``, kept; and, where a keep is on trial at that setting, what its trial
    says: "put back" where the weights before it were better, and they are put back; "kept" where its trial ended
    without that; None while it goes on or where none is open."""
    from core.agency.searching_by_kind import Weights

    ladder = Ladder.from_dict(getattr(rehearsal, "results_vary", None) or {})
    ladder.came(float(setting), float(points))
    rehearsal.results_vary = ladder.as_dict()
    held = getattr(rehearsal, "on_trial", None)
    if not held or float(held["setting"]) != float(setting):
        return None
    trial = KeepOnTrial.from_dict(held)
    said = trial.came(points)
    rehearsal.on_trial = None if said else trial.as_dict()
    if said == "put back":
        rehearsal.weights = Weights(**trial.weights_before)
    return said


def open_a_trial(rehearsal: Any, weights_before: Any) -> bool:
    """A keep of her practice put on trial at the setting where she has played most with her results varying, against
    her games there before it; whether one was opened. None is opened while another is, or where no setting yet has
    games that went more than one way."""
    from dataclasses import asdict

    if getattr(rehearsal, "on_trial", None):
        return False
    ladder = Ladder.from_dict(getattr(rehearsal, "results_vary", None) or {})
    varied = {s: p for s, p in ladder.results.items() if len(set(p)) > 1}
    if not varied:
        return False
    setting = max(varied, key=lambda s: len(varied[s]))
    rehearsal.on_trial = KeepOnTrial(setting=setting, before=list(varied[setting][-GAMES_A_TRIAL:]),
                                     weights_before=asdict(weights_before)).as_dict()
    return True
