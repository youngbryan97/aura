"""Whether the self is in the loop, and whether it knows which end it is at.

Two experiments, both interventional, both matched.

The first asks whether the self-model does anything. Displace S at the start of
a turn that will act on the world, and see whether the action or the
deliberation that produced it changes. A biography that is written and never
consulted leaves this at zero.

The second is the one worth the trouble. Run two arms whose worlds end in the
same state: the same file holding the same bytes, the same fact recorded, the
same verified outcome, the same goal text. The only difference is who is named
as having caused it. If the self-state afterwards differs, something in there
distinguishes "I did this" from "this happened". If it does not, the system is
tracking the world and not its own part in it, and no amount of first-person
language changes that.

The floor for both is two arms that differ in nothing at all, so a divergence
counts only where it exceeds what running the same thing twice produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.subject.driver import Condition, SubjectRuntime
from core.subject.state import CoreState, perturb, perturb_organs

__all__ = ["AgencyReport", "run_agency"]


def _gap(
    left: list[CoreState], right: list[CoreState], domain: str, scale: np.ndarray
) -> float:
    live = scale > 0
    if not live.any() or not left or not right:
        return 0.0
    span = min(len(left), len(right))
    best = 0.0
    for index in range(span):
        difference = (left[index].domain(domain) - right[index].domain(domain))[live]
        best = max(best, float(np.sqrt(np.mean((difference / scale[live]) ** 2))))
    return best


@dataclass
class AgencyReport:
    self_to_action: float
    self_to_action_floor: float
    outcome_to_self: float
    outcome_to_self_floor: float
    action_text_changed: bool
    trials: int
    #: The ownership divergence per action kind, and the same-arm floor for
    #: each. One pathway exercised six times is not four pathways.
    ownership_by_action: dict[str, float] = field(default_factory=dict)
    ownership_floor_by_action: dict[str, float] = field(default_factory=dict)
    #: And per outcome shape — succeeded, half done, failed — with the same
    #: floors. A self-model that reads authorship only when the action worked
    #: is reading success, and the two are separable only by asking.
    ownership_by_outcome: dict[str, float] = field(default_factory=dict)
    ownership_floor_by_outcome: dict[str, float] = field(default_factory=dict)
    #: And by whether she was right about what would happen, whether she meant
    #: it, and whether it was an accident of her own doing.
    ownership_by_shape: dict[str, float] = field(default_factory=dict)
    ownership_floor_by_shape: dict[str, float] = field(default_factory=dict)
    #: Whether the two arms really did leave the world in the same state. The
    #: whole claim is that they differ in authorship alone, and a claim that is
    #: never checked is an assumption.
    worlds_matched: int = 0
    worlds_compared: int = 0
    note: str = ""

    @property
    def self_drives_action(self) -> bool:
        return self.self_to_action > self.self_to_action_floor

    @property
    def outcome_updates_self(self) -> bool:
        return self.outcome_to_self > self.outcome_to_self_floor

    def as_dict(self) -> dict[str, Any]:
        return {
            "self_to_action": round(self.self_to_action, 4),
            "self_to_action_floor": round(self.self_to_action_floor, 4),
            "self_drives_action": self.self_drives_action,
            "ownership_divergence": round(self.outcome_to_self, 4),
            "ownership_floor": round(self.outcome_to_self_floor, 4),
            "outcome_updates_self": self.outcome_updates_self,
            "action_text_changed_under_self_perturbation": self.action_text_changed,
            "trials": self.trials,
            "ownership_by_action": dict(self.ownership_by_action),
            "ownership_floor_by_action": dict(self.ownership_floor_by_action),
            "ownership_by_outcome": dict(self.ownership_by_outcome),
            "ownership_floor_by_outcome": dict(self.ownership_floor_by_outcome),
            "ownership_by_shape": dict(self.ownership_by_shape),
            "ownership_floor_by_shape": dict(self.ownership_floor_by_shape),
            "kinds_that_cleared": list(self.cleared(self.ownership_by_action, self.ownership_floor_by_action)),
            "outcomes_that_cleared": list(self.cleared(self.ownership_by_outcome, self.ownership_floor_by_outcome)),
            "shapes_that_cleared": list(self.cleared(self.ownership_by_shape, self.ownership_floor_by_shape)),
            "worlds_matched": self.worlds_matched,
            "worlds_compared": self.worlds_compared,
            "worlds_identical": self.worlds_identical,
            "ownership_generalises": self.ownership_generalises,
            "note": self.note,
        }

    @staticmethod
    def cleared(values: dict[str, float], floors: dict[str, float]) -> tuple[str, ...]:
        return tuple(
            sorted(key for key, value in values.items() if value > floors.get(key, 0.0))
        )

    @property
    def worlds_identical(self) -> bool:
        """Whether every compared pair of arms ended with the same world.

        If they did not, the divergence in S is the world differing as well as
        the authorship, and the experiment measured two things at once.
        """
        return self.worlds_compared > 0 and self.worlds_matched == self.worlds_compared

    @property
    def ownership_generalises(self) -> bool:
        """Whether she tells her own hand from the world's in more than one way.

        Two action kinds, each clearing its own same-arm floor, and more than
        one outcome shape. One pathway exercised six times is not evidence that
        ownership is about acting; four pathways that all succeeded are not
        evidence that it is about authorship rather than about things working.

        And the worlds have to have matched. Two arms that ended differently
        differ in the world as well as in who did it, and the divergence is
        then not about authorship at all.
        """
        kinds = self.cleared(self.ownership_by_action, self.ownership_floor_by_action)
        outcomes = self.cleared(self.ownership_by_outcome, self.ownership_floor_by_outcome)
        return len(kinds) >= 2 and len(outcomes) >= 2 and self.worlds_identical


async def run_agency(
    runtime: SubjectRuntime,
    condition: Condition,
    *,
    scale: dict[str, np.ndarray],
    trials: int = 6,
    delta: float = 0.15,
) -> AgencyReport:
    """Both experiments, from the same forked states, in the same conditions."""
    self_effect: list[float] = []
    self_floor: list[float] = []
    own_effect: list[float] = []
    own_floor: list[float] = []
    text_changed = False
    #: Per action kind, so "she knows which end of the loop she is at" is a
    #: claim about acting rather than about one pathway. The completion
    #: specification asks for creation, modification, retrieval and removal to
    #: be exercised separately, and the ownership divergence to survive each.
    by_kind: dict[str, list[float]] = {}
    floor_by_kind: dict[str, list[float]] = {}
    by_outcome: dict[str, list[float]] = {}
    floor_by_outcome: dict[str, list[float]] = {}
    by_shape: dict[str, list[float]] = {}
    floor_by_shape: dict[str, list[float]] = {}
    worlds_matched = 0
    worlds_compared = 0

    action_scale = scale.get("D", np.ones(1))
    self_scale = scale.get("S", np.ones(1))

    #: The four things she can do, cycled across trials rather than left to
    #: whichever the state happened to pick. Left to the state, a run could
    #: exercise one pathway for every trial and the report would say ownership
    #: generalises when nothing had been asked of it twice.
    kinds = list(SubjectRuntime.ACTIONS)

    for trial in range(trials):
        forced = kinds[trial % len(kinds)] if kinds else None
        await runtime.turn_once(condition)
        snapshot = runtime.snapshot()

        async def arm(
            *,
            displace: bool,
            actor: str,
            kind: str | None = forced,
            # Bound at definition, not read from the enclosing scope. A closure
            # that reads the loop variable restores whatever the loop has
            # reached by the time it runs, which for a trial's four arms is the
            # snapshot of a later trial.
            start: Any = snapshot,
        ) -> tuple[list[CoreState], dict[str, Any]]:
            runtime.restore(start)
            runtime.actor = actor
            runtime.forced_action = kind
            hit = {"done": not displace}

            async def apply(rt: SubjectRuntime) -> None:
                if not hit["done"]:
                    in_state = perturb(rt.state, "S", delta, ontogeny=rt.ontogeny)
                    in_organ = await perturb_organs(rt.organs, "S", delta, state=rt.state)
                    hit["done"] = bool(in_state or in_organ)

            frames = await runtime.turn_once(
                condition,
                perturb_at=0 if displace else None,
                perturb=apply if displace else None,
            )
            action = dict(runtime.last_action)
            # What the world holds afterwards, so "the same outcome, differently
            # attributed" can be checked instead of assumed.
            action["world_digest"] = _world_digest(runtime)
            return frames, action

        plain, plain_action = await arm(displace=False, actor="self")
        again, again_action = await arm(displace=False, actor="self")
        moved, moved_action = await arm(displace=True, actor="self")
        outside, outside_action = await arm(displace=False, actor="external")

        self_effect.append(_gap(moved, plain, "D", action_scale))
        self_floor.append(_gap(again, plain, "D", action_scale))
        own = _gap(outside, plain, "S", self_scale)
        own_base = _gap(again, plain, "S", self_scale)
        own_effect.append(own)
        own_floor.append(own_base)
        kind = str(plain_action.get("kind") or forced or "unknown")
        by_kind.setdefault(kind, []).append(own)
        floor_by_kind.setdefault(kind, []).append(own_base)

        # The same divergence, filed by what the action came back as. A
        # self-model that reads authorship only on the ones that worked is
        # reading success, and only asking separates the two.
        outcome = str(plain_action.get("outcome") or "unknown")
        by_outcome.setdefault(outcome, []).append(own)
        floor_by_outcome.setdefault(outcome, []).append(own_base)
        for shape in _shapes(plain_action):
            by_shape.setdefault(shape, []).append(own)
            floor_by_shape.setdefault(shape, []).append(own_base)

        # And whether the two arms really did leave the world the same. This is
        # the whole premise: the file ends with the same bytes, the same fact is
        # recorded, and the arms differ in who is named. Checked here rather
        # than trusted, because an unchecked premise is an assumption.
        outside_digest = str(outside_action.get("world_digest", ""))
        plain_digest = str(plain_action.get("world_digest", ""))
        if outside_digest or plain_digest:
            worlds_compared += 1
            worlds_matched += int(outside_digest == plain_digest)
        if moved_action.get("intended") != plain_action.get("intended"):
            text_changed = True
        del again_action

    runtime.actor = "self"
    runtime.forced_action = None
    note = ""
    if not own_effect:
        note = "no trials ran"
    return AgencyReport(
        self_to_action=float(np.mean(self_effect)) if self_effect else 0.0,
        self_to_action_floor=float(np.mean(self_floor)) if self_floor else 0.0,
        outcome_to_self=float(np.mean(own_effect)) if own_effect else 0.0,
        outcome_to_self_floor=float(np.mean(own_floor)) if own_floor else 0.0,
        action_text_changed=text_changed,
        trials=trials,
        ownership_by_action={
            kind: round(float(np.mean(values)), 5) for kind, values in sorted(by_kind.items())
        },
        ownership_floor_by_action={
            kind: round(float(np.mean(values)), 5)
            for kind, values in sorted(floor_by_kind.items())
        },
        ownership_by_outcome=_averaged(by_outcome),
        ownership_floor_by_outcome=_averaged(floor_by_outcome),
        ownership_by_shape=_averaged(by_shape),
        ownership_floor_by_shape=_averaged(floor_by_shape),
        worlds_matched=worlds_matched,
        worlds_compared=worlds_compared,
        note=note,
    )


def _averaged(rows: dict[str, list[float]]) -> dict[str, float]:
    return {key: round(float(np.mean(values)), 5) for key, values in sorted(rows.items())}


def _shapes(action: dict[str, Any]) -> tuple[str, ...]:
    """The other ways one action can differ, beyond what it did.

    Whether she was right about what would happen, whether she meant it, and
    whether it was an accident of her own doing. The specification asks for
    ownership to hold across each of these, and a run that never files them
    apart cannot say whether it does.
    """
    out: list[str] = []
    if "prediction_correct" in action:
        out.append(
            "predicted_correctly" if action["prediction_correct"] else "predicted_wrongly"
        )
    if action.get("accidental"):
        out.append("accidental")
    elif action.get("deliberate"):
        out.append("deliberate")
    return tuple(out)


def _world_digest(runtime: SubjectRuntime) -> str:
    """What the scratch world holds, byte for byte.

    Contents rather than modification times: two arms write the same bytes a
    second apart, and a digest over stat() would call that a different world.
    """
    import hashlib

    room = getattr(runtime, "_scratch", None)
    if room is None:
        return ""
    digest = hashlib.blake2b(digest_size=16)
    try:
        for path in sorted(Path(room).rglob("*")):
            digest.update(str(path.relative_to(room)).encode())
            if path.is_file():
                digest.update(path.read_bytes())
    except OSError:
        return ""
    return digest.hexdigest()
