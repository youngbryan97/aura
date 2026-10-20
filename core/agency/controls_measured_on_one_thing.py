"""A kind can suggest a trial; only that thing's input receipts establish control.

Offline 2026-10-12 pooled movements of a kind lent control to a new piece of
scenery. Its own measurements said nothing moved. Keep each identity's
evidence with that identity, including exclusions found by other readings.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable
from typing import Any

from core.agency.causal_identification import CausalWitness
from core.verify import invariant


def own_control_witness(identity: Hashable, evidence: Any, *, epoch: int,
                        excluded: bool = False, a_gauge: bool = False,
                        with_the_world: bool = False, movement_over: float = 0.) -> CausalWitness | None:
    """Read only the current entity's independent press samples, or leave its control unknown.

    Earlier samples from another member of its kind cannot supply a trial,
    statistic or effect here. A caller clears the entity's experiment when
    continuity is lost; a kind's longer history remains a discovery aid.
    """
    if identity is None or evidence is None or excluded or a_gauge or with_the_world:
        return None
    if movement_over > 0 and not motion_is_resolved(evidence, movement_over):
        return None
    trials: dict[str, int] = defaultdict(int)
    for (key, began), values in evidence.by_press.items():
        if not isinstance(key, str) or not math.isfinite(began) or len(values) < 2:
            continue
        if not all(math.isfinite(value) for pair in values for value in pair):
            continue
        trials[key] += 1
    if sum(count >= 2 for count in trials.values()) < 2:
        return None
    statistic, effect = evidence.ratio()
    if not math.isfinite(statistic) or not math.isfinite(effect):
        return None
    return CausalWitness(identity, epoch, tuple(sorted(trials.items())), statistic, effect)


@invariant("agency.a_kind_cannot_lend_control_authority_to_a_new_thing", scope="agency",
           owner="core/agency/controls_measured_on_one_thing.py", observational=False)
def _own_control_witness_invariant() -> tuple:
    from types import SimpleNamespace

    blank = SimpleNamespace(by_press={}, ratio=lambda: (1000., 1000.))
    assert own_control_witness("new", blank, epoch=0) is None
    repeated = {("left", 1.): [(-30., 0.), (-30., 0.)],
                ("left", 3.): [(-30., 0.), (-30., 0.)],
                ("right", 5.): [(30., 0.), (30., 0.)],
                ("right", 7.): [(30., 0.), (30., 0.)]}
    measured = SimpleNamespace(by_press=repeated, ratio=lambda: (80., 60.))
    witness = own_control_witness("current", measured, epoch=4)
    assert witness is not None and witness.identity == "current" and witness.epoch == 4
    assert own_control_witness("current", measured, epoch=4, with_the_world=True) is None
    assert own_control_witness("current", measured, epoch=4, a_gauge=True) is None
    return ()


def own_trajectory_witness(thing: Any, receipts: Any, *, epoch: int,
                           statistic_over: float = 20., effect_over: float = 20.) -> CausalWitness | None:
    """Compare repeated paths on one continuing body, including a path that returns to its start."""
    import itertools
    from core.agency.a_contrast_that_meets_both_tests import contrast_rank

    from core.agency.what_a_press_does import _where

    groups: dict[str, list] = defaultdict(list)
    ended = -math.inf
    for trial in sorted(receipts, key=lambda value: value.began):
        if (trial.number != thing.number or trial.born != getattr(thing, "born", None) or trial.epoch != epoch
                or trial.began < ended or len(trial.samples) < 3
                or not all(math.isfinite(v) for point in trial.samples for v in point)
                or not math.isfinite(trial.began) or not math.isfinite(trial.ended)
                or trial.ended <= trial.began or trial.ended > getattr(thing, "seen", -math.inf)):
            continue
        times = [point[0] for point in trial.samples]
        if times[0] <= 0 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
            continue
        groups[trial.key].append(trial)
        ended = trial.ended
    groups = {key: trials[-4:] for key, trials in groups.items() if len(trials) >= 2}
    best = None
    for first, second in itertools.combinations(groups, 2):
        trials = groups[first] + groups[second]
        start = max(trial.samples[0][0] for trial in trials)
        end = min(trial.samples[-1][0] for trial in trials)
        if end - start < .05:
            continue
        points = [start + (end - start) * i / 8 for i in range(9)]
        paths = {key: [[_where(trial.samples, at) for at in points] for trial in groups[key]] for key in (first, second)}
        means = {key: [tuple(sum(path[i][axis] for path in paths[key]) / len(paths[key]) for axis in (0, 1))
                       for i in range(len(points))] for key in paths}
        noise = max(math.dist(path[i], means[key][i]) for key in paths for path in paths[key] for i in range(len(points)))
        effect = max(math.dist(a, b) for a, b in zip(means[first], means[second], strict=True))
        statistic = effect ** 2 / (noise ** 2 + .25)
        witness = CausalWitness(thing.number, epoch, ((first, len(groups[first])), (second, len(groups[second]))),
                                statistic, effect)
        if best is None or contrast_rank((witness.statistic, witness.effect), statistic_over=statistic_over,
                effect_over=effect_over, strict=True) > contrast_rank((best.statistic, best.effect),
                statistic_over=statistic_over, effect_over=effect_over, strict=True):
            best = witness
    return best


@invariant("agency.trajectory_control_keeps_its_entity_and_trials", scope="agency",
           owner="core/agency/controls_measured_on_one_thing.py", observational=False)
def _trajectory_control_invariant() -> tuple:
    from types import SimpleNamespace

    from core.agency.what_a_press_does import PressTrajectory

    body = SimpleNamespace(number=1, born=0., seen=10.)
    trials = [PressTrajectory(key, 1, 3, began, began + .4, 0.,
                              ((.1, 0., dy), (.2, 0., 2 * dy), (.4, 0., 0.)))
              for key, began, dy in (("a", 1., 30.), ("", 3., 0.), ("a", 5., 30.), ("", 7., 0.))]
    witness = own_trajectory_witness(body, trials, epoch=3)
    assert witness is not None and witness.effect > 20 and witness.statistic > 20
    assert own_trajectory_witness(body, trials, epoch=4) is None
    assert own_trajectory_witness(body, [trials[0], trials[0], trials[1], trials[1]], epoch=3) is None
    return ()


def motion_is_resolved(evidence: Any, least: float) -> bool:
    """Velocity derived from pixels needs an observed excursion beyond the tracker resolution.

    Offline 2026-10-12 a nearly stationary fragment's fitted acceleration
    passed a control test. Its measured positions had never left the pixel
    uncertainty. Differentiating that uncertainty did not create movement.
    """
    low, high = getattr(evidence, "lowest", None), getattr(evidence, "highest", None)
    if low is None or high is None or len(low) != 2 or len(high) != 2:
        return False
    if not all(math.isfinite(value) for value in (*low, *high, least)) or least <= 0:
        return False
    return max(upper - lower for lower, upper in zip(low, high, strict=True)) > least


@invariant("agency.pixel_control_requires_resolved_displacement", scope="agency",
           owner="core/agency/controls_measured_on_one_thing.py", observational=False)
def _resolved_motion_invariant() -> tuple:
    from types import SimpleNamespace

    evidence = SimpleNamespace(lowest=(50., 50.), highest=(53., 50.))
    assert not motion_is_resolved(evidence, 4.)
    evidence.highest = (70., 50.)
    assert motion_is_resolved(evidence, 4.)
    evidence.lowest = (math.inf, math.inf)
    assert not motion_is_resolved(evidence, 4.)
    return ()
