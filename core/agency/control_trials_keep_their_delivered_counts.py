"""Share delivered discovery probes across strategies before a body is known.

Exposure counts order experiments. They supply no observed response or
independent trial evidence to the identity estimator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from core.agency.calibration_for_a_current_body import MAX_CALIBRATION_CONTROLS
from core.agency.unknown_controls_get_their_turn import _bounded_controls
from core.verify import invariant


@dataclass
class DeliveredProbes:
    scope: tuple[int, float]
    counts: dict[str, int] = field(default_factory=dict)
    through: float = -math.inf


def _ledger(run: Any, hers: Any) -> DeliveredProbes | None:
    epoch, began = getattr(getattr(hers, 'identification', None), 'epoch', None), getattr(run, 'began', None)
    if (type(epoch) is not int or epoch < 0 or type(began) not in (int, float)
            or not math.isfinite(began)):
        return None
    scope = epoch, began
    kept = getattr(run, '_delivered_body_probes', None)
    if not isinstance(kept, DeliveredProbes) or kept.scope != scope:
        through = -math.inf
        if isinstance(kept, DeliveredProbes) and kept.scope[1] == began:
            if epoch < kept.scope[0]:
                return None
            # A new observation epoch cannot redeliver an old physical hold.
            through = kept.through
        kept = run._delivered_body_probes = DeliveredProbes(scope, through=through)
    return kept


def next_discovery_control(run: Any, hers: Any) -> str | None:
    """Choose among current authorized keys by actual prior delivery, with stable ties."""
    keys = _bounded_controls(run.keys)
    ledger = _ledger(run, hers)
    if not keys or ledger is None or len(set(ledger.counts) | set(keys)) > MAX_CALIBRATION_CONTROLS:
        return None
    return min(keys, key=lambda key: ledger.counts.get(key, 0))


def counter_probes_already_delivered(run: Any, hers: Any, key: str, *, phases: int) -> bool:
    """A preceding strategy can already have spent this pulse/rest programme's pulses."""
    keys = _bounded_controls(run.keys)
    if type(phases) is not int or phases < 2 or not keys or key not in keys:
        return False
    ledger = _ledger(run, hers)
    return ledger is not None and ledger.counts.get(key, 0) >= (phases+1)//2


def record_discovery_delivery(run: Any, hers: Any, key: str, began: float, *, independent: bool) -> bool:
    """A current matching physical hold counts once, including a counter-movement probe."""
    keys = _bounded_controls(run.keys)
    if (independent is not True or not keys or key not in keys or getattr(run, 'held', None) != key
            or getattr(run, 'held_trying', None) is not True or type(began) not in (int, float)
            or not math.isfinite(began) or began != getattr(run, 'held_since', None)):
        return False
    ledger = _ledger(run, hers)
    if (ledger is None or began < ledger.scope[1] or began <= ledger.through
            or len(set(ledger.counts) | {key}) > MAX_CALIBRATION_CONTROLS):
        return False
    ledger.counts[key] = ledger.counts.get(key, 0)+1
    ledger.through = began
    return True


@invariant('agency.discovery_strategies_share_delivered_probe_counts', scope='agency',
           owner='core/agency/control_trials_keep_their_delivered_counts.py', observational=False)
def _delivered_probe_count_invariant() -> tuple:
    from types import SimpleNamespace

    run = SimpleNamespace(keys=['a', 'b', 'c'], began=0., held='a', held_trying=True, held_since=1.)
    hers = SimpleNamespace(identification=SimpleNamespace(epoch=0))
    assert record_discovery_delivery(run, hers, 'a', 1., independent=True)
    assert not counter_probes_already_delivered(run, hers, 'a', phases=4)
    assert not record_discovery_delivery(run, hers, 'a', 1., independent=True)
    run.held_since = 2.
    assert record_discovery_delivery(run, hers, 'a', 2., independent=True)
    assert counter_probes_already_delivered(run, hers, 'a', phases=4)
    assert next_discovery_control(run, hers) == 'b'
    assert not record_discovery_delivery(run, hers, 'b', 2., independent=True)
    assert not record_discovery_delivery(run, hers, 'a', 2., independent=False)
    hers.identification.epoch += 1
    assert next_discovery_control(run, hers) == 'a'
    return ()
