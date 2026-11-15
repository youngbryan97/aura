"""Give each owned body a finite programme of control experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.agency.which_way_she_faces import PRESSES
from core.verify import invariant

BodyOwner = tuple[int, int, float]
MAX_CALIBRATION_OWNERS = 64
MAX_CALIBRATION_CONTROLS = 32


def _identity_names(hers: Any, mine: Any) -> tuple[int, int] | None:
    epoch = getattr(getattr(hers, "identification", None), "epoch", None)
    number = getattr(hers, "number", None)
    seen_number = getattr(mine, "number", None)
    if (type(epoch) is not int or epoch < 0 or type(number) is not int or number < 0
            or type(seen_number) is not int or seen_number != number):
        return None
    return epoch, number


def calibration_owner(hers: Any, mine: Any) -> BodyOwner | None:
    """A current matching body needs its own finite birth and observation epoch."""
    names = _identity_names(hers, mine)
    born = getattr(mine, "born", None)
    if names is None or type(born) not in (int, float):
        return None
    try:
        birth = float(born)
    # not a failure: a birth too large to be a time is none
    except OverflowError:
        return None
    return (*names, birth) if math.isfinite(birth) else None


def uses_total_calibration_cap(hers: Any, mine: Any) -> bool:
    """An older caller without a birth keeps the original total cap, without renewal."""
    return _identity_names(hers, mine) is not None and getattr(mine, "born", None) is None


@dataclass
class CurrentBodyCalibration:
    """Issued requests count even when their motion cannot be measured."""

    owner: BodyOwner
    controls: frozenset[str]
    limit: int
    issued: int = 0

    def take_trial(self) -> bool:
        if self.issued >= self.limit:
            return False
        self.issued += 1
        return True


def calibration_for_current_body(run: Any, hers: Any, mine: Any, *,
                                 minimum: int) -> CurrentBodyCalibration | None:
    """Keep the same owner's count through missing pictures and changed key ordering."""
    owner = calibration_owner(hers, mine)
    if owner is None or type(minimum) is not int or minimum < 1:
        return None
    controls = set()
    for key in run.keys:
        if isinstance(key, str) and key:
            controls.add(key)
            if len(controls) > MAX_CALIBRATION_CONTROLS:
                return None
    programmes = getattr(run, "_facing_calibration_programmes", None)
    if programmes is None:
        programmes = {}
    if owner not in programmes:
        if len(programmes) >= MAX_CALIBRATION_OWNERS:
            return None
        # Each repeated control needs an anchor; neutral travel needs its own repeats.
        limit = max(minimum, 2 * PRESSES * len(controls) + PRESSES)
        programmes[owner] = CurrentBodyCalibration(owner, frozenset(controls), limit)
        run._facing_calibration_programmes = programmes
    return programmes[owner]


@invariant("agency.calibration_belongs_to_the_current_owned_body", scope="agency",
           owner="core/agency/calibration_for_a_current_body.py", observational=False)
def _current_body_calibration_invariant() -> tuple:
    from types import SimpleNamespace

    run = SimpleNamespace(keys=["drive", "bend", "bend", ""])
    hers = SimpleNamespace(number=4, identification=SimpleNamespace(epoch=2))
    mine = SimpleNamespace(number=4, born=1.)
    first = calibration_for_current_body(run, hers, mine, minimum=10)
    assert first is not None and first.limit == 10
    assert all(first.take_trial() for _ in range(first.limit)) and not first.take_trial()
    assert calibration_owner(hers, SimpleNamespace(number=4)) is None
    assert calibration_for_current_body(run, hers, SimpleNamespace(number=4), minimum=10) is None
    assert calibration_for_current_body(run, hers, mine, minimum=10) is first
    hers.identification.epoch += 1
    second = calibration_for_current_body(run, hers, mine, minimum=10)
    assert second is not None and second is not first and second.issued == 0
    assert second.take_trial() and first.issued == first.limit
    for epoch in range(4, MAX_CALIBRATION_OWNERS + 2):
        hers.identification.epoch = epoch
        assert calibration_for_current_body(run, hers, mine, minimum=10) is not None
    hers.identification.epoch += 1
    assert calibration_for_current_body(run, hers, mine, minimum=10) is None
    hers.identification.epoch = first.owner[0]
    assert calibration_for_current_body(run, hers, mine, minimum=10) is first
    assert not first.take_trial()
    oversized = SimpleNamespace(keys=[f"key_{n}" for n in range(MAX_CALIBRATION_CONTROLS + 1)])
    assert calibration_for_current_body(oversized, hers, mine, minimum=10) is None
    return ()
