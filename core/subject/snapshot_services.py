"""What the fork carries of the services the container has built and of the layers the harness steps.

Lifted whole out of `snapshot`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _service_state(
    only: set[str] | None = None,
    skip: frozenset[int] = frozenset(),
    organs: frozenset[int] = frozenset(),
) -> dict[str, dict[str, Any]]:
    """The mutable state of everything the container has already built.

    Each object once. A service registered under two names is one object, and
    a service holding another service as a field (the authority keeps the
    somatic gate, the gate keeps interoception, three of them keep the
    neurochemical system) is holding something `skip` says is carried under
    its own name. Copying it again under every owner made one snapshot 230 MB,
    and a worker holding 128 of them held 29 GB: six workers restarted the
    machine on 22 September. A service that is one of the organs (`organs`)
    is carried with them and not again here.
    """
    from .snapshot import (
        _UNFORKED_SERVICES,
        _built_services,
        _organ_state,
    )

    out: dict[str, dict[str, Any]] = {}
    captured_ids: set[int] = set(organs)
    for name, instance in _built_services().items():
        if name in _UNFORKED_SERVICES:
            continue
        if only is not None and name not in only:
            continue
        if id(instance) in captured_ids:
            continue
        captured_ids.add(id(instance))
        try:
            captured = _organ_state(instance, skip=skip)
        except (
            ArithmeticError,
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            # A service that cannot be read is skipped, by kind rather than by
            # catching everything.
            continue
        if captured:
            out[name] = captured
    return out


def _restore_services(saved: Mapping[str, dict[str, Any]]) -> None:
    from .snapshot import (
        _built_services,
        _is_published_value,
        _republish,
        _restore_organ,
    )

    if not saved:
        return
    built = _built_services()
    # Published values first. Each is republished as a new object, and an organ
    # that keeps one as a field is relinked to the live service of that name
    # (`_LINK`); restored before the republish, it was pointed at the object the
    # registry was about to replace, and the unity runtime's six `_last_*`
    # records came back as objects nothing else held.
    for name, fields in saved.items():
        instance = built.get(name)
        if instance is not None and _is_published_value(instance):
            _republish(name, instance, fields)
    for name, fields in saved.items():
        instance = built.get(name)
        if instance is not None and not _is_published_value(instance):
            _restore_organ(instance, fields)


def _is_published_value(instance: Any) -> bool:
    """A frozen dataclass the container holds: a value published each tick, not an organ.

    `mind_moment`, `aura_now`, `ghost_snapshot` and `continuous_experience_frame`
    are replaced every tick rather than changed, and a frozen dataclass refuses
    field writes. The in-place restore skips anything that guards its own
    writes, so all four came back from a restore holding whatever the arm had
    published last, and the fork check found them there.
    """
    from .snapshot import (
        is_dataclass,
    )

    params = getattr(type(instance), "__dataclass_params__", None)
    return bool(is_dataclass(instance) and params is not None and params.frozen)


def _layer_state(organism: Any, carried: frozenset[int]) -> dict[str, dict[str, Any]]:
    """Each stepped layer's object that no service or organ already carries.

    The harness steps these once a frame (core/subject/steppable.py), so each
    holds state an arm moves. The consciousness bridge the organism builds for
    itself is registered under no name, so neither the services nor the organs
    carried it: `_chemistry_tick_seen` survived from one arm into the next, the
    first arm pulled her substrate towards her chemistry and the second did
    not, and two untouched forks parted the first frame the bridge ticked
    (seed 7, 28 September). What a layer object holds that is carried under a
    name of its own is skipped, as it is for services.
    """
    from .snapshot import (
        _organ_state,
    )

    from core.subject.steppable import layers_of

    if organism is None:
        return {}
    return {
        name: _organ_state(target, skip=carried)
        for name, target in layers_of(organism).items()
        if id(target) not in carried
    }


def _restore_layers(organism: Any, saved: Mapping[str, dict[str, Any]] | None) -> None:
    from .snapshot import (
        _restore_organ,
    )

    from core.subject.steppable import layers_of

    if not saved or organism is None:
        return
    for name, target in layers_of(organism).items():
        fields = saved.get(name)
        if fields is not None:
            _restore_organ(target, fields)


def _republish(name: str, current: Any, saved: Mapping[str, Any]) -> None:
    """Publish the saved value under its name again, as a new object.

    The object the arm published is left alone: anything else holding it holds
    a value, and a value is rewound by replacing it, not by writing into it.
    """
    from .snapshot import (
        _NESTED,
        _place,
        _restore_organ,
        copy,
        record_degradation,
    )

    rebuilt = copy.copy(current)
    for field_name, value in saved.items():
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _NESTED:
            _restore_organ(getattr(rebuilt, field_name, None), value[1])
            continue
        # A new object nothing else holds yet, so its frozen guard has no one
        # to protect.
        object.__setattr__(rebuilt, field_name, _place(value))
    try:
        from core.container import ServiceContainer

        ServiceContainer.set(name, rebuilt, required=False)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "subject_snapshot",
            exc,
            action=f"left {name} holding the value the arm published",
        )


