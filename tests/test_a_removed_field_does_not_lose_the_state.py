"""Removing a field must not make every saved state unloadable.

LIVE, 2026-09-21 boot: `IdentityKernel.__init__() got an unexpected keyword
argument 'met_as_a_type'`. The field had been removed from the class and the
state on disk still carried it, so the WHOLE identity kernel failed to
rehydrate and the runtime came up with a default one — reported as a single
degradation line among many.

Removing a field is an ordinary thing to do. This is what it costs now: the
field is dropped, by name, in the log, and the rest of the state loads.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import pytest

from core.state.state_repository import StateRepository, _declared


@dataclasses.dataclass
class _Kept:
    here: str = "yes"
    also: int = 1


def test_a_field_the_class_no_longer_has_is_dropped() -> None:
    kept = _declared(
        _Kept, {"here": "value", "also": 2, "a_field_that_was_removed": "gone"}
    )
    assert kept == {"here": "value", "also": 2}


def test_it_says_which_field_it_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """A state that quietly loses values is the other way to get this wrong."""
    with caplog.at_level(logging.WARNING, logger="Aura.StateRepository"):
        _declared(_Kept, {"here": "v", "a_field_that_was_removed": "gone"})
    said = " ".join(r.getMessage() for r in caplog.records)
    assert "a_field_that_was_removed" in said
    assert "_Kept" in said


def test_nothing_is_said_when_nothing_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    """The null. A line per load would be a line nobody reads."""
    with caplog.at_level(logging.WARNING, logger="Aura.StateRepository"):
        kept = _declared(_Kept, {"here": "v", "also": 3})
    assert kept == {"here": "v", "also": 3}
    assert not [r for r in caplog.records if "no longer declares" in r.getMessage()]


def test_a_missing_block_is_not_an_error() -> None:
    assert _declared(_Kept, None) == {}
    assert _declared(_Kept, {}) == {}


def test_a_state_with_a_removed_field_still_loads() -> None:
    """End to end: the shape that failed the live boot."""
    from core.state.aura_state import AuraState

    repository = StateRepository.__new__(StateRepository)
    payload = json.loads(repository._serialize(AuraState()))
    payload["identity"]["a_field_that_was_removed"] = "nothing declares this"
    payload["a_whole_block_that_went_away"] = {"x": 1}

    restored = repository._deserialize(json.dumps(payload))

    assert isinstance(restored, AuraState)
    assert not hasattr(restored.identity, "a_field_that_was_removed")
    assert not hasattr(restored, "a_whole_block_that_went_away")
    assert restored.identity.name == AuraState().identity.name


def test_the_fields_it_does_have_survive() -> None:
    """The null for the end-to-end: dropping must not drop everything."""
    from core.state.aura_state import AuraState

    repository = StateRepository.__new__(StateRepository)
    state = AuraState()
    state.identity.name = "Aura"
    payload = json.loads(repository._serialize(state))
    payload["identity"]["a_field_that_was_removed"] = "gone"

    restored = repository._deserialize(json.dumps(payload))
    assert restored.identity.name == "Aura"
