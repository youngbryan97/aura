"""A subsystem that moves the sampler must be declared, or it moves nothing.

The gate filters kwargs to the declared request fields and logs the rest at
debug level. A bias published by a subsystem and absent from that schema is
computed on every turn and dropped in silence.

LIVE, 2026-08-28: the cognitive-situation frame was the fourth such channel and
was in neither the gate's list of bias keys nor the typed request schema. Its
own test proved the engine handed the bias to a stub router and stopped there,
which is the shape of a half-wired channel: a writer, a test of the writer, and
no reader.

This is the check that would have caught it, and it is written from the source
rather than from a list, so a fifth channel added tomorrow cannot go quiet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.brain.inference_gate import _SAMPLING_BIAS_KEYS
from core.brain.request_contract import REQUEST_FIELDS

_ENGINE = Path("core/brain/cognitive_engine.py")
_PUBLISHES = re.compile(
    r'response_modifiers\[\s*"([a-z_]*sampling_bias)"\s*\]\s*=', re.IGNORECASE
)


def _published_channels() -> set[str]:
    """Every sampling bias the engine writes for a subsystem."""

    return set(_PUBLISHES.findall(_ENGINE.read_text()))


def test_the_engine_publishes_the_channels_we_think_it_does() -> None:
    published = _published_channels()
    assert len(published) >= 4, published
    assert "cognitive_situation_sampling_bias" in published


@pytest.mark.parametrize("channel", sorted(_published_channels()))
def test_a_published_bias_is_declared_in_the_request_schema(channel: str) -> None:
    assert channel in REQUEST_FIELDS, (
        f"{channel} is published by a subsystem and not declared, so the gate "
        "filters it out and the subsystem moves nothing"
    )


@pytest.mark.parametrize("channel", sorted(_published_channels()))
def test_a_published_bias_is_read_by_the_gate(channel: str) -> None:
    assert channel in _SAMPLING_BIAS_KEYS, (
        f"{channel} is declared but not among the gate's bias keys"
    )


def test_the_generation_phase_reads_the_same_set() -> None:
    """Three lists of the same thing is how one of them goes stale."""

    body = Path("core/phases/response_generation.py").read_text()
    for channel in _published_channels():
        assert channel in body, channel
