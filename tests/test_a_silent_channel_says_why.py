"""Zero states had four possible causes and reported one number.

`phi_residual_history: {"grassmann_states": 0}` is what the live runtime said
about the activation-grounded Φ complex on 2026-08-25, and it had said it for
the channel's whole existence. Zero is consistent with all of:

  - no hook was ever installed, so nothing sampled;
  - hooks installed but the ring never reached them;
  - the encoder is still filling its 24-vector window;
  - the encoder is raising on every single call.

The last one was invisible by construction: `_encode_grassmann_state` caught
ImportError, AttributeError, RuntimeError, TypeError and ValueError and
returned None with no record, on the correct principle that a telemetry sample
is never worth a generation — and the incorrect corollary that it therefore
need not be mentioned. An encoder failing every call looked exactly like an
encoder nobody called.

So the publisher counts what happened to each sample, and the health surface
reads those counters beside the depth. Failing open is kept; failing silently
is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.affective_steering import AffectiveSteeringHook


@pytest.fixture
def hook() -> AffectiveSteeringHook:
    return AffectiveSteeringHook.__new__(AffectiveSteeringHook)


def _prepare(hook: AffectiveSteeringHook, channel: object | None) -> None:
    hook._phi_residual_channel = channel
    hook._phi_sampled = 0
    hook._phi_encoded_none = 0
    hook._phi_published = 0
    hook._phi_encode_errors = 0
    hook._phi_last_error = ""
    hook._phi_sample_every = 32
    hook._inject_count = 0
    hook._grassmann_encoder = None


def test_the_counters_separate_the_four_causes(hook):
    """Each cause leaves a different fingerprint in the counters."""
    _prepare(hook, channel=None)

    # Nothing attached: nothing is even sampled.
    assert hook._phi_sampled == 0
    assert hook._phi_published == 0


def test_an_encoder_that_withholds_is_counted_separately_from_one_that_raises(
    hook, monkeypatch
):
    published: list[int] = []
    _prepare(hook, channel=object())
    monkeypatch.setattr(
        "core.consciousness.phi_residual_channel.publish_state",
        lambda channel, state: published.append(state) or True,
    )

    # Warming up: the window is not full, so the encoder withholds a state.
    monkeypatch.setattr(hook, "_encode_grassmann_state", lambda sample: None)
    hook._maybe_record_phi_residual(np.zeros((1, 1, 8)))
    assert hook._phi_sampled == 1
    assert hook._phi_encoded_none == 1
    assert hook._phi_encode_errors == 0

    # A withheld state must not be counted as an error: warming up and
    # broken are different problems and the counters keep them apart.
    assert hook._phi_published == 0
    assert published == []


def test_a_published_sample_is_counted(hook, monkeypatch):
    published: list[int] = []
    _prepare(hook, channel=object())
    monkeypatch.setattr(
        "core.consciousness.phi_residual_channel.publish_state",
        lambda channel, state: published.append(state) or True,
    )
    monkeypatch.setattr(hook, "_encode_grassmann_state", lambda sample: 7)

    hook._maybe_record_phi_residual(np.zeros((1, 1, 8)))

    assert published == [7]
    assert hook._phi_published == 1
    assert hook._phi_encoded_none == 0


def test_the_encoder_records_a_degradation_once_not_per_token(monkeypatch):
    """It runs inside the forward pass; a record per token costs more than the
    sample it describes."""
    recorded: list[tuple] = []
    monkeypatch.setattr(
        "core.runtime.errors.record_degradation",
        lambda subsystem, exc, **kw: recorded.append((subsystem, type(exc).__name__)),
    )

    hook = AffectiveSteeringHook.__new__(AffectiveSteeringHook)
    _prepare(hook, channel=object())

    class _Exploding:
        def observe(self, vector):
            raise RuntimeError("no subspace")

    for _ in range(5):
        hook._grassmann_encoder = _Exploding()
        assert hook._encode_grassmann_state(np.zeros(8)) is None

    assert hook._phi_encode_errors == 5
    assert len(recorded) == 1, recorded
    assert recorded[0][0] == "affective_steering.phi_residual"
    assert "no subspace" in hook._phi_last_error


def test_the_health_surface_reads_the_publishers_not_only_the_depth():
    """A depth of zero beside 'sampled 97, published 0' is a diagnosis."""
    import inspect

    from core.runtime import health_contract

    source = inspect.getsource(health_contract)
    block = source[source.index("grassmann_history_depth") :]
    block = block[: block.index("except Exception")]
    assert "publishers" in block
    assert "get_diagnostics" in block
