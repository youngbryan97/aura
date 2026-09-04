"""grassmann_states: 0 for the channel's whole existence, and nothing was broken.

Every part worked. The ring carried states, the publisher's gates were right,
the consumer drained, cortex binding resolved, and the logs said the steering
engine was ONLINE with three hooks. The channel still reported zero on every
boot.

The cause was arithmetic between two correct decisions. The Grassmann encoder
needs a sliding window of 24 residual samples before it can return its first
state — a subspace basis needs a window, and 24 is the encoder's. The hook
sampled one decode step in 32, to keep a per-token measurement off the hot
path. Multiplied, the first Φ reading needed 768 decode steps FROM ONE HOOK.

The cortex lane's median reply is 24 decode steps, measured across 4,927
recorded turns. So the first state needed 32 replies inside a single worker
lifetime, and every worker restart handed the hook a fresh encoder with an
empty window. The warmup was longer than the lane it ran on.

Neither number was wrong on its own, which is why reading either in isolation
found nothing.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

from core.consciousness.affective_steering import (
    _PHI_SAMPLE_EVERY,
    AffectiveSteeringHook,
)
from core.consciousness.phi_residual_channel import create_channel, drain

#: Decode steps in a typical cortex-lane reply, measured over the recorded
#: corpus rather than assumed. The warmup is stated in replies because that is
#: the unit a worker lifetime is counted in.
CORTEX_MEDIAN_REPLY_STEPS = 24

#: The first Φ reading must exist within a conversation. Eight replies is a
#: short one; thirty-two, which is what the old stride required, is not a
#: conversation at all.
WARMUP_BUDGET_REPLIES = 8


def _hook(channel: object) -> AffectiveSteeringHook:
    hook = AffectiveSteeringHook.__new__(AffectiveSteeringHook)
    hook._phi_residual_channel = channel
    hook._phi_sampled = 0
    hook._phi_encoded_none = 0
    hook._phi_published = 0
    hook._phi_encode_errors = 0
    hook._phi_last_error = ""
    hook._grassmann_encoder = None
    hook._phi_sample_every = _PHI_SAMPLE_EVERY
    return hook


def _decode(hook: AffectiveSteeringHook, steps: int) -> None:
    """Drive `steps` single-token decode steps through the publisher."""
    for step in range(steps):
        hook._inject_count = step
        hook._maybe_record_phi_residual(
            np.random.default_rng(step).normal(size=(1, 1, 512)).astype(np.float32)
        )


def test_the_first_phi_state_arrives_inside_a_conversation():
    channel = create_channel(mp.get_context("spawn"))
    hook = _hook(channel)

    _decode(hook, WARMUP_BUDGET_REPLIES * CORTEX_MEDIAN_REPLY_STEPS)

    states, _cursor = drain(channel, 0)
    assert states, (
        f"no Φ state after {WARMUP_BUDGET_REPLIES} replies. The encoder window "
        f"and the sample stride multiply: window x {_PHI_SAMPLE_EVERY} decode "
        "steps is the warmup, and it has grown past the lane it runs on again"
    )


def test_the_old_stride_could_not_have_filled_in_a_conversation():
    """The defect, kept as a measurement rather than a story."""
    channel = create_channel(mp.get_context("spawn"))
    hook = _hook(channel)
    hook._phi_sample_every = 32

    _decode(hook, WARMUP_BUDGET_REPLIES * CORTEX_MEDIAN_REPLY_STEPS)

    states, _cursor = drain(channel, 0)
    assert not states, (
        "the old stride now fills in a conversation, so this no longer "
        "records why the channel was silent"
    )


def test_a_warming_channel_is_distinguishable_from_a_broken_one():
    """Zero states with the window half full is not the same finding as zero
    states with the window full, and both reported the same number."""
    channel = create_channel(mp.get_context("spawn"))
    hook = _hook(channel)

    _decode(hook, 4 * CORTEX_MEDIAN_REPLY_STEPS)

    filled = len(getattr(hook._grassmann_encoder, "_buf", ()) or ())
    needed = int(getattr(hook._grassmann_encoder, "window", 0) or 0)
    assert 0 < filled < needed, f"expected a partly full window, got {filled}/{needed}"
    assert hook._phi_sampled > 0
    assert hook._phi_encode_errors == 0


def test_the_stride_is_not_quietly_raised_back():
    """The window and the stride are a product, and only their product matters."""
    assert _PHI_SAMPLE_EVERY * 24 <= WARMUP_BUDGET_REPLIES * CORTEX_MEDIAN_REPLY_STEPS, (
        f"window(24) x stride({_PHI_SAMPLE_EVERY}) = {24 * _PHI_SAMPLE_EVERY} "
        f"decode steps, which is more than the "
        f"{WARMUP_BUDGET_REPLIES * CORTEX_MEDIAN_REPLY_STEPS} a conversation "
        "gives it"
    )
