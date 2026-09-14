"""One layer's residual stream, sampled inside the forward pass for Φ.

The steering hook runs inside the forward pass of every instrumented block,
which is the one place the activations are. Φ's TPM wants an 8-bit state per
decode step, so the sample is encoded here — ~5120 floats in, one byte out —
and published across the process boundary, rather than sending a per-token
megabyte on the path where latency decides whether a turn survives.

The counters exist because `grassmann_states: 0` in a health report has four
causes — the hook was never called, the encoder is still filling its window,
the encoder fails on every call, nothing drains the channel — and they were
indistinguishable while the encoder swallowed its own exceptions.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("Aura.Consciousness.PhiResidual")

#: Decode steps between residual samples, chosen so the Grassmann encoder's
#: window can FILL inside a conversation rather than a lifetime of them.
#:
#: It was 32, and the encoder needs 24 samples before it can return its first
#: state, so the first Φ reading needed 768 decode steps from one hook. The
#: cortex lane's median reply is 24 decode steps, measured over 4,927 recorded
#: turns, which makes that 32 replies — inside a SINGLE worker lifetime, since
#: every restart gives the hook a fresh encoder with an empty window. The
#: channel therefore reported grassmann_states: 0 for its whole existence
#: while every part of it worked.
#:
#: 8 puts the window inside eight replies at that median. The cost is one
#: 5120-float slice per eight tokens per instrumented layer; the expensive
#: thing this avoids is sampling during prefill, which is gated separately and
#: is where the 58-82s first tokens came from.
PHI_SAMPLE_EVERY = 8

#: How the sampler reports a fault: (exc, action=, severity=, stage=, extra=).
FaultReport = Callable[..., None]


def sample_every_from_environment(default: int = PHI_SAMPLE_EVERY) -> int:
    try:
        return max(1, int(os.getenv("AURA_PHI_RESIDUAL_SAMPLE_EVERY", str(default))))
    except (TypeError, ValueError):
        return default


def residual_recording_is_off() -> bool:
    return os.getenv("AURA_PHI_RECORD_RESIDUALS", "1").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }


class PhiResidualSampler:
    """Samples one layer's residual stream and hands Φ an 8-bit state."""

    def __init__(
        self,
        *,
        layer_idx: int,
        report: FaultReport,
        sample_every: int | None = None,
    ) -> None:
        self.layer_idx = int(layer_idx)
        self._report = report
        #: Set by the worker when it is running out-of-process. When present,
        #: Grassmann states go across this ring instead of to a PhiCore that
        #: does not exist on this side of the fork.
        self.channel: Any = None
        self.sampled = 0
        self.encoded_none = 0
        self.published = 0
        self.encode_errors = 0
        self.last_error = ""
        self.encoder: Any = None
        self.sample_every = (
            sample_every_from_environment() if sample_every is None else max(1, int(sample_every))
        )

    def maybe_record(self, h: Any, *, inject_count: int) -> None:
        if residual_recording_is_off():
            return
        if inject_count % self.sample_every != 0:
            return
        # PhiCore does `np.asarray(hidden_state)`, which on MLX is a blocking
        # device sync AND a full materialisation of whatever it is handed. This
        # runs inside the forward pass of all 64 blocks, so measuring here
        # collapses MLX's lazy pipeline on the one path where latency decides
        # whether a turn survives.
        #
        # During PREFILL `h` is the whole sequence — [1, seq, 5120] — so a
        # single sample copied tens of megabytes off the GPU and stalled the
        # graph, repeatedly. Measured live 2026-07-26: ~3k-token prompts took
        # 58-82s to a first token, roughly 50 tok/s, about twenty times slower
        # than this model should prefill; turns 5-7 of a conversation died on
        # that alone.
        #
        # Prefill is not a thought moment anyway — the signal Φ wants is the
        # per-token dynamics of generation. So sample only single-token decode
        # steps, and hand over one already-sliced position rather than a
        # sequence, so the transfer is a 5120-float vector instead of a tensor.
        try:
            shape = tuple(getattr(h, "shape", ()) or ())
        except (AttributeError, TypeError):
            return
        if len(shape) >= 3 and shape[-2] > 1:
            return
        if len(shape) == 2 and shape[0] > 1:
            return
        try:
            sample = h[0, -1, :] if len(shape) >= 3 else h

            # IN THE WORKER PROCESS there is no PhiCore to hand this to — the
            # hook runs inside the MLX worker subprocess and PhiCore is
            # registered in the main runtime, so the container lookup below
            # returned False on every token and the activation-grounded complex
            # never filled. Encode here, where the activations are, and publish
            # the 8-bit state across the boundary.
            channel = self.channel
            if channel is not None:
                self.sampled += 1
                state = self.encode(sample)
                if state is not None:
                    from core.consciousness.phi_residual_channel import publish_state

                    publish_state(channel, state)
                    self.published += 1
                    return
                self.encoded_none += 1
                return

            from core.container import ServiceContainer

            if not ServiceContainer.has("phi_core"):
                return
            phi_core = ServiceContainer.get("phi_core", default=None)
            if phi_core is not None and hasattr(phi_core, "record_residual_stream"):
                phi_core.record_residual_stream(
                    sample, layer_idx=self.layer_idx, token_position=-1
                )
        except (ImportError, AttributeError, RuntimeError) as exc:
            self._report(
                exc,
                action="continued generation after optional phi residual sample failed",
                severity="warning",
                stage="phi_residual_sample",
                extra={"layer_idx": self.layer_idx},
            )
            logger.debug("Residual phi sample failed at layer %d: %s", self.layer_idx, exc)

    def encode(self, sample: Any) -> int | None:
        """Reduce a residual vector to the 8-bit state Φ's TPM is built from."""
        try:
            if self.encoder is None:
                from core.consciousness.grassmann_phi import GrassmannResidualComplex
                from core.consciousness.phi_core import _grassmann_anchor_count

                self.encoder = GrassmannResidualComplex(n_anchors=_grassmann_anchor_count())
            import numpy as _np

            vector = _np.asarray(sample, dtype=_np.float32).reshape(-1)
            state = self.encoder.observe(vector)
            if state is None:
                return None
            # Fold rather than truncate: `& 0xFF` would keep modes 0-7 and drop
            # everything above, so a wider encoder would subtract information.
            from core.consciousness.phi_core import _fold_modes_to_byte

            return _fold_modes_to_byte(int(state))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # A telemetry sample is never worth a generation, so this still
            # fails open. What it must not do is fail SILENTLY: an encoder
            # raising on every call looks exactly like an encoder that is never
            # reached, and both report zero states. Recorded once — this runs
            # inside the forward pass, and a per-token degradation record would
            # cost more than the sample it describes.
            self.encode_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            if self.encode_errors == 1:
                from core.runtime.errors import record_degradation

                record_degradation(
                    "affective_steering.phi_residual",
                    exc,
                    severity="warning",
                    action=(
                        "the Grassmann encoder refused a residual sample; the "
                        "activation-grounded complex will not fill while this "
                        "persists"
                    ),
                )
            return None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "channel_attached": self.channel is not None,
            "sampled": self.sampled,
            "published": self.published,
            "encoder_withheld": self.encoded_none,
            "encoder_errors": self.encode_errors,
            "last_error": self.last_error,
            "sample_every": self.sample_every,
            # How full the encoder's window is. Without this a warming
            # channel and a broken one both report zero states.
            "window_filled": len(getattr(self.encoder, "_buf", ()) or ()),
            "window_needed": int(getattr(self.encoder, "window", 0) or 0),
        }
