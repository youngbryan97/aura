"""core/conation/aesthetic.py — value in the structure of the thing itself.

The remaining origin is the one with nothing downstream of it. No budget is
restored, no fact is acquired, nobody else is involved. What draws attention
is the shape of the thing.

Schmidhuber's formulation is the one that survives contact with a machine:
what is interesting is not what compresses well and not what compresses
badly, but what is *currently getting easier to compress*. A blank page
compresses perfectly and is dull. Noise does not compress at all and is also
dull. Something whose regularities are being found right now is the interesting
case, and interest ends when the finding does.

    aesthetic value = (cost_before - cost_after) / cost_before

That derivative form is what separates this from the epistemic origin next
door. Epistemic value asks how much uncertainty an observation would remove.
This asks how much cheaper the thing has become to hold in mind, which is a
question about the observer's representation rather than about the world.

The two come apart in a case worth naming: a proof you already believe teaches
you nothing and can still be beautiful, because the encoding got shorter even
though the belief did not move.

## The measurement is real

Compression here is not a metaphor. The cost of an encoding is measured by
running one — the length of the serialised, compressed representation of a
target. That makes this origin the cheapest of the five to falsify: if the
bytes do not get smaller, there is no aesthetic value, whatever anything else
reports.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from typing import Any

from core.conation.origins import OriginReading, ValueOrigin

EPS = 1e-12


@dataclass
class CompressionTrace:
    """Encoding cost for one target, against a model that grows as it is seen.

    The cost has to be *conditional*. Compressing a payload on its own measures
    how regular that payload is, which is a property of the payload and does
    not change however many times it is encountered — a constant, and a
    constant has no derivative to be interested in. Compressing it against
    what the observer has already absorbed measures how much of it is
    already accounted for, and that is what falls with familiarity.

    The conditional form has a second benefit that matters at these sizes.
    Any real compressor spends bytes on a header, and on short payloads the
    header dominates the measurement. Taking the difference between the joint
    encoding and the model's own encoding cancels it exactly.
    """

    key: str
    costs: list[int] = field(default_factory=list)
    #: What has been absorbed about this target so far. Bounded: the point is
    #: a working model, not an archive, and an unbounded buffer keyed on
    #: arbitrary payloads is a leak with a respectable name.
    model: bytes = b""
    last_seen: float = field(default_factory=time.time)

    MAX_HISTORY = 32
    MAX_MODEL_BYTES = 8192

    def observe(self, cost: int, payload: bytes = b"") -> None:
        self.costs.append(int(cost))
        if len(self.costs) > self.MAX_HISTORY:
            self.costs.pop(0)
        if payload:
            self.model = (self.model + payload)[-self.MAX_MODEL_BYTES:]
        self.last_seen = time.time()

    def progress(self) -> float | None:
        """Fractional fall in encoding cost since the previous encounter.

        ``None`` on a first encounter: one measurement has no derivative, and
        reporting zero would say the thing had stopped becoming interesting
        before anyone had looked at it twice.
        """
        if len(self.costs) < 2:
            return None
        before, after = self.costs[-2], self.costs[-1]
        if before <= 0:
            return None
        return max(0.0, (before - after) / before)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "encounters": len(self.costs),
            "last_cost": self.costs[-1] if self.costs else None,
            "progress": self.progress(),
        }


class AestheticValuation:
    """Compression progress over what the observer has actually encoded."""

    MAX_TRACES = 256

    def __init__(self) -> None:
        self._traces: dict[str, CompressionTrace] = {}

    @staticmethod
    def encoding_cost(payload: str | bytes, *, model: bytes = b"") -> int:
        """Bytes needed to hold this, given what is already known.

        ``model`` is prior context — the observer's existing representation.
        Compressing the payload against it and subtracting the model's own
        cost gives the *conditional* cost, which is what falls as a target
        becomes familiar, and cancels the compressor's own header along the
        way. Compressing the payload alone would produce a constant, and a
        constant has no derivative to be interested in.
        """
        data = payload.encode("utf-8", "replace") if isinstance(payload, str) else payload
        if not model:
            return len(zlib.compress(data, 6))
        joint = len(zlib.compress(model + data, 6))
        alone = len(zlib.compress(model, 6))
        return max(1, joint - alone)

    def value(
        self,
        key: str,
        *,
        payload: str | bytes | None = None,
        model: bytes = b"",
        cost: int | None = None,
    ) -> OriginReading:
        """Price a target by how much cheaper it just became to hold.

        Supply either a ``payload`` to measure or a precomputed ``cost``. With
        neither, the origin reports unavailable, because an aesthetic claim
        with nothing encoded behind it is the exact decorative number this
        package exists to refuse.
        """
        origin = ValueOrigin.AESTHETIC
        if cost is None and payload is None:
            return OriginReading.unavailable(origin, "nothing supplied to encode")

        trace = self._traces.get(key)
        if trace is None:
            if len(self._traces) >= self.MAX_TRACES:
                stalest = min(self._traces.values(), key=lambda t: t.last_seen)
                self._traces.pop(stalest.key, None)
            trace = CompressionTrace(key=key)
            self._traces[key] = trace

        data = b""
        if payload is not None:
            data = (
                payload.encode("utf-8", "replace")
                if isinstance(payload, str)
                else payload
            )
        if cost is None:
            # Conditional on everything absorbed about this target so far,
            # plus any context the caller supplies. A second encounter with
            # the same structure costs less than the first because the model
            # now carries it.
            measured = self.encoding_cost(data, model=model + trace.model)
        else:
            measured = int(cost)
        trace.observe(measured, data)

        progress = trace.progress()
        if progress is None:
            return OriginReading.unavailable(
                origin, f"first encoding of this target at {measured} bytes"
            )

        return OriginReading(
            origin=origin,
            magnitude=max(0.0, min(1.0, progress)),
            available=True,
            evidence=(
                f"encoding cost {trace.costs[-2]} -> {measured} bytes "
                f"over {len(trace.costs)} encounters"
            ),
            detail={
                "cost": float(measured),
                "previous_cost": float(trace.costs[-2]),
                "progress": progress,
            },
        )

    def status(self) -> dict[str, Any]:
        return {
            "traces": len(self._traces),
            "tracked": [t.to_dict() for t in list(self._traces.values())[:5]],
        }
