"""The prediction is written down and sealed before the run.

Every result in this package is worth exactly as much as the honesty of the
moment it was scored, and the failure mode is not fraud. It is the ordinary
human motion of looking at a number and remembering what one expected. A
protocol that decides afterwards which direction counted as success has
measured nothing, however careful the rest of it was.

So the predictions are hashed and the hash is published before the run
starts. Scoring reads the sealed file, and a mismatch voids the run rather
than adjusting it. This is the mechanism the IIT/GNWT adversarial
collaboration used and it is the only reason its null results were worth
reading: both camps had committed to what would count as losing.

The same seal carries the pass/fail thresholds, the direction, and the
minimum effect. "It moved a bit in roughly the right way" is not a result and
cannot become one after the fact.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Prediction",
    "Preregistration",
    "SealBrokenError",
    "seal",
    "open_seal",
]


class SealBrokenError(RuntimeError):
    """The registered predictions do not match what is being scored."""


@dataclass(frozen=True)
class Prediction:
    """What one protocol must show, decided before it is run."""

    protocol: str
    #: The direction the measured quantity must move, in words a reader can
    #: check against the number afterwards.
    direction: str
    #: The smallest effect that counts. Written down because "significant"
    #: without a magnitude is a judgement made after seeing the data.
    minimum_effect: float
    #: What the measured quantity IS. Named so that a protocol cannot quietly
    #: switch to a friendlier metric.
    measure: str
    #: What result would kill H1 here. A protocol that cannot fail is not a
    #: test, and this field is what stops the battery being a formality.
    falsifier: str
    #: Which hypothesis a pass supports.
    supports: str = "H1"

    def __post_init__(self) -> None:
        if not self.falsifier.strip():
            raise ValueError(
                f"prediction for {self.protocol!r} names no falsifier. A "
                "protocol that cannot come out the other way is a ceremony"
            )
        if self.minimum_effect <= 0.0:
            raise ValueError(
                f"prediction for {self.protocol!r} has no minimum effect; any "
                "movement would count and the threshold would be chosen after "
                "the fact"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "direction": self.direction,
            "minimum_effect": self.minimum_effect,
            "measure": self.measure,
            "falsifier": self.falsifier,
            "supports": self.supports,
        }


@dataclass(frozen=True)
class Preregistration:
    """A frozen set of predictions and the digest that pins them."""

    predictions: tuple[Prediction, ...]
    registered_at: float = field(default_factory=time.time)
    #: What the run is against, so a result cannot be carried over to a
    #: different model, a different commit, or a different body.
    model_digest: str = ""
    source_commit: str = ""
    note: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "schema": "aura.phenomenology.preregistration.v1",
            "predictions": [p.as_dict() for p in self.predictions],
            "registered_at": round(self.registered_at, 3),
            "model_digest": self.model_digest,
            "source_commit": self.source_commit,
            "note": self.note,
        }

    def digest(self) -> str:
        payload = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def for_protocol(self, protocol: str) -> Prediction | None:
        for prediction in self.predictions:
            if prediction.protocol == protocol:
                return prediction
        return None


def seal(registration: Preregistration, path: Path | str) -> str:
    """Write the predictions and return the digest to publish.

    The digest goes somewhere the experimenter cannot quietly change — a
    commit message, an issue, a message to the person replicating. The file
    itself is not the seal; the published digest is.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {**registration.body(), "digest": registration.digest()}
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return registration.digest()


def open_seal(path: Path | str, *, expect_digest: str = "") -> Preregistration:
    """Read the predictions back, refusing a file that has been edited.

    ``expect_digest`` is the value published before the run. Passing it is
    what makes this a seal rather than a filename: without it, a rewritten
    file reads as a valid registration of whatever it now says.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = str(document.pop("digest", ""))
    registration = Preregistration(
        predictions=tuple(
            Prediction(**entry) for entry in document.get("predictions", ())
        ),
        registered_at=float(document.get("registered_at", 0.0)),
        model_digest=str(document.get("model_digest", "")),
        source_commit=str(document.get("source_commit", "")),
        note=str(document.get("note", "")),
    )
    recomputed = registration.digest()
    if stored and stored != recomputed:
        raise SealBrokenError(
            f"the registration file has been edited since it was sealed: it "
            f"stores {stored[:16]} and now hashes to {recomputed[:16]}"
        )
    if expect_digest and expect_digest != recomputed:
        raise SealBrokenError(
            f"this is not the registration that was published: expected "
            f"{expect_digest[:16]}, found {recomputed[:16]}"
        )
    return registration
