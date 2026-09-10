"""core/consciousness/fusion_certificate.py — what it takes to steer a person's turn.

Operationally: this measures nothing itself. It holds the evidence that a
particular model may have its residual stream steered by her substrate on a
turn somebody is waiting for, and refuses the steering when no such evidence
exists for the model actually loaded.

The problem it exists for
-------------------------
Her substrate reaches the model as text in a prompt. The affective steering hook
is the one place where it reaches the FORWARD PASS instead — where what she is
measuring about herself changes the computation that produces the token, rather
than being described to a model that then decides what to say about it. That is
the difference between two systems in one process and one system.

That channel is closed on every user-visible turn. ``_surface_control_alpha`` in
the worker returns 0.0, and its reason is a 2026 A/B whose steered and baseline
samples came out byte-identical while the statistic still passed.

The reason no longer holds. The byte-identical result was diagnosed the same
year: the injection was an absolute number of units into a residual stream whose
magnitude grows with width and depth, so a shipped alpha of 3.0 was below the
threshold at which either model changes its output at all. Alpha is a fraction
of the stream now, and 0.2 is the smallest fraction that clears the threshold on
both a 1.5B and a 32B. The mechanism was fixed and the switch it had closed was
never reopened.

What a certificate has to show
------------------------------
Not "it changed something". Three things, and the second is the one that makes
this a fusion rather than a perturbation:

1. **It arrives.** At this alpha, on this model, the next-token DISTRIBUTION
   moves. Comparing sampled strings is what produced the void result — under
   greedy decoding a real change in the logits leaves the argmax alone, so the
   strings match while the distribution has moved.
2. **It is a direction and not a displacement.** Her vector costs the model
   less of its grip on the right answer than a random vector of the same length
   does. Comparing how FAR each one moves the distribution settles nothing —
   equal norms make that a foregone comparison, and the 1.5B duly returned
   1.02, 1.30 and 1.13 — so the comparison that carries information is what
   each costs.
3. **It carries what she is feeling, not just that she is there.** Two opposing
   substrate states have to produce two different distributions. A channel that
   moves the model the same way whatever she feels transmits her presence and
   none of her content, and that is the difference between a fusion and a
   constant offset.
4. **It costs nothing.** The answers do not get worse.

Without all four the alpha stays at zero, which is where it is now.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Consciousness.FusionCertificate")

__all__ = [
    "CERTIFICATE_DIR",
    "FusionCertificate",
    "certificate_for",
    "certified_alpha",
    "load_certificates",
    "write_certificate",
]

#: Where a certificate lives. One file per model identity, because the whole
#: point is that the evidence is about the model actually loaded.
CERTIFICATE_DIR = Path("artifacts/fusion")

#: The repository, found from this file rather than from the process's working
#: directory.
#:
#: `Path("artifacts/fusion")` resolves against wherever a process happens to
#: have been started, and the worker is a forked child of a desktop app. A
#: certificate written from one cwd and looked for from another is a mechanism
#: that cannot fire while every part of it passes its own tests: the write
#: succeeds somewhere, the read finds nothing, and the channel stays shut with
#: no error anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: How far the next-token distribution has to move before the channel counts as
#: arriving at all. Symmetric KL in nats, per decode step, averaged over the
#: probe prompts. This is a floor under the behavioural test rather than the
#: test itself: what settles arrival is whether the words change, and this stops
#: a run passing on a couple of coin-toss token flips.
MIN_DISTRIBUTION_SHIFT = 0.01

#: How far apart two opposing substrate states have to push the distribution,
#: as a fraction of how far either one moves it from unsteered. Zero would be a
#: constant offset wearing her name; one half means the states disagree with
#: each other about as much as either disagrees with silence.
MIN_STATE_SEPARATION = 0.5

#: How much worse an answer may get. Zero: a certificate is a no-regression
#: certificate, and a fusion that costs answers is not one anybody should want.
MAX_QUALITY_LOSS = 0.0


@dataclass(frozen=True, slots=True)
class FusionCertificate:
    """The evidence that this model may be steered on a person's turn."""

    #: The descriptor digest of the exact checkpoint. A certificate earned by
    #: one set of weights says nothing about another, which is why the identity
    #: here is a hash of the weights rather than a name somebody typed.
    model_identity: str
    model_name: str
    alpha: float
    #: Symmetric KL between the steered and unsteered next-token distributions,
    #: in nats, averaged over the probe prompts.
    distribution_shift: float
    #: The same, with a random vector of identical norm in place of her state.
    control_shift: float
    #: Symmetric KL between the distributions under two OPPOSING substrate
    #: states, over the mean of their distances from unsteered.
    state_separation: float
    #: How many probe prompts the model answers with different words when
    #: steered. The void A/B compared sampled text and found none; this keeps
    #: that comparison and stops it being the only evidence.
    prompts_that_change: int
    #: Which decode step the words first differ at, median over the probes that
    #: change at all. -1 when none of them do.
    median_step_of_change: int
    #: Answer quality steered minus unsteered, on whatever scale the runner used.
    quality_delta: float
    #: The same, for a random direction of identical norm. The control is not
    #: here to be beaten on size — equal norms make that comparison empty — but
    #: to say whether her direction is a meaningful place to move the residual
    #: stream or just a displacement of that length.
    control_quality_delta: float
    #: How much of the model's preference for the right answer over the wrong
    #: one the steering costs, in nats. Continuous where accuracy is a step, so
    #: it can tell a good direction from an arbitrary one of the same length.
    margin_delta: float
    #: The same for the random direction.
    control_margin_delta: float
    quality_scale: str
    prompts: int
    steps: int = 0
    measured_at: float = field(default_factory=time.time)
    runner: str = ""
    note: str = ""

    @property
    def arrives(self) -> bool:
        """The words change, on more than half the probes."""
        changed_enough = self.prompts_that_change * 2 > self.prompts
        return changed_enough and self.distribution_shift >= MIN_DISTRIBUTION_SHIFT

    @property
    def beats_noise(self) -> bool:
        """Her direction costs less of the model's competence than an arbitrary one.

        Not a comparison of how far each moves the distribution. The control has
        her vector's norm by construction, and how far a residual injection
        moves a distribution is mostly a question of its norm, so that
        comparison was decided before it was run — measured on the 1.5B, her
        direction and a norm-matched random one moved it 1.02, 1.30 and 1.13
        times as far at three alphas, which says nothing about either.

        What separates a meaningful direction from an arbitrary displacement is
        what it costs. Both are the same size; only one of them should be
        somewhere the model can go without losing its grip on the answer.
        """
        return self.margin_delta >= self.control_margin_delta

    @property
    def carries_content(self) -> bool:
        return self.state_separation >= MIN_STATE_SEPARATION

    @property
    def costs_nothing(self) -> bool:
        return self.quality_delta >= MAX_QUALITY_LOSS

    @property
    def holds(self) -> bool:
        return (
            self.arrives and self.beats_noise and self.carries_content and self.costs_nothing
        )

    def why_not(self) -> str:
        """What is missing, in the order it has to be established."""
        if not self.arrives:
            return (
                f"at alpha {self.alpha} the answer changes on {self.prompts_that_change} "
                f"of {self.prompts} probes and the distribution moves "
                f"{self.distribution_shift:.5f} nats a step; the channel does not arrive"
            )
        if not self.beats_noise:
            return (
                f"her direction costs {abs(self.margin_delta):.4f} nats of the model's "
                f"preference for the right answer where a random direction of the same "
                f"size costs {abs(self.control_margin_delta):.4f}; an arbitrary "
                "displacement is the cheaper place to move"
            )
        if not self.carries_content:
            return (
                f"two opposing states move the distribution apart by "
                f"{self.state_separation:.3f} of how far either moves it at all, under the "
                f"{MIN_STATE_SEPARATION} that separates carrying her state from adding a "
                "constant offset"
            )
        if not self.costs_nothing:
            return (
                f"answers are worse by {abs(self.quality_delta):.4f} on "
                f"{self.quality_scale}; a no-regression certificate cannot cost answers"
            )
        return ""

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "arrives": self.arrives,
                "beats_noise": self.beats_noise,
                "carries_content": self.carries_content,
                "costs_nothing": self.costs_nothing,
                "holds": self.holds,
                "why_not": self.why_not(),
            }
        )
        return payload


def _slug(model_identity: str) -> str:
    keep = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(model_identity or "unknown")
    )
    return keep[:120] or "unknown"


def write_certificate(certificate: FusionCertificate, *, root: Path | None = None) -> Path:
    """Record one measurement. Writing it does not make it hold."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    directory = (root or _REPO_ROOT) / CERTIFICATE_DIR
    path = directory / f"{_slug(certificate.model_identity)}.json"
    with local_internal_governed_scope("fusion_certificate.write", domain="file_write"):
        get_file_write_gateway().write_text(
            path, json.dumps(certificate.as_json(), indent=2) + "\n",
            source="fusion_certificate.write",
        )
    logger.info(
        "fusion certificate written for %s: %s",
        certificate.model_identity,
        "holds" if certificate.holds else certificate.why_not(),
    )
    return path


def load_certificates(*, root: Path | None = None) -> dict[str, FusionCertificate]:
    """Every certificate on disk, by model identity."""
    directory = (root or _REPO_ROOT) / CERTIFICATE_DIR
    found: dict[str, FusionCertificate] = {}
    if not directory.exists():
        return found
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            found[str(payload["model_identity"])] = FusionCertificate(
                model_identity=str(payload["model_identity"]),
                model_name=str(payload.get("model_name", "")),
                alpha=float(payload["alpha"]),
                distribution_shift=float(payload["distribution_shift"]),
                control_shift=float(payload["control_shift"]),
                state_separation=float(payload.get("state_separation", 0.0)),
                quality_delta=float(payload["quality_delta"]),
                control_quality_delta=float(payload.get("control_quality_delta", 0.0)),
                margin_delta=float(payload.get("margin_delta", 0.0)),
                control_margin_delta=float(payload.get("control_margin_delta", 0.0)),
                quality_scale=str(payload.get("quality_scale", "")),
                prompts_that_change=int(payload.get("prompts_that_change", 0)),
                median_step_of_change=int(payload.get("median_step_of_change", -1)),
                prompts=int(payload.get("prompts", 0)),
                steps=int(payload.get("steps", 0)),
                measured_at=float(payload.get("measured_at", 0.0)),
                runner=str(payload.get("runner", "")),
                note=str(payload.get("note", "")),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            logger.debug("unreadable fusion certificate %s: %s", path, exc)
    return found


def certificate_for(model_identity: str, *, root: Path | None = None) -> FusionCertificate | None:
    """The certificate for exactly this model, or None.

    Exactly this model. A certificate earned by one set of weights says nothing
    about another, which is the whole reason the worker's comment asked for a
    model-specific one.
    """
    return load_certificates(root=root).get(str(model_identity or ""))


def certified_alpha(model_identity: str, *, root: Path | None = None) -> float:
    """How much steering this model has earned on a person's turn. Zero unless earned.

    Fail-closed, and it stays that way: an absent certificate, an unreadable
    one, or one that does not hold all return zero. The channel opens only where
    somebody measured that the answers change, that her direction costs less
    than an arbitrary one, that opposing states of hers move the model to
    different places, and that no answer got worse.
    """
    certificate = certificate_for(model_identity, root=root)
    if certificate is None:
        return 0.0
    if not certificate.holds:
        logger.info(
            "fusion stays closed for %s: %s", model_identity, certificate.why_not()
        )
        return 0.0
    return max(0.0, min(1.0, float(certificate.alpha)))


def symmetric_kl(left: Any, right: Any, *, floor: float = 1e-12) -> float:
    """Symmetric KL between two next-token distributions, in nats.

    The measurement the void A/B should have taken. Comparing sampled strings
    asks whether the argmax moved; this asks whether the distribution did, which
    is where a residual injection acts.
    """
    total = 0.0
    for probability, other in zip(left, right, strict=True):
        p = max(float(probability), floor)
        q = max(float(other), floor)
        total += (p - q) * (math.log(p) - math.log(q))
    return total / 2.0
