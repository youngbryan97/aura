"""Standing invariants for the endogenous language pathway.

Importing this module registers them. Three things must hold whenever the
pathway is up, and each of them has already been the shape of a real defect
somewhere in this codebase:

* An artifact that cannot be used must not read as one that can. A head bound
  to a stale channel layout will refuse on every generation forever, and
  without a check the only symptom is a bias that never applies.
* A refusal that means a fault must not be counted with refusals that mean an
  absence. "No head yet" and "a head that will not attach" look identical in a
  counter that lumps them.
* A head must not claim to be trained when its own report says the fit
  measured nothing. That is the file-on-disk version of an untrained random
  projection being served as a readout.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant

_OWNER = "core/brain/llm/endogenous_vocab_head.py"


def _resident_head():
    from core.brain.llm.endogenous_decode import load_head

    head, reason = load_head()
    return head, reason


@invariant(
    "endogenous.head_matches_the_channel_layout",
    scope="endogenous_language",
    owner=_OWNER,
    description="a head on disk is bound to the channel layout this build declares",
)
def _head_layout_matches() -> Iterator[Violation]:
    from core.brain.llm.endogenous_state import layout_digest

    head, _reason = _resident_head()
    if head is None:
        return
    if head.layout != layout_digest():
        yield Violation(
            subject="endogenous_vocab_head",
            message=(
                f"head was fitted against layout {head.layout} and this build "
                f"declares {layout_digest()}"
            ),
            remedy="refit with tools/train_endogenous_readout.py, or restore the layout",
        )


@invariant(
    "endogenous.trained_head_measured_something",
    scope="endogenous_language",
    owner="core/brain/llm/endogenous_readout_training.py",
    description="a head marked trained carries a verdict that earned it",
)
def _trained_head_has_a_verdict() -> Iterator[Violation]:
    head, _reason = _resident_head()
    if head is None or not head.trained:
        return
    verdict = str((head.report or {}).get("verdict") or "")
    if verdict not in {"style_prior", "content_bearing"}:
        yield Violation(
            subject="endogenous_vocab_head",
            message=(
                f"head is marked trained and its report says {verdict or 'nothing'}"
            ),
            remedy="delete the head; a fit that measured nothing must not attach",
        )


@invariant(
    "endogenous.refusals_are_absences_not_faults",
    scope="endogenous_language",
    owner="core/brain/llm/endogenous_decode.py",
    severity=Severity.WARNING,
    description="no generation was refused for a reason that means something is wrong",
)
def _no_unexpected_refusals() -> Iterator[Violation]:
    from core.brain.llm.endogenous_decode import pathway_health

    health = pathway_health()
    count = int(health.get("unexpected_refusals") or 0)
    if count <= 0:
        return
    reasons = {
        reason: n
        for reason, n in (health.get("reasons") or {}).items()
        if reason
        not in {
            "applied",
            "no_state_on_job",
            "alpha_disabled",
            "state_coverage_below_floor",
            "head_untrained",
            "bias_is_flat",
            "not_evaluated",
        }
        and not reason.startswith("no_head")
    }
    yield Violation(
        subject="endogenous_language_pathway",
        message=f"{count} generations refused the bias for a fault reason: {reasons}",
        remedy="check the head's tokenizer and layout against the resident model",
    )


__all__ = [
    "_head_layout_matches",
    "_no_unexpected_refusals",
    "_trained_head_has_a_verdict",
]
