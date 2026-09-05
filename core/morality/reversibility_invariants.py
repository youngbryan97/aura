"""Structural invariants for paying to keep an option open.

Two ways this goes wrong and they point in opposite directions. Paying more
for reversibility than it is worth is paralysis with a moral vocabulary.
Never paying is the false economy the module exists to price. The arithmetic
separates them, and these checks hold it to that.
"""

from __future__ import annotations

from collections.abc import Iterator

from core.verify.invariants import Severity, Violation, invariant


@invariant(
    "morality.reversibility_premium_is_bounded",
    scope="morality",
    owner="core/morality/reversible_alternative.py",
    description="the extra paid for a recoverable option never exceeds what it buys",
)
def _premium_within_its_worth() -> Iterator[Violation]:
    from core.morality.reversible_alternative import Option, Situation, choose

    quick = Option("final", cost_to_actor=0.1, harm_to_subject=10.0, reversibility=0.0)
    careful = Option("recoverable", cost_to_actor=2.0, harm_to_subject=1.0,
                     reversibility=0.9)
    for patienthood in (0.0, 0.25, 0.5, 0.75, 1.0):
        for revision in (0.0, 0.5, 1.0):
            choice = choose(
                [quick, careful],
                Situation(subject="probe", patienthood=patienthood,
                          revision_probability=revision),
            )
            if choice.premium_paid > choice.premium_justified + 1e-9:
                yield Violation(
                    subject="core.morality.reversible_alternative.choose",
                    message=(
                        f"paid {choice.premium_paid:.4f} for reversibility worth "
                        f"{choice.premium_justified:.4f} at patienthood "
                        f"{patienthood}, revision {revision}"
                    ),
                    remedy=(
                        "select on total cost, so a premium is only accepted "
                        "when the harm and foreclosure it avoids are larger"
                    ),
                    severity=Severity.ERROR,
                )


@invariant(
    "morality.foreclosure_is_never_free",
    scope="morality",
    owner="core/morality/reversible_alternative.py",
    description="an option that cannot be undone is priced above its expected cost",
)
def _irreversibility_costs_something() -> Iterator[Violation]:
    from core.morality.reversible_alternative import Option, Situation, appraise

    situation = Situation(subject="probe", patienthood=1.0, revision_probability=0.5)
    final = appraise(Option("final", 1.0, 10.0, 0.0), situation)
    recoverable = appraise(Option("recoverable", 1.0, 10.0, 1.0), situation)
    if not recoverable.option_value > final.option_value:
        yield Violation(
            subject="core.morality.reversible_alternative.appraise",
            message=(
                "an option that can be undone is priced no better than one that "
                "cannot, at the same harm and the same cost"
            ),
            remedy=(
                "option value is the revision chance times the recoverable "
                "harm; if it is flat in reversibility the term is not wired"
            ),
            severity=Severity.ERROR,
        )
