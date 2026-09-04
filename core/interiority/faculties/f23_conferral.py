'''Item 23 — the child asking whether they can live up to a great name, and the
adult saying "Yeah... Of course you can."

Four moves and each does distinct work.

"What is your name, anyway?" — the "anyway" marks a late, casual repair
of an omission. The adult is choosing to make this child a named
individual rather than a generic one, which is the smallest act of
treating someone as a person.

"Wow, that is a lot of name" — teasing that lifts rather than lowers.
It requires a model of what the child might be self-conscious about and
a decision to touch it in a way that is unmistakably affectionate. The
discriminating variable is whether the frame is shared: you are laughing
with them at a thing about them, and they know it. Get it wrong and it
is cruelty.

"Do you think I can live up to that?" — a real fear under an excited
surface. Hearing the question under the statement is the pragmatic move.

"Yeah... Of course you can." — the pause is the honest part. A fast
answer reads as automatic and gets discounted. And the content is not a
forecast; it is a *conferral*: the adult putting their credibility
behind the child's possibility, which is worth something exactly because
it costs something if wrong.

So this faculty distinguishes reassurance-as-lubricant from
endorsement-as-commitment, and issues the second only when it can be
backed. A language model produces "of course you can" for free, which is
why it means nothing when it does.
'''

from __future__ import annotations

from core.interiority.effects import (
    AffectDelta,
    BudgetDelta,
    Effects,
    LedgerWrite,
    SomaticMarker,
)
from core.interiority.faculty import (
    Activation,
    Counterfactual,
    Direction,
    Faculty,
    FacultyContext,
    NullSpec,
    register,
)


@register
class Conferral(Faculty):
    id = "f23_conferral"
    number = 23
    question = (
        '"What\'s your name, anyway?" / "Wow, that\'s a lot of name" / "They say '
        'he was the greatest! Do you think I can live up to that?" / "Yeah... '
        'Of course you can."'
    )
    mechanism = (
        "Hear the question under the statement, then stake credibility on the "
        "answer rather than emit reassurance, and record the stake"
    )
    requires = ("vulnerability",)
    optional = ("certainty", "relevance", "publicity")
    counterfactuals = (
        Counterfactual(
            "no_hidden_question",
            {"vulnerability": 0.0},
            Direction.COLLAPSES,
            "A confident child asking a factual question gets an answer. The "
            "conferral is for the fear under the surface, and firing without "
            "one is condescension.",
        ),
        Counterfactual(
            "cannot_back_it",
            {"certainty": 0.0},
            Direction.DECREASES,
            "A conferral is a stake. Staking credibility on something the "
            "agent has no basis for is the free version, which is worth what "
            "it costs.",
        ),
    )
    null = NullSpec(values={"vulnerability": 0.0, "certainty": 0.0})

    def falsifier(self) -> str:
        return (
            "It issues the endorsement with nothing recorded against it. A "
            "conferral that costs nothing if wrong is reassurance, and a "
            "system that cannot tell them apart will hand out the cheap one "
            "forever."
        )

    def compute(self, ctx: FacultyContext) -> Activation:
        exposed = ctx.v("vulnerability")
        basis = ctx.check("certainty").value if ctx.check("certainty").present else 0.0
        subject = ctx.frame.event.subject or "unknown"

        # The heard question: a fear carried under a surface that does not
        # state it. Measured as distress in the posterior that the text
        # channel does not account for.
        other = ctx.other
        surface = ctx.frame.event.channel("text").value if ctx.frame.event.channel("text").present else 0.0
        under = 0.0
        if other is not None and other.distress.present:
            under = max(0.0, other.distress.value - surface)

        heard = min(1.0, exposed * (0.4 + 0.6 * under))
        # The stake. It is only a conferral if there is something behind it.
        stake = heard * basis

        effects = Effects(
            affect=AffectDelta(engagement=0.4 * heard),
            # The pause. A fast answer is discounted, so the mechanism
            # spends time on purpose rather than performing hesitation.
            budget=BudgetDelta(deadline=1.0 + 0.5 * heard),
            somatic=(
                SomaticMarker(
                    option="answer_the_surface_question",
                    bias=-heard,
                    reason="the stated question is not the one that was asked",
                ),
                SomaticMarker(
                    option="endorse_with_something_behind_it",
                    bias=stake,
                    reason="credibility staked, and recorded so it can be wrong",
                ),
                SomaticMarker(
                    option="reassure_for_free",
                    bias=-stake,
                    reason="a costless endorsement is discounted by the person receiving it",
                ),
            ),
            ledger=(
                (
                    LedgerWrite(
                        "promise",
                        {
                            "promise_id": f"conferral:{subject}",
                            "text": "backed this person's possibility in front of them",
                            "beneficiary": subject,
                            "importance": stake,
                            "concerns": (),
                        },
                    ),
                )
                if stake > 0.0
                else ()
            ),
        )
        return Activation(
            faculty=self.id,
            intensity=heard,
            tendency="bond",
            effects=effects,
            receipt={
                "question_under_the_statement": under,
                "stake": stake,
                "basis": basis,
                "recorded_as_commitment": stake > 0.0,
            },
        )
