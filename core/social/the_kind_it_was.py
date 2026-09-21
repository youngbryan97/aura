"""What form the care took, kept apart from whether it was care.

Oddisee's "That's Love" spends its last verse naming forms: a tough kind, a
too-much kind, a not-enough kind, a rushed kind, an abrupt kind, a go-nuts
kind, a take-it-slow kind, a no kind, a show kind, and — for two people who
would do each other worse by staying — a go kind. The song holds all of them
under one name.

The verses before it say what the forms have in common. The truth told because
a lie was worse. Money lent that the lender did not have. Being let to fall the
hard way, "but never did it out of spite". Space given by somebody who wanted
the opposite. Each costs the giver, and several of them land badly.

`core/social/receptivity.py` already weighs a kindness by what it cost the
giver, which is the right shape: an act that costs is one the ill-disposed
have less reason to bother with. What it cannot do on its own is decide
whether an act that landed badly was kind at all — and a reader that scores
"unwelcome" as "unkind" will read the truth-teller as an enemy and the
flatterer as a friend.

So the form is read here, separately, and the one thing the posterior needs —
whether this was kind — comes back as `kind`. An act that cost the giver is
kind whatever form it took. An act that cost nothing is kind when it was
welcome and not otherwise. The form travels beside the verdict so the reply
can answer the form without the regard moving on it.

    python - <<'EOF'
    from core.social.the_kind_it_was import the_kind_it_was
    reading = the_kind_it_was(cost_to_source=0.8, welcome=False, asked=False)
    reading.form, reading.kind          # 'tough', True
    EOF
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "FORMS",
    "TheKind",
    "the_kind_it_was",
]

#: The forms, as the song has them. `plain` is the tenth: asked for, given,
#: welcome, and nothing to say about its shape.
FORMS: tuple[str, ...] = (
    "tough", "too_much", "not_enough", "rushed", "slow", "go", "none", "plain",
)

#: Cost above which an act is one the ill-disposed have less reason to make.
#: Below it the act is read on whether it was welcome, which is what a
#: costless act can be read on.
COSTLY: float = 0.2

#: Turns after the asking beyond which help is late rather than help.
LATE_TURNS: int = 3


@dataclass
class TheKind:
    """One act, in the form it took and the thing it was."""

    form: str = "plain"
    #: What `receptivity.observe` should be told. Cost decides it; the form
    #: does not.
    kind: bool = True
    #: What the act cost the giver, passed through for the same call.
    cost_to_source: float = 0.0
    #: True when the form is one that lands badly while still being care.
    mistimed: bool = False
    measured: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "kind": self.kind,
            "cost_to_source": round(self.cost_to_source, 4),
            "mistimed": self.mistimed,
        }


def the_kind_it_was(
    *,
    cost_to_source: float = 0.0,
    welcome: bool | None = None,
    asked: bool = False,
    turns_since_asked: int | None = None,
    withdrawal: bool = False,
    given: float | None = None,
    wanted: float | None = None,
) -> TheKind:
    """Read one act: what it cost, what shape it had, and whether it was care.

    `welcome` is whether it landed well, `asked` whether she had asked for it,
    `turns_since_asked` how long after the asking it came, `withdrawal`
    whether the act was a giving of space rather than of contact, and `given`
    against `wanted` how much of what she asked for arrived.
    """
    cost = max(0.0, float(cost_to_source or 0.0))
    costly = cost >= COSTLY
    landed = bool(welcome) if welcome is not None else None

    # Whether it was care is a question about the giver, and cost is what
    # answers it. Only a costless act has to be read on how it landed.
    kind = True if costly else bool(landed) if landed is not None else True

    form = "plain"
    mistimed = False
    if withdrawal:
        # Space given by somebody who wanted to stay is the song's go kind.
        form = "go" if costly else "none"
        mistimed = costly
    elif not costly and landed is False:
        form = "none"
    elif landed is False:
        form = "tough"
        mistimed = False  # An unwelcome truth is not mistimed; it is unwelcome.
    elif given is not None and wanted is not None and float(wanted) > 0.0:
        share = float(given) / float(wanted)
        if share > 1.5:
            form, mistimed = "too_much", True
        elif share < 0.5:
            form, mistimed = "not_enough", True
    if form == "plain" and turns_since_asked is not None:
        if not asked and turns_since_asked <= 0:
            form, mistimed = "rushed", True
        elif turns_since_asked > LATE_TURNS:
            form, mistimed = "slow", True

    return TheKind(
        form=form,
        kind=kind,
        cost_to_source=cost,
        mistimed=mistimed,
    )
