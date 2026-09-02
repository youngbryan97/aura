"""How much evidence a claim about herself needs, and how much of the budget to spend.

Four things, and each of them replaces a number somebody would otherwise have
had to choose.

**How many probes.** A change that looked better on three families might have
looked better by luck. Hoeffding's bound says how many bounded observations a
claim of a given size needs at a given confidence, and it is arithmetic rather
than a convention: `n ≥ ln(2/δ) / (2ε²)`. `enough_families_to_say` is that,
and `sure_enough` is the sequential version — stop as soon as the interval
clears zero, which usually happens well before the worst case.

**Which action, when several might do.** Taking the best estimate every time
means an action that failed once early is never tried again, and its estimate
never improves. Drawing from what the record supports instead — a sample from
the Beta the counts imply — tries the uncertain one exactly as often as the
uncertainty warrants. Nothing is tuned; the counts do it.

**How much to spend on developing at all.** Not a fraction. The share follows
what development has actually returned against what answering returns, both
measured, so a stretch where nothing pays shrinks it and a stretch where things
pay opens it.

**When the bar should be higher.** A live question in front of her raises the
threshold, because the cost of developing now includes the answer that did not
get written. That is the opportunity cost, and it is the one term the value
function could not see from the record alone.

The winner's curse
------------------
Choosing the largest of several noisy estimates and then believing it is how a
system convinces itself that everything works. `after_the_winners_curse`
shrinks a winning estimate by how many rivals it beat, which is the correction
that stops a promotion policy from being an optimist.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from typing import Any, Sequence

__all__ = [
    "after_the_winners_curse",
    "enough_families_to_say",
    "how_much_to_spend_on_developing",
    "how_sure",
    "more_likely_than_not_better",
    "sure_enough",
    "the_bar_right_now",
    "which_to_try",
]

logger = logging.getLogger("Aura.HowSureSheIs")


def enough_families_to_say(*, at_least: float, sure: float = 0.95) -> int:
    """How many bounded observations a claim of this size needs.

    Hoeffding: `n ≥ ln(2/δ) / (2ε²)`. Nothing chosen — the size of the claim
    and the confidence wanted determine it, and both are stated by whoever is
    making the claim rather than hidden in a constant.
    """
    gap = max(1e-9, float(at_least))
    doubt = max(1e-9, min(0.5, 1.0 - float(sure)))
    return int(math.ceil(math.log(2.0 / doubt) / (2.0 * gap * gap)))


def how_sure(seen: Sequence[float], *, sure: float = 0.95) -> tuple[float, float]:
    """The mean of these and how far it could be out, at this confidence.

    Hoeffding again, read the other way: given `n`, what interval does that
    confidence buy. Returns the mean and the half-width.
    """
    if not seen:
        return 0.0, float("inf")
    doubt = max(1e-9, min(0.5, 1.0 - float(sure)))
    width = math.sqrt(math.log(2.0 / doubt) / (2.0 * len(seen)))
    return statistics.mean(seen), width


def sure_enough(
    seen: Sequence[float], *, better_by: float = 0.0, sure: float = 0.95
) -> tuple[bool, str]:
    """Is this better by that much, with the interval clear of it?

    The sequential form: keep looking until the whole interval is on one side
    of the claim. A single comparison that came out ahead is not evidence and
    a rule that treats it as evidence promotes noise.
    """
    middle, width = how_sure(seen, sure=sure)
    if width == float("inf"):
        return False, "nothing seen"
    if middle - width > better_by:
        return True, f"{middle:.3f} ± {width:.3f}, clear of {better_by:.3f}"
    if middle + width < better_by:
        return False, f"{middle:.3f} ± {width:.3f}, below {better_by:.3f}"
    return False, f"{middle:.3f} ± {width:.3f}, still straddles {better_by:.3f}"


def more_likely_than_not_better(
    wins: int, of: int, *, than: float
) -> tuple[bool, str]:
    """Is it more likely than not that this beats that rate, given what was seen?

    A decision rather than a significance test, and the difference matters. A
    confidence level is a number somebody picks; this asks whether the
    posterior puts more than half its mass above the rate she is used to, and
    there is nothing left to choose.

    It also has to be reachable. Hoeffding at ninety-five per cent wants a
    hundred and eighty-five families to support a ten per cent claim, and with
    four families in hand no evidence whatever clears it — which is a gate that
    cannot fire, not a careful one. The exact posterior on four families says
    plenty.

    Beta(wins + 1, of - wins + 1), the posterior after Laplace, and the mass
    above `than` computed exactly from the regularised incomplete beta.
    """
    of = max(0, int(of))
    wins = max(0, min(of, int(wins)))
    if of == 0:
        return False, "nothing seen"
    a, b = wins + 1, of - wins + 1
    # P(rate > than) for Beta(a, b) is 1 - I_than(a, b), and for whole a and b
    # the regularised incomplete beta is a finite sum.
    mass = 0.0
    for k in range(a, a + b):
        mass += (
            math.comb(a + b - 1, k)
            * (than**k)
            * ((1.0 - than) ** (a + b - 1 - k))
        )
    above = 1.0 - mass
    return above > 0.5, (
        f"{wins} of {of}, so {above:.2f} of the posterior is above {than:.2f}"
    )


def after_the_winners_curse(best: float, over: int, *, spread: float = 0.0) -> float:
    """A winning estimate, shrunk by how many rivals it beat.

    The largest of several noisy numbers is larger than the truth by an amount
    that grows with how many were compared. Ignoring that is how a promotion
    policy convinces itself that everything it picked worked.

    The correction is the expected maximum of that many standard normals,
    approximated by `sqrt(2 ln k)`, times the spread among them.
    """
    if over < 1 or spread <= 0:
        return best
    return best - spread * math.sqrt(2.0 * math.log(over + 1))


def which_to_try(
    among: Sequence[Any],
    *,
    pays: Any,
    gains: Any,
    rng: random.Random | None = None,
) -> Any | None:
    """Draw one, from what the counts support rather than from the best guess.

    Taking the argmax means an action that failed once early is never tried
    again and its estimate never improves. A draw from the Beta the counts
    imply tries the uncertain one exactly as often as the uncertainty warrants,
    and there is nothing to tune: the counts are the whole of it.
    """
    if not among:
        return None
    draw = rng or random
    best: tuple[float, Any] | None = None
    for one in among:
        kept, taken = pays(one)
        # Beta(kept + 1, taken - kept + 1): the posterior after Laplace, and a
        # draw from it is what "as often as the uncertainty warrants" means.
        sampled = draw.betavariate(max(1e-6, kept + 1), max(1e-6, taken - kept + 1))
        worth = sampled * float(gains(one))
        if best is None or worth > best[0]:
            best = (worth, one)
    return best[1] if best else None


def how_much_to_spend_on_developing() -> float:
    """The share of the budget development may have, from what it has returned.

    Not a fraction anybody set. What development has returned per candidate
    against what answering returns per candidate, both off the record. A
    stretch where nothing pays shrinks it towards nothing; a stretch where
    things pay opens it.
    """
    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE

    answering = [one.walked for one in the_record().kept if one.route == "an answer"]
    if not answering:
        return 0.5  # nothing to compare against, so neither is favoured
    per_answer = sum(answering) / len(answering)
    gained = [
        one.what_it_gains for one in WHAT_THEY_HAVE_DONE.values() if one.gained
    ]
    if not gained:
        return 0.5
    per_change = sum(gained) / len(gained)
    whole = per_answer + per_change
    return 0.5 if whole <= 0 else max(0.0, min(1.0, per_change / whole))


def the_bar_right_now(*, a_question_is_waiting: bool = False) -> float:
    """How much better than nothing a change has to be, here and now.

    Zero when nothing is waiting: a change that pays at all is worth making.
    With a question in front of her the bar is what answering it is worth,
    because the cost of developing now includes the answer that did not get
    written — and that is the one term the record cannot see, since an answer
    she never gave leaves no episode.
    """
    if not a_question_is_waiting:
        return 0.0
    from core.cognition.the_record_of_her_own_work import the_record

    answering = [one.walked for one in the_record().kept if one.route == "an answer"]
    return float(sum(answering) / len(answering)) if answering else 0.0
