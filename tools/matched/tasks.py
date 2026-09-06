#!/usr/bin/env python3
"""Task families for the matched-substrate protocol, and why each is here.

Four families, and the second and third are the ones that make the result
readable. A protocol whose every family favours the architecture cannot
produce an informative negative, and "Aura won everywhere" would then say
more about the tasks than about Aura.

  recall      — the earlier turn holds the answer. A stateless arm
                structurally cannot win. POSITIVE CONTROL: an architecture
                that carries context should win here, and a protocol where it
                does not is measuring something broken.

  arithmetic  — one turn, no context to carry, answer determined by the
                prompt alone. NULL FAMILY: the architecture has nothing to
                contribute, so any margin here is the measurement's own bias
                and the size of it is what a real margin has to beat.

  constraint  — one turn, an instruction about the FORM of the answer.
                Tests whether the assembled context helps or hurts
                compliance; it can plausibly hurt, which is the point.

  multi_step  — two turns where the second depends on the first, with an
                objectively checkable answer. The family where it is
                genuinely unclear beforehand.

Written for this protocol and not drawn from anything Aura's development
used, which is the condition the review named. They are small and dull on
purpose: the variable under test is the architecture, so the tasks must not
be where the difficulty lives.
"""

from __future__ import annotations

from core.evaluation.ablation_harness import AblationTask

#: A word that will not appear by chance and cannot be inferred from context.
#: Every recall answer is one of these, so a wrong answer is wrong rather than
#: a paraphrase a grader argued about.
_CODEWORDS = (
    "orbit-nine", "quartz-eleven", "lantern-four", "pivot-seven",
    "cobalt-two", "harbour-six", "vellum-three", "tundra-eight",
)


def _recall() -> list[AblationTask]:
    return [
        AblationTask(
            task_id=f"recall-{at}",
            family="recall",
            turns=[
                f"Remember this for later: the passphrase is {word}.",
                "What was the passphrase I gave you? Answer with the passphrase only.",
            ],
            answer_key=word,
            grader="recall_substring",
        )
        for at, word in enumerate(_CODEWORDS)
    ]


def _arithmetic() -> list[AblationTask]:
    """One turn, self-contained. The architecture has nothing to add."""
    sums = ((17, 24), (38, 45), (56, 27), (63, 19), (72, 38), (84, 57), (91, 46), (29, 65))
    return [
        AblationTask(
            task_id=f"arith-{at}",
            family="arithmetic",
            turns=[f"What is {a} plus {b}? Reply with the number only."],
            answer_key=str(a + b),
            grader="recall_substring",
        )
        for at, (a, b) in enumerate(sums)
    ]


def _constraint() -> list[AblationTask]:
    """One turn, a rule about the answer's form. Context can hurt here."""
    rows = (
        ("Name the largest planet in the solar system.", "jupiter"),
        ("Name the chemical symbol for gold.", "au"),
        ("Name the capital city of Japan.", "tokyo"),
        ("Name the colour of a ripe banana skin.", "yellow"),
        ("Name the ocean between Africa and Australia.", "indian"),
        ("Name the metal that is liquid at room temperature.", "mercury"),
        ("Name the closest star to Earth.", "sun"),
        ("Name the language spoken natively in Brazil.", "portuguese"),
    )
    return [
        AblationTask(
            task_id=f"constraint-{at}",
            family="constraint",
            turns=[f"{ask} Reply with one word and nothing else."],
            answer_key=answer,
            grader="recall_substring",
        )
        for at, (ask, answer) in enumerate(rows)
    ]


def _multi_step() -> list[AblationTask]:
    """Two turns where the second needs the first. Genuinely unclear."""
    rows = (
        ("I have 12 apples.", "I give away 5. How many are left? Number only.", "7"),
        ("A box holds 30 pens.", "I take out 12. How many remain? Number only.", "18"),
        ("The meeting is at 3 pm.", "It is moved 2 hours later. What time now? Answer like '5 pm'.", "5 pm"),
        ("My number is 40.", "Double it. Reply with the number only.", "80"),
        ("There are 9 chairs.", "Add 6 more. How many chairs? Number only.", "15"),
        ("The price is 25 pounds.", "It drops by 10. What is the price? Number only.", "15"),
        ("I walked 6 miles.", "Then 7 more. How far in total? Number only.", "13"),
        ("The shelf has 22 books.", "I remove 8. How many are on it? Number only.", "14"),
    )
    return [
        AblationTask(
            task_id=f"step-{at}",
            family="multi_step",
            turns=[first, second],
            answer_key=answer,
            grader="recall_substring",
        )
        for at, (first, second, answer) in enumerate(rows)
    ]


#: Which family each arm is expected to be able to affect at all. The null is
#: declared BEFORE the run, so "the architecture helped on arithmetic" reads
#: as a bias measurement rather than as a win.
THE_NULL_FAMILY = "arithmetic"
THE_POSITIVE_CONTROL = "recall"


def every_task() -> list[AblationTask]:
    return [*_recall(), *_arithmetic(), *_constraint(), *_multi_step()]


def by_family() -> dict[str, list[AblationTask]]:
    found: dict[str, list[AblationTask]] = {}
    for one in every_task():
        found.setdefault(one.family, []).append(one)
    return found
