"""What kind of problem a request is, read from its words and its shape.

Moved out of core/brain/reasoning_amplifier_v2.py, which re-exports these
names: the amplifier picks its verifier by the kind, and the declared
capability check reads whether a prose problem is a quantitative puzzle.
"""
from __future__ import annotations

import re

__all__ = ["asks_a_reference_question", "classify_task_type"]


# Naming a language is naming code. Without these, "how would you reverse a
# string in python?" matched only "how would you" and classified as PLANNING,
# so a code question asked the ordinary way never reached the code verifier
# and was answered without anything running it.
_CODE_HINT = re.compile(
    r"\b(code|function|bug|patch|compile|traceback|stack ?trace|def |class |import |"
    r"python|javascript|typescript|rust|golang|java|kotlin|swift|ruby|perl|scala|"
    r"c\+\+|c#|sql|bash|shell script|regex|regular expression|"
    r"script|method|algorithm|docstring|unit test|api call)\b",
    re.I,
)
_MATH_HINT = re.compile(
    r"\b(calculate|compute|how many|sum|product|equation|solve|factorial|prime|"
    r"to the power of|raised to|power of|mod(?:ulo)?|gcd|greatest common divisor|"
    r"remainder|divisible|\d+\s*[-+*/^]\s*\d+)\b",
    re.I,
)
_REPO_HINT = re.compile(r"\b(repo|codebase|module|where is|which file|architecture|how does .* work|implemented)\b", re.I)
_PLAN_HINT = re.compile(
    r"\b(plan|steps|how (?:do|would) (?:i|we|you)|approach|strategy|roadmap|"
    r"schedul\w*|makespan|deadline\w*|prerequisite\w*|dependenc\w*|"
    r"resource allocation|execution order|horizon)\b",
    re.I,
)
# Checked last, so widening it only rescues turns that were falling through to
# "generic" — where no verifier plan applies at all. "Who wrote Hamlet?" is as
# factual a question as "who is Hamlet's author", and only the second one was
# recognised.
_FACT_HINT = re.compile(
    r"\b(what is|what are|what was|what were|who is|who was|who are|"
    r"who (?:wrote|made|created|invented|discovered|founded|built|painted|"
    r"composed|directed|designed|won)|"
    r"when (?:did|was|were|is)|where (?:is|was|did|are)|"
    r"which (?:year|country|city|company|language|element)|"
    r"define|explain|fact|true that)\b",
    re.I,
)
_LOGIC_HINT = re.compile(
    r"\b(causal|counterfactual|intervention|premise|claim|deduce|infer|"
    r"constraint|actual winner|highest score)\w*\b",
    re.I,
)


#: A problem stated in prose, with quantities and constraints and one answer.
#:
#: LIVE DEFECT, 2026-08-19. The bridge-and-torch puzzle — four people, times
#: 1, 2, 7 and 10, at most two crossing, torch must come back — classified as
#: `generic`, so the verifier-backed amplifier never saw it. She derived the
#: correct schedule and then stated the total as 19 where her own steps sum to
#: 17. Sampling and checking is exactly what catches that, and it was switched
#: off by vocabulary: every hint above is a keyword list, and a novel problem
#: uses none of those words. That is the definition of a task outside the
#: template.
#:
#: Recognised by shape instead: several quantities, a rule constraining them,
#: and a definite question. Conservative on purpose — amplification costs
#: samples, so a chatty message that merely contains numbers must not trigger
#: it.
_CONSTRAINT_HINT = re.compile(
    r"\b(?:at most|at least|no more than|only|must|cannot|can't|each|every|"
    r"exactly|per|apiece|both|neither|either|between them|in total)\b",
    re.I,
)
_DEFINITE_QUESTION_HINT = re.compile(
    r"\b(?:who|which|what|how many|how long|how much|in what order|what order|"
    r"how do|how would|how can|minimum|maximum|fastest|shortest|optimal)\b",
    re.I,
)
#: Quantities are quantities whether or not they are written as digits: the
#: three-switches puzzle names none in figures and is no less a puzzle.
_NUMBER_TOKEN = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|twenty|thirty|forty|fifty|hundred|dozen|"
    r"once|twice|thrice|single|pair|both)\b",
    re.I,
)


def _looks_like_a_quantitative_puzzle(text: str) -> bool:
    """True when a prose problem has quantities, a constraint and one answer."""
    body = str(text or "")
    if len(body) < 60:
        return False
    numbers = _NUMBER_TOKEN.findall(body)
    if len(numbers) < 3:
        return False
    if not _DEFINITE_QUESTION_HINT.search(body):
        return False
    return bool(_CONSTRAINT_HINT.search(body))


def classify_task_type(text: str) -> str:
    """Classify a problem, checking the SOURCE-DEPENDENT class first.

    CP126 93e56508. The code regex ran before the repository regex, and
    _CODE_HINT matches ordinary words like "code", "function", "class",
    "import" and "bug". So "where is the retry logic implemented in this
    codebase" classified as `code` — a source-INDEPENDENT type, which
    changes verifier selection and admits the answer to the solved cache and
    to self-improvement traces.

    That is unsound for a question whose answer depends on the mutable
    source tree: the file moves, the function is renamed, and a cached
    answer keeps being served as current truth. Repository questions are
    therefore recognised first, so a question that is about the codebase is
    classified as being about the codebase even when it mentions code.
    """
    t = str(text or "")
    if _REPO_HINT.search(t):
        return "repo_audit"
    if _CODE_HINT.search(t):
        return "code"
    if _MATH_HINT.search(t):
        return "math"
    if _PLAN_HINT.search(t):
        return "planning"
    if _LOGIC_HINT.search(t):
        return "logic"
    if _looks_like_a_quantitative_puzzle(t):
        return "math"
    if _FACT_HINT.search(t):
        return "factual"
    return "generic"


def asks_a_reference_question(text: str) -> bool:
    """Whether an encyclopedia would help answer this, independent of class.

    `classify_task_type` returns ONE label and checks the source-dependent
    classes first, so "explain Dijkstra\'s algorithm" comes back `code` — the
    word algorithm wins before the word explain is reached. That ordering is
    right for picking a verifier and wrong for deciding whether to fetch
    reference evidence: the question still wants a definition.

    A repository question is excluded. Its answer lives in the mutable source
    tree, and offline reference material cannot say anything true about it.
    """
    body = str(text or "")
    if _REPO_HINT.search(body):
        return False
    return bool(_FACT_HINT.search(body))
