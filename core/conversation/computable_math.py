"""Questions with one exact answer that Python can produce.

Aura has Python. When a question has a single mechanical answer, generating
prose about it is strictly worse than computing it: the model is guessing at
something the machine can know.

`arithmetic_check` already computes expressions — 7919 * 6367, 2^31 - 1, 15%
of 240 — after a live miss where "7919 times 6421" came back 50864799 for a
product of 50847899. The same argument covers the named functions that were
still going to the model: a factorial's digit count, a primality test, the nth
Fibonacci number, a square root, a remainder, a gcd, a binomial coefficient.
Each is exact, and each is exactly what a person asks a machine rather than
work out by hand.

Every form declares the questions it claims WITH their answers, so the
registry checks itself: a pattern that stops matching its own example, or
computes a different number for it, fails the suite rather than quietly
returning None and handing the turn back to the model.

Bounded on purpose. Each form caps its input where the computation stops being
instant — measured, not guessed, in the test beside it — because a checker
that hangs the turn it was meant to answer is a worse failure than not
answering. Out-of-range questions return None and the ordinary path handles
them.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COMPUTABLE_FORMS",
    "ComputedNumber",
    "computable_result",
    "is_prime_answer",
    "ComputableForm",
    "computable_answer",
    "form_failures",
]


@dataclass(frozen=True)
class ComputedNumber:
    """An exact number together with the code object that produced it."""

    value: int | float
    form: str
    module: str
    function: str

    @property
    def source(self) -> str:
        return f"{self.module}.{self.function}"

    def provenance(self) -> str:
        return (
            f"computed by {self.source} (form {self.form!r}), "
            "run as Python, not generated"
        )


@dataclass(frozen=True, slots=True)
class ComputableForm:
    """One shape of question, and the computation that answers it."""

    name: str
    pattern: re.Pattern[str]
    #: Takes the match, returns the exact answer, or None when out of bounds.
    compute: Callable[[re.Match[str]], int | float | None]
    #: Questions it MUST answer, with the answer it must give.
    examples: tuple[tuple[str, int | float], ...] = ()
    #: Questions it must NOT claim — usually a neighbour form's.
    counter_examples: tuple[str, ...] = field(default=())

    def failures(self) -> list[str]:
        found: list[str] = []
        for question, expected in self.examples:
            got = computable_answer(question)
            if got is None:
                found.append(f"{self.name}: computed nothing for {question!r}")
            elif got != expected:
                found.append(
                    f"{self.name}: {question!r} -> {got!r}, expected {expected!r}"
                )
        for question in self.counter_examples:
            match = self.pattern.search(question)
            if match is not None and self.compute(match) is not None:
                found.append(f"{self.name}: wrongly claimed {question!r}")
        return found


# ── bounds ───────────────────────────────────────────────────────────────────
#
# These cap the WORK. What can actually be served is capped separately by
# _renderable, since CPython refuses to write down an integer past
# sys.get_int_max_str_digits(). The test beside this file measures that every
# in-range question finishes inside a turn.
_MAX_FACTORIAL = 10_000
_MAX_FIBONACCI = 100_000
_MAX_PRIMALITY = 2**64
_MAX_CHOOSE = 10_000


# ── ordinary arithmetic ──────────────────────────────────────────────────────
#
# LIVE DEFECT, 2026-08-19. `computable_answer("what is 17 * 4839")` returned
# None. Eight exotic forms were covered — primality, factorials, Fibonacci,
# gcd — and plain arithmetic was not, so the commonest computable question in
# existence depended entirely on the model getting it right. When it did not,
# `arithmetic_answer_missing` destroyed every draft and the turn ended in a
# canned apology, live and repeatedly.
#
# Evaluated through `ast`, not `eval`: only literals and the arithmetic
# operators exist in this grammar, so there is no name to resolve, no call to
# make and nothing to reach.
_ARITHMETIC_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

#: Exponents are capped because 9**9**9 is a denial of service, not a sum.
_MAX_EXPONENT = 4096
_MAX_BASE_DIGITS = 100

_WORD_OPERATORS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), "**"),
    (re.compile(r"\braised\s+to\b", re.IGNORECASE), "**"),
    (re.compile(r"\b(?:multiplied\s+by|times)\b", re.IGNORECASE), "*"),
    (re.compile(r"\b(?:divided\s+by|over)\b", re.IGNORECASE), "/"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    (re.compile(r"\bmodulo\b", re.IGNORECASE), "%"),
    (re.compile(r"\b[x×]\b", re.IGNORECASE), "*"),
    (re.compile(r"[÷]"), "/"),
    # A caret is exponentiation everywhere except Python. Without this,
    # "what is 2^31 - 1?" matched the FRAGMENT "31 - 1" and answered 30.
    (re.compile(r"\^"), "**"),
)

#: An expression: numbers joined by operators, at least one operator present.
_ARITHMETIC_EXPRESSION_RE = re.compile(
    r"(?<![\w.])"
    r"\(*\s*-?\d[\d,]*(?:\.\d+)?\s*\)*"
    r"(?:\s*(?:\*\*|//|[-+*/%])\s*\(*\s*-?\d[\d,]*(?:\.\d+)?\s*\)*)+"
)


def _evaluate_node(node: ast.AST) -> int | float | None:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _evaluate_node(node.operand)
        if inner is None:
            return None
        return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp):
        handler = _ARITHMETIC_OPS.get(type(node.op))
        if handler is None:
            return None
        left, right = _evaluate_node(node.left), _evaluate_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, int) or abs(right) > _MAX_EXPONENT:
                return None
            if isinstance(left, int) and len(str(abs(left))) > _MAX_BASE_DIGITS:
                return None
        try:
            return handler(left, right)
        except (ArithmeticError, TypeError, ValueError):
            return None
    return None


def _arithmetic(match: re.Match[str]) -> int | float | None:
    """Evaluate the expression this question is asking for."""
    raw = match.group(0).replace(",", "").strip()
    if not raw:
        return None
    try:
        tree = ast.parse(raw, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    value = _evaluate_node(tree)
    if value is None:
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        # A float that is exactly an integer reads as one. 6/3 is 2, not 2.0.
        if value.is_integer() and abs(value) < 2**53:
            value = int(value)
        else:
            return round(value, 10)
    if isinstance(value, int):
        return _renderable(value)
    return value


#: Asking for a sum, rather than containing one.
#:
#: Numbers with an operator between them are everywhere in ordinary prose:
#: "the 2015 - 2020 period" is -5, "call me at 555-1234" is -679, and "she
#: scored 9/10" is 0.9. A form that answers those is worse than one that
#: answers nothing, so the text has to ASK.
_ASKS_TO_COMPUTE_RE = re.compile(
    r"\b(?:what(?:'s|\s+is|\s+are|\s+does)|how\s+much(?:\s+is)?|"
    r"calculate|compute|work\s+out|evaluate|solve|equals?|"
    r"(?:the\s+)?(?:sum|product|total|difference|quotient)\s+of)\b",
    re.IGNORECASE,
)


#: How far an expression may sit from the words that ask for it.
_MAX_WORDS_BEFORE_EXPRESSION = 3


class _ArithmeticPattern:
    """Normalises words to operators before looking for an expression.

    "2 to the power of 40" and "17 times 4839" are the same question as
    "2 ** 40" and "17 * 4839"; only the spelling differs.
    """

    def search(self, text: str) -> re.Match[str] | None:
        body = str(text or "")
        for pattern, symbol in _WORD_OPERATORS:
            body = pattern.sub(symbol, body)
        found = _ARITHMETIC_EXPRESSION_RE.search(body)
        if found is None:
            return None
        # Never answer a FRAGMENT. An expression that begins immediately after
        # an operator is the tail of a longer one, and its value is not the
        # answer to anything that was asked.
        before = body[: found.start()].rstrip()
        if before and before[-1] in "+-*/%^(":
            return None
        # A bare expression IS the request; anything else has to ask for one,
        # and the asking has to come first.
        if found.group(0).strip() == body.strip().rstrip("?=. "):
            return found
        asked = _ASKS_TO_COMPUTE_RE.search(body)
        if asked is None or asked.start() > found.start():
            return None
        # The expression has to be what the question is ABOUT, not something
        # inside the noun phrase it asks about.
        #
        # LIVE, 2026-08-19: "what is the optimal total time for the classic
        # 1/2/7/10 bridge and torch puzzle" was answered "0.0071428571." — the
        # slashes read as division, "what is" satisfied the gate, and because a
        # computed answer replaces the draft, that number became the entire
        # reply to a logic puzzle. Seven words stood between the question and
        # those digits; a real arithmetic question has almost none.
        if len(body[asked.end() : found.start()].split()) > _MAX_WORDS_BEFORE_EXPRESSION:
            return None
        # Digits that MODIFY a noun are a label, not a sum: a 2/3/5 split, a
        # 4/4 time signature, the 1/2/7/10 puzzle. A real computation is
        # followed by punctuation or by nothing at all.
        if re.match(r"\s*[A-Za-z]", body[found.end() :]):
            return None
        # The mirror of the fragment check above. An expression FOLLOWED by an
        # operator is the head of a longer one, and its value is not the answer
        # either.
        #
        # "What is 144 / 6 + seven?" captured "144 / 6" — the operand after the
        # plus is a word, so the expression regex stopped short — and answered
        # 24. The real answer is 31. A confidently wrong arithmetic check is
        # worse than none: it would have rejected a correct reply and accepted
        # a wrong one.
        if re.match(r"\s*[-+*/%^]", body[found.end() :]):
            return None
        return found



def _int(text: str) -> int:
    return int(str(text).replace(",", "").strip())


def _renderable(value: int) -> int | None:
    """The value, or None when this runtime cannot write it down.

    CPython refuses int-to-string conversion past sys.get_int_max_str_digits()
    (4300 by default), and it raises rather than truncating. 10000! is 35660
    digits, so asking for its digit count raised ValueError out of the answer
    path — a crash where a decline belonged. An answer that cannot be rendered
    is not an answer, so the render limit IS the bound.
    """
    try:
        str(value)
    except ValueError:
        return None
    return value


def _is_prime(number: int) -> bool:
    """Deterministic below 2**64 with these witnesses."""
    if number < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if number % small == 0:
            return number == small
    exponent, remainder = 0, number - 1
    while remainder % 2 == 0:
        remainder //= 2
        exponent += 1
    for witness in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        value = pow(witness, remainder, number)
        if value in (1, number - 1):
            continue
        for _ in range(exponent - 1):
            value = value * value % number
            if value == number - 1:
                break
        else:
            return False
    return True


def _fibonacci(index: int) -> int:
    """Fast doubling: F(100000) has 20899 digits and costs one pass."""
    def pair(n: int) -> tuple[int, int]:
        if n == 0:
            return (0, 1)
        a, b = pair(n >> 1)
        c = a * ((b << 1) - a)
        d = a * a + b * b
        return (d, c + d) if n & 1 else (c, d)

    return pair(index)[0]


# ── the forms ────────────────────────────────────────────────────────────────

def _factorial_digits(match: re.Match[str]) -> int | None:
    number = _int(match.group("n"))
    if not 0 <= number <= _MAX_FACTORIAL:
        return None
    computed = _renderable(math.factorial(number))
    return None if computed is None else len(str(computed))


def _factorial(match: re.Match[str]) -> int | None:
    number = _int(match.group("n") or match.group("n2"))
    if not 0 <= number <= _MAX_FACTORIAL:
        return None
    return _renderable(math.factorial(number))


def is_prime_answer(question: str) -> bool | None:
    """Whether a primality question is true, or None if it is not one.

    Deliberately NOT one of the numeric forms below. Those feed a channel that
    checks the reply CONTAINS the computed number, and the honest answer to
    "is 1000003 prime?" is "yes" — which contains no 1, so serving a 1 here
    would have the reply rejected for missing its own answer. The computation
    is exact and ready for a channel that can carry a boolean.
    """
    match = _PRIMALITY_RE.search(str(question or ""))
    if match is None:
        return None
    number = _int(match.group("n"))
    if not 0 <= number < _MAX_PRIMALITY:
        return None
    return _is_prime(number)


def _nth_fibonacci(match: re.Match[str]) -> int | None:
    index = _int(match.group("n") or match.group("n2"))
    if not 0 <= index <= _MAX_FIBONACCI:
        return None
    return _renderable(_fibonacci(index))


def _square_root(match: re.Match[str]) -> int | None:
    """Perfect squares only.

    The checker that consumes this asks whether the reply contains the value.
    sqrt(2) is 1.4142135623730951 and a good answer says "about 1.414", which
    would be scored as the wrong number — so an irrational root is left to the
    ordinary path, where a rounded answer is correct.
    """
    text = match.group("n").replace(",", "")
    if not text.isdigit():
        return None
    number = int(text)
    root = math.isqrt(number)
    return root if root * root == number else None


def _remainder(match: re.Match[str]) -> int | None:
    divisor = _int(match.group("d") or match.group("d2"))
    if divisor == 0:
        return None
    return _int(match.group("n") or match.group("n2")) % divisor


def _gcd(match: re.Match[str]) -> int:
    return math.gcd(_int(match.group("a")), _int(match.group("b")))


def _lcm(match: re.Match[str]) -> int:
    return math.lcm(_int(match.group("a")), _int(match.group("b")))


def _choose(match: re.Match[str]) -> int | None:
    total, taken = _int(match.group("n")), _int(match.group("k"))
    if not 0 <= taken <= total <= _MAX_CHOOSE:
        return None
    return _renderable(math.comb(total, taken))


def _choose_from(match: re.Match[str]) -> int | None:
    total = _int(match.group("n") or match.group("n2"))
    taken = _int(match.group("k") or match.group("k2"))
    if not 0 <= taken <= total <= _MAX_CHOOSE:
        return None
    return _renderable(math.comb(total, taken))


_N = r"(?P<n>[0-9][0-9,]*)"
_PRIMALITY_RE = re.compile(r"\bis\s+(?P<n>[0-9][0-9,]*)\s+(?:a\s+)?prime\b", re.IGNORECASE)

def _rectangle_area(match: re.Match[str]) -> int | float | None:
    """Length times width, wherever the sentence puts the two numbers.

    The dimensions of a rectangle are written in whatever order the sentence
    wants — "the rectangle is 3 by 4, what is its area", "the area of a 3 by 4
    rectangle", "a 3 by 4 rectangle, area?". A single pattern that fixed the
    order answered the first and refused the rest.
    """
    left = match.group("a") or match.group("a2")
    right = match.group("b") or match.group("b2")
    if not left or not right:
        return None
    try:
        if "." in left or "." in right:
            return float(left) * float(right)
        return int(left) * int(right)
    except (ArithmeticError, ValueError):
        return None


COMPUTABLE_FORMS: tuple[ComputableForm, ...] = (
    ComputableForm(
        "factorial_digits",
        re.compile(
            rf"how\s+many\s+digits?\b(?s:.){{0,40}}?\b{_N}\s*(?:!|factorial)",
            re.IGNORECASE,
        ),
        _factorial_digits,
        examples=(
            ("how many digits are in 100 factorial?", 158),
            ("how many digits does 20! have?", 19),
        ),
        counter_examples=("what is 5 factorial?",),
    ),
    ComputableForm(
        "factorial",
        re.compile(
            rf"what(?:'s| is)\s+{_N}\s*(?:!|factorial)\b|"
            r"\bfactorial\s+of\s+(?P<n2>[0-9][0-9,]*)\b",
            re.IGNORECASE,
        ),
        _factorial,
        examples=(
            ("what is 5 factorial?", 120),
            ("what is the factorial of 10", 3_628_800),
        ),
    ),
    ComputableForm(
        "nth_fibonacci",
        re.compile(
            rf"{_N}(?:st|nd|rd|th)?\s+fibonacci|"
            r"fibonacci\s+(?:number\s+)?(?P<n2>[0-9][0-9,]*)\b",
            re.IGNORECASE,
        ),
        _nth_fibonacci,
        examples=(
            ("what is the 10th fibonacci number?", 55),
            (
                "what is the 200th fibonacci number?",
                280571172992510140037611932413038677189525,
            ),
        ),
    ),
    ComputableForm(
        "square_root",
        re.compile(
            rf"(?:square\s+root|sqrt)\s+of\s+{_N}\b", re.IGNORECASE
        ),
        _square_root,
        examples=(
            ("what is the square root of 144?", 12),
            ("what is the square root of 1000000?", 1000),
        ),
    ),
    ComputableForm(
        "remainder",
        re.compile(
            # `mod`/`modulo` spelled out is unambiguous. The bare `%` SYMBOL is
            # not — "what is 10 % 3" reads as a percentage at least as
            # naturally as a remainder, and arithmetic_check refuses modulo for
            # exactly that reason: an operator that cannot be honoured
            # unambiguously must be declined, because refusing costs a fallback
            # and answering costs a confident wrong number. Accepting the
            # symbol here reached past that refusal and answered 1.
            rf"what\s+is\s+{_N}\s*mod(?:ulo)?\s*(?P<d>[0-9][0-9,]*)|"
            r"remainder\s+(?:of|when)\s+(?P<n2>[0-9][0-9,]*)\s+"
            r"(?:is\s+)?(?:divided\s+by|/)\s*(?P<d2>[0-9][0-9,]*)",
            re.IGNORECASE,
        ),
        _remainder,
        examples=(
            ("what is 17 mod 5?", 2),
            ("what is the remainder when 123456 is divided by 7", 4),
        ),
    ),
    ComputableForm(
        "gcd",
        re.compile(
            r"(?:gcd|hcf|(?:greatest|highest|largest)\s+common\s+(?:divisor|factor))\s+of\s+"
            r"(?P<a>[0-9][0-9,]*)\s*(?:and|,|&)\s*(?P<b>[0-9][0-9,]*)",
            re.IGNORECASE,
        ),
        _gcd,
        examples=(
            ("what is the gcd of 462 and 1071?", 21),
            ("greatest common divisor of 48 and 180", 12),
            ("what is the highest common factor of 54 and 24", 6),
        ),
    ),
    ComputableForm(
        "lcm",
        re.compile(
            r"(?:lcm|(?:least|lowest|smallest)\s+common\s+multiple)\s+of\s+"
            r"(?P<a>[0-9][0-9,]*)\s*(?:and|,|&)\s*(?P<b>[0-9][0-9,]*)",
            re.IGNORECASE,
        ),
        _lcm,
        examples=(
            ("what is the lcm of 4 and 6?", 12),
            ("lowest common multiple of 21 and 6", 42),
            ("least common multiple of 8 and 12", 24),
        ),
    ),
    ComputableForm(
        "choose",
        re.compile(
            r"(?:how\s+many\s+ways(?s:.){0,40}?choose\s+(?P<k>[0-9][0-9,]*)"
            r"(?s:.){0,20}?\bfrom\s+(?P<n>[0-9][0-9,]*)"
            r"|(?P<n2>[0-9][0-9,]*)\s+choose\s+(?P<k2>[0-9][0-9,]*))",
            re.IGNORECASE,
        ),
        _choose_from,
        examples=(
            ("52 choose 5", 2_598_960),
            ("how many ways are there to choose 3 from 10?", 120),
        ),
    ),
)


COMPUTABLE_FORMS = COMPUTABLE_FORMS + (
    ComputableForm(
        "arithmetic",
        _ArithmeticPattern(),  # type: ignore[arg-type]
        _arithmetic,
        examples=(
            ("what is 17 * 4839", 82_263),
            ("what is 2 to the power of 40", 1_099_511_627_776),
            ("what's 1024 / 8?", 128),
            ("how much is 12,500 + 3,750", 16_250),
            ("what is 7 times 6", 42),
        ),
        counter_examples=(
            "what is 5 factorial?",
            "is 97 prime",
            "tell me about the year 1984",
        ),
    ),
)


COMPUTABLE_FORMS = COMPUTABLE_FORMS + (
    ComputableForm(
        "rectangle_area",
        re.compile(
            r"\brectangl\w*\b(?s:.){0,60}?"
            r"(?P<a>[0-9]+(?:\.[0-9]+)?)\s*(?:by|x|\u00d7|\*)\s*"
            r"(?P<b>[0-9]+(?:\.[0-9]+)?)"
            r"|(?P<a2>[0-9]+(?:\.[0-9]+)?)\s*(?:by|x|\u00d7|\*)\s*"
            r"(?P<b2>[0-9]+(?:\.[0-9]+)?)(?s:.){0,30}?\brectangl\w*\b",
            re.IGNORECASE,
        ),
        _rectangle_area,
        examples=(
            ("the rectangle is 3 by 4, what is its area?", 12),
            ("what is the area of a 3 by 4 rectangle", 12),
            ("area of a rectangle 2.5 by 4", 10.0),
        ),
        counter_examples=(
            "what is the area of a circle with radius 3",
            "how many rectangles are on the screen",
        ),
    ),
)


def computable_result(question: str) -> ComputedNumber | None:
    """The exact answer and the code object that produced it.

    Same reason as the text side: an answer with no account of its own
    mechanism gets one invented for it when the person asks how it was done.
    """
    text = str(question or "")
    if not text.strip():
        return None
    for form in COMPUTABLE_FORMS:
        match = form.pattern.search(text)
        if match is None:
            continue
        answer = form.compute(match)
        if answer is not None:
            return ComputedNumber(
                value=answer,
                form=form.name,
                module=getattr(form.compute, "__module__", __name__),
                function=getattr(form.compute, "__qualname__", form.name),
            )
    return None


def computable_answer(question: str) -> int | float | None:
    """The exact answer, or None when no form claims the question.

    The first matching form wins, and the forms are ordered so a narrower one
    is asked first: "how many digits are in 100 factorial" is a question about
    the digit count, not about 100!.
    """
    text = str(question or "")
    if not text.strip():
        return None
    for form in COMPUTABLE_FORMS:
        match = form.pattern.search(text)
        if match is None:
            continue
        answer = form.compute(match)
        if answer is not None:
            return answer
    return None


def form_failures() -> list[str]:
    """Every declared example a form gets wrong."""
    return [failure for form in COMPUTABLE_FORMS for failure in form.failures()]


def capability_vocabulary() -> tuple[str, ...]:
    """The words a person uses to ask for these, taken from the forms.

    See the note on the text side: derived, so a form added later is a
    capability she knows she has without anyone writing it down twice.
    """
    words: list[str] = []
    for form in COMPUTABLE_FORMS:
        words.append(form.name.replace("_", " "))
        words.extend(question for question, _answer in form.examples)
    return tuple(words)
