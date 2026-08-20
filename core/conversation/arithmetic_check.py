"""core/conversation/arithmetic_check.py — the answer this runtime can check itself.

Extracted from ``response_reliability.py``, where it did not belong.

That module's job — and the job the lexical-debt ratchet watches it for — is
deciding what Aura may SAY. This code does something different in kind: it
reads the arithmetic a person asked for, computes the answer, and hands back a
number. It bans no phrase and suppresses no reply. It is the clearest example
of what that module's own docstring calls the checks that are "genuinely
causal — the arithmetic check recomputes the sum".

Living in the watched file, seventeen compiled patterns of arithmetic PARSING
were counted as output-filter debt, so a legitimate natural-language number
parser pushed a gate red that exists to catch phrase-banning. Deleting working
arithmetic to satisfy that count would have been optimising the measure
instead of the thing. Moving it puts it where it belongs and lets the ratchet
measure what it claims to.

Why the parser is this elaborate:

LIVE DEFECT, 2026-08-10. "what is 7919 times 6421? just the number." came back
50864799; the product is 50847899. The deterministic verifier that exists to
catch exactly this returned None, because the question pattern wanted a
lead-in verb AND symbol operators. It computed nothing for "times", nothing
for "multiply 7919 by 6421", and nothing for a bare "2+2" — so the check that
knows the right answer never ran, on any phrasing a person actually uses.

Bounded on purpose. Evaluation is an AST walk over a whitelisted node set, not
``eval``; exponents are capped; division by zero and non-finite results return
None. A wrong "computed" value injected as authoritative is worse than none,
so intent is required before anything is computed — numbers with operators
between them also appear in version strings, dates and ranges.
"""

from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, InvalidOperation

from core.conversation.computable_math import computable_result
from typing import Any

ArithmeticResult = int | float

__all__ = [
    "ARITHMETIC_NUMBER_RE",
    "ArithmeticResult",
    "arithmetic_answer_matches",
    "requested_arithmetic_provenance",
    "requested_arithmetic_result",
]


# Arithmetic a reply can be CHECKED against. The 2026-07-25 probe asked
# "What is 144 / 6 + 7? Just the number." and was answered "Will do. Searched
# web for 'simple cognitive tasks aging'. Dementia affects simple cognitive
# tasks first…" — retrieved memory served as the answer. Nothing caught it:
# the topicality check needs topic anchors, and a bare sum has almost none, so
# a short computable question was unjudgeable by every gate in the path.
#
# It does not have to be. An arithmetic question has one right answer and the
# runtime can do the arithmetic itself, which turns "sounds plausible" into
# "is correct" for the whole class — including the hijack, which contains no
# number at all.
_ARITHMETIC_QUESTION_RE = re.compile(
    r"(?:what(?:'s| is)|calculate|compute|how much is|solve)\s*:?\s*"
    r"([0-9][0-9\s\.\+\-\*/x×÷\^\(\)]{2,60})",
    re.IGNORECASE,
)
ARITHMETIC_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

#: People write arithmetic in words at least as often as in symbols.
#:
#: LIVE DEFECT, 2026-08-10: "what is 7919 times 6421? just the number." came
#: back 50864799; the product is 50847899. The deterministic verifier that
#: exists to catch exactly this returned None, because _ARITHMETIC_QUESTION_RE
#: wanted a lead-in verb AND symbol operators. It computed nothing for "times",
#: nothing for "multiply 7919 by 6421", and nothing for a bare "2+2" — so the
#: check that knows the right answer never ran, on any phrasing a person is
#: likely to use.
_WORD_OPERATOR_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmultiplied\s+by\b|\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), "/"),
    (re.compile(r"\bplus\b|\badded\s+to\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
    # "7919 x 6421" and the typographic signs. Bounded by digits so the letter
    # x in ordinary words is untouched.
    (re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)", re.IGNORECASE), "*"),
    (re.compile(r"(?<=\d)\s*÷\s*(?=\d)"), "/"),
)
#: "multiply 7919 by 6421", "add 12 and 30" — the operator leads the operands.
_PREFIX_OPERATION_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\bmultiply\s+([\d,.]+)\s+(?:by|and|with)\s+([\d,.]+)", re.IGNORECASE
        ),
        "*",
    ),
    (re.compile(r"\badd\s+([\d,.]+)\s+(?:to|and)\s+([\d,.]+)", re.IGNORECASE), "+"),
    (
        re.compile(r"\bdivide\s+([\d,.]+)\s+by\s+([\d,.]+)", re.IGNORECASE),
        "/",
    ),
    (
        re.compile(r"\bsubtract\s+([\d,.]+)\s+from\s+([\d,.]+)", re.IGNORECASE),
        "rsub",
    ),
)
#: A bare expression anywhere in the turn: "2+2", "7919 * 6421".
#: An expression has to stand on its own, not sit inside a name.
#:
#: LIVE DEFECT, 2026-08-19. "there's a python project at
#: /private/tmp/claude-501/-Users-bryan--.../7a6cdc9e-da7f-47f7-8c38-.../ledger
#: - one of its tests is failing. read the code, work out why..." was answered
#: with a bare number. "work out" satisfied the intent gate and this pattern
#: found "7-8" INSIDE the UUID `47f7-8c38`, so a request to debug a repository
#: came back as arithmetic.
#:
#: Paths, UUIDs, version strings and hyphenated names are full of digits with
#: operators between them. The guards say the expression may not begin or end
#: flush against a word character, a dot, a slash or a hyphen — which is what
#: separates "2+2" from the middle of an identifier.
_BARE_EXPRESSION_RE = re.compile(
    r"(?<![\w./-])\d[\d.]*(?:\s*[-+*/^]\s*\d[\d.]*)+(?![\w./-])"
)
#: Only compute when the turn is actually ASKING for a computation. Numbers with
#: operators between them appear in version strings, dates and ranges, and a
#: wrong "computed" value injected as authoritative is worse than none.
_ARITHMETIC_INTENT_RE = re.compile(
    r"\b(?:what(?:'s| is| are)|calculate|compute|how much is|how many is|solve|"
    r"multiply|multiplied|divide|divided|times|plus|minus|add|subtract|"
    r"work\s+out|figure\s+out|product\s+of|sum\s+of)\b",
    re.IGNORECASE,
)


#: How far an expression may sit from the words that ask for it.
#:
#: "What is 7919 * 6367" has none between; "what is the optimal total time for
#: the classic 1/2/7/10 bridge and torch puzzle" has seven, and those digits
#: are a label rather than a sum.
_MAX_WORDS_BEFORE_EXPRESSION = 3


def _arithmetic_expression_in(text: str) -> str | None:
    """The arithmetic expression a turn is asking about, in symbol form."""
    raw = str(text or "")
    if not raw.strip():
        return None
    # A message that is nothing but an expression is a computation request even
    # with no verb in front of it: people type "2+2".
    bare_only = bool(
        re.fullmatch(r"[\d\s.,+\-*/x×÷()]+[?=.]*", raw.strip())
        and re.search(r"\d", raw)
    )
    asked_to_compute = _ARITHMETIC_INTENT_RE.search(raw)
    if not bare_only and not asked_to_compute:
        return None
    if not bare_only and asked_to_compute:
        # The expression has to be what the question is ABOUT, not something
        # inside the noun phrase it asks about.
        #
        # LIVE DEFECT, 2026-08-19. "what is the optimal total time for the
        # classic 1/2/7/10 bridge and torch puzzle" was answered with
        # "0.0071428571." — the slashes read as division, "what is" satisfied
        # the intent gate, and the computed number replaced the entire reply.
        # Seven words stood between the question and those digits; in a real
        # arithmetic question there are almost none.
        found = _BARE_EXPRESSION_RE.search(raw)
        if found and found.start() > asked_to_compute.end():
            between = raw[asked_to_compute.end() : found.start()]
            if len(between.split()) > _MAX_WORDS_BEFORE_EXPRESSION:
                return None
        if found and re.match(r"\s*[A-Za-z]", raw[found.end() :]):
            # Digits that MODIFY a noun are a label, not a sum: a 2/3/5 split,
            # a 4/4 time signature, the 80/20 rule. LIVE 2026-08-19 the second
            # of those was answered "0.1333…". A real computation is followed
            # by punctuation or by nothing.
            return None

    for pattern, operator in _PREFIX_OPERATION_RES:
        match = pattern.search(raw)
        if match:
            left = match.group(1).replace(",", "")
            right = match.group(2).replace(",", "")
            if operator == "rsub":
                # "subtract 5 from 20" is 20 - 5.
                return f"{right}-{left}"
            return f"{left}{operator}{right}"

    normalized = raw
    for pattern, symbol in _WORD_OPERATOR_SUBS:
        normalized = pattern.sub(symbol, normalized)
    # Thousands separators only, never the decimal comma: "1,000 * 2".
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)

    candidates: list[str] = []
    for match in _BARE_EXPRESSION_RE.finditer(normalized):
        # Never compute only a valid prefix of an invalid expression.
        remainder = normalized[match.end() :]
        # Includes the operators this evaluator cannot honour. Without ^ here,
        # "what is 2 + 2 ^ 3" matched "2 + 2" and answered 4.
        if re.match(r"\s*[-+*/%!]", remainder):
            continue
        # ...nor a valid SUFFIX of one. This looked only forward, so
        # "compute 2^31 - 1" matched "31 - 1" with the "2^" sitting behind it
        # unexamined, and returned 30.
        #
        # LIVE 2026-08-17 that reached the user as the whole answer: the turn
        # serves a computed value directly now, so a fragment does not merely
        # mislead a sample, it IS the reply. An operator this evaluator cannot
        # honour — exponent, modulo, root — means the expression is not one it
        # may answer, and refusing costs a fallback while answering costs a
        # confident wrong number.
        preceding = normalized[: match.start()]
        if re.search(r"[%!]\s*$", preceding):
            continue
        if re.search(r"[\d.)]\s*$", preceding):
            continue
        candidates.append(match.group(0))
    if not candidates:
        return None
    return max(candidates, key=len)


def _evaluate_arithmetic(expression: str) -> ArithmeticResult | None:
    """Evaluate a simple arithmetic expression, or None if it is not one."""
    cleaned = (
        str(expression or "")
        .replace("x", "*").replace("X", "*")
        .replace("×", "*").replace("÷", "/")
        # "^" is how people write exponentiation outside Python. Refusing it was
        # the safe answer and not the right one: "compute 2^31 - 1" should
        # return 2147483647, which is exactly the kind of value a person asks a
        # machine for rather than working out by hand.
        .replace("^", "**")
        .strip().rstrip("?.=").strip()
    )
    if not cleaned or not re.fullmatch(r"[0-9\s\.\+\-\*/\(\)]+", cleaned):
        return None
    if not any(op in cleaned for op in "+-*/"):
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def _eval(node: ast.AST) -> ArithmeticResult:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return left / right
            if isinstance(node.op, ast.Pow):
                # Bounded so a runaway exponent cannot become this checker's
                # own problem: 9**9**9 would hang the turn it is meant to
                # answer. The same limits the dedicated power branch uses.
                if not isinstance(left, int) or not isinstance(right, int):
                    raise ValueError("non-integer power")
                if not (0 <= right <= 64) or abs(left) > 10_000:
                    raise ValueError("power out of range")
                return left**right
        raise ValueError("unsupported expression")

    try:
        result = _eval(tree.body)
    except (ArithmeticError, ValueError, TypeError, RecursionError):
        return None
    if isinstance(result, float) and not math.isfinite(result):
        return None
    return result


# Word forms with exactly one mechanical answer. The bare-expression pattern
# covered 2 of the 8 math questions the 2026-07-25 probe actually asks; these
# two forms are the other computable ones. Everything left — rates, catch-up,
# pages-per-day — needs reasoning and is deliberately NOT claimed here.
_PERCENT_OF_RE = re.compile(
    r"what(?:'s| is)\s+([0-9]+(?:\.[0-9]+)?)\s*%\s+of\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"what(?:'s| is)\s+([0-9]+)\s+to\s+the\s+([0-9]+)(?:st|nd|rd|th)?\s+power",
    re.IGNORECASE,
)
#: Rectangle area moved to computable_math as a self-checking form, where it
#: also answers "the area of a 3 by 4 rectangle" — the phrasing this pattern
#: refused, because it required the dimensions before the word "area".


def _resolve(text: str) -> tuple[ArithmeticResult, str] | None:
    """The answer and the name of the branch that produced it.

    One traversal serves both public entry points, so the account of the
    method cannot drift from the method.
    """

    # Named functions — factorials, primality, Fibonacci, roots, remainders,
    # gcd, binomials — are asked first because they are the narrower claim:
    # "how many digits are in 100 factorial" carries no operator for the
    # expression parser below, so it used to reach the model, which has no way
    # to know that the answer is 158.
    named = computable_result(text)
    if named is not None:
        return named.value, named.source

    match = _PERCENT_OF_RE.search(text)
    if match:
        try:
            return (
                float(match.group(1)) / 100.0 * float(match.group(2)),
                f"{__name__}._PERCENT_OF_RE",
            )
        except (ArithmeticError, ValueError):
            return None

    match = _POWER_RE.search(text)
    if match:
        try:
            base, exponent = int(match.group(1)), int(match.group(2))
        except ValueError:
            return None
        # Bounded: a runaway exponent must not become the check's own problem.
        if not (0 <= exponent <= 64) or abs(base) > 10_000:
            return None
        try:
            value = base**exponent
        except ArithmeticError:
            return None
        return value, f"{__name__}._POWER_RE"

    match = _ARITHMETIC_QUESTION_RE.search(text)
    if match:
        # The captured character class stops at an unsupported operator, so
        # "what is 2 + 2 ^ 3" captures "2 + 2 " and evaluates to 4. Whatever
        # follows the capture decides whether it was the whole expression.
        _tail = text[match.end() :]
        if not re.match(r"\s*[-+*/%!0-9]", _tail):
            result = _evaluate_arithmetic(match.group(1))
            if result is not None:
                return result, f"{__name__}._evaluate_arithmetic"

    expression = _arithmetic_expression_in(text)
    if expression is None:
        return None
    evaluated = _evaluate_arithmetic(expression)
    if evaluated is None:
        return None
    return evaluated, f"{__name__}._evaluate_arithmetic"


def requested_arithmetic_result(user_message: Any) -> ArithmeticResult | None:
    """The single correct answer to a computable arithmetic question, if any."""
    resolved = _resolve(str(user_message or ""))
    return resolved[0] if resolved is not None else None


def requested_arithmetic_provenance(user_message: Any) -> str | None:
    """What actually computed the answer, named so she can say it.

    A correct number with no account of its own mechanism gets an invented
    one: asked how she reversed a string she reported "a model capability",
    about a Python slice.
    """
    resolved = _resolve(str(user_message or ""))
    if resolved is None:
        return None
    return f"computed by {resolved[1]}, run as Python, not generated"


def arithmetic_answer_matches(expected: ArithmeticResult, candidate: Any) -> bool:
    """Compare exact integers exactly and decimal results with bounded tolerance."""

    token = str(candidate or "").strip().replace(",", "")
    if not ARITHMETIC_NUMBER_RE.fullmatch(token):
        return False
    if isinstance(expected, int):
        try:
            return Decimal(token) == Decimal(expected)
        except InvalidOperation:
            return False
    try:
        value = float(token)
    except (OverflowError, ValueError):
        return False
    if not math.isfinite(value):
        return False
    return math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-6)


def capability_vocabulary() -> tuple[str, ...]:
    """The words this reader answers to, from the forms behind it.

    It evaluates plain expressions and delegates the named forms — factorial,
    gcd, roots — to computable_math, so its vocabulary is that module's plus
    the arithmetic words a person writes in a sentence.
    """
    from core.conversation.computable_math import (
        capability_vocabulary as math_vocabulary,
    )

    return math_vocabulary() + (
        "arithmetic",
        "calculate",
        "compute",
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "product",
        "percent",
        "what is 47 * 89",
    )
