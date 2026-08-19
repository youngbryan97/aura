"""Text questions with one exact answer, computed rather than spelled out.

LIVE 2026-08-19: "spell 'necessary' backwards" returned the canned refusal.
The model had produced a nine-character answer — the right length for
yrassecen — and the turn was killed downstream before it reached anyone.

Letter-level work is the classic place a language model is unreliable and a
machine is exact: reversing a word, counting the r's in "strawberry", checking
a palindrome. Asking a model to do it and then policing the result is the
wrong shape when `[::-1]` is available.

This is the text half of computable_math. Same contract: every form declares
the questions it claims WITH their answers, so the registry checks itself and
a pattern that stops matching its own example fails the suite.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "COMPUTED_TEXT_HEADER",
    "TEXT_FORMS",
    "TextForm",
    "ComputedAnswer",
    "computed_text_answer",
    "computed_text_result",
    "text_form_failures",
]

COMPUTED_TEXT_HEADER = "## THE EXACT ANSWER, COMPUTED"

#: The operand, in quotes or bare. A quoted word is unambiguous; a bare one is
#: taken only when the pattern pins its position.
def _named(name: str = "word") -> str:
    """A quoted operand, or one announced as a word/string.

    A bare noun is not an operand where the verb is ambiguous. "Reverse the
    polarity of the flow" is an idiom, and "uppercase the word aura" puts a
    filler word before the target — a loose pattern reversed "polarity" and
    capitalised "the".
    """
    return (
        rf"(?:[\"'“‘](?P<{name}>[A-Za-z][A-Za-z\-]{{0,40}})[\"'”’]"
        rf"|(?:word|string|letters?\s+of)\s+(?P<{name}_alt>[A-Za-z][A-Za-z\-]{{0,40}}))"
    )


def _bare(name: str = "word") -> str:
    """A plain operand, for patterns that pin its position themselves.

    "How many r's IN strawberry" and "is racecar A PALINDROME" leave no doubt
    about which token is the subject, so quoting it is not required of anyone.
    """
    return rf"[\"'“‘]?(?P<{name}>[A-Za-z][A-Za-z\-]{{0,40}})[\"'”’]?"


def _operand(match: "re.Match[str]") -> str:
    """Whichever operand group actually matched."""
    groups = match.groupdict()
    for key in ("word", "word_alt", "word2", "word2_alt"):
        value = groups.get(key)
        if value:
            return value
    return ""


@dataclass(frozen=True, slots=True)
class TextForm:
    """One shape of question about a string, and the answer it must give."""

    name: str
    pattern: re.Pattern[str]
    compute: Callable[[re.Match[str]], str | None]
    examples: tuple[tuple[str, str], ...] = ()
    counter_examples: tuple[str, ...] = ()

    def failures(self) -> list[str]:
        found: list[str] = []
        for question, expected in self.examples:
            got = computed_text_answer(question)
            if got is None:
                found.append(f"{self.name}: computed nothing for {question!r}")
            elif got != expected:
                found.append(f"{self.name}: {question!r} -> {got!r}, wanted {expected!r}")
        for question in self.counter_examples:
            if self.pattern.search(question) and self.compute(
                self.pattern.search(question)  # type: ignore[arg-type]
            ):
                found.append(f"{self.name}: wrongly claimed {question!r}")
        return found


def _reversed(match: re.Match[str]) -> str:
    return _operand(match)[::-1]


def _letter_count(match: re.Match[str]) -> str:
    word = _operand(match)
    return str(len([c for c in word if c.isalpha()]))


def _occurrences(match: re.Match[str]) -> str | None:
    letter = (match.group("letter") or "").lower()
    word = (_operand(match) or "").lower()
    if not letter or not word:
        return None
    return str(word.count(letter))


def _is_palindrome(match: re.Match[str]) -> str:
    cleaned = "".join(c for c in _operand(match).lower() if c.isalnum())
    return "yes" if cleaned and cleaned == cleaned[::-1] else "no"


def _upper(match: re.Match[str]) -> str:
    return _operand(match).upper()


def _sorted_letters(match: re.Match[str]) -> str:
    return "".join(sorted(_operand(match).lower()))


TEXT_FORMS: tuple[TextForm, ...] = (
    TextForm(
        "reverse",
        re.compile(
            rf"\b(?:spell|write|say|type)\b[^.?!]{{0,20}}?{_named()}[^.?!]{{0,20}}?"
            r"\b(?:backwards?|in\s+reverse|reversed)\b"
            rf"|\breverse\s+(?:the\s+)?{_named('word2')}",
            re.IGNORECASE,
        ),
        _reversed,
        examples=(
            ("spell 'necessary' backwards", "yrassecen"),
            ("reverse the word stressed", "desserts"),
        ),
        counter_examples=("reverse the polarity of the flow",),
    ),
    TextForm(
        "occurrences",
        re.compile(
            r"how\s+many\s+[\"']?(?P<letter>[A-Za-z])[\"']?(?:'s|s)?\s+"
            rf"(?:are\s+)?(?:there\s+)?in\s+{_bare()}",
            re.IGNORECASE,
        ),
        _occurrences,
        examples=(
            ("how many r's in strawberry", "3"),
            ("how many s in mississippi", "4"),
        ),
    ),
    TextForm(
        "letter_count",
        re.compile(
            rf"how\s+many\s+letters\s+(?:are\s+)?(?:there\s+)?in\s+{_bare()}",
            re.IGNORECASE,
        ),
        _letter_count,
        examples=(("how many letters in necessary", "9"),),
    ),
    TextForm(
        "palindrome",
        re.compile(rf"\bis\s+{_bare()}\s+a\s+palindrome\b", re.IGNORECASE),
        _is_palindrome,
        examples=(
            ("is racecar a palindrome?", "yes"),
            ("is necessary a palindrome?", "no"),
        ),
    ),
    TextForm(
        "uppercase",
        re.compile(
            rf"\b(?:uppercase|capitali[sz]e|all\s+caps)\b[^.?!]{{0,20}}?{_named()}",
            re.IGNORECASE,
        ),
        _upper,
        examples=(("uppercase the word aura", "AURA"),),
    ),
    TextForm(
        "sorted_letters",
        re.compile(
            rf"\b(?:sort|alphabeti[sz]e)\b[^.?!]{{0,24}}?{_named()}"
            rf"|\bletters?\s+of\s+{_bare('word2')}\s+(?:in\s+)?alphabetical\s+order",
            re.IGNORECASE,
        ),
        _sorted_letters,
        examples=(("sort the letters of aura", "aaru"),),
    ),
)


@dataclass(frozen=True)
class ComputedAnswer:
    """An exact answer together with what produced it.

    Asked to reverse a string she returned "desserts", correctly, and then
    said she had "a model capability for string manipulation" and had
    "requested the reverse operation". Nothing of the sort happened: a regex
    matched and a Python slice ran. The answer travelled and its provenance
    did not, so the only account of the method available to the reply was an
    invented one.

    Every field here is read off the code object that ran, so there is no
    sentence anywhere that can drift from what actually happened.
    """

    value: str
    form: str
    module: str
    function: str

    @property
    def source(self) -> str:
        """`module.function`, the thing a person could go and read."""
        return f"{self.module}.{self.function}"

    def provenance(self) -> str:
        """One line naming the mechanism, in words she can say."""
        return (
            f"computed by {self.source} (form {self.form!r}), "
            "run as Python, not generated"
        )


def computed_text_result(question: str) -> ComputedAnswer | None:
    """The exact answer and the code object that produced it."""
    text = str(question or "")
    if not text.strip():
        return None
    for form in TEXT_FORMS:
        match = form.pattern.search(text)
        if match is None:
            continue
        answer = form.compute(match)
        if answer:
            return ComputedAnswer(
                value=str(answer),
                form=form.name,
                module=getattr(form.compute, "__module__", __name__),
                function=getattr(form.compute, "__qualname__", form.name),
            )
    return None


def computed_text_answer(question: str) -> str | None:
    """The exact answer, or None when no form claims the question."""
    result = computed_text_result(question)
    return result.value if result is not None else None


def text_form_failures() -> list[str]:
    """Every declared example a form gets wrong."""
    return [failure for form in TEXT_FORMS for failure in form.failures()]


def capability_vocabulary() -> tuple[str, ...]:
    """The words a person uses to ask for these, taken from the forms.

    Self-knowledge reads this. Deriving it from the declared forms rather than
    writing a sentence about them means a form added later describes itself,
    and a description can never drift from what the code actually answers.
    """
    words: list[str] = []
    for form in TEXT_FORMS:
        words.append(form.name.replace("_", " "))
        words.extend(question for question, _answer in form.examples)
    return tuple(words)
