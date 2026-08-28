"""Failures are data. What they have in common is a concept.

Every recorded failure in this repository was repaired by hand, one at a time,
by somebody reading it and writing the distinction into a pattern. Three of them
on one day were the same mistake:

    "is three examples enough to pin it DOWN?"      answered with a health report
    "tell me what's ACTUALLY happening"             read as an instruction for now
    "STEP THROUGH /tmp/proj and tell me where"      read as a game to be played

Each was fixed separately, with a separate frame, in a separate module. Nobody
formed the thing they share, because nothing looks at failures together. That
is the gap: a system that is taught an abstraction after each failure has not
discovered one.

What they share is computable, and it is not lexical. In each case a pattern
decided from a span in isolation, while the meaning of that span was fixed by
what surrounded it. A pattern with that property will keep being wrong in that
way, on wordings nobody has seen yet, and the repair is the same repair every
time: the span is not enough, ask for the role.

So the signature of a failure is a property of the DECISION rather than of the
words: did the pattern that fired need the sentence, or only the fragment? When
enough failures share a signature, the constraint that answers all of them is
formed once and named, and it applies to every pattern with that property —
including the ones that have not failed yet.

This reads the record the repository already keeps. It invents no vocabulary of
its own and consults no language model: a failure is a sentence somebody wrote
down next to the code that got it wrong, and the signature is measured by
running the code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FormedConstraint",
    "RecordedFailure",
    "asks_for_a_role",
    "cluster_by_signature",
    "form_constraints",
    "harvest_recorded_failures",
    "signature_of",
    "span_local",
]

#: A dated note beside the code that got something wrong, with the wording that
#: caused it. The repository has kept these for months; this is the reader.
_LIVE_NOTE = re.compile(
    r"LIVE[,:]?\s+(?P<date>\d{4}-\d{2}-\d{2})\s*:?\s*(?P<body>.{0,600})",
    re.DOTALL,
)

#: The wording itself, as it was quoted. Both quote styles appear.
_QUOTED = re.compile(r"[\"“]([^\"”]{8,400})[\"”]")


@dataclass(frozen=True)
class RecordedFailure:
    """One wording that was handled wrongly, and where that was written down."""

    text: str
    module: Path
    date: str = ""

    def __str__(self) -> str:
        return f"{self.module.name}: {self.text[:60]!r}"


#: Marks in a pattern's source that make it ask for more than the token: a
#: lookaround, an adjacency requirement, a sequence of more than one thing. A
#: pattern with none of these decides from the token alone, whatever else is in
#: it. Read from the source rather than declared, so no pattern can claim to
#: honour the constraint without doing so.
_ASKS_FOR_MORE = ("(?=", "(?!", "(?<=", "(?<!", r"\s+", r"\s*", "]{", "){")


def asks_for_a_role(pattern: re.Pattern[str], token: str) -> bool:
    """Whether this pattern wants anything besides the token itself.

    The formed constraint says a token is not a decision. A pattern honours it
    by asking for the token IN a role — something before it, something after
    it, a longer unit it belongs to. One that names the token in a bare
    alternation and nothing else does not.
    """

    source = str(getattr(pattern, "pattern", "") or "")
    word = str(token or "").strip().lower()
    if not word or not _names_the_token(source, word):
        return True
    return any(mark in source for mark in _ASKS_FOR_MORE)


def _names_the_token(source: str, word: str) -> bool:
    """Whether the pattern names this token, as a token.

    As a word, not as letters that happen to be there: "i" sits inside "in",
    "is" and "with", so a substring test made almost every pattern look as if
    it decided from almost every token. That is the same mistake this constraint
    is about, made while checking for it.
    """

    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", source.lower()) is not None


@dataclass(frozen=True)
class FormedConstraint:
    """A concept formed from failures that turned out to be the same mistake.

    ``applies_to`` is what makes it more than a description: the patterns in
    the codebase that have the property, whether or not they have failed yet.
    """

    name: str
    signature: str
    statement: str
    formed_from: tuple[RecordedFailure, ...] = ()
    applies_to: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.name}: {self.statement}"

    def violations(self) -> list[str]:
        """Decisions this concept covers that do not honour it.

        Measured at the decision, not at the pattern. A pattern that names an
        under-determined token is not yet a fault: _FOREGROUND_ACTION_VERB_RE
        fires on "copy their approach" and on "that move was a good one", and
        its caller conjoins it with a surface word, so the decision asks for
        the role even though the pattern does not. Counting patterns put that
        site in the list and inflated the number from 63 to 246.

        A fault is a pattern that names such a token AND is consulted somewhere
        as the whole of a decision, with nothing beside it.
        """

        # Only the tokens that MADE the cluster. A token that decided one
        # failure is an incident; the finding is the ones that decided several,
        # and the constraint is about those.
        seen: dict[str, set[tuple[Path, str]]] = {}
        for failure in self.formed_from:
            for token in _single_token_matches(failure):
                seen.setdefault(token, set()).add((failure.module, failure.text))
        tokens = sorted(token for token, where in seen.items() if len(where) >= 2)
        offending: set[str] = set()
        for failure in self.formed_from:
            alone = _consulted_alone(failure.module)
            if not alone:
                continue
            for name, pattern in _patterns_in(failure.module):
                if name not in alone:
                    continue
                for token in tokens:
                    if not _names_the_token(
                        str(getattr(pattern, "pattern", "")), token
                    ):
                        continue
                    if not asks_for_a_role(pattern, token):
                        offending.add(
                            f"{failure.module.as_posix()}:{name} decides from {token!r}"
                        )
        return sorted(offending)


def _consulted_alone(module: Path) -> set[str]:
    """Patterns this module tests as the whole of a decision, with nothing beside.

    A pattern conjoined with something else has already asked for more than the
    token, whatever the pattern itself says. Read from the source, because a
    call site cannot be asked at runtime whether it was the only condition.
    """

    import ast

    try:
        tree = ast.parse(module.read_text())
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return set()

    def named(node: ast.AST) -> str:
        # PATTERN.search(x) / PATTERN.match(x) / PATTERN.fullmatch(x)
        if not isinstance(node, ast.Call):
            return ""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return ""
        if func.attr not in {"search", "match", "fullmatch"}:
            return ""
        holder = func.value
        return holder.id if isinstance(holder, ast.Name) else ""

    def unwrap(node: ast.AST) -> ast.AST:
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            node = node.operand
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "bool" and node.args:
                return unwrap(node.args[0])
        return node

    alone: set[str] = set()
    for node in ast.walk(tree):
        tests: list[ast.AST] = []
        if isinstance(node, (ast.If, ast.While)):
            tests.append(node.test)
        elif isinstance(node, ast.Return) and node.value is not None:
            tests.append(node.value)
        for test in tests:
            settled = unwrap(test)
            if isinstance(settled, ast.BoolOp):
                # Conjoined or alternated with something: more was asked for.
                continue
            found = named(settled)
            if found:
                alone.add(found)
    return alone


def _clean(note: str) -> str:
    """A quoted wording with the comment furniture taken off."""

    body = note.replace("\n#:", " ").replace("\n#", " ").replace("\n", " ")
    return " ".join(body.split())


def harvest_recorded_failures(
    roots: Sequence[Path | str] = ("core", "interface"),
) -> list[RecordedFailure]:
    """Every dated failure note in the tree, with the wording it recorded.

    A note without a quoted wording is skipped rather than guessed at: the
    signature is measured by running the code on the words, so a note with no
    words in it has nothing to measure.
    """

    found: list[RecordedFailure] = []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                body = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            if "LIVE" not in body:
                continue
            for note in _LIVE_NOTE.finditer(body):
                text = _clean(note.group("body"))
                quoted = _QUOTED.search(text)
                if quoted is None:
                    continue
                wording = quoted.group(1).strip()
                if len(wording.split()) < 3:
                    continue
                found.append(
                    RecordedFailure(
                        text=wording, module=path, date=note.group("date")
                    )
                )
    return found


def span_local(pattern: re.Pattern[str], text: str) -> bool | None:
    """Whether this pattern's decision needed the sentence or only the fragment.

    Fires on the sentence, then on nothing but the fragment it matched. A
    pattern that fires on the fragment alone decided from the fragment alone,
    however much sentence was in front of it. None when the pattern does not
    fire at all, which is not a property of the pattern but of this sentence.
    """

    try:
        hit = pattern.search(text)
    except (re.error, TypeError):
        return None
    if hit is None:
        return None
    fragment = hit.group(0)
    if not fragment.strip() or fragment.strip() == text.strip():
        return None
    try:
        return pattern.search(fragment) is not None
    except (re.error, TypeError):
        return None


def _patterns_in(module: Path) -> list[tuple[str, re.Pattern[str]]]:
    """The compiled patterns a module exposes, by the names it gave them."""

    import importlib

    parts = module.with_suffix("").parts
    try:
        loaded = importlib.import_module(".".join(parts))
    except Exception:  # noqa: BLE001 - a module that will not import has no patterns
        return []
    found: list[tuple[str, re.Pattern[str]]] = []
    for name in dir(loaded):
        value = getattr(loaded, name, None)
        if isinstance(value, re.Pattern):
            found.append((name, value))
    return found


def _single_token_matches(failure: RecordedFailure) -> set[str]:
    """The one-word spans a pattern in this module decided this wording from.

    One word, because that is where the property bites: a pattern matching a
    phrase has already taken some of the surroundings into account, and a
    pattern matching a single token has taken none.
    """

    tokens: set[str] = set()
    for _name, pattern in _patterns_in(failure.module):
        if not span_local(pattern, failure.text):
            continue
        try:
            hit = pattern.search(failure.text)
        except (re.error, TypeError):
            continue
        if hit is None:
            continue
        span = hit.group(0).strip().lower()
        if span and len(span.split()) == 1 and span.isalpha():
            tokens.add(span)
    return tokens


def cluster_by_signature(
    failures: Iterable[RecordedFailure],
) -> dict[str, list[RecordedFailure]]:
    """Failures grouped by the kind of mistake, ignoring what they were about.

    Span-locality on its own is nearly every pattern — a regex almost always
    fires on the text it just matched — so on its own it groups half the record
    and says nothing. Measured over the whole corpus, 186 of 323 failures had
    it, which is a property of regular expressions rather than a finding about
    these failures.

    What is a finding is the same single token deciding more than one failure.
    That is the token being asked to carry a judgement it cannot carry, shown
    by it being wrong about more than one thing, and it is exactly the case
    where the surroundings are what settle the meaning.
    """

    observed = list(failures)
    by_token: dict[str, list[RecordedFailure]] = {}
    for failure in observed:
        for token in _single_token_matches(failure):
            by_token.setdefault(token, []).append(failure)
    grouped: dict[str, list[RecordedFailure]] = {}
    for token, group in by_token.items():
        if len(group) < 2:
            continue
        distinct = {(item.module, item.text) for item in group}
        if len(distinct) < 2:
            continue
        grouped.setdefault("decided from one token, more than once", []).extend(
            item for item in group if item not in grouped.get(
                "decided from one token, more than once", []
            )
        )
    return grouped


def signature_of(failure: RecordedFailure) -> str:
    """The kind of mistake this one was, or "" when it is not a kind yet.

    A single failure has no kind: what makes a mistake a kind is that it
    happened again. Kept for callers that hold one failure, and answered from
    the corpus rather than from the failure alone.
    """

    for mark, group in cluster_by_signature(harvest_recorded_failures()).items():
        if failure in group:
            return mark
    return ""


#: What each signature, once seen often enough, means and asks for. The
#: statement is the concept; the name is what it is called afterwards.
_CONCEPTS: dict[str, tuple[str, str]] = {
    "decided from one token, more than once": (
        "a token is not a decision",
        "One word decided more than one of these wrongly. A word that has been "
        "wrong about several different things is not carrying the judgement "
        "put on it: what settles its meaning is the role it plays in the "
        "phrase around it, so the pattern has to ask for that role.",
    ),
}

#: Below this a cluster is a coincidence. Set from what it takes to see a
#: repetition at all: two occurrences are a pair, three are a pattern.
_ENOUGH_TO_BE_A_KIND = 3


def form_constraints(
    failures: Iterable[RecordedFailure],
    *,
    modules: Sequence[Path | str] = ("core", "interface"),
) -> list[FormedConstraint]:
    """Concepts formed from failures that turned out to be the same mistake.

    ``applies_to`` is the point: once formed, the concept names every pattern
    with the property, not only the ones that have already been wrong.
    """

    formed: list[FormedConstraint] = []
    for mark, group in sorted(cluster_by_signature(failures).items()):
        if len(group) < _ENOUGH_TO_BE_A_KIND or mark not in _CONCEPTS:
            continue
        name, statement = _CONCEPTS[mark]
        formed.append(
            FormedConstraint(
                name=name,
                signature=mark,
                statement=statement,
                formed_from=tuple(group),
                applies_to=tuple(_patterns_with(mark, group, modules)),
            )
        )
    return formed


def _patterns_with(
    mark: str,
    group: Sequence[RecordedFailure],
    modules: Sequence[Path | str],
) -> list[str]:
    """Every pattern the formed concept covers, failed or not."""

    if mark != "decided from one token, more than once":
        return []
    covered: set[str] = set()
    for failure in group:
        for name, pattern in _patterns_in(failure.module):
            if span_local(pattern, failure.text):
                covered.add(f"{failure.module.as_posix()}:{name}")
    return sorted(covered)
