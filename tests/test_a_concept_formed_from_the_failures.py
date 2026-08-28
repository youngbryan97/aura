"""Failures that were fixed one at a time turn out to be one mistake.

Every recorded failure here was repaired by hand, by somebody reading it and
writing the distinction into a pattern. Three on one day were the same thing —
"pin it DOWN" read as a fault report, "what's ACTUALLY happening" read as an
instruction, "STEP THROUGH /tmp" read as a game — and each was fixed separately
because nothing looks at failures together.

This checks that the record is enough to form the concept they share, without
being told what it is, and that the concept then names patterns that have not
failed yet. That last part is what makes it a concept rather than a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.language.formed_constraints import (
    FormedConstraint,
    asks_for_a_role,
    cluster_by_signature,
    form_constraints,
    harvest_recorded_failures,
    span_local,
)

_BASELINE = Path("config/formed_constraint_baseline.json")


@pytest.fixture(scope="module")
def failures():
    return harvest_recorded_failures()


@pytest.fixture(scope="module")
def formed(failures) -> list[FormedConstraint]:
    return form_constraints(failures)


def test_the_record_the_repository_already_keeps_is_readable(failures) -> None:
    """Months of dated failures, with the wording that caused each."""

    assert len(failures) > 200
    assert all(item.text for item in failures)
    assert all(item.date for item in failures)
    # Spread over time rather than one afternoon's worth.
    assert len({item.date for item in failures}) > 10


def test_span_locality_alone_says_nothing(failures) -> None:
    """The first signature grouped half the record, which is not a finding.

    A regular expression almost always matches the text it just matched, so
    "decided from a fragment" is a property of regular expressions. Kept as a
    check because the mechanism is only worth anything if its signature
    discriminates, and this is the measurement that showed the first one did
    not.
    """

    import re

    always = re.compile(r"\bdown\b")
    assert span_local(always, "is three examples enough to pin it down") is True
    assert span_local(always, "nothing here matches") is None


def test_one_concept_is_formed_and_it_is_about_tokens(formed) -> None:
    assert len(formed) == 1
    concept = formed[0]
    assert concept.name == "a token is not a decision"
    assert concept.signature == "decided from one token, more than once"
    assert len(concept.formed_from) >= 20


def test_the_concept_recovers_the_case_that_motivated_it(failures) -> None:
    """"down" is in it, and nobody put it there.

    It was hand-fixed on 2026-08-27, in one module, as one frame. The
    mechanism finds it again from the record alone.
    """

    from core.language.formed_constraints import _single_token_matches

    tokens: dict[str, set[tuple[Path, str]]] = {}
    for failure in failures:
        for token in _single_token_matches(failure):
            tokens.setdefault(token, set()).add((failure.module, failure.text))
    repeated = {token for token, where in tokens.items() if len(where) >= 2}
    assert "down" in repeated
    # And the words the outside review named as the problem, found the same way.
    assert {"copy", "move", "open", "read", "file", "screen"} <= repeated


def test_the_concept_names_patterns_that_have_not_failed(formed) -> None:
    """A summary describes what happened; a concept covers what has not."""

    concept = formed[0]
    assert len(concept.applies_to) > 50
    covered_modules = {name.split(":")[0] for name in concept.applies_to}
    failed_modules = {item.module.as_posix() for item in concept.formed_from}
    assert covered_modules <= failed_modules
    assert len(concept.applies_to) > len(failed_modules)


def test_asking_for_a_role_is_read_from_the_pattern_not_declared() -> None:
    import re

    bare = re.compile(r"\b(?:copy|move|open)\b")
    in_a_role = re.compile(r"\b(?:copy|move)\s+(?:the\s+)?\w+\s+to\b")
    assert not asks_for_a_role(bare, "copy")
    assert asks_for_a_role(in_a_role, "copy")
    # A pattern that does not name the token is not the constraint's business.
    assert asks_for_a_role(bare, "diagnose")


def test_the_token_check_is_a_word_check() -> None:
    """The same substring mistake the constraint is about, made while checking.

    "i" sits inside "in", "is" and "with". A substring test made almost every
    pattern look as if it decided from almost every token, and the violation
    count came out three times too high.
    """

    import re

    unrelated = re.compile(r"\b(?:within|listing)\b")
    assert asks_for_a_role(unrelated, "i")
    assert asks_for_a_role(unrelated, "is")


def test_the_violations_only_go_down(formed) -> None:
    """A ratchet, like every other count in this repository."""

    concept = formed[0]
    found = len(concept.violations())
    assert found > 0, "a constraint nothing violates is not being enforced"
    try:
        recorded = int(json.loads(_BASELINE.read_text())["patterns_not_honouring"])
    except (OSError, KeyError, TypeError, ValueError):
        pytest.fail(f"{_BASELINE} must record the baseline this ratchet holds")
    assert found <= recorded, (
        f"{found} patterns decide from a token that has been wrong more than "
        f"once, up from {recorded}. The constraint is formed; honour it in the "
        "new pattern rather than raising the baseline."
    )


def test_the_named_example_from_the_outside_review_is_caught(formed) -> None:
    """"Copy only means a filesystem operation under certain conditions.\""""

    violations = formed[0].violations()
    assert any("decides from 'copy'" in line for line in violations)
