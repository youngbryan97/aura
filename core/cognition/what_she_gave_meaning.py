"""Keeping the meanings she induced across a restart.

She can induce the meaning of a kind of rule nobody wrote and have the
interpreter run it. It lived in process memory, so it died with the process,
and a node saved with that kind came back as a name with nothing behind it.

Kept here rather than in the runtime, beside the thing it keeps. A package that
holds what a rule means is where knowledge of what a rule means belongs;
core/agency keeps the properties she invented the same way, and neither has to
reach across for the other.

What is kept is the RECIPE, never a pickled object: which two places a value is
read from and what is done with the pair. It reconstructs exactly, a person can
read it, and it cannot execute anything that was not already in the space she
searches.

The words themselves are kept the same way. A meaning is written in a language,
and a meaning recalled into a language missing the word it was written in is a
name with nothing behind it — so the derived addressings, the derived
operations and the ways of building come back first, and the meanings are read
against them. A language that resets every morning has not grown.

A way of building is code, so what is kept is its name and the constructor is
looked up in the registry the code lives in. Nothing here can name a
constructor that does not already exist in the source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["forget_everything", "keep", "recall"]

logger = logging.getLogger("Aura.WhatSheGaveMeaning")

_KEPT_AT = Path.home() / ".aura" / "state" / "meanings_she_induced.json"

#: A bound on the file. What she has given meaning to is a handful of recipes.
_MOST_KEPT = 200_000


def _words_she_derived() -> dict[str, Any]:
    """The derived words and ways of building, as recipes."""
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, WHAT_OF_IT, WHERE_FROM
    from core.cognition.widening_the_language import DerivedAddressing, DerivedOperation

    addressings: dict[str, Any] = {}
    for name, word in WHERE_FROM.items():
        if isinstance(word, DerivedAddressing):
            addressings[name] = {
                "name": word.name,
                "at": {str(size): list(found) for size, found in word.at.items()},
            }
    operations: dict[str, Any] = {}
    for name, word in WHAT_OF_IT.items():
        if isinstance(word, DerivedOperation):
            try:
                does = [[one, other, got] for (one, other), got in word.does.items()]
                json.dumps(does)
            except (TypeError, ValueError):
                continue
            row: dict[str, Any] = {"name": word.name, "does": does}
            if word.rule is not None:
                from core.cognition.an_operation_that_generalises import written_down

                row["rule"] = written_down(word.rule)
            operations[name] = row
    # A way she BUILT is saved as its recipe, because a name can only ever
    # resolve against constructors the source already has, and the point of
    # building one is that the source does not have it.
    from core.cognition.a_constructor_she_built import written_down as recipe_data
    from core.cognition.one_algebra import written_down as term_data

    built: dict[str, Any] = {}
    wrote: dict[str, Any] = {}
    named: list[str] = []
    for name, build in WAYS_TO_BUILD.items():
        recipe = getattr(build, "recipe", None)
        term = getattr(build, "term", None)
        if term is not None:
            # A way she WROTE is saved as its term. There is no registry it
            # could be looked up in, because the whole point of writing one is
            # that nothing had written it down.
            wrote[name] = term_data(term)
        elif recipe is not None:
            built[name] = recipe_data(recipe)
        else:
            named.append(name)
    return {
        "addressings": addressings,
        "operations": operations,
        "ways": sorted(named),
        "built": built,
        "wrote": wrote,
    }


def keep() -> bool:
    """Write down the language she worked out, and what she said in it."""
    from core.cognition.an_invented_kind import KINDS

    kinds = {
        kind: {
            "where_from": meaning.where_from,
            "and_from": meaning.and_from,
            "what_of_it": meaning.what_of_it,
            "held_back": float(meaning.held_back),
            "from_examples": int(meaning.from_examples),
        }
        for kind, meaning in KINDS.items()
    }
    words = _words_she_derived()
    if not kinds and not any(words.values()):
        return False
    body: dict[str, Any] = {"kinds": kinds, "language": words}
    try:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        written = json.dumps(body)
        if len(written) > _MOST_KEPT:
            logger.info("what she gave meaning to is too big to keep (%d)", len(written))
            return False
        # Making the directory is a write like any other, and it was outside
        # the scope — so every keep logged a governance violation while the
        # write beside it was fine.
        with local_internal_governed_scope(
            "what_she_gave_meaning.keep", domain="state_mutation"
        ):
            get_file_write_gateway().ensure_directory(
                _KEPT_AT.parent, source="what_she_gave_meaning"
            )
            get_file_write_gateway().write_text(
                _KEPT_AT, written, source="what_she_gave_meaning"
            )
        logger.info(
            "kept %d meaning(s) and %d derived word(s) in %d way(s) of building",
            len(kinds),
            len(words["addressings"]) + len(words["operations"]),
            len(words["ways"]),
        )
        return True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "what_she_gave_meaning", exc, severity="info",
            action="keep the meanings she induced",
        )
        return False


def _put_the_language_back(language: dict[str, Any]) -> int:
    """Put back the derived words and ways of building. Returns how many."""
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, WHAT_OF_IT, WHERE_FROM
    from core.cognition.widening_the_language import (
        CONSTRUCTORS,
        DerivedAddressing,
        DerivedOperation,
    )

    back = 0
    from core.cognition.a_constructor_she_built import build as rebuild
    from core.cognition.a_constructor_she_built import read_back as read_recipe

    from core.cognition.one_algebra import as_a_maker
    from core.cognition.one_algebra import read_back as read_term

    for name, row in (language.get("wrote") or {}).items():
        term = read_term(row)
        if term is None:
            logger.info("a term she wrote does not read back: %r", name)
            continue
        WAYS_TO_BUILD[str(name)] = as_a_maker(term)
        back += 1
    for name, row in (language.get("built") or {}).items():
        recipe = read_recipe(row)
        if recipe is None:
            logger.info("a recipe she kept does not read back: %r", name)
            continue
        WAYS_TO_BUILD[str(name)] = rebuild(recipe)
        back += 1
    for name in language.get("ways") or ():
        build = CONSTRUCTORS.get(str(name))
        if build is None:
            # Named a way of building this source does not have. Silence is
            # wrong and guessing is worse, so it is recorded and skipped.
            logger.info("a way of building she kept is not in this source: %r", name)
            continue
        WAYS_TO_BUILD[str(name)] = build
        back += 1
    for name, row in (language.get("addressings") or {}).items():
        try:
            at = {
                int(size): tuple(int(where) for where in found)
                for size, found in (row.get("at") or {}).items()
            }
        except (AttributeError, TypeError, ValueError):
            continue
        if not at:
            continue
        WHERE_FROM[str(name)] = DerivedAddressing(name=str(row.get("name") or name), at=at)
        back += 1
    for name, row in (language.get("operations") or {}).items():
        try:
            does = {
                (one, other): got for one, other, got in (row.get("does") or ())
            }
        except (AttributeError, TypeError, ValueError):
            continue
        if not does and not row.get("rule"):
            continue
        from core.cognition.an_operation_that_generalises import read_back

        WHAT_OF_IT[str(name)] = DerivedOperation(
            name=str(row.get("name") or name),
            does=does,
            rule=read_back(row.get("rule")),
        )
        back += 1
    return back


def recall() -> int:
    """Put back the language she worked out, then what she said in it.

    The words first. A meaning read back before the word it is written in has
    nothing to resolve against, and would be dropped as unreadable.
    """
    try:
        held = json.loads(_KEPT_AT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if not isinstance(held, dict):
        return 0
    from core.cognition.an_invented_kind import KINDS, Induced

    if "kinds" in held or "language" in held:
        language = held.get("language") or {}
        kinds = held.get("kinds") or {}
    else:
        # Written before the language was kept alongside the meanings.
        language, kinds = {}, held
    words = _put_the_language_back(language) if isinstance(language, dict) else 0

    back = 0
    for kind, row in kinds.items():
        if not isinstance(row, dict):
            continue
        try:
            KINDS[str(kind)] = Induced(
                where_from=str(row["where_from"]),
                and_from=str(row["and_from"]),
                what_of_it=str(row["what_of_it"]),
                held_back=float(row.get("held_back") or 0.0),
                from_examples=int(row.get("from_examples") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        back += 1
    if back or words:
        logger.info(
            "she remembered %d meaning(s), in a language %d word(s) wider than "
            "the one she was given",
            back,
            words,
        )
    return back


def forget_everything() -> bool:
    """Drop the lot. What was induced on evidence can be dropped."""
    try:
        from core.runtime.file_write_gateway import get_file_write_gateway

        get_file_write_gateway().delete_file(_KEPT_AT, source="what_she_gave_meaning")
        return True
    except (OSError, RuntimeError, AttributeError):
        return False
