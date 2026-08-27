"""Appending a path must not change what a prose reader thinks was said.

Twice, in two different functions, the word "aura" inside a filesystem path
made a question about somebody's project read as a question about her
capabilities. The first was in chat_preflight; the second, found the same way a
week later, in skill_task_bridge. Both were fixed by masking opaque spans —
machinery that already existed for exactly this and cites the same directory in
its own docstring.

Fixing the second copy by hand does not find the third. This sweeps every prose
predicate in the intent surfaces and asserts the property directly: a path is an
address, and adding one moves no judgement about what the sentence means.

Readers that are SUPPOSED to notice a path are listed with their reason. That
list is the interesting half — anything not on it must be address-blind.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

#: A real path, containing a word that means something else in prose.
_LOADED_PATH = (
    "/private/tmp/claude-501/-Users-bryan--aura-live-source/"
    "7a6cdc9e-da7f-47f7-8c38-8cfadf95a75e/scratchpad/invoice-tools"
)

#: Sentences with no path in them, covering the shapes that have gone wrong.
_SENTENCES = (
    "what is the capital of Peru",
    "why is the second invoice picking up the first one's lines",
    "what's the actual cause, and what do I change?",
    "what tools do you have",
    "write me a one-pager about the migration",
    "how are you doing?",
)

#: Modules whose prose predicates this sweeps.
_SURFACES = (
    "core.runtime.skill_task_bridge",
    "core.runtime.desktop_objective_intent",
    "core.phases.response_contract",
    "core.introspection.self_evidence",
    "core.intent.artifact_request",
    "core.conversation.asks_about_the_world",
)

#: Readers that are about paths, so a path is exactly what they should notice.
_ABOUT_PATHS = frozenset(
    {
        "looks_like_filesystem_observation",
        "looks_like_desktop_objective",
        "asks_for_own_source",
        "requested_file_read",
        "first_named_url",
        "without_opaque_spans",
        "named_paths",
        "first_existing_path",
        "asks_to_build_software",
        "looks_like_execution_report",
        "derive_capability_set",
        "requested_effect_ceiling",
    }
)


def _prose_predicates(module_name: str):
    """Every one-argument boolean reader this module exposes."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:  # pragma: no cover - a surface that moved
        pytest.skip(f"{module_name} is not importable")
    found = []
    for name, value in vars(module).items():
        if name.startswith("_") or not inspect.isfunction(value):
            continue
        if value.__module__ != module_name:
            continue
        if name in _ABOUT_PATHS:
            continue
        if not (name.startswith(("looks_like_", "asks_", "is_", "wants_", "names_"))):
            continue
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            continue
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(required) == 1:
            found.append((name, value))
    return found


@pytest.mark.parametrize("module_name", _SURFACES)
def test_no_prose_reader_changes_its_mind_because_of_a_path(module_name: str) -> None:
    readers = _prose_predicates(module_name)
    assert readers, f"{module_name} exposed no prose readers to check"
    moved: list[str] = []
    for name, reader in readers:
        for sentence in _SENTENCES:
            try:
                plain = bool(reader(sentence))
                with_path = bool(reader(f"{sentence} at {_LOADED_PATH}"))
            except (RuntimeError, TypeError, ValueError, AttributeError, OSError):
                continue
            if plain != with_path:
                moved.append(
                    f"{module_name}.{name}: {sentence!r} was {plain}, "
                    f"and {with_path} once a path was appended"
                )
    assert not moved, "an address changed what a sentence means:\n" + "\n".join(moved)


def test_the_two_readers_that_actually_broke_agree_now() -> None:
    """Both copies of "is this an inventory question", on the live sentence."""
    from core.runtime.skill_task_bridge import (
        looks_like_capability_inventory_dialogue_request,
    )
    from interface.routes.chat_preflight import (
        _is_explicit_capability_inventory_request,
    )

    asked = (
        f"ok this is driving me nuts. {_LOADED_PATH} — clean run, nothing raises, "
        "but invoice two comes out holding invoice one's lines. what's the actual "
        "cause, and what do I change?"
    )
    assert looks_like_capability_inventory_dialogue_request(asked) is False
    assert _is_explicit_capability_inventory_request(asked) is False
    # And a real inventory question still reaches both.
    assert looks_like_capability_inventory_dialogue_request("what tools do you have")
    assert _is_explicit_capability_inventory_request("what tools do you have")
