"""core/interiority/vocabulary.py — the action classes the faculties speak in.

A faculty emits a somatic marker or a constraint against a named action class:
``conceal``, ``state_the_boundary_and_the_cost``, ``refuse_with_attention``.
Those names are not tasks. They are stances — ways of meeting a situation that
she has by construction, the way a person has "say nothing" available without
needing an instrument for it.

The distinction matters at exactly one place, and it was wrong there. The
capability model counts acts that could bring an option about, and an option
with no such act is judged impossible. Run that check against a stance and it
returns zero, because no skill in the catalogue is called "state the boundary
and the cost" — so the deliberation concluded she was incapable of a thing she
does in most conversations.

This module supplies the set to exclude, read out of the faculty sources
rather than written down. A list written down would drift the first time a
faculty was added, and drift here means a stance silently becoming impossible.
Extraction is by AST and never executes a faculty.
"""

from __future__ import annotations

import ast
import functools
import logging
import pathlib

logger = logging.getLogger("Aura.Interiority.Vocabulary")

#: The effect types whose first field names an action class.
_ACTION_CLASS_EFFECTS = {"SomaticMarker": "option", "ActionConstraint": "action_class"}

_FACULTY_DIR = pathlib.Path(__file__).resolve().parent / "faculties"


def _literal_action_classes(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        field = _ACTION_CLASS_EFFECTS.get(str(name))
        if field is None:
            continue
        # Positional first argument, or the keyword by name.
        candidates: list[ast.expr] = list(node.args[:1])
        candidates += [kw.value for kw in node.keywords if kw.arg == field]
        for candidate in candidates:
            if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
                found.add(candidate.value)
            elif isinstance(candidate, ast.JoinedStr):
                # An f-string like f"pursue:{subject}" names a class with a
                # subject attached. The class is the part before the colon.
                head = candidate.values[0] if candidate.values else None
                if isinstance(head, ast.Constant) and isinstance(head.value, str):
                    stem = head.value.split(":")[0].strip()
                    if stem:
                        found.add(stem)
    return found


@functools.lru_cache(maxsize=1)
def action_classes() -> frozenset[str]:
    """Every action class any faculty can name.

    Cached: this reads forty-three files and the answer cannot change while
    the process is running.
    """
    found: set[str] = set()
    if not _FACULTY_DIR.is_dir():
        return frozenset()
    for path in sorted(_FACULTY_DIR.glob("*.py")):
        try:
            found |= _literal_action_classes(ast.parse(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            logger.debug("Could not read action classes from %s: %s", path.name, exc)
    return frozenset(found)


def is_stance(option: str) -> bool:
    """Whether this option is a way of meeting a situation rather than a task.

    A stance needs no instrument, so asking the capability catalogue whether
    she can do it is asking the wrong question and getting no for an answer.
    """
    return str(option).split(":")[0].strip() in action_classes()


__all__ = ["action_classes", "is_stance"]
