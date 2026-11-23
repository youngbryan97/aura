"""Her model's proposals for mending a fault of hers, asked for only once her own Python has none left (Master Todo F16).

Her repair search is engineered: the exact changes her own Python writes, semantic repair from her program's values,
and the fix templates come first (core/agency/mending_a_fault_of_hers.her_own_python_first). Bryan, 22 November 2026:
her model may propose too, "as a last resort ... so long as its checked and verified the same way". So this proposer
is the search's last, and what it proposes is a candidate like any other: it must make the fault's own test pass and
break no test that passed (core/self_modification/repair_by_selection.py), then pass her gates, the outside check and
probation before it stays. Nothing it says is taken on its word.

It asks her model already loaded, privately and in the background, through the typed edits her outside repairs already
use (core/self_modification/edits_her_model_proposes.py): no second model is loaded. It is shown the function the
fault was raised in and what the failing test printed, and each edit it returns is handed to the search as the edit
itself, labelled as her model's so a landing says where it came from.
"""
from __future__ import annotations

import ast
import textwrap
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

#: How a mend from this proposer is labelled in the search's lineage and in the commit that offers it.
ORIGIN = "her model, last of her proposers"
#: The longest tail of what the failing test printed that her model is shown, in characters.
_FAILURE_SHOWN = 4000


def the_function_around(source: str, line: int) -> tuple[int, int] | None:
    """The lines (first, last) of the innermost function holding ``line``, decorators included; None outside any."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found = None
    for node in ast.walk(tree):
        first = min([getattr(node, "lineno", 0), *(item.lineno for item in getattr(node, "decorator_list", ()))])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and first <= line <= (node.end_lineno or 0):
            if found is None or first >= found[0]:
                found = (first, node.end_lineno or node.lineno)
    return found


def one_function_still(before: str, after: str) -> bool:
    """Whether ``after`` is still the one function ``before`` is: one definition of the same kind and name, its
    decorators kept, and nothing else beside it. An edit whose new text also defines ``is_admin`` below the function it
    mends changes more than it says (round nine of the red team)."""
    try:
        was, now = (ast.parse(textwrap.dedent(text)).body for text in (before, after))
    except SyntaxError:
        return False
    return (len(was) == 1 == len(now) and type(was[0]) is type(now[0])
            and isinstance(now[0], (ast.FunctionDef, ast.AsyncFunctionDef)) and now[0].name == was[0].name
            and [ast.dump(item) for item in now[0].decorator_list] == [ast.dump(item) for item in was[0].decorator_list]
            and before[:len(before) - len(before.lstrip())] == after[:len(after) - len(after.lstrip())])


def her_model_as_a_last_resort(path: str, line: int, *,
                               propose_edits: Callable[..., Awaitable[list[Any]]] | None = None) -> Callable[..., Awaitable[list[tuple[str, Any]]]]:
    """A proposer for repair_by_selection: her model's edits to the function at ``path``:``line``, each handed over as
    the edit itself, never as text the search reads back, and only where what it makes is still that one function.
    ``propose_edits`` is what is asked (edits_her_model_proposes, by default), so a test can stand in for her model."""

    async def propose(defect: Any, start: Mapping[str, str], parent: Any, n: int) -> list[tuple[str, Any]]:
        source = start.get(path, "")
        span = the_function_around(source, line)
        if not source or span is None:
            return []
        lines = source.splitlines(keepends=True)
        function = "".join(lines[span[0] - 1:span[1]]).rstrip("\n")
        if source.count(function) != 1:
            return []
        failure = parent.fitness.failures if parent is not None and getattr(parent, "fitness", None) else defect.seen
        seen_wrong = {"the failing test": str(failure)[-_FAILURE_SHOWN:], "the tests that must pass": ", ".join(defect.tests)}
        if propose_edits is None:
            from core.self_modification.edits_her_model_proposes import edits_her_model_proposes as asked
        else:
            asked = propose_edits
        from core.self_modification.code_that_looks_wrong import applied
        from core.self_modification.repair_by_selection import Edit

        out: list[tuple[str, Any]] = []
        for suspicion in (await asked(function, seen_wrong))[:max(int(n), 1)]:
            mended = applied(function, list(getattr(suspicion, "edits", None) or []))
            if mended != function and one_function_still(function, mended):
                out.append((ORIGIN, (Edit(path, function, mended),)))
        return out

    return propose
