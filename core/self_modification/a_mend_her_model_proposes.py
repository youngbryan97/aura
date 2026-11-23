"""Her model's proposals for mending a fault of hers, asked for only once her own Python has none left (Master Todo F16).

Her repair search is engineered: the exact changes her own Python writes, semantic repair from her program's values,
and the fix templates come first (core/agency/mending_a_fault_of_hers.her_own_python_first). Bryan, 22 November 2026:
her model may propose too, "as a last resort ... so long as its checked and verified the same way". So this proposer
is the search's last, and what it proposes is a candidate like any other: it must make the fault's own test pass and
break no test that passed (core/self_modification/repair_by_selection.py), then pass her gates, the outside check and
probation before it stays. Nothing it says is taken on its word.

It asks her model already loaded, privately and in the background, through the typed edits her outside repairs already
use (core/self_modification/edits_her_model_proposes.py): no second model is loaded. It is shown the function the
fault was raised in and what the failing test printed, and each edit it returns is written back as one SEARCH/REPLACE
block the search reads, labelled as her model's so a landing says where it came from.
"""
from __future__ import annotations

import ast
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= (node.end_lineno or 0):
            first = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            if found is None or first >= found[0]:
                found = (first, node.end_lineno or node.lineno)
    return found


def as_a_block(path: str, before: str, after: str) -> str:
    """The change from ``before`` to ``after`` as one SEARCH/REPLACE block, its SEARCH text found once in ``before``."""
    return f"{path}\n<<<<<<< SEARCH\n{before}\n=======\n{after}\n>>>>>>> REPLACE"


def her_model_as_a_last_resort(path: str, line: int, *,
                               propose_edits: Callable[..., Awaitable[list[Any]]] | None = None) -> Callable[..., Awaitable[list[tuple[str, str]]]]:
    """A proposer for repair_by_selection: her model's edits to the function at ``path``:``line``, each as a block.
    ``propose_edits`` is what is asked (edits_her_model_proposes, by default), so a test can stand in for her model."""

    async def propose(defect: Any, start: Mapping[str, str], parent: Any, n: int) -> list[tuple[str, str]]:
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

        out: list[tuple[str, str]] = []
        for suspicion in (await asked(function, seen_wrong))[:max(int(n), 1)]:
            mended = applied(function, list(getattr(suspicion, "edits", None) or []))
            if mended != function:
                out.append((ORIGIN, as_a_block(path, function, mended)))
        return out

    return propose
