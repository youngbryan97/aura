"""Edits her own model proposes for what a program is seen doing wrong: candidates, never fixes.

The shapes of code that look wrong (code_that_looks_wrong.py) are a few that
are almost never meant, and a bug of another shape is not among them. When
watching a program still shows something wrong and none of those shapes'
edits settles it, she thinks about it the way a person does before trying
anything: what in this code could make it do that? Her model is asked that,
privately, with the code and what was seen. What comes back is a list of
small edits in a typed form, each a guess. Nothing it says is taken for true:
every proposal is tried on a copy and watched like any other edit, and kept
only for the difference it is seen to make (repairing_by_behaviour.py).

The same shape as consult_semantic_sources: the model is a proposer whose
text is data, never an authority.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from core.self_modification.code_that_looks_wrong import Edit, Suspicion

__all__ = ["ProposedEdit", "ProposedEdits", "edits_her_model_proposes", "edits_from_proposals"]

logger = logging.getLogger("SelfModification.EditsHerModelProposes")

#: The longest program, in lines, shown whole; past this the proposals would
#: be guesses about code the model never saw.
MOST_LINES = 600

#: How far from the line it names a proposal's text may be found.
NEAR_LINES = 3


class ProposedEdit(BaseModel):
    line: int = Field(ge=1)
    old: str = Field(default="", max_length=600)
    new: str = Field(default="", max_length=600)
    why: str = Field(default="", max_length=240)


class ProposedEdits(BaseModel):
    edits: list[ProposedEdit] = Field(default_factory=list, max_length=8)


def _numbered(source: str) -> str:
    return "\n".join(f"{n:04d}: {line}" for n, line in enumerate(source.splitlines(), start=1))


def edits_from_proposals(source: str, proposals: ProposedEdits) -> list[Suspicion]:
    """Each proposal that names text really on (or near) its line, as an edit that can be tried.

    A proposal whose ``old`` text is not there is dropped: an edit to code
    that is not in the program cannot be tried. An empty ``old`` inserts
    ``new`` at the start of the named line.
    """
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    found: list[Suspicion] = []
    seen: set[tuple[int, int, str]] = set()
    for proposal in proposals.edits:
        if proposal.line > len(lines) or proposal.old == proposal.new:
            continue
        index = proposal.line - 1
        if not proposal.old:
            edit = Edit(starts[index], starts[index], proposal.new if proposal.new.endswith("\n") else proposal.new + "\n")
        else:
            low, high = max(0, index - NEAR_LINES), min(len(lines), index + NEAR_LINES + 1)
            window = source[starts[low] : starts[high]]
            at = window.find(proposal.old)
            if at < 0:
                continue
            start = starts[low] + at
            edit = Edit(start, start + len(proposal.old), proposal.new)
        key = (edit.start, edit.end, edit.text)
        if key in seen:
            continue
        seen.add(key)
        found.append(Suspicion(
            pattern="proposed by her model", line=proposal.line,
            why=" ".join(proposal.why.split())[:200] or "her model suggested it", edits=[edit],
            function=_enclosing_function(source, edit.start),
        ))
    return found


def _enclosing_function(source: str, offset: int) -> str:
    """The name of the last function opened before ``offset``, by its declaration's shape."""
    names = re.findall(r"(?:function\s+([A-Za-z_$][\w$]*)|def\s+([A-Za-z_]\w*))", source[:offset])
    return next((a or b for a, b in reversed(names)), "")


async def edits_her_model_proposes(
    source: str, seen_wrong: dict[str, str], *, advisor: Any = None, deadline_s: float = 120.0
) -> list[Suspicion]:
    """Ask her model, privately, which small edits could make what was seen wrong come right."""
    if not seen_wrong or source.count("\n") > MOST_LINES:
        return []
    if advisor is None:
        try:
            from core.brain.llm.structured_llm import StructuredLLM

            advisor = StructuredLLM(ProposedEdits, max_retries=1)
        except Exception as why:  # noqa: BLE001 - no model here (offline, or not yet registered) means no proposals
            logger.info("no model to ask about the code: %s", why)
            return []
    payload = {"program": _numbered(source), "seen_wrong": seen_wrong}
    try:
        batch = await advisor.generate(
            "Propose small edits to this program that could make the behaviour seen wrong come right. "
            "Each edit replaces exact text from one numbered line (old) with new text; an empty old "
            "inserts new before that line. Return the typed schema; do not claim any edit works. Data: "
            + json.dumps(payload),
            is_background=True, deadline_s=deadline_s,
        )
    except (RuntimeError, TimeoutError, ValueError, TypeError, OSError) as why:
        logger.info("her model could not be asked about the code: %s", why)
        return []
    if batch is None:
        return []
    if not isinstance(batch, ProposedEdits):
        batch = ProposedEdits.model_validate(batch)
    proposals = edits_from_proposals(source, batch)
    logger.info("her model proposed %d edit(s), %d of them to code that is there", len(batch.edits), len(proposals))
    return proposals
