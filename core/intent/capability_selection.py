"""Which capabilities a turn needs. One answer, for every caller that asks.

LIVE DEFECT, 2026-08-19. Two mechanisms decided this and disagreed. The
capability router matched a request to read a repository against skill
declarations and nominated ``uplink_local`` — whose description mentions a
state-repository — while omitting ``file_operation`` entirely. The tool
handoff, computing the same thing a different way, had the right five. So the
router sent the turn to a skill that could not do the job while the loop knew
which one could.

The selection has three parts, and each exists because the one before it is
not enough on its own:

* **Ranked declaration matches** — what the skills say they do, scored against
  the request. Precise when the request happens to use a word a skill
  declared.
* **Foundational capabilities** — reading, computing and looking things up,
  offered to any request-shaped turn. Nouns are an open class: "read
  README.md" names nothing any skill declares, so ranking alone hands a real
  task nothing at all.
* **Admissibility** — a skill is only offered if some action of it is within
  what the turn may do. Scope is a property of the CALL, so a skill that can
  delete is still offered for its reader.

Takes the skills mapping rather than resolving a service, so the router can
ask during its own construction and the contract can ask from anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["DEFAULT_CAPABILITY_SET", "select_capabilities"]

#: How many capabilities one turn may be handed at once. Every capacity
#: offered is also context the model must hold, so this is a working set
#: rather than the catalogue.
DEFAULT_CAPABILITY_SET = 5


def select_capabilities(
    objective: str,
    skills: Mapping[str, Any],
    *,
    ceiling: str,
    admissible_scopes: frozenset[str] | set[str],
    limit: int = DEFAULT_CAPABILITY_SET,
) -> list[str]:
    """The capabilities this request needs, most specific first.

    ``ceiling`` is the most dangerous effect scope this turn may reach, and
    ``admissible_scopes`` are the scopes a skill may carry outright. A skill
    outside both is still offered when one of its ACTIONS is within the
    ceiling, because the dispatch refuses the actions that are not.
    """
    text = str(objective or "").strip()
    if not text or not skills:
        return []

    # The deliverable owner is shared with turn analysis. A request to explain,
    # compare, describe, or otherwise answer in this reply is not an external
    # effect merely because its grammar is imperative. Without this boundary,
    # "show the distance updates" in a Dijkstra explanation nominated five
    # unrelated tools after the resident model had already produced an answer.
    from core.runtime.skill_task_bridge import looks_like_inline_answer_request

    if looks_like_inline_answer_request(text):
        return []

    from core.intent.declared_capability import (
        computation_capabilities,
        declared_vocabulary,
        distinctive_objects,
        foundational_capabilities,
        looks_like_a_request,
        rank_declaration_matches,
        requested_foundational_domains,
        settles_by_computation,
    )
    from core.skills.action_scope import resolve_skill_target, skill_has_action_within

    catalogue = {
        name: declared_vocabulary(name, str(getattr(meta, "description", "") or ""))
        for name, meta in skills.items()
        if getattr(meta, "enabled", True)
    }
    if not catalogue:
        return []

    ranked = rank_declaration_matches(text, catalogue, distinctive_objects(catalogue))
    # A semantic neighbour is not automatically a plausible tool. "Run
    # Python" gave code_repl 1.75 but also admitted a program-DNA equivalence
    # battery at 0.75 because program and Python share a domain class. Keep the
    # strongest evidence class; materially different domains are added below
    # from the request's objects, so compound tasks retain their working set.
    strongest = ranked[0][1] if ranked else 0.0
    ordered = [name for name, score in ranked if score == strongest]
    if looks_like_a_request(text):
        domains = requested_foundational_domains(text)
        for name in foundational_capabilities(catalogue, domains):
            if name not in ordered:
                ordered.append(name)
    elif settles_by_computation(text):
        # A problem to work out asks for no capability by name, so the mood
        # gate above leaves it with nothing — and a finite constraint problem
        # is exactly the case where enumeration is not a heuristic but the
        # definition of the answer. Only the computing primitives: searching
        # the web for a seating puzzle is noise, and every capability offered
        # is context the model has to hold.
        for name in computation_capabilities(catalogue):
            if name not in ordered:
                ordered.append(name)

    chosen: list[str] = []
    for name in ordered:
        meta = skills.get(name)
        scope = str(getattr(meta, "effect_scope", "") or "").strip().lower()
        if scope in admissible_scopes or skill_has_action_within(
            resolve_skill_target(meta), scope, ceiling
        ):
            chosen.append(name)
        if len(chosen) >= max(1, int(limit)):
            break
    return chosen
