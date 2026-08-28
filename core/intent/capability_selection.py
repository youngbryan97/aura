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

__all__ = [
    "DEFAULT_CAPABILITY_SET",
    "points_at_something_real",
    "select_capabilities",
]

#: How many capabilities one turn may be handed at once. Every capacity
#: offered is also context the model must hold, so this is a working set
#: rather than the catalogue.
DEFAULT_CAPABILITY_SET = 5


def _asks_for_a_thing(text: str) -> bool:
    """Whether the turn asks for something that exists when it is over."""
    try:
        from core.intent.artifact_request import asks_for_an_artifact

        return bool(asks_for_an_artifact(text))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _wants_more_than_an_answer(text: str) -> bool:
    """Whether the turn needs something the reply alone cannot be.

    Two ways that happens, and each cost a live turn on 2026-08-22. The
    request points at something only looking can answer — a path on this disk,
    an address. Or it asks for a thing that exists afterwards: "Six slides, no
    fluff" was read as a request for prose, so not one capability was offered
    and the model invented a tool to call.
    """
    if points_at_something_real(text):
        return True
    try:
        from core.intent.artifact_request import asks_for_an_artifact

        return bool(asks_for_an_artifact(text))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def points_at_something_real(text: str) -> bool:
    """Whether the request names an artifact the answer depends on.

    A path that resolves on this disk, or an address. Both are already read
    elsewhere for the same reason: the bytes are AT that place, so no amount
    of prose substitutes for looking.
    """
    try:
        from core.intent.opaque_spans import first_named_url

        if first_named_url(text):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    try:
        from core.conversation.filesystem_check import requested_file_read

        named = requested_file_read(text)
        if named is not None and getattr(named, "exists", False):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    # A directory named outright, which the file check does not claim.
    #
    # LIVE, 2026-08-25: the pattern here ended at `[\w.\-~/]+`, so a path at
    # the end of a sentence came back with the full stop attached and did not
    # resolve. Nothing pointed at anything real, no tool was selected, and a
    # request to diagnose a project was answered with a capability catalogue.
    # The reader lives in one place now.
    try:
        from core.language.named_paths import first_existing_path

        if first_existing_path(text) is not None:
            return True
    except (ImportError, OSError, ValueError):
        pass
    return False


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

    # ...unless the request points at something only looking can answer.
    #
    # LIVE, 2026-08-22: "there's a small python project at <path> — one of its
    # tests fails and I can't see why. can you look at it and tell me what's
    # actually wrong?" was read as a request for prose, so no capability was
    # offered at all, and the turn fell through to the desktop lane, where
    # os_automation spent thirty-seven seconds failing to compile AppleScript
    # for a Python question.
    #
    # The grammar was right: it does ask to be told something. What it asks to
    # be told is not knowable without running the project. A named address or
    # a path on this disk is the difference between answering and looking.
    if looks_like_inline_answer_request(text) and not _wants_more_than_an_answer(text):
        return []

    from core.intent.declared_capability import (
        asks_why_something_behaves,
        behaviour_capabilities,
        computation_capabilities,
        declared_vocabulary,
        distinctive_objects,
        foundational_capabilities,
        looks_like_a_request,
        producing_capabilities,
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
    # A question that names something real is a request, whatever its mood.
    #
    # LIVE, 2026-08-22: "why is the test failing in <path>" is not imperative,
    # so nothing foundational was offered and the turn was left with no way to
    # look at the thing it was asked about.
    if looks_like_a_request(text) or _wants_more_than_an_answer(text):
        domains = requested_foundational_domains(text)
        for name in foundational_capabilities(catalogue, domains):
            if name not in ordered:
                ordered.append(name)
    if asks_why_something_behaves(text) and points_at_something_real(text):
        # A cause is not in the request and not in memory. It is in the thing,
        # and the only way to it is to run the thing and look.
        for name in behaviour_capabilities(catalogue):
            if name not in ordered:
                ordered.insert(0, name)
    if _asks_for_a_thing(text):
        # A thing asked for by name. Not an `elif`: a request can both look
        # like a request and ask for something to exist, and the first
        # arrangement of this made the foundational branch swallow it.
        #
        # LIVE, 2026-08-22: "Six slides, no fluff" ranked nothing, because
        # ranking reads a verb acting on an object and that is a noun with a
        # count in front of it. The reader that decides whether a thing was
        # asked for had already said yes and nothing turned that into an
        # offer, so the model invented a tool to call.
        for name in producing_capabilities(catalogue):
            if name not in ordered:
                ordered.append(name)
    if not ordered and settles_by_computation(text):
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
