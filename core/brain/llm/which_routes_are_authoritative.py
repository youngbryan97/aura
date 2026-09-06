"""Which semantic routes decide an answer, and which only watch one.

An external review put it plainly: several recurrent and compositional
semantic routes are shadow- or qualification-gated rather than generally
authoritative for arbitrary production cognition. That is a true statement
about the code and it was nowhere in the code, which meant the difference
between a route that answers and a route that watches was something a reader
had to reconstruct from imports.

Three states, and they are not degrees of the same thing:

* **authoritative** — its output can be the answer. Nothing downstream has to
  agree first.
* **qualified** — it answers only where a qualification says this case is one
  it has been shown to handle. Real, and narrower than it looks.
* **shadow** — it runs and its output is compared, never served. A shadow
  route contributes no answers however good it gets.

The count of shadow routes is the number that has to fall, and it falls by a
route being promoted rather than deleted. Deleting one would also lower it,
so the authoritative count is tracked beside it.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("Aura.WhichRoutesAreAuthoritative")

__all__ = [
    "THE_SEMANTIC_ROUTES",
    "how_the_routes_stand",
    "which_are_only_shadows",
]

#: Every recurrent or compositional semantic route, what it is, and why.
#: "why" is the field that makes this reviewable: a route listed as shadow
#: with no reason is a route somebody forgot to promote.
THE_SEMANTIC_ROUTES: dict[str, dict[str, str]] = {
    "semantic_neural_serving": {
        "where": "core/brain/llm/semantic_neural_serving.py",
        "state": "qualified",
        "why": "serves under qualified_exact_semantic_v1: an activation "
        "document has to name this case before the route may answer it",
    },
    "unified_recurrent_shadow": {
        "where": "core/brain/llm/unified_recurrent_shadow.py",
        "state": "shadow",
        "why": "the package is inspected and run against canary cases; its "
        "output is compared with the served answer and never replaces it",
    },
    "compositional_semantic_qualification": {
        "where": "core/learning/compositional_semantic_qualification.py",
        "state": "qualified",
        "why": "builds the activation the serving route reads, so it decides "
        "what may be answered rather than answering",
    },
    "sequence_induction": {
        "where": "core/cognition/sequence_induction.py",
        "state": "authoritative",
        "why": "answer_sequence_question is applied in the live chat path and "
        "its answer replaces the model's when it has one",
    },
    "relation_language": {
        "where": "core/cognition/relation_language.py",
        "state": "authoritative",
        "why": "the hypothesis space the induction searches; its rules are "
        "what the served answer is derived from",
    },
    "latent_cortex": {
        "where": "core/brain/latent_cortex_service.py",
        "state": "qualified",
        "why": "runs where the router admits it, and the router admits it on "
        "measured readiness rather than on every turn",
    },
}


def how_the_routes_stand() -> dict[str, Any]:
    """Every route by state, and whether each says why it is in that state."""
    by_state: dict[str, list[str]] = {}
    unexplained: list[str] = []
    for name, row in sorted(THE_SEMANTIC_ROUTES.items()):
        by_state.setdefault(row["state"], []).append(name)
        if len(row.get("why", "").split()) < 6:
            unexplained.append(name)
    return {
        "routes": len(THE_SEMANTIC_ROUTES),
        "authoritative": len(by_state.get("authoritative", ())),
        "qualified": len(by_state.get("qualified", ())),
        "shadow": len(by_state.get("shadow", ())),
        "by_state": by_state,
        "unexplained": unexplained,
        "what_this_means": (
            "a shadow route contributes no answers however good it gets; a "
            "qualified one answers only where something else said it may"
        ),
    }


def which_are_only_shadows() -> list[str]:
    """Routes that run and never answer. The number that has to fall."""
    return sorted(
        name
        for name, row in THE_SEMANTIC_ROUTES.items()
        if row["state"] == "shadow"
    )


def register_the_route_states() -> None:
    """Offer the states through the registry, so health can read them.

    core/runtime may not import core.brain — its DEPS is one of the seven
    hand-written foundation rules — so the provider is registered here and
    resolved there, the same way the control-policy sweep is.
    """
    from core.container import ServiceContainer
    from core.runtime.service_registry import register_runtime_service

    ServiceContainer.register_instance(
        "which_routes_are_authoritative", how_the_routes_stand, required=False
    )
    register_runtime_service(
        "which_routes_are_authoritative",
        how_the_routes_stand,
        required=False,
        owner="core/brain/llm/which_routes_are_authoritative.py",
        registered_by="register_the_route_states",
    )
