"""One seam every cognitive source plugs into (CP241).

The integrated eval proved the workspace amplifies whatever relevant
material it is fed: facts placed in it took the model from 0% to 56% on
questions it could not answer alone. Retrieval was the first proven
producer. But the workspace does not care where material comes from, and
Aura has many sources that could feed it:

    retrieval      -> facts from memory / Wikipedia (situations WITH knowledge)
    imagination    -> generated scenarios, counterfactuals (NOVEL situations)
    world model    -> predicted consequences of actions
    reasoning aids -> intermediate results from amplifiers

The temptation -- and the exact shape that killed the RLC's seven
mechanisms -- is to wire all of them and hope. This module is the discipline
that prevents that. Every source implements ONE interface and produces
tagged, budgeted, provenance-carrying material. The composer merges them
without letting any source crowd the question out of the window, and every
piece of material is labelled by its source and its TRUST class, because the
sources are not equal:

* Retrieval and world-model outputs are grounded (real memory, learned
  dynamics). They can be offered to the workspace as context.
* Imagination GENERATES content, which can be wrong. Its material is a
  HYPOTHESIS, never a fact, and is labelled as such so the model (and any
  verifier) treats it as something to test, not trust. Anima Rationis line
  220's warning applies at the source: feeding fabricated "facts" into the
  workspace would strengthen confident mistakes.

Nothing here is load-bearing until it is PROVEN load-bearing: each producer
earns its place through the same ablation factorial retrieval passed
(source on/off, does it move the number), or it does not go live. This
module makes that measurable; it does not assume it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("Aura.Learning.WorkspaceProducers")

WORKSPACE_PRODUCERS_SCHEMA = "aura.workspace_producers.v1"

# Trust classes. The composer and any downstream verifier key on these:
# grounded material may be offered as context; hypothetical material must be
# tested before it is believed.
GROUNDED = "grounded"        # real facts / learned dynamics
HYPOTHETICAL = "hypothetical"  # generated / imagined -> must be verified
TRUST_CLASSES = (GROUNDED, HYPOTHETICAL)


@dataclass(frozen=True)
class WorkspaceMaterial:
    """One piece of material offered to the workspace, with provenance."""

    text: str
    source: str
    trust: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("workspace material must carry text")
        if not self.source.strip():
            raise ValueError("material must name its source for provenance")
        if self.trust not in TRUST_CLASSES:
            raise ValueError(f"trust must be one of {TRUST_CLASSES}")

    def as_line(self) -> str:
        # Hypothetical material is visibly marked so the model never mistakes
        # an imagined scenario for a retrieved fact.
        prefix = "[hypothesis] " if self.trust == HYPOTHETICAL else ""
        return f"{prefix}{self.text}"


class WorkspaceProducer(Protocol):
    """The one interface. Retrieval already fits it; imagination will too.

    ``produce(query, limit)`` returns material for the workspace. A producer
    that finds nothing returns an empty list -- it never fabricates to keep a
    score alive.
    """

    name: str
    trust: str

    def produce(self, query: str, *, limit: int) -> list[WorkspaceMaterial]: ...


def _record_producer_degradation(
    exc: BaseException, *, producer: str, action: str
) -> None:
    """Name a producer failure. A workspace short one input should say so."""
    try:
        from core.runtime.errors import record_degradation

        record_degradation(
            "workspace_producers",
            exc,
            severity="warning",
            action=action,
            extra={"producer": producer},
        )
    except Exception as recorder_exc:  # noqa: BLE001 - reporting must not raise
        logger.debug(
            "workspace producer degradation unrecorded (%s); original: %s",
            recorder_exc,
            exc,
        )


@dataclass
class RetrievalProducer:
    """Adapts any RetrievalSource (e.g. FacadeRetrieval) to the seam.

    Retrieval surfaces real stored facts, so its material is GROUNDED.
    """

    source: Any
    name: str = "retrieval"
    trust: str = GROUNDED

    def produce(self, query: str, *, limit: int) -> list[WorkspaceMaterial]:
        passages = self.source.retrieve(query, limit=limit)
        return [
            WorkspaceMaterial(text=p, source=self.name, trust=self.trust)
            for p in passages
            if str(p).strip()
        ]


@dataclass
class ImaginationProducer:
    """Adapts a scenario/counterfactual generator to the seam.

    ``generator(query, limit) -> list[str]`` produces imagined scenarios.
    Its material is HYPOTHETICAL by construction: it is generated, not
    recalled, so it is labelled as something to test, never to trust. This
    is the source that covers NOVEL situations retrieval cannot -- there is
    nothing to retrieve, so the workspace reasons over generated
    possibilities instead, under verification.
    """

    generator: Any
    name: str = "imagination"
    trust: str = HYPOTHETICAL

    def produce(self, query: str, *, limit: int) -> list[WorkspaceMaterial]:
        run = getattr(self.generator, "imagine", None) or getattr(
            self.generator, "generate", None
        )
        if run is None:
            return []
        try:
            scenarios = run(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 - a producer must not break the workspace
            # The comment here already said this is a degradation. It was not
            # recorded as one, so the workspace silently lost a producer and
            # nothing anywhere said which. Returning [] is still right —
            # fabricating material would be worse — but the loss is named.
            _record_producer_degradation(
                exc,
                producer=self.name,
                action="returned no imagination material after the generator failed",
            )
            return []
        return [
            WorkspaceMaterial(text=str(s), source=self.name, trust=self.trust)
            for s in (scenarios or [])
            if str(s).strip()
        ][:limit]


@dataclass
class WorkspaceComposer:
    """Merge material from many producers without crowding the question.

    The failure this prevents: a source dumps so much material that the
    actual question is pushed out of the attention window, which looks like
    a reasoning failure and is a plumbing one (the same length-bounding
    lesson FacadeRetrieval learned, now enforced across ALL sources).
    """

    producers: list[WorkspaceProducer] = field(default_factory=list)
    per_source_limit: int = 4
    total_limit: int = 10

    def compose(self, query: str) -> dict[str, Any]:
        gathered: list[WorkspaceMaterial] = []
        per_source: dict[str, int] = {}
        for producer in self.producers:
            try:
                material = producer.produce(query, limit=self.per_source_limit)
            except Exception as exc:  # noqa: BLE001 - one source must not fail the gather
                # `per_source[producer.name] = 0` below is indistinguishable
                # from a source that legitimately had nothing, so a producer
                # crashing on every query looked exactly like a quiet corpus.
                logger.warning(
                    "Workspace producer %s failed for query %r: %s",
                    getattr(producer, "name", producer),
                    str(query)[:80],
                    exc,
                )
                material = []
            per_source[producer.name] = len(material)
            gathered.extend(material)

        # Grounded material first, then hypothetical -- the model sees facts
        # before possibilities, and if the budget forces a cut it is the
        # unverified material that is dropped, never the grounded facts.
        gathered.sort(key=lambda m: (m.trust != GROUNDED, -m.score))
        kept = gathered[: self.total_limit]
        return {
            "schema": WORKSPACE_PRODUCERS_SCHEMA,
            "material": kept,
            "lines": [m.as_line() for m in kept],
            "per_source": per_source,
            "grounded": sum(1 for m in kept if m.trust == GROUNDED),
            "hypothetical": sum(1 for m in kept if m.trust == HYPOTHETICAL),
            "dropped": max(0, len(gathered) - len(kept)),
        }

    def context_block(self, query: str) -> tuple[str, dict[str, Any]]:
        """The text block to prepend to the prompt, plus the receipt."""
        composed = self.compose(query)
        if not composed["lines"]:
            return "", composed
        block = "Known context:\n" + "\n".join(
            f"- {line}" for line in composed["lines"]
        )
        return block, composed


def ablation_variants(composer: WorkspaceComposer) -> dict[str, WorkspaceComposer]:
    """Each source on/off, plus none -- the factorial that PROVES a source.

    Returns composers to run the same query through. A source earns its
    place only if turning it OFF measurably lowers the score; otherwise it
    is not load-bearing and does not go live. This is the harness that keeps
    integration from becoming the RLC's seven unproven mechanisms.
    """
    variants: dict[str, WorkspaceComposer] = {
        "all": composer,
        "none": WorkspaceComposer(
            producers=[], per_source_limit=composer.per_source_limit,
            total_limit=composer.total_limit,
        ),
    }
    for producer in composer.producers:
        without = [p for p in composer.producers if p.name != producer.name]
        variants[f"without_{producer.name}"] = WorkspaceComposer(
            producers=without,
            per_source_limit=composer.per_source_limit,
            total_limit=composer.total_limit,
        )
    return variants


__all__ = [
    "GROUNDED",
    "HYPOTHETICAL",
    "TRUST_CLASSES",
    "WORKSPACE_PRODUCERS_SCHEMA",
    "ImaginationProducer",
    "RetrievalProducer",
    "WorkspaceComposer",
    "WorkspaceMaterial",
    "WorkspaceProducer",
    "ablation_variants",
]
