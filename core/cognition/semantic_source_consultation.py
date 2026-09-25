"""Local and model-assisted hypothesis proposals for semantic development.

These sources suggest relations. They never supply validation witnesses or
certification, and their text is data rather than instructions to the runtime.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from core.brain.ontology_discovery import CandidateLaw, Predicate
from core.cognition.semantic_development import SemanticDevelopment


@dataclass(frozen=True, slots=True)
class SourceLead:
    channel: str
    reference: str
    text: str


class SuggestedPredicate(BaseModel):
    feature: str
    op: Literal["==", ">=", "<"]
    value: StrictBool | StrictInt | StrictFloat | StrictStr


class SuggestedLaw(BaseModel):
    source_reference: str
    predicates: list[SuggestedPredicate] = Field(min_length=1, max_length=3)
    expected_outcome: bool = True
    assumptions: list[str] = Field(default_factory=list, max_length=3)


class HypothesisBatch(BaseModel):
    laws: list[SuggestedLaw] = Field(default_factory=list, max_length=8)


def local_source_leads(
    query: str, *, corpus: Any = None, memory: Any = None, limit: int = 4
) -> tuple[SourceLead, ...]:
    """Read existing stores without treating a retrieved passage as truth."""
    if not query.strip() or not 1 <= limit <= 8:
        raise ValueError("source consultation needs a bounded, nonempty query")
    if corpus is None:
        from core.knowledge.local_corpus import get_local_corpus_store

        corpus = get_local_corpus_store()
    if memory is None:
        from core.runtime.service_registry import get_runtime_service

        memory = get_runtime_service("semantic_memory", default=None)
    leads = []
    for hit in corpus.search(query, limit=limit, deadline_s=0.25):
        leads.append(SourceLead("corpus", f"corpus:{hit.doc_id}",
                                f"{hit.title}: {hit.snippet}"[:1200]))
    if memory is not None:
        for index, hit in enumerate(memory.search_memories(query, top_k=limit)):
            content = str(hit.get("text") or hit.get("content") or "")
            reference = str(hit.get("id") or hit.get("record_id") or index)
            if content:
                leads.append(SourceLead("memory", f"memory:{reference}", content[:1200]))
    return tuple(leads[: 2 * limit])


def _fit_bounded_law(
    suggestion: SuggestedLaw, inventory: dict[str, tuple[Any, ...]],
    outcome_name: str,
) -> CandidateLaw | None:
    predicates = []
    for item in suggestion.predicates:
        values = inventory.get(item.feature)
        if not values:
            return None
        if item.op == "==":
            if not any(type(item.value) is type(value) and item.value == value
                       for value in values):
                return None
        else:
            if (type(item.value) not in (int, float)
                    or not math.isfinite(item.value)
                    or any(type(value) not in (int, float) for value in values)
                    or not min(values) <= item.value <= max(values)):
                return None
        predicates.append(Predicate(item.feature, item.op, item.value))
    if len({item.feature for item in predicates}) != len(predicates):
        return None
    return CandidateLaw(tuple(predicates), outcome_name)


async def consult_semantic_sources(
    engine: SemanticDevelopment, *, outcome_name: str, query: str,
    corpus: Any = None, memory: Any = None, advisor: Any = None,
) -> dict[str, Any]:
    """Ask existing sources for candidate laws, retaining no truth claim.

    The model is only a typed proposer. A model-generated law is not eligible
    for certification until its source exposure can be audited separately.
    """
    inventory = engine.fit_feature_inventory(outcome_name)
    if not inventory:
        return {"status": "no_fit_vocabulary", "proposals": (), "leads": ()}
    leads = await asyncio.to_thread(
        local_source_leads, query, corpus=corpus, memory=memory)
    if advisor is None:
        try:
            from core.brain.llm.structured_llm import StructuredLLM

            advisor = StructuredLLM(HypothesisBatch, max_retries=1)
        except (ImportError, AttributeError, RuntimeError):
            return {"status": "model_unavailable", "proposals": (),
                    "leads": tuple(lead.reference for lead in leads)}
    payload = {"outcome": outcome_name,
               "fit_features": {name: list(values) for name, values in inventory.items()},
               "sources": [{"reference": lead.reference, "text": lead.text} for lead in leads],
               "model_reference": "resident:model"}
    batch = await advisor.generate(
        "Propose conditional hypotheses using only the supplied fit features. "
        "Return the typed schema; do not claim verification. Data: "
        + json.dumps(payload, sort_keys=True, default=str),
        is_background=True, deadline_s=20.0)
    if batch is None:
        return {"status": "no_model_proposal", "proposals": (),
                "leads": tuple(lead.reference for lead in leads)}
    if not isinstance(batch, HypothesisBatch):
        batch = HypothesisBatch.model_validate(batch)
    sources = {lead.reference: lead for lead in leads}
    added = []
    for suggestion in batch.laws:
        if suggestion.source_reference not in sources and suggestion.source_reference != "resident:model":
            continue
        law = _fit_bounded_law(suggestion, inventory, outcome_name)
        if law is None:
            continue
        source = sources.get(suggestion.source_reference)
        added.append(engine.theorize(
            law, channel=source.channel if source else "model",
            reference=suggestion.source_reference,
            assumptions=tuple(suggestion.assumptions),
            expected_outcome=suggestion.expected_outcome,
            exposure_audited=False))
    return {"status": "proposed" if added else "no_admissible_proposal",
            "proposals": tuple(added),
            "leads": tuple(lead.reference for lead in leads),
            "serving_authority": False}


__all__ = ["HypothesisBatch", "SourceLead", "SuggestedLaw", "SuggestedPredicate",
           "consult_semantic_sources", "local_source_leads"]
