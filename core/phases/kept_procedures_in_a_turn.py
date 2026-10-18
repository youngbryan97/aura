"""A kept procedure's answer in a turn: found before she drafts, and her reply kept when it says the answer.

A procedure she wrote for a kind of problem, kept because it agreed with
every answer she knew for that kind, computes the answer in milliseconds
(core/learning/procedures_from_solved_examples.py). A request worded like the
kind's problems is answered as it stands. One that means the kind but is
worded otherwise is found by her embedding (core/learning/kinds_by_meaning.py),
restated in the kind's form by her model, and answered from the restatement;
that answer is evidence only, since nothing checked the restatement.

Both response phases use this, and so does the path that answers her desktop
chat (core/brain/cognitive_engine_quick_reply.py), which runs before the
phases and, when it answers, instead of them. The unitary phase her kernel
runs had only the first: a request had to be worded like a benchmark's
problems to reach a procedure at all, and the computed answer replaced her
draft rather than being put in front of her before it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.KeptProceduresInATurn")

_HER_EMBEDDER: dict[str, Any] = {}


def _a_problem() -> Any:
    from core.brain.llm.an_envelope_the_decoder_enforces import AnEnvelope

    # restatement_request shows each problem under "Problem:", so the next one would begin there.
    return AnEnvelope(opens="Problem:\n", closes="\n\nProblem:")


A_PROBLEM = _a_problem()


def her_query_vector(text: str, task: str) -> Any:
    """Her embedding of a request, only from an embedding model the runtime already holds; None otherwise."""
    try:
        from core.memory.embedding_runtime import (
            acquire_shared_embedding_engine,
            shared_embedding_runtime_snapshot,
        )

        if not shared_embedding_runtime_snapshot().get("engine_live"):
            return None
        engine = _HER_EMBEDDER.get("lease")
        if engine is None:
            engine = _HER_EMBEDDER["lease"] = acquire_shared_embedding_engine("kinds_by_meaning")
        return engine.embed_query(text, task=task)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("no embedding of the request by meaning: %s", exc)
        return None


def states_the_answer(draft: str, answer: str) -> bool:
    """Whether her reply's closing paragraph states the answer a kept procedure computed.

    A lettered option counts when it is named as one ("(b)", "b)", "option b",
    "answer is b"); a bare letter would match an article.
    """
    paragraphs = [part for part in re.split(r"\n\s*\n", str(draft or "")) if part.strip()]
    if not paragraphs:
        return False
    closing = " ".join(paragraphs[-1].split()).casefold()
    wanted = " ".join(str(answer or "").split()).casefold()
    if wanted and wanted in closing:
        return True
    lettered = re.fullmatch(r"\(([a-z])\)", wanted)
    return bool(lettered) and re.search(
        rf"(?:\(|\boption\s+|\banswer\s+is:?\s+){lettered.group(1)}\b|\b{lettered.group(1)}\)", closing
    ) is not None


async def kept_answer_by_meaning(objective: str, *, router: Any, origin: str, tier: str | None) -> Any:
    """A kept procedure's answer to a request that means its kind in other words; None when there is none.

    Her embedding model is used only if the runtime already holds it, so a
    turn never waits on loading one; the restatement is one short decode of
    hers with the private channel closed.
    """
    from core.brain.reasoning_amplifier_v2 import amplified_from_kept_receipt
    from core.learning.kinds_by_meaning import TASK, restatement_request, restates
    from core.learning.procedures_from_solved_examples import kept_procedure_book

    if router is None:
        return None
    book = await asyncio.to_thread(kept_procedure_book)
    if book is None or not book.meanings:
        return None
    vector = await asyncio.to_thread(her_query_vector, objective, TASK)
    if vector is None:
        return None
    kind, similarity, why = book.route_by_meaning(vector)
    if kind is None or kind not in book.procedures or not book.examples.get(kind):
        logger.debug("no kept kind meant by this request (%s, %.3f)", why, similarity)
        return None
    examples = book.examples[kind]
    # About as long as the kind's longest shown problem and the request together, at under two
    # characters a token: room for the problem written out, and no more.
    cap = (max(len(e["problem"]) for e in examples) + len(objective)) // 2
    # Her reply opens inside the label her shown problems carry, so what she writes is a problem and not an
    # account of how she would write one (live 2026-10-16: twice "To write the problem in the same form as the
    # examples, we need to ..."), and it stops where the next problem would begin.
    from core.brain.llm.an_envelope_the_decoder_enforces import the_request_for

    try:
        out = await router.think(
            **the_request_for(A_PROBLEM, [{"role": "user", "content": restatement_request(objective, examples)}]),
            priority=1.0,
            origin=f"kept_procedure_restatement_{origin}",
            purpose="kept_procedure_restatement",
            prefer_tier=tier or "primary",
            is_background=False,
            foreground_request=True,
            allow_cloud_fallback=False,
            skip_runtime_payload=True,
            clean_user_surface_contract=True,
            temperature=0.0,
            max_tokens=max(1, cap),
        )
    except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("kept_procedure_restatement", exc, severity="warning",
                           action="drafted without restating the request for its kind")
        return None
    restated = str((out.get("content") or out.get("response") or "") if isinstance(out, dict) else (out or ""))
    restated = restated.strip().removeprefix(A_PROBLEM.opens.strip()).split(A_PROBLEM.closes.strip())[0]
    if not restated.strip():
        return None
    if not restates(objective, restated, examples):
        logger.info("kept procedure for %s not consulted: the restatement is not the problem asked (%r)",
                    kind, restated[:160])
        return None
    receipt = await asyncio.to_thread(book.answer_restated, kind, restated.strip())
    receipt["similarity"] = round(similarity, 4)
    return amplified_from_kept_receipt(receipt)


async def kept_answer_among_the_word_near(objective: str) -> Any:
    """A kept answer when the words leave several kinds near and her meaning picks one of them.

    Typed in plain words, "What are the prime factors of 21000063?" is as near by its words to
    the kind that lists a number's prime factors as to the kind that computes the number first
    from equations, which keeps no procedure, so it was declined (live, 17 October). The plain
    kind's procedure answers the other kind's problems wrongly, so the words alone must not
    choose. Her embedding chooses among those kinds only: the one the request is as typical of as
    its least typical sealed problem was, when exactly one is.
    """
    from core.brain.reasoning_amplifier_v2 import amplified_from_kept_receipt
    from core.learning.kinds_by_meaning import TASK
    from core.learning.procedures_from_solved_examples import SOLVED_EXAMPLE_PROCEDURES_SCHEMA, kept_procedure_book

    book = await asyncio.to_thread(kept_procedure_book)
    if book is None or not book.meanings:
        return None
    receipt = await asyncio.to_thread(book.answer, objective)
    near = list(receipt.get("near_enough") or [])
    if receipt.get("answer") is not None or len(near) < 2:
        return None
    vector = await asyncio.to_thread(her_query_vector, objective, TASK)
    if vector is None:
        return None
    kind = book.typical_among(vector, near)
    if kind is None or not book.procedures.get(kind):
        logger.debug("the kinds near by words are not told apart by meaning (%s)", near)
        return None
    run = await asyncio.to_thread(book._run_kind, kind, objective, {
        "schema": SOLVED_EXAMPLE_PROCEDURES_SCHEMA, "family": kind, "answer": None,
        "form": "words_then_meaning", "near_by_words": near,
    })
    return amplified_from_kept_receipt(run)


async def kept_answer_for(objective: str, *, router: Any, origin: str, tier: str | None) -> dict[str, Any] | None:
    """What a kept procedure computed for this request, by its words or else by its meaning; None when none did.

    The record is what a phase keeps in ``response_modifiers["kept_procedure"]``
    and puts in front of her: the answer, the kind, how the request reached it,
    and a one-line summary of where the answer came from.
    """
    try:
        from core.brain.reasoning_amplifier_v2 import answer_from_kept_procedures
    except ImportError as exc:
        record_degradation("kept_procedure_consult", exc, severity="warning",
                           action="drafted without consulting her kept procedures")
        return None
    kept = await asyncio.to_thread(answer_from_kept_procedures, objective)
    form = "words"
    if kept is None:
        kept = await kept_answer_among_the_word_near(objective)
        form = "words_then_meaning"
    if kept is None:
        kept = await kept_answer_by_meaning(objective, router=router, origin=origin, tier=tier)
        form = "restated"
    if kept is None:
        return None
    operation = dict(kept.receipt.cognitive_operations[0]) if kept.receipt.cognitive_operations else {}
    known = int(operation.get("agreed_known") or 0)
    family = str(operation.get("family") or "")
    summary = (
        f"computed by the procedure she keeps for problems of the kind {family!r}, "
        f"which agreed with all {known} answers she knows for that kind"
    )
    if form == "restated":
        summary += f"; the request was first written in that kind's form, as: {operation.get('restated', '')}"
    return {
        "objective": objective,
        "answer": kept.answer,
        "family": family,
        "form": form,
        "agreed_known": known,
        "confidence": float(kept.confidence),
        "receipt": kept.receipt.to_dict(),
        "generation_metadata": dict(kept.generation_metadata),
        "summary": summary,
    }


def kept_answer_evidence(planted: dict[str, Any]) -> str:
    """A kept answer as she is shown it before drafting: the skill result the phases put in front of her."""
    return f"[SKILL RESULT: kept_procedure] \u2705 Answer: {planted['answer']}\nSummary: {planted['summary']}"


def planted_for(state: Any, objective: str) -> dict[str, Any] | None:
    """The kept answer found before this turn's draft, only if it was found for this very request."""
    planted = (getattr(state, "response_modifiers", None) or {}).get("kept_procedure")
    if isinstance(planted, dict) and planted.get("answer") and planted.get("objective") == objective:
        return planted
    return None


def after_the_draft(draft: str, planted: dict[str, Any]) -> tuple[bool, str]:
    """(her reply stands, how): her words stand when they state the computed answer, or when the answer rests on
    a restatement nothing checked; otherwise the computed answer is served."""
    if planted.get("form") == "restated":
        return True, "evidence_for_a_restated_request"
    if states_the_answer(draft, str(planted.get("answer") or "")):
        return True, "her_reply_states_it"
    return False, "computed_answer_served"


__all__ = ["after_the_draft", "her_query_vector", "kept_answer_among_the_word_near", "kept_answer_by_meaning",
           "kept_answer_evidence", "kept_answer_for", "planted_for", "states_the_answer"]
