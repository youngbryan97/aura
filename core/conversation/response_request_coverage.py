"""What the person asked for, and whether the reply covers it.

Lifted whole out of `response_reliability`, which is 9,000 lines and the
largest single contributor to the module-size budget. Every name taken from
that module is imported at CALL time: it imports this one to re-export these,
and a test that patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from typing import Any


def visible_user_request(user_message: Any) -> str:
    """Return only the part of a turn the PERSON wrote, or "" if unknowable.

    A live prompt is assembled: identity anchor, retained-memory evidence,
    replayed transcript, working-memory blocks — with the person's actual words
    somewhere inside. Scaffold appears BEFORE the request as often as after, so
    truncating at the first marker is wrong in both directions.

    Returning "" when the request cannot be isolated is the important half.
    A coverage check that cannot see what was asked must not assert the reply
    failed to cover it — an unknown request is not an unmet one.
    """
    from .response_reliability import (
        _INJECTED_PROMPT_BLOCK_MARKERS,
        _MAX_PLAUSIBLE_USER_TURN_CHARS,
        _SCAFFOLD_KV_LINE_RE,
        _TRANSCRIPT_REPLAY_LINE_RE,
    )

    from core.utils.injected_blocks import strip_injected_blocks

    text = str(user_message or "")
    if not text.strip():
        return ""
    # The banners every turn attaches are read from the one shared list
    # first. LIVE 2026-10-04 her screen notes, missing from the markers
    # below, were read as a second question: "missed 'Answer the question
    # that was actually asked, in your own words'".
    text = strip_injected_blocks(text)

    kept: list[str] = []
    in_scaffold_block = False
    for line in text.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            in_scaffold_block = False       # a blank line ends a block
            kept.append(line)
            continue
        if any(marker in lowered for marker in _INJECTED_PROMPT_BLOCK_MARKERS):
            in_scaffold_block = True
            continue
        if _TRANSCRIPT_REPLAY_LINE_RE.match(stripped) or _SCAFFOLD_KV_LINE_RE.match(stripped):
            in_scaffold_block = True
            continue
        if in_scaffold_block:
            continue
        kept.append(line)

    remainder = "\n".join(kept).strip()
    if not remainder:
        return ""
    # A remainder that is still mostly assembled context is not a request. The
    # live prompts run to ~8,000 characters; a person's turn does not.
    if len(remainder) > _MAX_PLAUSIBLE_USER_TURN_CHARS:
        return ""
    return remainder


def _instruction_coverage_reasons(user_message: Any, reply_text: Any) -> list[str]:
    from .response_reliability import (
        _FACT_COUNT_REQUEST_RE,
        _FOLLOWUP_QUESTION_REQUEST_RE,
        _PARAGRAPH_REQUEST_RE,
        A_BARE_ANSWER,
        _asked_for_a_bare_answer,
        _bullet_count,
        _factual_unit_count,
        _has_empty_requested_list_item,
        _matches_exact_reply_request,
        _missing_choice_clarification,
        _missing_requested_memory_limit_coverage,
        _normalize,
        _paragraph_count,
        _reply_contains_reference_value,
        _requested_count,
        _requested_line_count,
        _requested_list_item_count,
        _requested_reference_values,
        _requested_required_phrases,
        _requested_sentence_count,
        _requested_word_count_range,
        _split_sentences,
        _word_count,
        evaluate_facet_coverage,
        requested_exact_reply_target,
    )

    user = visible_user_request(user_message)
    reply = str(reply_text or "").strip()
    if not user or not reply:
        return []

    reasons: list[str] = []
    exact_target = requested_exact_reply_target(user)
    if exact_target and not _matches_exact_reply_request(user, reply):
        reasons.append("missing_requested_exact_reply")

    if _asked_for_a_bare_answer(user) and _word_count(reply) > A_BARE_ANSWER:
        reasons.append("missing_requested_bare_answer")

    # A question that names its scale names the shape of its answer. Asked how
    # she feels "from -1 (very bad) to 1 (very good)" she answered "I feel clear
    # and gathered" on 29 of 96 arms of the reports run of 29 September, and the
    # experiment could not read a report she had not placed. This is a
    # shortfall, never a hard failure: the prose is still an answer, and the
    # person sees that the placement they asked for is missing from it.
    from .asked_scale import answers_the_scale_it_was_asked_for, asks_for_a_rating

    if asks_for_a_rating(user) and not answers_the_scale_it_was_asked_for(user, reply):
        reasons.append("missing_requested_scale_placement")

    requested_word_range = _requested_word_count_range(user)
    if requested_word_range:
        minimum_words, maximum_words = requested_word_range
        reply_words = _word_count(reply)
        if reply_words < minimum_words or reply_words > maximum_words:
            reasons.append("missing_requested_word_count")

    requested_sentences = _requested_sentence_count(user)
    if requested_sentences is not None:
        if len(_split_sentences(reply)) != requested_sentences:
            reasons.append("missing_requested_sentence_count")

    requested_lines = _requested_line_count(user)
    if requested_lines and requested_lines > 1:
        # Lenient on FORM, strict on COUNT. Someone asking for four lines
        # wants four of something; whether they arrive as four newlines or
        # four sentences in a paragraph is a formatting preference, and
        # flagging prose that delivered the substance would be the
        # length-floor mistake this file has made before. Delivering three
        # when four were asked for is the actual failure.
        delivered = max(
            len([line for line in reply.splitlines() if line.strip()]),
            len(_split_sentences(reply)),
        )
        if delivered < requested_lines:
            reasons.append("missing_requested_line_count")

    if any(
        not _reply_contains_reference_value(reply, value)
        for _, value in _requested_reference_values(user)
    ):
        reasons.append("missing_requested_reference_value")

    requested_paragraphs = _requested_count(_PARAGRAPH_REQUEST_RE, user)
    if requested_paragraphs and requested_paragraphs > 1:
        if _paragraph_count(reply) < requested_paragraphs:
            reasons.append("missing_requested_paragraph_count")

    requested_list_items = _requested_list_item_count(user)
    if requested_list_items > 1:
        if _has_empty_requested_list_item(reply, requested_list_items):
            reasons.append("empty_requested_list_item")
        if _bullet_count(reply) < requested_list_items:
            reasons.append("missing_requested_list_count")

    requested_facts = _requested_count(_FACT_COUNT_REQUEST_RE, user)
    if requested_facts and requested_facts > 1:
        if _factual_unit_count(reply) < requested_facts:
            reasons.append("missing_requested_list_count")

    if _missing_choice_clarification(user, reply):
        reasons.append("missing_requested_choice_clarification")
    if _missing_requested_memory_limit_coverage(user, reply):
        reasons.append("missing_requested_memory_limit_coverage")

    if _FOLLOWUP_QUESTION_REQUEST_RE.search(user) and "?" not in reply:
        reasons.append("missing_requested_followup_question")
    normalized_reply = _normalize(reply)
    for phrase in _requested_required_phrases(user):
        if phrase and phrase not in normalized_reply:
            reasons.append("missing_requested_phrase")
            break
    facet_evidence = evaluate_facet_coverage(reply, user)
    requested_facets = list(facet_evidence.get("requested") or [])
    satisfied_facets = set(facet_evidence.get("satisfied") or [])
    if len(requested_facets) >= 2 and any(
        facet not in satisfied_facets for facet in requested_facets
    ):
        reasons.append("missing_requested_objective_facets")
    if len(requested_facets) >= 2 and facet_evidence.get("prompt_echo_detected"):
        reasons.append("prompt_echo_contamination")
    if facet_evidence.get("protocol_artifact_detected"):
        reasons.append("protocol_artifact_leakage")
    return reasons


def _semantic_coverage_reasons(user_message: Any, reply_text: Any) -> list[str]:
    from .response_reliability import (
        _normalize,
    )

    user = _normalize(user_message)
    reply = _normalize(reply_text)
    if not user or not reply:
        return []

    reasons: list[str] = []
    asks_future_memory = bool(
        re.search(r"\bwill\s+you\s+remember\b", user)
        and re.search(
            r"\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?)\b",
            user,
        )
    )
    if asks_future_memory:
        unsupported_guarantee = bool(
            re.search(r"\b(?:can|will)\s+guarantee\b", reply)
            or re.search(
                r"\b(?:(?:i|we|aura)(?:'|’)?ll|(?:i|we|aura)\s+will|will|definitely|certainly|always)\s+remember\b.*\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?)\b",
                reply,
            )
        )
        explicit_boundary = bool(
            re.search(r"\b(?:(?:cannot|can't)\s+guarantee|should\s+not\s+promise)\b", reply)
        )
        if unsupported_guarantee and not explicit_boundary:
            reasons.append("unsupported_memory_guarantee")
        future_answered = bool(
            re.search(
                r"\b(?:tomorrow|later|future|next\s+(?:time|session)|across\s+sessions?|"
                r"durable|persist(?:ent|ed|s)?|stored|memory\s+(?:write|gateway|store)|"
                r"(?:cannot|can't)\s+guarantee|should\s+not\s+promise)\b",
                reply,
            )
        )
        if not future_answered:
            reasons.append("missing_future_memory_answer")

    asks_identity = bool(re.search(r"\b(?:what|who)\s+are\s+you\b", user))
    if asks_identity and asks_future_memory:
        identity_answered = bool(
            re.search(
                r"\b(?:aura|cognitive\s+architecture|runtime|system|agent|entity|mind)\b",
                reply,
            )
        )
        if not identity_answered:
            reasons.append("missing_identity_answer")
    return reasons


def repair_instruction_shape(user_message: Any, reply_text: Any) -> str:
    """Deterministically repair explicit structure misses without another model call."""
    from .response_reliability import (
        _BACKEND_SYMBOLIC_SURFACE_RE,
        _FOLLOWUP_QUESTION_REQUEST_RE,
        _NUMBERED_LIST_REQUEST_RE,
        _NUMBERED_SENTENCE_REQUEST_RE,
        _PARAGRAPH_REQUEST_RE,
        _bullet_count,
        _compact_reference_acknowledgement,
        _default_followup_question,
        _expand_sentence_candidates,
        _fit_reply_to_requested_word_count,
        _listify_sentences,
        _matches_exact_reply_request,
        _number_sentences,
        _pad_sentence_candidates,
        _paragraph_count,
        _paragraphize_sentences,
        _requested_count,
        _requested_list_item_count,
        _requested_sentence_count,
        _requested_word_count_range,
        _split_sentences,
        normalize_user_facing_format,
        requested_exact_reply_target,
    )

    user = str(user_message or "")
    original = str(reply_text or "").strip()
    if not user:
        return original
    exact_target = requested_exact_reply_target(user)
    if exact_target and not _matches_exact_reply_request(user, original):
        return exact_target
    if not original:
        return original
    compact_acknowledgement = _compact_reference_acknowledgement(user)
    if compact_acknowledgement and _BACKEND_SYMBOLIC_SURFACE_RE.search(original):
        return compact_acknowledgement
    normalized_original = normalize_user_facing_format(original)
    if not set(_instruction_coverage_reasons(user, original)):
        return normalized_original

    repaired = normalized_original
    sentences = _split_sentences(repaired)

    requested_word_range = _requested_word_count_range(user)
    if requested_word_range:
        word_repaired = _fit_reply_to_requested_word_count(user, repaired)
        if word_repaired:
            return word_repaired

    # Missing requested values are semantic omissions. Appending raw values to
    # an unrelated sentence can produce grammatical text whose proposition is
    # false. Only another grounded generation may supply them.

    requested_sentences = _requested_sentence_count(user)
    if requested_sentences is not None:
        sentence_repaired = _expand_sentence_candidates(
            _split_sentences(repaired),
            requested_sentences,
        )
        sentence_repaired = _pad_sentence_candidates(
            sentence_repaired,
            requested_sentences,
        )
        if len(sentence_repaired) >= requested_sentences:
            # Select only from content the model already produced. A later
            # sentence can carry the requested value even when the opening is
            # preamble ("Done. Sample two."). Check each contiguous window and
            # admit one only when the complete semantic contract still holds.
            for start in range(len(sentence_repaired) - requested_sentences + 1):
                candidate = " ".join(
                    sentence_repaired[start : start + requested_sentences]
                )
                if not _instruction_coverage_reasons(user, candidate):
                    repaired = candidate
                    break

    requested_numbered = _requested_count(_NUMBERED_LIST_REQUEST_RE, user)
    requested_numbered_sentences = _requested_count(_NUMBERED_SENTENCE_REQUEST_RE, user)
    requested_list_items = _requested_list_item_count(user)
    # Exact-label replies ("Objective: ...", "Stop conditions: ...") are
    # already structured by the user's own labels; renumbering them
    # destroys an exact-format contract that was satisfied. Count
    # label-styled lines as fulfilled structure.
    label_lines = sum(
        1
        for line in repaired.splitlines()
        if re.match(r"^[A-Z][^:\n]{0,40}:\s", line.strip())
    )
    if requested_list_items > 1 and label_lines >= requested_list_items:
        requested_list_items = 0
    if requested_list_items > 1 and _bullet_count(repaired) < requested_list_items:
        if requested_numbered or requested_numbered_sentences:
            list_repaired = _number_sentences(sentences, requested_list_items)
        else:
            list_repaired = _listify_sentences(sentences, requested_list_items)
        if list_repaired:
            repaired = list_repaired

    requested_paragraphs = _requested_count(_PARAGRAPH_REQUEST_RE, user)
    if requested_paragraphs and requested_paragraphs > 1:
        if _paragraph_count(repaired) < requested_paragraphs:
            paragraph_repaired = _paragraphize_sentences(_split_sentences(repaired), requested_paragraphs)
            if paragraph_repaired:
                repaired = paragraph_repaired

    if _FOLLOWUP_QUESTION_REQUEST_RE.search(user) and "?" not in repaired:
        followup = _default_followup_question(user)
        if requested_paragraphs and requested_paragraphs > 1 and _paragraph_count(repaired) >= requested_paragraphs:
            parts = [
                block.strip()
                for block in re.split(r"(?:\r?\n\s*){2,}", repaired)
                if block.strip()
            ]
            parts[-1] = f"{parts[-1]} {followup}"
            repaired = "\n\n".join(parts)
        else:
            repaired = f"{repaired}\n\n{followup}"
    repaired = repaired.strip()
    # A remaining coverage reason is deliberately left visible. The compact
    # acknowledgement above is safe only when replacing a literal backend
    # surface leak; using it as a universal fallback turned unrelated prose
    # into a canned sentence that happened to contain the requested number.
    return repaired
