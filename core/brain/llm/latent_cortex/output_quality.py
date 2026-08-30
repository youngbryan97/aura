"""Deterministic product-quality contract for resident latent answers.

The worker receipt proves how an episode ran. This module separately proves
that the user-visible text is complete enough to count as an answer. It never
generates replacement prose and therefore cannot create a second model owner.
"""
from __future__ import annotations

import hashlib
import math
import re
from difflib import SequenceMatcher
from typing import Any

from core.language import relational_request

OUTPUT_QUALITY_SCHEMA = "aura.latent_output_quality.v1"

_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)\S", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"[.!?](?:[\"')\]]+)?(?=\s|$)")
_REQUEST_FACETS = {
    "compare": re.compile(
        r"\b(?:compar(?:e|es|ed|ing)|contrast(?:s|ed|ing)?|difference|versus|vs\.?)\b",
        re.I,
    ),
    "select": re.compile(
        r"\b(?:choose|recommend|prefer|stronger|better|best\s+(?:design|option|architecture))\b",
        re.I,
    ),
    "verify": re.compile(r"\b(?:verify|test|prove|validate|certif(?:y|ication))\b", re.I),
    "explain": re.compile(r"\b(?:explain|why|caus(?:e|al|ality))\b", re.I),
    "enumerate": re.compile(r"\b(?:list|enumerate|steps|each)\b", re.I),
}
_HOW_RE = re.compile(r"\bhow\b", re.I)
_HOW_COMPARISON_RE = re.compile(
    r"\bhow\s+(?:does|do|did|would|will|is|are|was|were)\b"
    r"[^?.;\n]{0,120}?\b(?:compar(?:e|es|ed|ing)|contrast(?:s|ed|ing)?|"
    r"differ(?:s|ed|ent|ently|ing)?)\b",
    re.I,
)
_ANSWER_FACETS = {
    "compare": re.compile(
        r"\b(?:whereas|while|unlike|versus|compared|by\s+contrast|in\s+contrast)\b",
        re.I,
    ),
    "select": re.compile(
        r"(?:\b(?:i|we)\s+(?:would\s+)?(?:choose|recommend|prefer)\b|"
        r"\b(?:is|remains)\s+(?:the\s+)?stronger\b|"
        r"\bshould\s+(?:use|choose|adopt)\b|\bthe\s+winner\b)",
        re.I,
    ),
    "verify": re.compile(
        r"\b(?:test(?:ing|ed|s)?|assert(?:ion|s|ed)?|inject(?:ion|ed|s)?|"
        r"simulate(?:d|s|ion)?|exercise(?:d|s)?|replay(?:ed|s)?|"
        r"measure(?:d|s|ment)?|check(?:ed|s)?|verify\s+(?:by|with|that))\b",
        re.I,
    ),
    "explain": re.compile(
        r"\b(?:because|therefore|thus|so\s+that|leads?\s+to|prevents?|"
        r"caus(?:e|es|ed|ing)|ensures?)\b",
        re.I,
    ),
    "enumerate": re.compile(r"\b(?:first|second|third|finally|steps?)\b", re.I),
}
_TEMPORAL_CONTRAST_RE = re.compile(
    r"\b(?:\d+|a|an|one|two|three|few|several)\s+"
    r"(?:seconds?|minutes?|hours?|days?)\s+ago\b[\s\S]{0,360}?"
    r"\b(?:now|currently|since|afterward|afterwards|at\s+present)\b|"
    r"\b(?:now|currently|at\s+present)\b[\s\S]{0,360}?"
    r"\b(?:\d+|a|an|one|two|three|few|several)\s+"
    r"(?:seconds?|minutes?|hours?|days?)\s+ago\b",
    re.I,
)
_STOPWORDS = {
    "about", "after", "again", "against", "also", "among", "answer", "because",
    "before", "being", "both", "could", "design", "does", "each", "every", "explain",
    "enumerate", "from", "have", "into", "itself", "list", "more", "most", "other",
    "should", "some", "stronger",
    "such", "than", "that", "their", "then", "there", "these", "they", "this", "through",
    "under", "using", "verify", "what", "when", "where", "which", "while", "with", "would",
}
# An explain-request that is ABOUT the comparison: satisfying the comparison
# is what explaining the difference means.
_COMPARATIVE_EXPLAIN_RE = re.compile(
    r"\b(?:explain|describe|tell\s+me)\b[^?.;\n]{0,40}?"
    r"\b(?:difference|distinction|contrast)\b",
    re.I,
)
_SUBJECT_TRAIL_RE = re.compile(
    r"\b(?:under|against|across|including)\s+([^?.;\n]{3,180})",
    re.I,
)
_SUBJECT_NOISE = {
    "and", "case", "cases", "condition", "conditions", "fault", "faults", "scenario",
    "scenarios", "the", "with",
}
_PROTOCOL_ARTIFACT_RE = re.compile(
    r"(?:<\s*/?\s*(?:request|response|system|assistant|user)\b|"
    r"<\s*div\b[^>]{0,160}\bresponse\b[^>]{0,160}\brequest\b)",
    re.I,
)
_FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def request_facets(objective: Any) -> list[str]:
    """Facets a request explicitly asks for (compare/select/verify/…).

    Public so allocation can shape the answer surface (token budget, decode
    discipline) with EXACTLY the same definition the quality gate will later
    judge the answer by — no drift between what is provisioned and what is
    demanded."""
    text = objective if isinstance(objective, str) else ""
    facets = [name for name, pattern in _REQUEST_FACETS.items() if pattern.search(text)]
    # "How" normally requests an explanation, except when it introduces a
    # comparison ("How does that compare...?"). That form asks one question.
    comparison_spans = tuple(match.span() for match in _HOW_COMPARISON_RE.finditer(text))
    how_requires_explanation = any(
        not any(start <= match.start() < end for start, end in comparison_spans)
        for match in _HOW_RE.finditer(text)
    )
    if how_requires_explanation and "explain" not in facets:
        facets.append("explain")
    return facets


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _compared_subjects(objective: str) -> tuple[list[str], list[str]] | None:
    """Return the shared language substrate's explicit comparison pair."""

    subjects = relational_request.compared_subjects(objective)
    if subjects is None:
        return None
    left, right = subjects
    return list(left), list(right)


def _covers_both_compared_subjects(analysis_text: str, objective: str) -> bool:
    """True when the answer substantively addresses BOTH compared subjects.

    A connective keyword ("whereas", "unlike") is neither necessary nor
    sufficient for a comparison: prose that holds two named things against
    each other IS a comparison, while filler can contain "while" and compare
    nothing. Requiring coverage of the two subjects the REQUEST named is the
    stronger test — filler that addresses neither still fails.
    """
    return relational_request.comparison_sides_are_covered(
        analysis_text, objective
    ) is True


def _analysis_surface(text: str, objective: str) -> dict[str, Any]:
    """Remove copied request spans before judging whether an answer did work.

    A model may legitimately reuse individual topic terms. What cannot earn
    credit is a long contiguous copy of the request: CP118 echoed the entire
    compare/choose/verify clause, and the old gate counted those instructions
    as if the answer had fulfilled them.
    """

    answer_tokens = _tokens(text)
    objective_tokens = _tokens(objective)
    copied_indexes: set[int] = set()
    longest_run = 0
    matcher = SequenceMatcher(
        None,
        objective_tokens,
        answer_tokens,
        autojunk=False,
    )
    for block in matcher.get_matching_blocks():
        if block.size < 4:
            continue
        longest_run = max(longest_run, block.size)
        copied_indexes.update(range(block.b, block.b + block.size))
    copied_count = len(copied_indexes)
    objective_echo_ratio = copied_count / max(1, len(objective_tokens))
    answer_echo_ratio = copied_count / max(1, len(answer_tokens))
    echo_threshold = max(8, math.ceil(len(objective_tokens) * 0.25))
    prompt_echo_detected = bool(
        longest_run >= echo_threshold
        or (copied_count >= 8 and objective_echo_ratio >= 0.35)
    )
    novel_tokens = (
        [
            token
            for index, token in enumerate(answer_tokens)
            if index not in copied_indexes
        ]
        if prompt_echo_detected
        else list(answer_tokens)
    )
    visible_without_code = _FENCED_BLOCK_RE.sub("", text)
    protocol_artifact_detected = bool(
        _PROTOCOL_ARTIFACT_RE.search(visible_without_code)
    )
    return {
        "analysis_text": " ".join(novel_tokens),
        "analysis_tokens": novel_tokens,
        "prompt_echo_detected": prompt_echo_detected,
        "protocol_artifact_detected": protocol_artifact_detected,
        "copied_objective_token_count": copied_count,
        "longest_objective_copy_tokens": longest_run,
        "objective_echo_ratio": objective_echo_ratio,
        "answer_echo_ratio": answer_echo_ratio,
    }


def evaluate_facet_coverage(text: Any, objective: Any) -> dict[str, Any]:
    """Judge requested facets only on answer-authored, non-echoed content."""

    rendered = text if isinstance(text, str) else ""
    objective_text = objective if isinstance(objective, str) else ""
    surface = _analysis_surface(rendered, objective_text)
    analysis_text = str(surface["analysis_text"])
    requested = request_facets(objective_text)
    list_items = len(_LIST_ITEM_RE.findall(rendered))
    satisfied: list[str] = []
    excerpts: dict[str, str] = {}
    unsupported_cues: list[str] = []
    broad_hints = {
        "compare": re.compile(r"\b(?:compare|contrast|early|late|whereas|while)\b", re.I),
        "select": re.compile(r"\b(?:choose|recommend|prefer|stronger|best|winner)\b", re.I),
        "verify": re.compile(r"\b(?:verif\w*|test\w*|fault|cancel|timeout|restart)\b", re.I),
        "explain": _ANSWER_FACETS["explain"],
        "enumerate": re.compile(r"\b(?:first|second|third|finally|steps?)\b", re.I),
    }
    for name in requested:
        matched = bool(_ANSWER_FACETS[name].search(analysis_text))
        if name == "verify" and not matched:
            matched = bool(
                re.search(
                    r"(?:\brun\b.{0,50}\bverif\w*\b|"
                    r"\bverif\w*\b.{0,35}\b(?:plan|procedure|turn|step|harness)\b)",
                    analysis_text,
                    re.I,
                )
            )
        if name == "compare":
            # When the request NAMES the pair, subject coverage is the test —
            # in both directions. A connective keyword is neither necessary
            # (prose that holds two named things against each other is a
            # comparison) nor sufficient (filler containing a bare "while"
            # compares nothing). Requests without an explicit pair fall back
            # to the connective/heuristic match.
            if _compared_subjects(objective_text) is not None:
                matched = _covers_both_compared_subjects(analysis_text, objective_text)
            elif not matched:
                matched = bool(
                    re.search(r"\bearly\b", analysis_text, re.I)
                    and re.search(r"\blate\b", analysis_text, re.I)
                    and re.search(
                        r"\b(?:pros?|cons?|advantage|disadvantage)\w*\b",
                        analysis_text,
                        re.I,
                    )
                )
            if not matched:
                # "A minute ago X. Now Y." performs a direct temporal
                # comparison without requiring an essay connective.
                matched = bool(_TEMPORAL_CONTRAST_RE.search(analysis_text))
        if name == "explain" and not matched and _COMPARATIVE_EXPLAIN_RE.search(objective_text):
            # "Explain the difference between A and B": explaining the
            # difference IS the comparison, so a covered comparison answers
            # the explain request too.
            matched = _covers_both_compared_subjects(analysis_text, objective_text)
        if name == "enumerate" and list_items >= 2:
            matched = True
        minimum_facet_words = {
            "compare": 10,
            "select": 8,
            "verify": 10,
            "explain": 8,
            "enumerate": 6,
        }[name]
        if len(surface["analysis_tokens"]) < minimum_facet_words:
            matched = False
        if matched:
            satisfied.append(name)
            match = _ANSWER_FACETS[name].search(analysis_text)
            start = max(0, (match.start() if match else 0) - 90)
            end = min(len(analysis_text), (match.end() if match else 0) + 110)
            excerpts[name] = analysis_text[start:end].strip()
        elif broad_hints[name].search(analysis_text):
            unsupported_cues.append(name)
    if surface["prompt_echo_detected"] or surface["protocol_artifact_detected"]:
        score = 0.0 if requested else None
    else:
        score = (len(satisfied) / len(requested)) if requested else None
    return {
        "requested": requested,
        "satisfied": satisfied,
        "unsupported_cues": unsupported_cues,
        "excerpts": excerpts,
        "score": score,
        "prompt_echo_detected": bool(surface["prompt_echo_detected"]),
        "protocol_artifact_detected": bool(surface["protocol_artifact_detected"]),
        "copied_objective_token_count": int(surface["copied_objective_token_count"]),
        "longest_objective_copy_tokens": int(surface["longest_objective_copy_tokens"]),
    }


def _concept(token: str) -> str:
    token = token.lower().replace("-", " ").strip()
    replacements = {
        "cancellation": "cancel",
        "cancelled": "cancel",
        "canceled": "cancel",
        "cancelling": "cancel",
        "verification": "verify",
        "verifications": "verify",
        "verified": "verify",
        "timeouts": "timeout",
        "restarted": "restart",
        "restarts": "restart",
        "restarting": "restart",
    }
    if token in replacements:
        return replacements[token]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _max_blank_line_run(text: str) -> int:
    maximum = current = 0
    for line in text.splitlines():
        if line.strip():
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _terminal_complete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    # An odd fence count means an unclosed code block ANYWHERE in the text —
    # the answer is structurally truncated no matter how its last line ends.
    if stripped.count("```") % 2 != 0:
        return False
    if stripped.endswith("```"):
        return True
    return stripped.endswith((".", "?", "!", ")", "]", "}"))


# How much of an objective is examined at all. The old cap of 32 silently
# ignored everything a long objective asked for beyond that point.
_MAX_OBJECTIVE_TERMS = 64
# A compound answer must engage with the objective in proportion to how much
# it asks, bounded so paraphrase is not punished.
_MIN_OBJECTIVE_TERM_MATCHES = 3
_MAX_REQUIRED_TERM_MATCHES = 8
_OBJECTIVE_TERM_MATCH_RATIO = 0.25
_MIN_OBJECTIVE_COVERAGE = 0.20
_WIDE_OBJECTIVE_COVERAGE = 0.15


def _listed_subjects(objective: str) -> list[dict[str, Any]]:
    subjects: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for match in _SUBJECT_TRAIL_RE.finditer(objective):
        for raw_part in re.split(r",|\band\b", match.group(1), flags=re.I):
            raw_tokens = [
                token
                for token in _tokens(raw_part)
                if token not in _SUBJECT_NOISE
            ]
            keys: list[str] = []
            for token in raw_tokens:
                keys.extend(
                    _concept(part)
                    for part in token.replace("-", " ").split()
                    if part and part not in _SUBJECT_NOISE
                )
            key_tuple = tuple(dict.fromkeys(key for key in keys if len(key) >= 3))
            if not key_tuple or key_tuple in seen:
                continue
            seen.add(key_tuple)
            subjects.append(
                {
                    "label": " ".join(raw_tokens)[:80],
                    "keys": list(key_tuple),
                }
            )
            if len(subjects) >= 8:
                return subjects
    return subjects


def evaluate_latent_output(
    text: Any,
    *,
    generated_tokens: Any,
    termination: Any,
    objective: Any,
) -> dict[str, Any]:
    """Return a self-contained, hash-bound acceptance receipt."""

    rendered = text if isinstance(text, str) else ""
    objective_text = objective if isinstance(objective, str) else ""
    generated = generated_tokens if type(generated_tokens) is int else 0
    stop = termination if isinstance(termination, str) else ""
    surface = _analysis_surface(rendered, objective_text)
    visible_words = _tokens(rendered)
    words = list(surface["analysis_tokens"])
    objective_words = _tokens(objective_text)
    normalized_nonempty_lines = [
        " ".join(_tokens(line))
        for line in rendered.splitlines()
        if line.strip()
    ]
    normalized_nonempty_lines = [line for line in normalized_nonempty_lines if line]
    list_items = len(_LIST_ITEM_RE.findall(rendered))
    code_fence_count = rendered.count("```")
    structured = bool(list_items >= 2 or (code_fence_count >= 2 and code_fence_count % 2 == 0))
    sentence_count = len(_SENTENCE_END_RE.findall(rendered))
    # Technical prose legitimately chains clauses with semicolons
    # ("cancellation revokes the token; timeouts trip the guard; …") — a
    # live 228-word answer satisfying every requested facet was rejected as
    # underdeveloped because only [.!?] counted as discourse boundaries.
    semicolon_clauses = rendered.count(";")
    discourse_units = max(sentence_count + semicolon_clauses, list_items)
    max_blank_lines = _max_blank_line_run(rendered)
    lexical_yield = len(visible_words) / max(1, generated)

    trigrams = list(
        zip(visible_words, visible_words[1:], visible_words[2:], strict=False)
    )
    trigram_diversity = len(set(trigrams)) / max(1, len(trigrams))
    line_duplication_ratio = (
        1.0 - len(set(normalized_nonempty_lines)) / len(normalized_nonempty_lines)
        if normalized_nonempty_lines
        else 0.0
    )

    facet_evidence = evaluate_facet_coverage(rendered, objective_text)
    requested_facets = list(facet_evidence["requested"])
    satisfied_facets = list(facet_evidence["satisfied"])
    missing_facets = sorted(set(requested_facets) - set(satisfied_facets))
    compound = len(requested_facets) >= 2

    objective_terms: list[str] = []
    for token in objective_words:
        concept = _concept(token)
        if len(concept) >= 4 and token not in _STOPWORDS and concept not in objective_terms:
            objective_terms.append(concept)
        if len(objective_terms) >= _MAX_OBJECTIVE_TERMS:
            break
    answer_concepts = {_concept(token) for token in words}
    matched_objective_terms = [
        term for term in objective_terms if term in answer_concepts
    ]
    objective_coverage = len(matched_objective_terms) / max(1, len(objective_terms))

    listed_subjects = _listed_subjects(objective_text)
    covered_subjects = [
        subject["label"]
        for subject in listed_subjects
        if all(key in answer_concepts for key in subject["keys"])
    ]
    listed_subject_coverage = len(covered_subjects) / max(1, len(listed_subjects))

    minimum_words = 5
    if compound:
        minimum_words = 28 if structured else 48
    elif generated >= 64:
        minimum_words = 16
    if stop == "token_limit" and generated >= 128 and not structured:
        minimum_words = max(minimum_words, math.ceil(generated * 0.25))

    reasons: list[str] = []
    if not rendered.strip():
        reasons.append("empty_output")
    if generated <= 0:
        reasons.append("invalid_generated_token_count")
    if len(words) < minimum_words:
        reasons.append("insufficient_lexical_content")
    if generated >= 64 and not structured and lexical_yield < 0.22:
        reasons.append("low_lexical_yield")
    if max_blank_lines > 2:
        reasons.append("excessive_blank_lines")
    if len(trigrams) >= 24 and trigram_diversity < 0.70:
        reasons.append("repetitive_language")
    if len(normalized_nonempty_lines) >= 4 and line_duplication_ratio > 0.35:
        reasons.append("repeated_lines")
    if compound and discourse_units < 3 and not structured:
        reasons.append("compound_answer_underdeveloped")
    if missing_facets:
        reasons.append("missing_requested_facets")
    if surface["prompt_echo_detected"]:
        reasons.append("prompt_echo_contamination")
    if surface["protocol_artifact_detected"]:
        reasons.append("protocol_artifact_leakage")
    # CP126 a3f8d861. A compound answer cleared this with TWO matched terms
    # and 8% coverage, so a long multi-part response could mention two topic
    # words, ignore almost every requested constraint, and still certify as
    # connected to its objective. A floor that low is not a check.
    #
    # The requirement now scales with how much was asked: a richer objective
    # has to be engaged with proportionally, subject to a bound so that
    # legitimate paraphrase is not punished. Explicit obligations are
    # handled separately and strictly by listed_subjects_uncovered above —
    # this is the backstop for objectives that state no explicit list.
    required_term_matches = min(
        len(objective_terms),
        max(
            _MIN_OBJECTIVE_TERM_MATCHES,
            min(
                _MAX_REQUIRED_TERM_MATCHES,
                math.ceil(len(objective_terms) * _OBJECTIVE_TERM_MATCH_RATIO),
            ),
        ),
    )
    required_coverage = (
        _MIN_OBJECTIVE_COVERAGE
        if len(objective_terms) <= 8
        else _WIDE_OBJECTIVE_COVERAGE
    )
    if compound and (
        len(matched_objective_terms) < required_term_matches
        or objective_coverage < required_coverage
    ):
        reasons.append("objective_disconnected")
    if len(listed_subjects) >= 2 and listed_subject_coverage < 1.0:
        reasons.append("listed_subjects_uncovered")
    terminal_complete = _terminal_complete(rendered)
    if rendered.strip() and not structured and not terminal_complete:
        reasons.append("terminal_fragment")

    return {
        "schema": OUTPUT_QUALITY_SCHEMA,
        "policy": "resident_latent_product_quality_v1",
        "passed": not reasons,
        "text_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "objective_sha256": hashlib.sha256(objective_text.encode("utf-8")).hexdigest(),
        "char_count": len(rendered),
        "word_count": len(visible_words),
        "novel_word_count": len(words),
        "generated_token_count": generated,
        "termination": stop,
        "lexical_yield": round(lexical_yield, 6),
        "sentence_count": sentence_count,
        "list_item_count": list_items,
        "structured_output": structured,
        "max_blank_line_run": max_blank_lines,
        "trigram_diversity": round(trigram_diversity, 6),
        "line_duplication_ratio": round(line_duplication_ratio, 6),
        "terminal_complete": terminal_complete,
        "minimum_word_count": minimum_words,
        "compound_request": compound,
        "requested_facets": requested_facets,
        "satisfied_facets": satisfied_facets,
        "missing_facets": missing_facets,
        "objective_term_count": len(objective_terms),
        "matched_objective_terms": matched_objective_terms,
        "objective_term_coverage": round(objective_coverage, 6),
        # Truncation is disclosed: a coverage ratio computed over a clipped
        # term list describes less of the objective than it appears to.
        "objective_terms_considered": len(objective_terms),
        "objective_terms_truncated": len(objective_terms) >= _MAX_OBJECTIVE_TERMS,
        "listed_subjects": [subject["label"] for subject in listed_subjects],
        "covered_listed_subjects": covered_subjects,
        "listed_subject_coverage": round(listed_subject_coverage, 6),
        "prompt_echo_detected": bool(surface["prompt_echo_detected"]),
        "protocol_artifact_detected": bool(surface["protocol_artifact_detected"]),
        "copied_objective_token_count": int(
            surface["copied_objective_token_count"]
        ),
        "longest_objective_copy_tokens": int(
            surface["longest_objective_copy_tokens"]
        ),
        "objective_echo_ratio": round(float(surface["objective_echo_ratio"]), 6),
        "answer_echo_ratio": round(float(surface["answer_echo_ratio"]), 6),
        "facet_evidence": facet_evidence,
        "reasons": reasons,
    }


__all__ = [
    "OUTPUT_QUALITY_SCHEMA",
    "evaluate_facet_coverage",
    "evaluate_latent_output",
    "request_facets",
]
