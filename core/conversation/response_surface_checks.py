"""Whether a reply's surface is broken: drift, truncation, or code cut short.

Lifted whole out of `response_reliability`, which is 9,000 lines and the
largest single contributor to the module-size budget. Every name taken from
that module is imported at CALL time: it imports this one to re-export these,
and a test that patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from typing import Any


def _is_promise_without_answer(user_message: Any, reply_text: Any) -> bool:
    """True when the whole reply is a promise to answer, not an answer.

    Deliberately narrow: it fires only when the ENTIRE reply is the promise
    AND the user asked for content. "Let me check — the answer is 42."
    carries content; "I'm thinking about it" answers "what are you doing?".
    The failure being caught is emptiness, not politeness.
    """
    from .response_reliability import (
        _ACTIVITY_QUESTION_RE,
        _PROMISE_ONLY_REPLY_RE,
        _REPLY_ABOUT_THE_REPLY_RE,
        _word_count,
    )

    raw = str(reply_text or "").strip()
    if not raw or len(raw) > 240:
        return False
    if _ACTIVITY_QUESTION_RE.search(str(user_message or "")):
        return False
    # Any sign of actual content redeems the reply: a promise that is
    # followed by the answer is just courtesy, not emptiness.
    lowered = raw.lower()
    carries_content = bool(
        re.search(r"\d", raw)
        or re.search(r"\b(?:is|are|was|were|because|means|so that|here'?s)\b", lowered)
        or ":" in raw
    )
    if not carries_content and _PROMISE_ONLY_REPLY_RE.match(raw):
        return True
    return bool(_REPLY_ABOUT_THE_REPLY_RE.search(raw)) and _word_count(raw) <= 40


def _has_pseudo_commitment_status_leak(user_message: Any, reply_text: Any) -> bool:
    from .response_reliability import (
        _PSEUDO_COMMITMENT_STATUS_RE,
        _normalize,
    )

    raw = str(reply_text or "").strip()
    if not raw or not _PSEUDO_COMMITMENT_STATUS_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    if any(marker in prompt for marker in ("last thing you committed", "what did you commit", "recent activity")):
        return False
    return True


def _has_camelcase_internal_jargon(user_message: Any, reply_text: Any) -> bool:
    from .response_reliability import (
        _CAMELCASE_INTERNAL_JARGON_RE,
        _normalize,
        is_operational_status_turn,
        is_practical_diagnostic_turn,
        is_reliability_concern,
        names_any,
    )

    raw = str(reply_text or "").strip()
    if not raw or not _CAMELCASE_INTERNAL_JARGON_RE.search(raw):
        return False
    prompt = _normalize(user_message)
    if (
        is_practical_diagnostic_turn(prompt)
        or is_reliability_concern(prompt)
        or is_operational_status_turn(prompt)
    ):
        return False
    if any(
        marker in prompt
        for marker in (
            "cognitiveengine",
            "cognitive engine",
            "cortex",
            "mind/cognition path",
            "cognition path",
            "cognitive path",
            "desktop route",
            "live desktop route",
            "desktop path",
            "live desktop path",
            "desktop ui path",
            "conversation lane",
            "model lane",
            "what path are you using",
            "path are you using right now",
        )
    ):
        return False
    if names_any(prompt, ("architecture", "system", "kernel", "runtime", "code", "debug", "log")):
        return False
    allowed = {"OpenAI", "ChatGPT", "YouTube", "GitHub", "JavaScript"}
    allowed.update(match.group(0) for match in _CAMELCASE_INTERNAL_JARGON_RE.finditer(str(user_message or "")))
    return any(match.group(0) not in allowed for match in _CAMELCASE_INTERNAL_JARGON_RE.finditer(raw))


def _has_unrequested_pop_culture_intrusion(user_message: Any, reply_text: Any) -> bool:
    from .response_reliability import (
        _UNREQUESTED_POP_CULTURE_INTRUSION_RE,
    )

    raw = str(reply_text or "")
    if not _UNREQUESTED_POP_CULTURE_INTRUSION_RE.search(raw):
        return False
    return not _UNREQUESTED_POP_CULTURE_INTRUSION_RE.search(str(user_message or ""))


def _has_unexpected_cjk_intrusion(user_message: Any, reply_text: Any) -> bool:
    from .response_reliability import (
        _CJK_INTRUSION_RE,
        _SCRIPT_SUBJECT_MARKERS,
        names_any,
    )

    raw = str(reply_text or "")
    # Inside a fence the characters are data — a test string, a sample input,
    # the very thing a question about CJK handling has to show. Judging them
    # as an intrusion rejected the answer the question asked for.
    prose = "\n".join(raw.split("```")[::2]) if "```" in raw else raw
    if not _CJK_INTRUSION_RE.search(prose):
        return False
    asked = str(user_message or "")
    if _CJK_INTRUSION_RE.search(asked):
        return False
    return not names_any(asked, _SCRIPT_SUBJECT_MARKERS)


def _has_surface_nonsense_drift(user_message: Any, reply_text: Any) -> bool:
    from .response_reliability import (
        _SURFACE_NONSENSE_DRIFT_RE,
    )

    raw = str(reply_text or "")
    # Source URLs are expected in grounded search/tool answers.  The legacy
    # drift pattern includes ``:/`` to catch malformed emotive fragments, which
    # would otherwise make every ``https://`` citation look like nonsense.
    raw_without_urls = re.sub(r"https?://\S+", "", raw)
    prompt_without_urls = re.sub(r"https?://\S+", "", str(user_message or ""))
    if not _SURFACE_NONSENSE_DRIFT_RE.search(raw_without_urls):
        return False
    return not _SURFACE_NONSENSE_DRIFT_RE.search(prompt_without_urls)


def _looks_like_structured_output(body: str) -> bool:
    """Code, JSON and list-shaped answers are legitimately function-word poor."""
    from .response_reliability import (
        _LABELLED_LINE_RE,
        _LIST_LINE_RE,
    )

    if "```" in body or "|---" in body:
        return True
    stripped = body.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        return True
    lines = [line for line in body.splitlines() if line.strip()]
    if lines and sum(1 for line in lines if _LIST_LINE_RE.match(line)) * 2 >= len(lines):
        return True
    # Scripted dialogue and labelled records ("Mainframe: First statement.")
    # are function-word poor by form, not by collapse. The discriminator is the
    # line: genuine labelled text puts each label on its own, which is exactly
    # what a single run-on line of "Introspection: ... CONFORMANCE Signal: ..."
    # does not do.
    if len(lines) >= 2:
        labelled = sum(1 for line in lines if _LABELLED_LINE_RE.match(line))
        if labelled >= 2 and labelled * 2 >= len(lines):
            return True
    return False


def _has_function_word_starvation(reply_text: Any) -> bool:
    from core.language.what_a_watcher_hears import in_her_own_words

    from .response_reliability import (
        _FUNCTION_WORDS,
        _MIN_FUNCTION_WORD_RATIO,
        _MIN_PROSE_WORDS_FOR_FUNCTION_TEST,
        _PROSE_WORD_RE,
    )

    body = str(reply_text or "").strip()
    if not body or _looks_like_structured_output(body):
        return False
    # Writing she quotes is someone else's words; a screen's labels read out
    # are short of small words by nature, and her prose is judged without them.
    prose = re.sub(r"`[^`]*`", " ", in_her_own_words(body))
    # Identifiers, hashes and telemetry blobs are not prose in either
    # direction. Left in, "[x_A_4521B_8A7C]" contributed two tokens that look
    # exactly like the article "a" and pushed a starved reply back over the
    # threshold on noise alone.
    prose = re.sub(r"\S*[\d_]\S*", " ", prose)
    words = [word.lower() for word in _PROSE_WORD_RE.findall(prose)]
    if len(words) < _MIN_PROSE_WORDS_FOR_FUNCTION_TEST:
        return False
    ratio = sum(1 for word in words if word in _FUNCTION_WORDS) / len(words)
    return ratio < _MIN_FUNCTION_WORD_RATIO


def _is_structured_payload(body: str) -> bool:
    """Whether this is a machine-readable object rather than prose.

    Only a complete JSON object or array counts. A reply that merely contains
    a brace is prose with a brace in it, and treating it as structured would
    hand every prose detector an escape hatch.
    """
    text = body.strip()
    if not text or text[0] not in "{[":
        return False
    try:
        import json

        json.loads(text)
    except (ValueError, TypeError):
        # not a failure: text that will not parse is not JSON, which is the question this answers.
        return False
    return True


def _terminal_word_is_a_unit(body: str, terminal_start: int) -> bool:
    """True when the short word ending the reply is a unit on a number.

    LIVE, 2026-08-20. "The temperature reported by the API is 11.6°C." was
    refused as a truncated tail and the person got "I couldn't get to an
    answer I'd stand behind on that one." The answer was right, and the rule
    was right in general: a reply ending in a one- or two-letter word is
    usually cut mid-word. A unit is the exception, and it is not a vocabulary
    question — °C, km, m, ft, kg, Hz and every unit nobody has thought of are
    identified by the number they attach to.
    """
    from .response_reliability import (
        _MEASUREMENT_LEAD_RE,
    )

    return bool(_MEASUREMENT_LEAD_RE.search(body[:terminal_start]))


def _has_truncated_tail(
    reply_text: Any,
    *,
    generation_stop_reason: Any = "",
) -> bool:
    from .response_reliability import (
        _ALLOWED_SHORT_TAIL_WORDS,
        _BARE_NUMERIC_RANGE_TAIL_RE,
        _DANGLING_FUNCTION_WORD_TAIL_RE,
        _DANGLING_GERUND_TAIL_RE,
        _INCOMPLETE_TAIL_WORDS,
        _LIST_LINE_RE,
        _PUNCTUATED_INCOMPLETE_TAIL_RE,
        _STRUCTURAL_INCOMPLETE_TAIL_RE,
        _STRUCTURAL_UNPUNCTUATED_TAIL_RE,
        _word_count,
        has_terminal_sentence_boundary,
        terminal_content,
        unfulfilled_commitments,
    )

    body = str(reply_text or "").strip()
    if unfulfilled_commitments(body):
        return True
    # Grammar first, length second. A sentence left hanging on a conjunction
    # is cut whether it is 23 characters or 230, and the floor below is about
    # not demanding punctuation from a legitimately terse reply — a different
    # question that was silently answering this one.
    if len(body.split()) >= 2 and _DANGLING_FUNCTION_WORD_TAIL_RE.search(body):
        return True
    if len(body) < 24:
        return False
    straight_quote_positions = [
        match.start() for match in re.finditer(r'(?<!\\)"', body)
    ]
    if len(straight_quote_positions) % 2:
        unmatched_position = straight_quote_positions[-1]
        preceding = body[unmatched_position - 1] if unmatched_position else ""
        # Preserve ordinary inch/second notation such as 6" while rejecting
        # prose that opens a quotation and never closes it.
        if not preceding.isdigit():
            return True
    if body.count("“") != body.count("”"):
        return True
    if re.search(r'(?<!\d)[.!?]["”’)]?\s*\d+[.)]\s*$', body):
        return True
    if _STRUCTURAL_INCOMPLETE_TAIL_RE.search(body):
        return True
    if _STRUCTURAL_UNPUNCTUATED_TAIL_RE.search(body):
        return True
    if _DANGLING_GERUND_TAIL_RE.search(body):
        return True
    if _PUNCTUATED_INCOMPLETE_TAIL_RE.search(body):
        return True
    if (
        len(body) >= 80
        and _word_count(body) >= 12
        and not has_terminal_sentence_boundary(body)
        and _BARE_NUMERIC_RANGE_TAIL_RE.search(body)
    ):
        return True
    terminal_word_match = re.search(r"([A-Za-z]+)[.!?\"'”’)\]]*$", body)
    if terminal_word_match and len(body) >= 40:
        terminal_word = terminal_word_match.group(1).lower()
        terminal_start = terminal_word_match.start(1)
        possessive_suffix = (
            terminal_word == "s"
            and terminal_start > 0
            and body[terminal_start - 1] in ("'", "’")
        )
        if (
            len(terminal_word) <= 2
            and terminal_word not in _ALLOWED_SHORT_TAIL_WORDS
            and not possessive_suffix
            and not _terminal_word_is_a_unit(body, terminal_start)
        ):
            return True
    if body.endswith(("...", "…")):
        return True
    if re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s*$", body):
        return True
    # Ending by asking its own question, at exactly the point the budget ran
    # out, is a thought that stopped rather than a turn that finished. It is
    # grammatically whole, so the boundary check below passes it — LIVE
    # 2026-08-30, asked whether "decided" is doing any work, she made the
    # case, raised "So, is it just the most fluent token?" and stopped there.
    #
    # Only where something says it ran out. Ending by asking the person
    # something is an ordinary way to finish a turn, and refusing those would
    # cost far more than this saves.
    if str(generation_stop_reason or "").strip().lower() not in {
        "",
        "eos",
        "configured_stop",
        "role_continuation",
    } and body.endswith(("?", '?"', "?'", "?)")):
        return True
    if has_terminal_sentence_boundary(body):
        return False
    trailing_block = None
    if re.search(r"(?:^|\n)\s*\d+\.\s+\S+", body) or re.search(r"\*\*[^*\n]{2,80}:\*\*", body):
        # A structured answer legitimately ends on its last item with no full
        # stop. This branch used to flag every one of them, so a well-formatted
        # worked answer — numbered steps, or a "**Both Red:**" heading over
        # bullets — was rejected as clipped no matter how complete it was.
        # Live 2026-07-26 that turned a correct marble derivation into "I
        # couldn't get to an answer I'd stand behind".
        #
        # What actually indicates a cut is ending mid-structure: on a bare
        # marker, on a heading with nothing under it, or on a fragment.
        structured_lines = [line for line in body.splitlines() if line.strip()]
        last_line = structured_lines[-1].strip() if structured_lines else ""
        marker_match = _LIST_LINE_RE.match(last_line)
        ends_on_complete_item = bool(
            marker_match and len((marker_match.group("body") or "").split()) >= 3
        )
        ends_on_bare_heading = bool(re.fullmatch(r"\*\*[^*\n]{2,80}:\*\*", last_line))
        # …and on inconsistency. If the earlier items in this list close with a
        # full stop and the final one does not, the list was cut, whatever the
        # last item looks like on its own. A list that never punctuates its
        # items is simply written that way.
        earlier_items = [
            match.group("body").strip()
            for match in (
                _LIST_LINE_RE.match(line) for line in structured_lines[:-1]
            )
            if match and (match.group("body") or "").strip()
        ]
        punctuated = [item for item in earlier_items if item.endswith((".", "!", "?"))]
        inconsistent_tail = bool(
            len(earlier_items) >= 2
            and len(punctuated) * 2 >= len(earlier_items)
            and marker_match
        )
        if ends_on_bare_heading or (
            marker_match and (not ends_on_complete_item or inconsistent_tail)
        ):
            return True
        if not marker_match:
            # A footer or concluding paragraph is not another list item.
            # Judge its own structure, without inheriting the preceding list's
            # punctuation or hiding clipped prose behind that list's shape.
            last_item = next(
                (
                    index
                    for index in range(len(structured_lines) - 2, -1, -1)
                    if _LIST_LINE_RE.match(structured_lines[index])
                ),
                None,
            )
            if last_item is not None:
                trailing_block = "\n".join(structured_lines[last_item + 1 :])
    unwrapped_body = terminal_content(body)
    if unwrapped_body.endswith(("-", "—", ":", ";", ",")):
        return True
    match = re.search(r"([A-Za-z]+)$", unwrapped_body)
    if not match:
        return False
    last_word = match.group(1).lower()
    if len(last_word) <= 2 and len(body) >= 40:
        return True
    if last_word in _INCOMPLETE_TAIL_WORDS:
        return True
    if trailing_block is not None:
        return _has_truncated_tail(
            trailing_block, generation_stop_reason=generation_stop_reason
        )
    # Prose that simply stops. Everything above looks for a SUSPICIOUS last
    # word — a dangling conjunction, a two-letter fragment — so a reply cut off
    # on an ordinary noun read as finished.
    #
    # Live 2026-07-26: "…we need to consider each case separately: Both Red"
    # was served as a complete answer, and assessed ok. It was a correct
    # derivation truncated at 239 tokens, and "Red" is not a suspicious word.
    # A reply of real length that ends on any ordinary word with no terminal
    # punctuation was cut, not finished.
    #
    # Prose only. A list, a table or a worked derivation legitimately ends on
    # its last item with no full stop, and flagging those turned a mostly
    # complete answer into a refusal — which is a worse outcome than the
    # clipped tail it was trying to prevent. The repair path in the worker
    # handles list-shaped clipping by dropping the final item instead.
    if _looks_like_structured_output(body):
        return False
    terminal_cause = str(generation_stop_reason or "").strip().lower()
    if terminal_cause in {"eos", "configured_stop", "role_continuation"}:
        return False
    return len(body) >= 80 and _word_count(body) >= 12


def _is_code_response(text: str) -> bool:
    from .response_reliability import (
        _CODE_FENCE_LANGS,
        _FENCED_BLOCK_RE,
        _LATEX_MATH_RE,
        _NON_CODE_FENCE_LANGS,
    )

    raw = str(text or "").strip()
    if not raw:
        return False
    fenced_blocks = list(_FENCED_BLOCK_RE.finditer(raw))
    # Unfenced maths is prose about numbers, not a code response. With a fence
    # present the block's own language wins, below — a code sample is allowed
    # to sit beside an equation.
    if not fenced_blocks and _LATEX_MATH_RE.search(raw):
        return False
    if fenced_blocks:
        for block in fenced_blocks:
            lang = (block.group("lang") or "").strip().lower()
            body = block.group("body") or ""
            if lang in _CODE_FENCE_LANGS or (lang in _NON_CODE_FENCE_LANGS and _looks_like_code_body(body)):
                return True
        return False
    if raw.startswith(("def ", "import ", "class ", "from ", "print(", "#", "var ", "const ", "let ", "function ")):
        return True

    # One implementation of "does this look like code", not two.
    #
    # This used to carry its own inline copy of the same heuristic — any line
    # containing "=", or a matched pair of brackets, counted as code. Fixing
    # that in _looks_like_code_body left this copy untouched, and the
    # consequence was worse than the original bug: classifying prose as code
    # SHORT-CIRCUITS every prose check above, so a truncated answer was served
    # as complete. Live 2026-07-26:
    #
    #   "Total number of marbles: 3 red + 4 blue + 5 green = 12
    #    2. Draw two without replacement means the probability changes…
    #    We need to calculate P(both red) + P(both blue) + P(both green)
    #    Calculating for"
    #
    # — assessed ok, truncated mid-word, because "=" and "(...)" made it code
    # and code is exempt from truncated_tail and final_answer_missing.
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 2 and _looks_like_code_body(raw):
        return True

    return False


def _looks_like_code_body(text: Any) -> bool:
    from .response_reliability import (
        _CODE_ASSIGNMENT_RE,
        _CODE_CALL_RE,
    )

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    code_like_lines = 0
    for line in lines:
        if (
            line.startswith(
                (
                    "def ",
                    "import ",
                    "class ",
                    "from ",
                    "return ",
                    "if ",
                    "elif ",
                    "else:",
                    "for ",
                    "while ",
                    "try:",
                    "except",
                    "with ",
                    "#",
                    "print(",
                    "const ",
                    "let ",
                    "var ",
                    "function ",
                )
            )
            or _CODE_ASSIGNMENT_RE.match(line)
            or _CODE_CALL_RE.match(line)
            or line.endswith((";", "{", "}", "):", "->"))
        ):
            code_like_lines += 1
    threshold = 0.5 if len(lines) <= 3 else 0.6
    return code_like_lines / len(lines) >= threshold


def _has_incomplete_code_response(text: Any) -> bool:
    from .response_reliability import (
        _FENCED_BLOCK_RE,
        _INCOMPLETE_CODE_TAIL_RE,
    )

    raw = str(text or "").strip()
    if not raw:
        return False
    if raw.count("```") % 2:
        return True

    blocks = list(_FENCED_BLOCK_RE.finditer(raw))
    bodies = [block.group("body") or "" for block in blocks] if blocks else [raw]
    for body in bodies:
        if not _looks_like_code_body(body):
            continue
        lines = [line.rstrip() for line in body.splitlines() if line.strip()]
        if not lines:
            continue
        last = lines[-1].strip()
        if _INCOMPLETE_CODE_TAIL_RE.search(last):
            return True
    return False
