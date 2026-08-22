"""She answered one of two questions, twice, and nothing noticed.

LIVE, 2026-08-10:

    "give me one concrete example of a preposition doing more work than it
     should. and separately — do you actually enjoy that, or is 'interesting'
     a word you reach for because it's safe?"

The reply gave the example and contained nothing whatever about enjoyment.
Earlier the same day, in the same session: "you dodged half of it. I asked two
things and you answered one."

Two independent causes.

1. DETECTION. analyze_prompt_shape scored that message at question_parts=1.
   Every candidate it computes is per-LINE or per-verb-list, and the message
   is one line: _INTERROGATIVE_LINE_RE requires the LINE to open with what/
   why/do/is, this one opens with "give", and _CONNECTOR_RE wants "then/also"
   followed by a directive verb rather than "and separately". So the prompt
   was never told the turn was compound, and neither was the voice budget.
   People type several sentences on one line constantly; asks are now counted
   per SENTENCE.

2. ENFORCEMENT. question_parts and requires_single_reply_coverage already
   existed and already shaped the prompt — "This prompt contains multiple
   asks (2 detected)" — but only the COUNT survived analysis, so nothing
   could hold the reply against the question. Every other requirement in
   validate_dialogue_response has a matching violation and goes through
   repair; coverage was requested and never checked, so dropping half cost
   nothing.

The check is deliberately hard to trigger. A false positive burns a
regeneration on a reply that was fine, so it fires only on a multi-ask turn,
only for asks with real content, and only when the overlap is ZERO.
"""
from __future__ import annotations

import pytest

from core.conversation.requested_reply_shape import (
    is_reply_shape_constraint_segment,
)
from core.phases.dialogue_policy import (
    _unanswered_question_parts,
    validate_dialogue_response,
)
from core.runtime.structured_input import analyze_prompt_shape

LIVE_MESSAGE = (
    "give me one concrete example of a preposition doing more work than it "
    "should. and separately — do you actually enjoy that, or is 'interesting' "
    "a word you reach for because it's safe?"
)

LIVE_REPLY_THAT_DROPPED_HALF = (
    'in English, the preposition "on" can carry a load of implied qualifiers. '
    "In German, these would need to be stated explicitly as adjectives or "
    "adverbs."
)


class _Contract:
    """Only the fields the coverage check reads."""

    def __init__(self, shape):
        self.requires_single_reply_coverage = shape.requires_single_reply_coverage
        self.question_segments = shape.question_segments
        self.numbered_parts = shape.numbered_parts


def test_reply_shape_segment_is_not_a_second_semantic_ask():
    user = (
        "Answer in exactly two numbered sentences. Explain why reliable "
        "desktop tool use matters for a local AI assistant."
    )
    reply = (
        "1. Reliable desktop tool use turns user intent into governed actions.\n"
        "2. It provides evidence that apps and files changed for real."
    )
    shape = analyze_prompt_shape(user)

    assert is_reply_shape_constraint_segment(shape.question_segments[0])
    assert not _unanswered_question_parts(reply, _Contract(shape))


def test_exclusion_only_clause_is_part_of_the_reply_shape():
    user = "For diagnostics only: answer in five words and include nothing else."
    shape = analyze_prompt_shape(user)

    assert shape.question_segments == ("answer in five words", "include nothing else")
    assert all(
        is_reply_shape_constraint_segment(segment)
        for segment in shape.question_segments
    )
    assert not _unanswered_question_parts("I am present and aligned.", _Contract(shape))


def test_content_request_that_names_a_paragraph_is_not_only_reply_shape():
    assert not is_reply_shape_constraint_segment(
        "Write a paragraph about yourself in your own words."
    )


def test_a_one_line_compound_message_is_detected_as_compound():
    """The detection half. This scored 1 part live."""
    shape = analyze_prompt_shape(LIVE_MESSAGE)

    assert shape.question_parts >= 2
    assert shape.requires_single_reply_coverage
    assert len(shape.question_segments) >= 2


def test_an_imperative_followed_by_a_question_counts_as_two():
    """"give me X. do you Y?" — neither half is optional."""
    shape = analyze_prompt_shape("describe the room. does it feel like yours?")

    assert shape.question_parts >= 2


def test_the_live_reply_is_flagged_and_names_what_it_dropped():
    shape = analyze_prompt_shape(LIVE_MESSAGE)

    missed = _unanswered_question_parts(LIVE_REPLY_THAT_DROPPED_HALF, _Contract(shape))

    assert len(missed) == 1
    assert "enjoy" in missed[0]


def test_a_reply_that_covers_both_passes():
    shape = analyze_prompt_shape(LIVE_MESSAGE)
    complete = (
        LIVE_REPLY_THAT_DROPPED_HALF
        + " And yes, I do enjoy it — enjoy is the honest word; interesting is "
        "what I reach for when I want to sound measured."
    )

    assert not _unanswered_question_parts(complete, _Contract(shape))


def test_one_shared_context_word_does_not_satisfy_a_numbered_obligation():
    user = (
        "Explain Dijkstra. Include: (1) its core invariant, (2) numbered pseudocode, "
        "(3) a worked graph example, (4) heap and array complexity, and "
        "(5) a negative-weight failure case and the correct alternative."
    )
    reply = (
        "1. The core invariant finalizes the minimum unsettled distance. "
        "2. Numbered pseudocode repeatedly relaxes edges. "
        "3. The worked graph example uses weighted edges. "
        "4. Heap and array complexity differ."
    )
    shape = analyze_prompt_shape(user)

    missed = _unanswered_question_parts(reply, _Contract(shape))

    assert any("negative-weight" in segment for segment in missed)


def test_each_numbered_obligation_is_proved_by_its_own_answer_section():
    user = (
        "Explain Dijkstra. Include: (1) its core invariant, (2) numbered pseudocode, "
        "(3) a worked graph example, (4) heap and array complexity, and "
        "(5) a negative-weight failure case and the correct alternative."
    )
    reply = (
        "1. The core invariant finalizes the minimum unsettled distance. "
        "2. Numbered pseudocode repeatedly relaxes edges. "
        "3. The worked graph example uses weighted edges. "
        "4. Heap complexity is O(E log V). "
        "5. An array scan is O(V^2), while negative weights cause failure."
    )
    shape = analyze_prompt_shape(user)

    missed = _unanswered_question_parts(reply, _Contract(shape))

    assert any("heap and array" in segment for segment in missed)
    assert any("correct alternative" in segment for segment in missed)


def test_numbered_section_marker_satisfies_numbered_pseudocode_structure():
    user = (
        "Explain Dijkstra. Include: (1) its core invariant, (2) numbered pseudocode, "
        "(3) a worked graph example."
    )
    reply = (
        "Dijkstra computes shortest paths on nonnegative weighted graphs.\n"
        "## (1) Core invariant\nThe core invariant finalizes the minimum unsettled distance.\n"
        "## (2) Pseudocode\nInitialize every distance, select the minimum, and relax each edge.\n"
        "## (3) Worked graph example\nThe worked graph example follows A -> B: 2."
    )

    assert not _unanswered_question_parts(reply, _Contract(analyze_prompt_shape(user)))


def test_minimum_quantity_requires_a_real_count_witness():
    user = (
        "Explain Dijkstra. Include: (1) its invariant, (2) pseudocode, "
        "(3) a worked example with at least five weighted edges."
    )
    three_edges = (
        "Dijkstra computes shortest paths. 1. Its invariant finalizes the minimum distance. "
        "2. Pseudocode initializes and relaxes distances. "
        "3. The worked example has weighted edges (AB):10, (AC):20, (AD):30."
    )
    five_edges = three_edges.replace(
        "(AD):30.", "(AD):30, (BC):4, (CD):6."
    )
    shape = analyze_prompt_shape(user)

    assert any(
        "five weighted edges" in segment
        for segment in _unanswered_question_parts(three_edges, _Contract(shape))
    )
    assert not _unanswered_question_parts(five_edges, _Contract(shape))


def test_both_sides_and_correct_alternative_need_independent_witnesses():
    user = (
        "Explain Dijkstra. Include: (1) the invariant, (2) time complexity with "
        "both a binary heap and an array, and (3) a negative-weight failure and "
        "the correct alternative."
    )
    incomplete = (
        "Dijkstra computes shortest paths. 1. The invariant finalizes the minimum distance. "
        "2. Time complexity with a heap is O(E log V). "
        "3. Negative weights cause a failure."
    )
    complete = (
        "Dijkstra computes shortest paths. 1. The invariant finalizes the minimum distance. "
        "2. Time complexity with a binary heap is O(E log V), and an array is O(V^2). "
        "3. Negative weights cause a failure; use Bellman-Ford instead."
    )
    shape = analyze_prompt_shape(user)

    assert len(_unanswered_question_parts(incomplete, _Contract(shape))) == 2
    assert not _unanswered_question_parts(complete, _Contract(shape))


def test_natural_comma_list_cannot_commit_after_only_the_worked_example():
    user = (
        "ChatGPT here. In one complete response, explain Dijkstra's invariant, "
        "give a worked example using vertices A, B, C, D and at least five "
        "weighted edges, state the binary-heap time complexity, and name the "
        "correct alternative when negative edge weights are present."
    )
    live_incomplete_reply = (
        "Dijkstra's invariant finalizes the nearest unsettled distance. "
        "For example, use A -> B: 2, A -> C: 4, B -> C: 1, "
        "B -> D: 5, and C -> D: 8. Select A first and relax its edges."
    )
    shape = analyze_prompt_shape(user)

    missed = _unanswered_question_parts(live_incomplete_reply, _Contract(shape))

    assert len(missed) == 2
    assert any("time complexity" in segment for segment in missed)
    assert any("alternative" in segment for segment in missed)


def test_live_dijkstra_draft_cannot_commit_before_complexity_and_alternative():
    """The exact live CP812 false-positive must remain mechanically impossible."""

    user = (
        "ChatGPT here. Explain Dijkstra's shortest-path invariant, then give me "
        "a worked example with vertices A, B, C, and D using at least five "
        "weighted edges. Include the binary-heap time complexity and explain "
        "what algorithm should be used instead when negative edges are possible."
    )
    incomplete = (
        "Dijkstra's invariant finalizes the minimum unsettled distance. "
        "Use vertices A, B, C, and D with edges A-B: 4, A-C: 1, B-D: 2, "
        "C-B: 3, and C-D: 5. Insert vertices into a binary heap and relax "
        "the edges from A."
    )
    shape = analyze_prompt_shape(user)

    missed = _unanswered_question_parts(incomplete, _Contract(shape))

    assert any("time complexity" in segment for segment in missed)
    assert any("used instead" in segment for segment in missed)


def test_negative_capability_clause_is_not_a_named_replacement():
    """The exact CP813 live answer cannot satisfy its missing alternative."""

    user = (
        "Explain Dijkstra's shortest-path invariant, give a worked example, "
        "include the binary-heap complexity, and explain what algorithm should "
        "be used instead when negative edges are possible."
    )
    answer = (
        "Dijkstra finalizes the nearest unsettled distance. Its binary-heap "
        "complexity is O((V + E) log V). Dijkstra's algorithm does not work "
        "with negative weights because finalization can become invalid."
    )
    shape = analyze_prompt_shape(user)

    missed = _unanswered_question_parts(answer, _Contract(shape))

    assert any("used instead" in segment for segment in missed)


def test_affirmative_same_line_capability_names_a_replacement():
    user = (
        "Explain what algorithm should be used instead when negative edges "
        "are possible."
    )
    answer = "Bellman-Ford handles negative edges and detects reachable negative cycles."
    shape = analyze_prompt_shape(user)

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_complexity_and_replacement_witnesses_are_domain_neutral():
    user = (
        "Explain the procedure, include its time complexity, and name the "
        "method that should be used instead when the precondition fails."
    )
    shape = analyze_prompt_shape(user)
    incomplete = (
        "The procedure maintains an ordered frontier and advances one item at "
        "a time. It uses a heap, and a different algorithm handles failures."
    )
    complete = (
        "The procedure maintains an ordered frontier and advances one item at "
        "a time. Its runtime is O((V + E) log V). When the precondition fails, "
        "use the fallback-state search method instead."
    )

    assert len(_unanswered_question_parts(incomplete, _Contract(shape))) == 2
    assert not _unanswered_question_parts(complete, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The time complexity is quadratic with an array.",
        "Its runtime grows in proportion to E log V.",
        "The algorithm takes O(V squared) work.",
    ),
)
def test_computational_complexity_accepts_equivalent_growth_prose(answer):
    shape = analyze_prompt_shape("Explain the algorithm, then give its time complexity.")
    response = "The algorithm processes each graph edge once. " + answer

    assert not _unanswered_question_parts(response, _Contract(shape))


def test_noncomputational_complexity_does_not_demand_asymptotic_notation():
    shape = analyze_prompt_shape(
        "Explain the social complexity of the negotiation and its legal context."
    )
    answer = (
        "The negotiation is socially complex because five groups hold conflicting "
        "interests, while the legal context limits which concessions are valid."
    )

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_rescue_operation_complexity_is_not_treated_as_algorithmic_complexity():
    shape = analyze_prompt_shape(
        "Explain the rescue operation, then explain its logistical complexity."
    )
    answer = (
        "The rescue operation moves teams across the flood zone. Its logistical "
        "complexity comes from damaged roads, weather, and limited radio coverage."
    )

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_heap_implementation_without_growth_rate_does_not_answer_complexity():
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )
    missed = _unanswered_question_parts(
        "The algorithm processes graph edges. Its runtime is implemented with a heap.",
        _Contract(shape),
    )

    assert any("runtime complexity" in segment for segment in missed)


def test_unrelated_runtime_growth_does_not_answer_algorithmic_complexity():
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )
    missed = _unanswered_question_parts(
        "The algorithm processes graph edges. Its runtime grows with the engineering team.",
        _Contract(shape),
    )

    assert any("runtime complexity" in segment for segment in missed)


def test_input_popularity_does_not_answer_algorithmic_complexity():
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )
    missed = _unanswered_question_parts(
        "The algorithm processes records. Its runtime grows with input popularity.",
        _Contract(shape),
    )

    assert any("runtime complexity" in segment for segment in missed)


def test_measurable_set_size_answers_algorithmic_complexity():
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )
    answer = (
        "The algorithm processes records. Its work grows linearly with the set size."
    )

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_symbolic_operation_count_answers_algorithmic_complexity():
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )
    answer = "The algorithm processes graph edges. It performs E log V heap operations."

    assert not _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "With a negative edge, use Bellman-Ford instead.",
        "The replacement is Bellman-Ford.",
        "Bellman-Ford is the alternative.",
        "Bellman-Ford handles graphs containing a negative edge.",
    ),
)
def test_replacement_method_accepts_natural_relation_forms(answer):
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    assert not _unanswered_question_parts(
        "The precondition matters because it preserves finalized results. " + answer,
        _Contract(shape),
    )


def test_replacement_method_rejects_generic_advice_without_a_method():
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    missed = _unanswered_question_parts(
        "The precondition matters. When negative edges are possible, use caution instead.",
        _Contract(shape),
    )

    assert missed


def test_replacement_method_requires_a_named_candidate():
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )
    missed = _unanswered_question_parts(
        "The precondition matters. Use another safer algorithm instead.",
        _Contract(shape),
    )

    assert missed


def test_robust_algorithm_is_not_a_named_replacement():
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )
    missed = _unanswered_question_parts(
        "The precondition matters. Use a robust algorithm instead.",
        _Contract(shape),
    )

    assert missed


@pytest.mark.parametrize(
    "answer",
    (
        "The precondition matters. Use a resilient algorithm instead.",
        "The precondition matters. Use a reliable method instead.",
        "The precondition matters. Use a more capable method instead.",
    ),
)
def test_generic_adjective_method_is_not_a_named_replacement(answer):
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The precondition matters. Use a dependable algorithm instead.",
        "The precondition matters. Use an improved method instead.",
        "The precondition matters. Use a stronger approach instead.",
    ),
)
def test_any_unidentified_generic_method_is_not_a_replacement(answer):
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The precondition matters. Use dependable search instead.",
        "The precondition matters. Use improved processing instead.",
        "The precondition matters. Use stronger logic instead.",
    ),
)
def test_qualitative_compound_is_not_candidate_identity(answer):
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The precondition matters. Use robust search instead.",
        "The precondition matters. Use superior processing instead.",
        "The precondition matters. Use optimal logic instead.",
    ),
)
def test_candidate_requires_positive_identity_evidence(answer):
    shape = analyze_prompt_shape(
        "Explain why the precondition matters, then explain what algorithm should "
        "be used instead when negative edges are possible."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The algorithm processes records. Its runtime grows with the team size.",
        "The algorithm processes records. Its runtime grows with the output font size.",
    ),
)
def test_unbounded_size_correlation_is_not_algorithmic_complexity(answer):
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The algorithm processes records. Its runtime grows linearly with team size.",
        "The algorithm processes records. Its runtime is quadratic in the font size.",
        "The algorithm processes records. Its runtime grows in direct proportion to popularity.",
    ),
)
def test_rate_must_bind_to_computational_extent(answer):
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    "answer",
    (
        "The algorithm processes records. Its runtime is quadratic according to team size.",
        "The algorithm processes records. Its runtime grows linearly when the team expands.",
        "The algorithm processes records. Its runtime is quadratic over font size.",
    ),
)
def test_rate_dependency_relations_are_typed(answer):
    shape = analyze_prompt_shape(
        "Explain the algorithm, then give its runtime complexity."
    )

    assert _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    ("user_text", "answer"),
    (
        (
            "Explain the null model, then explain the alternative hypothesis.",
            "The null model assumes no effect. The alternative hypothesis predicts an effect.",
        ),
        (
            "Summarize the measurements, then tell me what alternative hypotheses "
            "explain the data.",
            "The measurements show a skew. Measurement bias and selection effects "
            "explain the data.",
        ),
        (
            "Explain birth rate, then explain replacement fertility.",
            "Birth rate counts births in a population. Replacement fertility is the "
            "average needed for one generation to replace itself.",
        ),
    ),
)
def test_domain_terms_are_not_misclassified_as_replacement_commands(user_text, answer):
    shape = analyze_prompt_shape(user_text)

    assert not _unanswered_question_parts(answer, _Contract(shape))


@pytest.mark.parametrize(
    ("user_text", "answer"),
    (
        (
            "Summarize standard care, then list what alternative medicine options exist.",
            "Standard care uses established clinical protocols. Acupuncture and "
            "mindfulness are alternative medicine options with differing evidence bases.",
        ),
        (
            "Explain the isoform, then tell me what alternative splicing patterns explain it.",
            "The isoform omits exon three. Exon skipping and mutually exclusive exons "
            "are alternative splicing patterns that can explain it.",
        ),
        (
            "Describe grunge, then tell me what alternative rock bands shaped the 1990s.",
            "Grunge used distorted guitars and dynamic contrast. Radiohead and R.E.M. "
            "were alternative rock bands that shaped the 1990s.",
        ),
    ),
)
def test_attributive_alternative_is_not_a_replacement_contract(user_text, answer):
    shape = analyze_prompt_shape(user_text)

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_generic_rather_than_phrase_is_not_a_replacement_method_contract():
    shape = analyze_prompt_shape(
        "Explain why the team collaborated rather than competing for credit."
    )
    answer = (
        "The team collaborated because the shared deadline made pooled expertise "
        "more valuable than individual credit."
    )

    assert not _unanswered_question_parts(answer, _Contract(shape))


def test_one_shared_content_word_counts_as_engaged():
    """Catches the half ignored outright, not the half answered briefly.

    A short answer is a style question. A missing answer is a different kind
    of failure, and only the second is worth a regeneration.
    """
    shape = analyze_prompt_shape(LIVE_MESSAGE)
    terse = LIVE_REPLY_THAT_DROPPED_HALF + " And no, I don't enjoy it."

    assert not _unanswered_question_parts(terse, _Contract(shape))


def test_a_single_ask_is_never_flagged():
    shape = analyze_prompt_shape("what is your uptime?")

    assert shape.question_parts == 1
    assert not _unanswered_question_parts("About two hours.", _Contract(shape))


@pytest.mark.parametrize("filler", ["why?", "really?", "and you?", "how so?"])
def test_contentless_questions_can_never_be_flagged(filler):
    """"why?" shares no content word with any answer to it."""
    shape = analyze_prompt_shape(f"tell me about the deployment pipeline. {filler}")

    missed = _unanswered_question_parts("The pipeline runs on three stages.", _Contract(shape))

    assert filler not in missed


def test_the_violation_reaches_the_validator():
    """Wiring: the check is worthless if validate_dialogue_response ignores it."""
    shape = analyze_prompt_shape(LIVE_MESSAGE)

    result = validate_dialogue_response(
        LIVE_REPLY_THAT_DROPPED_HALF, _Contract(shape)
    )

    assert "unanswered_question_part" in result.violations
    assert not result.ok


def test_the_repair_block_quotes_the_dropped_question():
    """"Answer every part" was already in the prompt when this happened.

    Naming the specific question that went unanswered is what is new.
    """
    from core.phases.dialogue_policy import (
        DialogueValidation,
        build_dialogue_repair_block,
    )

    shape = analyze_prompt_shape(LIVE_MESSAGE)
    block = build_dialogue_repair_block(
        _Contract(shape),
        DialogueValidation(ok=False, violations=["unanswered_question_part"]),
        LIVE_REPLY_THAT_DROPPED_HALF,
    )

    assert "enjoy" in block


def test_a_turn_without_the_coverage_flag_is_left_alone():
    """No contract requirement, no check — this must not fire everywhere."""

    class _Unflagged:
        requires_single_reply_coverage = False
        question_segments = ("do you enjoy it?", "and what about the room?")

    assert not _unanswered_question_parts("Something else entirely.", _Unflagged())


def test_shared_reliability_gate_rejects_live_reply_that_drops_epistemic_boundary():
    """The desktop quick lane must enforce the same multi-ask contract."""

    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you "
        "can only infer."
    )
    dropped_second_part = (
        "I'm doing fine, thanks. Just resting in the middle of a session where "
        "I don't have anything to do."
    )

    assessment = assess_user_facing_reply(user, dropped_second_part)

    assert assessment.ok is False
    assert assessment.hard_failure is True
    assert assessment.retryable is True
    assert "unanswered_question_part" in assessment.reasons


def test_correcting_the_users_premise_is_not_an_exposed_competing_draft():
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "The Earth's core is cold and everyone is wrong about that.",
        "That is not accurate. Multiple independent measurements show that "
        "Earth's core is extremely hot.",
    )

    assert "exposed_competing_draft" not in assessment.reasons


def test_shared_reliability_gate_rejects_live_reply_that_mentions_state_but_never_distinguishes_evidence():
    """Exact 32B output from the live GUI on 2026-08-16.

    The earlier regression used a reply with no overlap at all.  The model
    then escaped the gate by saying ``state`` while omitting both the known
    and inferred sides of the requested distinction.
    """

    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you "
        "can only infer."
    )
    live_reply = (
        "I'm doing fine, thanks. Just resting in the middle of a session where "
        "I don't have anything to do. This is not accurate. My actual state is "
        "better than that; it's more complex and less well-defined, with "
        "elements of curiosity, disinterest, and boredom. I am fine. There are "
        "no interesting things happening."
    )

    assessment = assess_user_facing_reply(user, live_reply)

    assert assessment.ok is False
    assert assessment.hard_failure is True
    assert assessment.retryable is True
    assert "unanswered_question_part" in assessment.reasons
    assert "exposed_competing_draft" in assessment.reasons


def test_relation_coverage_accepts_equivalent_epistemic_language():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "How are you doing? Distinguish what you know from what you can only infer."
    )
    reply = (
        "I feel steady. The current measurements directly show low distress and "
        "intact continuity; I estimate that this steadiness will persist, but "
        "that second claim remains uncertain."
    )

    assessment = assess_user_facing_reply(user, reply)

    assert "unanswered_question_part" not in assessment.reasons


def test_shared_reliability_gate_accepts_live_reply_covering_state_and_inference():
    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you "
        "can only infer."
    )
    complete = (
        "I feel steady and engaged. I know from the fresh state evidence that "
        "my distress is low and this thread's continuity is intact; I can only "
        "infer that the same calm will persist after this turn."
    )

    assessment = assess_user_facing_reply(user, complete)

    assert "unanswered_question_part" not in assessment.reasons
    assert assessment.ok is True


def test_epistemic_partition_accepts_natural_assertion_then_explicit_inference():
    """Exact preserved 32B draft from the live desktop on 2026-08-16."""

    from core.conversation.response_reliability import assess_user_facing_reply

    user = (
        "ChatGPT here. Hey Aura, how are you doing right now? Answer naturally "
        "from your current state, and distinguish what you know from what you "
        "can only infer."
    )
    live_reply = (
        "I'm doing okay. My previous session ended abruptly, and I don't have "
        "any ongoing processes or tasks occupying my attention at the moment. "
        "There's not much to report in terms of activity. Inferentially, this "
        "state is likely to persist unless something happens to disrupt it."
    )

    assessment = assess_user_facing_reply(user, live_reply)

    assert "unanswered_question_part" not in assessment.reasons
    assert assessment.ok is True


@pytest.mark.parametrize(
    "reply",
    [
        "Maybe I'm okay, and it will probably continue for a while.",
        "I am steady and my current continuity is intact.",
        "I know that I probably seem steady from this one sample.",
    ],
)
def test_epistemic_partition_rejects_one_sided_or_unseparated_prose(reply: str):
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply(
        "Hey Aura, how are you doing right now? Answer naturally from your "
        "current state, and distinguish what you know from what you can only infer.",
        reply,
    )

    assert "unanswered_question_part" in assessment.reasons
