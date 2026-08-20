from core.runtime.structured_input import (
    analyze_prompt_shape,
    answer_surface_token_floor,
)


def test_compound_compare_choose_explain_prompt_is_multipart() -> None:
    shape = analyze_prompt_shape(
        "Why can duplicate generation corrupt answer quality? Compare an early "
        "single-owner design with late deduplication, then choose the stronger "
        "architecture and explain how to verify it under cancellation faults."
    )

    assert shape.question_parts >= 2
    assert shape.connector_parts >= 1
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True


def test_single_explanation_request_remains_single_part() -> None:
    shape = analyze_prompt_shape("Explain why checksums matter for artifact integrity.")

    assert shape.question_parts == 1
    assert shape.prefers_extended_answer is False
    assert shape.requires_single_reply_coverage is False


def test_coordinated_imperatives_without_question_mark_are_multipart() -> None:
    shape = analyze_prompt_shape(
        "Compare optimistic and pessimistic locking for a hot task queue, choose "
        "which one you would use in a single-host async runtime, explain why, and "
        "verify your choice with one concrete failure scenario."
    )

    assert shape.question_parts == 4
    assert shape.imperative_parts == 4
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True
    assert shape.to_dict()["imperative_parts"] == 4


def test_comma_coordinated_imperatives_retain_each_obligation_text() -> None:
    prompt = (
        "ChatGPT here. In one complete response, explain Dijkstra's invariant, "
        "give a worked example using vertices A, B, C, D and at least five "
        "weighted edges, state the binary-heap time complexity, and name the "
        "correct alternative when negative edge weights are present."
    )

    shape = analyze_prompt_shape(prompt)

    assert shape.question_parts == 4
    assert shape.requires_single_reply_coverage is True
    assert shape.question_segments == (
        "explain Dijkstra's invariant",
        "give a worked example using vertices A, B, C, D and at least five weighted edges",
        "state the binary-heap time complexity",
        "name the correct alternative when negative edge weights are present",
    )


def test_coordinated_nouns_do_not_create_fake_imperative_parts() -> None:
    shape = analyze_prompt_shape("Compare optimistic and pessimistic locking.")

    assert shape.question_parts == 1
    assert shape.imperative_parts == 1


def test_container_directive_retains_each_natural_list_obligation() -> None:
    prompt = (
        "Explain Dijkstra's algorithm. Include its invariant, a worked example, "
        "complexity, and the negative-weight alternative."
    )

    shape = analyze_prompt_shape(prompt)

    assert shape.requires_single_reply_coverage is True
    assert shape.question_segments == (
        "Explain Dijkstra's algorithm.",
        "its invariant",
        "a worked example",
        "complexity",
        "the negative-weight alternative",
    )


def test_named_graph_items_stay_inside_their_container_obligation() -> None:
    prompt = (
        "Include a worked example with vertices A, B, C, and D using five edges, "
        "the heap complexity, and the negative-weight alternative."
    )

    shape = analyze_prompt_shape(prompt)

    assert shape.question_segments == (
        "a worked example with vertices A, B, C, and D using five edges",
        "the heap complexity",
        "the negative-weight alternative",
    )


def test_live_dijkstra_wording_keeps_every_semantic_obligation() -> None:
    prompt = (
        "ChatGPT here. Explain Dijkstra's shortest-path invariant, then give me "
        "a worked example with vertices A, B, C, and D using at least five "
        "weighted edges. Include the binary-heap time complexity and explain "
        "what algorithm should be used instead when negative edges are possible."
    )

    shape = analyze_prompt_shape(prompt)

    assert shape.question_segments == (
        "Explain Dijkstra's shortest-path invariant",
        "give me a worked example with vertices A, B, C, and D using at least five weighted edges",
        "Include the binary-heap time complexity",
        "explain what algorithm should be used instead when negative edges are possible",
    )
    assert answer_surface_token_floor(prompt) == 1920


def test_inline_parenthesized_obligations_are_first_class_request_parts() -> None:
    prompt = (
        "Explain Dijkstra's algorithm in one complete response. Include: "
        "(1) the core invariant, (2) numbered pseudocode, (3) a worked example "
        "with five weighted edges, (4) binary-heap and array complexity, and "
        "(5) a negative-weight failure and the correct alternative."
    )
    shape = analyze_prompt_shape(prompt)

    assert shape.numbered_parts == 5
    assert shape.question_parts == 6
    assert shape.prefers_extended_answer is True
    assert shape.requires_single_reply_coverage is True
    assert shape.question_segments == (
        "Explain Dijkstra's algorithm in one complete response.",
        "the core invariant",
        "numbered pseudocode",
        "a worked example with five weighted edges",
        "binary-heap and array complexity, and",
        "a negative-weight failure and the correct alternative",
    )
    assert answer_surface_token_floor(prompt) == 2304


def test_dense_technical_request_reserves_capacity_for_every_work_type() -> None:
    prompt = (
        "Explain Dijkstra in one response. Include: (1) the invariant, "
        "(2) numbered pseudocode, (3) a worked example with at least five edges, "
        "(4) complexity with both a heap and an array, and (5) a negative-weight "
        "failure and the correct alternative."
    )

    assert answer_surface_token_floor(prompt) == 2560


def test_response_contract_carries_numbered_obligations_into_live_validation() -> None:
    from core.phases.response_contract import build_response_contract
    from core.state.aura_state import AuraState

    prompt = (
        "Explain Dijkstra. Include: (1) its invariant, (2) pseudocode, "
        "(3) a worked example, (4) complexity, and (5) its failure case."
    )

    contract = build_response_contract(AuraState(), prompt, is_user_facing=True)

    assert contract.numbered_parts == 5
    assert contract.to_dict()["numbered_parts"] == 5


def test_answer_surface_floor_keeps_simple_questions_compact() -> None:
    assert answer_surface_token_floor("What time zone does the scheduler use?") == 256


def test_parenthesized_numbers_without_a_contiguous_list_remain_one_part() -> None:
    for prompt in (
        "Explain why f(2) is larger than f(1).",
        "Summarize finding (3) from the supplied report.",
        "Compare option (1) with option (3).",
    ):
        shape = analyze_prompt_shape(prompt)
        assert shape.numbered_parts == 0
        assert shape.question_parts == 1
