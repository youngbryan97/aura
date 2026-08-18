from __future__ import annotations

import pytest

QUESTION = (
    "Explain Dijkstra's shortest-path algorithm in one complete response. Include: "
    "(1) the core invariant, (2) numbered pseudocode, (3) a worked example on a "
    "graph with vertices A, B, C, D and at least five weighted edges, (4) time "
    "complexity with both a binary heap and an array, and (5) one failure case "
    "involving negative weights and the correct alternative."
)

COMPLETE = """
1. Core invariant: when Dijkstra removes the minimum-distance vertex from the queue,
that distance is final because every edge weight is non-negative.
2. Numbered pseudocode: 1) set dist[source] = 0 and every other distance to infinity;
2) push the source; 3) pop the smallest tentative distance; 4) relax every outgoing
edge; 5) repeat until the queue is empty.
3. Worked example: use A-B = 4, A-C = 1, C-B = 2, B-D = 1, and C-D = 5. Starting
at A, relaxing C gives 1, then C improves B to 3 and D to 6, and B improves D to 4.
The final distances are A=0, C=1, B=3, D=4.
4. Complexity: a binary heap takes O((V + E) log V); an array takes O(V^2 + E),
usually written O(V^2) for a dense graph.
5. Negative weights invalidate the final-distance invariant. Use Bellman-Ford when
negative edges are possible; it also detects reachable negative cycles.
""".strip()

LOSSY = """
1. Core invariant: a removed minimum-distance vertex is final when weights are non-negative.
2. Pseudocode: initialize distances and repeatedly relax the nearest vertex.
3. Worked example: A-B = 1, B-C = 1, C-D = 1, so the distance from A to D is 3.
4. A heap takes O((V + E) log V), while an array takes O(V^2).
5. Dijkstra fails on negative weights; use Bellman-Ford instead.
""".strip()


def test_transform_cannot_trade_away_a_quantified_request_obligation() -> None:
    from core.conversation.surface_disposition import repair_is_an_improvement

    assert repair_is_an_improvement(COMPLETE, COMPLETE, QUESTION)
    assert not repair_is_an_improvement(COMPLETE, LOSSY, QUESTION)


@pytest.mark.asyncio
async def test_dialogue_cleaner_cannot_own_a_semantically_weaker_answer(monkeypatch) -> None:
    from core.phases import dialogue_policy
    from core.runtime.structured_input import analyze_prompt_shape

    contract = analyze_prompt_shape(QUESTION)
    real_validate = dialogue_policy.validate_dialogue_response

    def forced_validation(text, candidate_contract, state=None):
        if text == COMPLETE:
            return dialogue_policy.DialogueValidation(
                ok=False,
                violations=["generic_assistant_language"],
            )
        if text == LOSSY:
            return dialogue_policy.DialogueValidation(ok=True, violations=[])
        return real_validate(text, candidate_contract, state)

    monkeypatch.setattr(dialogue_policy, "validate_dialogue_response", forced_validation)
    monkeypatch.setattr(dialogue_policy, "repair_dialogue_surface", lambda *_: LOSSY)

    text, validation, retried = await dialogue_policy.enforce_dialogue_contract(
        COMPLETE,
        contract,
        user_message=QUESTION,
    )

    assert text == COMPLETE
    assert validation.violations == ["generic_assistant_language"]
    assert retried is False


def test_sovereign_surface_shaper_keeps_authored_answer_when_cleanup_loses_work(
    monkeypatch,
) -> None:
    import core.synthesis as synthesis
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    monkeypatch.setattr(synthesis, "cure_personality_leak", lambda _text: LOSSY)
    monkeypatch.setattr(
        synthesis,
        "stabilize_user_facing_response",
        lambda text, _question: text,
    )

    assert UnitaryResponsePhase._shape_user_facing_response(COMPLETE, QUESTION) == COMPLETE
