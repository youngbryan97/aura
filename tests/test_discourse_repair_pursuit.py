from core.conversation.discourse_repair_pursuit import (
    apply_repair_pursuit_to_history,
    build_repair_pursuit,
    question_focus,
)


def test_question_focus_distinguishes_method_from_state_and_quantity():
    assert question_focus("How did you know that?").kind == "mechanism_or_manner"
    assert question_focus("How are you right now?").kind == "state_or_degree"
    assert question_focus("How many files changed?").kind == "quantity"


def test_repeated_how_pursuit_reopens_method_focus():
    contract = build_repair_pursuit(
        "Well then how did you know",
        [
            {
                "user": "How’d you know that",
                "aura": (
                    "I didn’t know it in the way you might mean; I did not have "
                    "a flash of insight."
                ),
            }
        ],
    )

    assert contract.active is True
    assert contract.focus_kind == "mechanism_or_manner"
    assert contract.focus_terms == ("know",)
    assert contract.relation == "user_reopens_same_interrogative_focus"


def test_same_interrogative_with_new_subject_is_not_a_repair_pursuit():
    contract = build_repair_pursuit(
        "Well then how do birds fly?",
        [{"user": "How do planes fly?", "aura": "By generating lift."}],
    )

    assert contract.active is False


def test_unmarked_related_question_is_not_silently_reclassified():
    contract = build_repair_pursuit(
        "How did you know the second result?",
        [
            {
                "user": "How did you know the first result?",
                "aura": "I calculated it from the supplied values.",
            }
        ],
    )

    assert contract.active is False


def test_repair_pursuit_withdraws_only_the_rejected_assistant_turn():
    exchanges = [
        {"role": "user", "content": "Who wrote Solaris?"},
        {"role": "assistant", "content": "Stanisław Lem."},
        {"role": "user", "content": "How’d you know that?"},
        {
            "role": "assistant",
            "content": "I didn’t know it through a flash of hidden intuition.",
        },
    ]
    contract = build_repair_pursuit(
        "Well then how did you know?",
        [
            {
                "user": "How’d you know that?",
                "aura": "I didn’t know it through a flash of hidden intuition.",
            }
        ],
    )

    filtered = apply_repair_pursuit_to_history(exchanges, contract)

    assert filtered == exchanges[:-1]
