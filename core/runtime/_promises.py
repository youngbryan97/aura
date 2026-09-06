"""What core.runtime guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "A hole in the spine's sequence is reported by its range rather "
        "than read past as though the log were whole.",
        "checked_by": "tests/test_the_kept_log_attacked.py::"
        "test_a_hole_in_the_sequence_is_named_by_its_range",
        "if_it_fails": "integrity() names the missing range; nothing downstream "
        "treats the log as replayable",
    },
    {
        "it": "An event may not name a causal parent that has not happened, so "
        "lineage cannot be fabricated after the fact.",
        "checked_by": "tests/test_the_kept_log_attacked.py::"
        "test_an_event_may_not_name_a_parent_that_has_not_happened",
        "if_it_fails": "LineageBroken is raised at append; the event is refused "
        "rather than written with a parent nobody can find",
    },
    {
        "it": "A failure that is a decision rather than a fault is never retried, "
        "however many attempts a policy declared.",
        "checked_by": "tests/test_what_must_never_be_retried.py::"
        "test_a_cancelled_turn_is_never_restarted_by_a_retry_loop",
        "if_it_fails": "how_to_treat returns AGAIN for a refusal and a governance "
        "denial becomes a loop; visible as a repeated refusal in the turn log",
    },
    {
        "it": "A required call that fails is not quietly replaced by a fallback "
        "answer.",
        "checked_by": "tests/test_how_a_call_is_made.py::"
        "test_a_required_call_does_not_quietly_fall_back",
        "if_it_fails": "the outcome carries the failure and the turn cannot be "
        "completed; recorded in how_the_calls_are_made()",
    },
    {
        "it": "A guardrail that cannot run decides nothing, and what that means "
        "for the answer is the rail's own declaration.",
        "checked_by": "tests/test_what_an_answer_must_pass.py::"
        "test_a_rail_that_does_not_say_refuses_when_it_cannot_run",
        "if_it_fails": "an undeclared rail would fail open; the report's "
        "fail_open_rails names every rail that may",
    },
    {
        "it": "Two resources are acquired in one global order, so two callers "
        "wanting the same pair cannot deadlock.",
        "checked_by": "tests/test_claiming_more_than_one.py::"
        "test_two_callers_wanting_the_same_pair_do_not_deadlock",
        "if_it_fails": "the claim never returns; who_holds_what() shows one holder "
        "per resource and both waiting",
    },
    {
        "it": "A published message cannot also name a recipient, so a handler "
        "never has to infer whether it was addressed.",
        "checked_by": "tests/test_what_a_message_carries.py::"
        "test_a_published_message_cannot_also_name_a_recipient",
        "if_it_fails": "the envelope is refused at construction; without it one "
        "message shape means two different things to two readers",
    },
    {
        "it": "Only one turn owns the answer at a time, and a second cannot begin "
        "while one is active.",
        "checked_by": "tests/test_whose_turn_it_is.py::"
        "test_a_second_turn_cannot_begin_while_one_is_active",
        "if_it_fails": "two turns write the same reply slot and the person sees "
        "whichever finished last",
    },
    {
        "it": "Spending past the end of a budget is refused rather than raised, "
        "so a caller decides what to do about it.",
        "checked_by": "tests/test_what_is_left_to_spend.py::"
        "test_spending_past_the_end_is_refused_and_not_raised",
        "if_it_fails": "the refusal is recorded with what wanted what; a raise "
        "here would make running out of budget look like a fault",
    },
    {
        "it": "A status transition that is not legal is refused and says why, "
        "rather than being applied and reported as fine.",
        "checked_by": "tests/test_what_a_status_may_become.py::"
        "test_an_illegal_move_is_refused_and_says_why",
        "if_it_fails": "the table's outcome says REFUSED with the reason; a "
        "silent apply would leave a workflow in a state nothing can leave",
    },
    {
        "it": "Every background task owner declares when its work may be cancelled "
        "and how long its drain may take.",
        "checked_by": "tests/test_how_a_task_should_end.py::"
        "test_an_owner_that_never_declared_is_named_not_defaulted_silently",
        "if_it_fails": "owners_that_have_not_said() names the owner; the runtime "
        "default applies and an orphan is not treated as a defect",
    },
)
