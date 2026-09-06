"""What core.cognition guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "Every destination a new term can go to has something that installs "
        "into it, so inventing a term is not the same as writing a word down.",
        "checked_by": "tests/test_where_a_term_can_go.py::"
        "test_every_destination_has_at_least_one_action_installing_into_it",
        "if_it_fails": "what_is_not_really_a_destination() names it; the term is "
        "recorded and nothing ever reads it",
    },
    {
        "it": "Every part a destination declares resolves to something that is "
        "actually in the tree.",
        "checked_by": "tests/test_where_a_term_can_go.py::"
        "test_each_part_resolves_to_something_that_is_there",
        "if_it_fails": "the destination names a module or attribute that is not "
        "there, and installing into it fails at the moment it matters",
    },
    {
        "it": "The same improvement rule over the same recorded life gives the "
        "same number, so a measured gain is not an artefact of replay order.",
        "checked_by": "tests/test_does_improving_compound.py::"
        "test_the_same_rule_over_the_same_record_gives_the_same_number",
        "if_it_fails": "two runs disagree about whether she improved and neither "
        "can be preferred",
    },
    {
        "it": "A fall inside the measured noise is not reported as an "
        "improvement, and the noise is measured rather than assumed.",
        "checked_by": "tests/test_does_improving_compound.py::"
        "test_a_fall_inside_the_noise_is_not_an_improvement",
        "if_it_fails": "a rule is promoted on a difference the record cannot "
        "distinguish from chance",
    },
)
