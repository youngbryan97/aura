"""What core.skills guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "Every skill gives back the same shape, so a caller reads one "
        "result rather than learning each skill's own spelling.",
        "checked_by": "tests/test_what_every_skill_gives_back.py::"
        "test_a_missing_required_field_is_named",
        "if_it_fails": "check_a_result names the field; a caller reading the "
        "shared shape gets None where a skill used its own key",
    },
    {
        "it": "The shared result requires only what every skill actually keeps, "
        "so declaring it does not force a skill to invent a field.",
        "checked_by": "tests/test_what_every_skill_gives_back.py::"
        "test_the_shared_result_requires_only_what_every_skill_keeps",
        "if_it_fails": "skills fill a required field with a placeholder, and the "
        "shape stops meaning anything",
    },
    {
        "it": "Code written by a model is checked for the async mistakes that "
        "make it look right and behave wrong before it is served.",
        "checked_by": "tests/test_is_this_async_code_correct.py::"
        "test_every_defect_from_the_live_run_is_found",
        "if_it_fails": "is_it_correct reports the mistakes; unchecked, delivery "
        "succeeds and semantic correctness is zero",
    },
)
