"""What core.perception guarantees, and the test that catches each breaking."""
from __future__ import annotations

THE_PROMISES: tuple[dict[str, str], ...] = (
    {
        "it": "A lattice worked out from where things happen is not rebuilt while "
        "the places it already holds still fit, so the frame does not move "
        "under her on the turn she learns something.",
        "checked_by": "tests/test_a_frame_that_fits_does_not_move.py::"
        "test_a_place_that_already_fits_leaves_the_lines_where_they_are",
        "if_it_fails": "two readings either side of that turn are in different "
        "frames however alike they look, and neither can be compared",
    },
    {
        "it": "Places that do not fit the frame she holds are still evidence of a "
        "new shape, so a frame can change when the world does.",
        "checked_by": "tests/test_a_frame_that_fits_does_not_move.py::"
        "test_places_that_do_not_fit_are_still_evidence_of_a_new_shape",
        "if_it_fails": "a frame that can never change is not a frame, it is an "
        "assumption, and every later reading is cropped to it",
    },
    {
        "it": "A set of places is built from only once it has stopped growing for "
        "longer than it has ever gone between growing.",
        "checked_by": "tests/test_a_frame_that_fits_does_not_move.py::"
        "test_more_lines_than_she_holds_is_a_bigger_view_of_the_same_thing",
        "if_it_fails": "a four-by-four board becomes a three-by-three lattice and "
        "every reading after it is scored against a board with a column gone",
    },
)
