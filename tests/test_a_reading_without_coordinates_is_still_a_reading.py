"""A reader that says WHAT without saying WHERE must not blind her.

Everything she does with a screen is asked of the arrangement: whether her
move changed anything, what the rule of this world is, what a move would lead
to, how near the goal she is. So a reading that comes back as words with no
coordinates — one backend's accessibility dump, one OCR path, one page that
answers in prose — used to produce an arrangement of nothing, and an
arrangement of nothing equals the one before it. Every move then graded
"nothing changed", on a screen that was changing perfectly well.
"""

from __future__ import annotations

from core.perception.where_it_responds import what_is_there


def _read(text: str) -> dict[str, object]:
    """A reading that carries words and no places, as some readers do."""
    return {"ok": True, "text": text, "layout": [], "bounds": []}


def test_words_without_places_still_arrive_as_an_arrangement():
    seen = what_is_there(_read("board 2"), None)
    assert seen.occupied() > 0, "a reading with words in it came back empty"
    assert "board" in seen.as_text()


def test_two_readings_that_differ_are_not_equal():
    before = what_is_there(_read("board 2"), None)
    after = what_is_there(_read("board 3"), None)
    assert before.as_text() != after.as_text(), (
        "a changed screen read as unchanged, which grades every move as stalled"
    )


def test_a_grid_printed_as_lines_comes_back_as_that_grid():
    seen = what_is_there(_read("2 4 8 2\n16 32 64 4\n. . 2 8\n2 4 . ."), None)
    assert (seen.rows, seen.columns) == (4, 4)
    assert seen.as_text().splitlines()[1].split() == ["16", "32", "64", "4"]


def test_columns_come_from_counting_not_from_character_positions():
    """Text is not drawn at an even pitch, so distance along a line lies."""
    # Row one is nine characters wide and row two is four, and both hold four
    # values. Binned by where their characters sit, they land in six columns.
    seen = what_is_there(_read("128 256 512 8\n2 4 8 2"), None)
    assert seen.columns == 4


def test_an_empty_reading_stays_empty():
    assert what_is_there(_read(""), None).occupied() == 0
    assert what_is_there(_read("   \n\n  "), None).occupied() == 0


def test_places_are_preferred_when_the_reader_gives_them():
    """The fallback is for a reader that says nothing, not a poor reader."""
    reading = {
        "ok": True,
        "text": "this text is ignored",
        "layout": [{"text": "real", "center_x": 0.5, "center_y": 0.5}],
        "bounds": [],
    }
    seen = what_is_there(reading, None)
    assert seen.as_text().strip() == "real"


def test_a_reading_without_places_can_be_cropped_to_the_thing_in_it():
    """The whole point: what comes back is something the rest can work on."""
    from core.perception.the_thing_itself import the_thing_itself

    seen = what_is_there(_read("2 4 8 2\n16 32 64 4\n2 8 4 2\n4 2 8 16"), None)
    thing = the_thing_itself(seen)
    assert thing.occupied() == 16
